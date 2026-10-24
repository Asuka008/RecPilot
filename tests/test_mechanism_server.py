"""Server-only CPU integration with synthetic frozen states, without training."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from content_hstu.artifacts import prepare_content
from content_hstu.model import TitleHSTU
from diagnostics.tail_admission import source_hashes
from scripts.audit_recommendation_mechanisms import ROOT, run, title_scores
from scripts.compare_title_validation import digest
from scripts.evaluate_selected_title import file_digest
from src.baselines import ItemCF
from src.data.dataset import SASRecEvalDataset
from src.evaluation import evaluate
from src.microlens_training import evaluate_hstu
from src.models.id_hstu import IDHSTU
from src.run_baselines import load_data
from tests.content_fixture import write_cache


class MechanismServerTest(unittest.TestCase):
    def test_title_mean_matches_hand_computation_and_ignores_padding(self):
        table = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]])
        ids = torch.tensor([[0, 1, 2], [0, 1, 3]])
        scores, zero = title_scores(table, ids, ids != 0)
        expected = torch.tensor([[2**-.5, 2**-.5, -2**-.5], [0., 0., 0.]])
        torch.testing.assert_close(scores, expected)
        self.assertEqual(zero, 1)

    def make_fixture(self, folder):
        """Cover every frequency and history slice using tiny synthetic records."""
        data = folder / "data"
        data.mkdir()
        records = [{"user_id": 101, "train": [1]*25, "val": 4, "test": 7},
                   {"user_id": 102, "train": [2]*12, "val": 1, "test": 7},
                   {"user_id": 103, "train": [3]*3 + [5]*3, "val": 2, "test": 8},
                   {"user_id": 104, "train": [6]*2, "val": 3, "test": 8}]
        (data / "sequences.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (data / "item_mapping.json").write_text(json.dumps({"padding_idx": 0,
            "item2idx": {str(i): i for i in range(1, 9)}, "idx2item": [None] + list(range(1, 9))}))
        (data / "metadata.json").write_text(json.dumps({"counts": {"users": 4, "items": 8,
            "interactions": sum(len(r["train"]) + 2 for r in records)}, "config": {"min_seq_len": 4}}))
        cache, features_sha, _ = write_cache(folder, num_items=8, data_dir=data)
        records, n, checksums = load_data(data)
        config = json.loads((ROOT / "configs/hstu-microlens-debug.json").read_text())
        config.update(mode="full", data_dir=str(data), output_dir=str(folder / "id"),
                      train_users=None, eval_users=None, epochs=1, eval_batch_size=2)
        config["model"].update(embedding_dim=64, sequence_length=32)
        reference_path = folder / "id.json"
        reference_path.write_text(json.dumps(config))
        recipe = json.loads((ROOT / "configs/recommendation-mechanism-audit-server.json").read_text())
        recipe["output_dir"] = str(folder / "diagnostic")
        shared = source_hashes(ROOT, ("src",))
        content_sources = source_hashes(ROOT, ("content_hstu", "title_features"))
        users = [r["user_id"] for r in records]
        identity = {"data_sha256": checksums, "train_users_sha256": digest(users), "eval_users_sha256": digest(users)}
        dataset = SASRecEvalDataset(data / "sequences.jsonl", 32, "val")
        for name in ("id", "title-init", "title-fusion"):
            c = copy.deepcopy(config)
            c["output_dir"] = str(folder / name)
            p = {**identity, "data_id": digest(identity), "source_sha256": shared, "source_id": digest(shared)}
            report = {"status": "PASS", "action": "train", "config": c, "runtime": {}, "checksums_unchanged": True,
                      "data": {"catalog_items": n, "train_users": 4, "evaluation_users": 4, "full_data_users": 4}}
            torch.manual_seed(42)
            if name == "id":
                model = IDHSTU(num_items=n, **c["model"])
            else:
                c.update(representation=name, reference_config=str(reference_path), feature_cache=str(cache),
                         features_sha256=features_sha, projection="train-item-pca-v1", fusion_weight=0 if name == "title-init" else .5)
                table, _, content, _ = prepare_content(c, checksums, n, records)
                p.update(content_source_sha256=content_sources, shared_source_id=digest(shared),
                         source_id=digest({"shared": shared, "content": content_sources}),
                         content_id=content["content_id"], content=content, reference_config=config,
                         reference_config_sha256=digest(config))
                report.update(content=content, frozen_content_unchanged=True)
                model = TitleHSTU(num_items=n, semantic=torch.from_numpy(table), representation=name,
                                  fusion_weight=c["fusion_weight"], **c["model"])
            p["config_sha256"] = digest(c)
            report["provenance"] = p
            evaluation = evaluate_hstu(model, dataset, list(range(4)), c, torch.device("cpu"), records)
            report["history"] = [{"epoch": 1, "validation": evaluation}]
            report["selection"] = {"metric": "NDCG@10", "best_epoch": 1, "split": "val",
                                   "best_score": evaluation["groups_by_target_train_frequency"]["all"]["metrics"]["NDCG@10"]}
            run_dir = folder / name
            run_dir.mkdir()
            checkpoint = run_dir / "best.pt"
            # These untrained fixtures test admission/inference, not training quality.
            torch.save({"schema_version": 1, "config": c, "epoch": 1, "provenance": p,
                        "model_state_dict": model.state_dict()}, checkpoint)
            report["checkpoints"] = {"best.pt": {"sha256": file_digest(checkpoint)}}
            path = run_dir / "results.json"
            path.write_text(json.dumps(report))
            config_path = folder / f"{name}.json"
            config_path.write_text(json.dumps(c))
            recipe["reports"][name], recipe["training_configs"][name] = str(path), str(config_path)
        bc = {"ks": [10, 20], "itemcf_neighbors": 3}
        model = ItemCF(records, n, 3)
        frequency = Counter(i for r in records for i in r["train"])
        baseline = {"config": bc, "split": "val", "data_sha256": checksums,
                    "models": {"itemcf": evaluate(model, records, "val", [10, 20], frequency)}}
        del baseline["models"]["itemcf"]["groups_by_history_length"]
        for key, value in (("baseline_config", bc), ("baseline_report", baseline)):
            path = folder / f"{key}.json"
            path.write_text(json.dumps(value))
            recipe[key] = str(path)
        path = folder / "audit.json"
        path.write_text(json.dumps(recipe))
        return path, recipe

    def test_complete_inference_audit_and_numerical_anchor_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path, recipe = self.make_fixture(folder)
            output = io.StringIO()
            with redirect_stdout(output):
                try:
                    report, directory = run(path)
                except SystemExit:
                    self.fail(output.getvalue())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["variants"]), 8)
            self.assertEqual(report["data"]["validation_users"], 4)
            self.assertTrue(report["checks"]["title-fusion_validation_reproduced"])
            self.assertEqual(report["anchor_coverage"]["itemcf"]["computed_without_historical_anchor"],
                             ["groups_by_history_length"])
            with np.load(directory / "per_user_diagnostics.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["user_id"], [101, 102, 103, 104])
                self.assertFalse(any("test" in name for name in arrays.files))
            baseline_path = Path(recipe["baseline_report"])
            baseline = json.loads(baseline_path.read_text())
            baseline["models"]["itemcf"]["groups_by_target_train_frequency"]["all"]["metrics"]["Recall@20"] = .123
            baseline_path.write_text(json.dumps(baseline))
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                run(path)
            failed = sorted(Path(recipe["output_dir"]).glob("*/results.json"))[-1]
            failure = json.loads(failed.read_text())
            self.assertEqual(failure["status"], "FAIL")
            self.assertIn("anchor differs", failure["error"]["message"])


if __name__ == "__main__":
    unittest.main()
