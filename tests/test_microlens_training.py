"""Server CPU tests on tiny synthetic files, including end-to-end continuation."""

from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src.data.dataset import SASRecEvalDataset, SASRecTrainDataset
from src.evaluation import evaluate
from src.microlens_training import evaluate_hstu, full_catalog_loss
from src.models.id_hstu import IDHSTU
from src.run_hstu_microlens import run


ROOT = Path(__file__).resolve().parents[1]
RECORDS = [{"user_id": 1, "train": [1, 2, 3], "val": 4, "test": 5},
           {"user_id": 2, "train": [2, 3], "val": 5, "test": 6}]


def write_data(root):
    """Use real processed-file schemas without the MicroLens dataset."""
    (root / "sequences.jsonl").write_text("".join(json.dumps(record) + "\n" for record in RECORDS))
    (root / "item_mapping.json").write_text(json.dumps({
        "padding_idx": 0, "item2idx": {str(i): i for i in range(1, 7)},
        "idx2item": [None, 1, 2, 3, 4, 5, 6],
    }))
    (root / "metadata.json").write_text(json.dumps({
        "counts": {"users": 2, "items": 6, "interactions": 9}, "config": {"min_seq_len": 4},
    }))


class MicroLensTrainingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_data(self.root)
        self.config = json.loads((ROOT / "configs/hstu-microlens-debug.json").read_text())
        self.config.update(mode="full", data_dir=str(self.root), output_dir=str(self.root / "runs"),
                           train_users=None, eval_users=None, ks=[1, 2], selection_metric="NDCG@1",
                           batch_size=2, eval_batch_size=2)
        self.config["model"].update(sequence_length=2, dropout=0.2)

    def test_objective_aligns_shifted_labels_and_ignores_padding(self):
        model = IDHSTU(num_items=6, **self.config["model"])
        model.eval()
        dataset = SASRecTrainDataset(self.root / "sequences.jsonl", 2)
        batch = {key: torch.stack([dataset[i][key] for i in range(2)]) for key in dataset[0]}
        mask = batch["padding_mask"]
        logits = model.score_candidates(model(batch["input_ids"], mask)[mask], torch.arange(1, 7)) / 0.2
        expected = F.cross_entropy(logits, batch["target_ids"][mask] - 1)
        torch.testing.assert_close(full_catalog_loss(model, batch, 0.2), expected)
        batch["target_ids"][~mask] = 999
        torch.testing.assert_close(full_catalog_loss(model, batch, 0.2), expected)
        batch["target_ids"][mask] = 0
        with self.assertRaises(ValueError):
            full_catalog_loss(model, batch, 0.2)

    def test_batched_evaluation_matches_shared_protocol_and_full_history_filter(self):
        class ConstantScorer:
            num_items = 6

            def eval(self):
                return self

            def __call__(self, ids, mask):
                return torch.zeros(*ids.shape, 1)

            def score_candidates(self, queries, candidates):
                return torch.zeros(len(queries), len(candidates))

            def score(self, history):
                return np.zeros(7)

        scorer = ConstantScorer()
        frequency = Counter(item for record in RECORDS for item in record["train"])
        for split in ("val", "test"):
            dataset = SASRecEvalDataset(self.root / "sequences.jsonl", 2, split)
            actual = evaluate_hstu(scorer, dataset, [0, 1], self.config, torch.device("cpu"), RECORDS)
            self.assertEqual(actual, evaluate(scorer, RECORDS, split, [1, 2], frequency))
            self.assertEqual(actual["groups_by_target_train_frequency"]["0"]["users"], 2)
            self.assertEqual(actual["groups_by_history_length"]["1-5"]["users"], 2)
        dataset = SASRecEvalDataset(self.root / "sequences.jsonl", 2, "val")
        actual = evaluate_hstu(scorer, dataset, [0], self.config, torch.device("cpu"), RECORDS)
        self.assertEqual(actual["groups_by_target_train_frequency"]["all"]["metrics"]["Recall@1"], 1.0)
        dataset.records[0]["val"] = 1
        with self.assertRaisesRegex(ValueError, "target is in filtered history"):
            evaluate_hstu(scorer, dataset, [0], self.config, torch.device("cpu"), RECORDS)

    def launch(self, resume=None, stop=None, action="train", checkpoint=None, split="val"):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                run(SimpleNamespace(config=config_path, action=action, resume=resume,
                                    stop_after_epoch=stop, checkpoint=checkpoint, split=split))
            except SystemExit:
                self.fail(output.getvalue())
        paths = sorted((self.root / "runs").glob("*/results.json"))
        result = json.loads(paths[-1].read_text())
        self.assertNotEqual(result["status"], "FAIL", output.getvalue())
        return paths[-1].parent, result

    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_same(a, b)
        else:
            self.assertEqual(left, right)

    def test_epoch_boundary_resume_matches_uninterrupted_training_and_explicit_evaluation(self):
        full_dir, full_report = self.launch()
        first_dir, first_report = self.launch(stop=1)
        self.assertEqual(first_report["status"], "PARTIAL")
        resumed_dir, resumed_report = self.launch(resume=first_dir / "last.pt")
        self.assertEqual(resumed_report["status"], "PASS")
        self.assertEqual(resumed_report["selection"], full_report["selection"])
        self.assertTrue(resumed_report["checks"]["checkpoint_output_exact"])
        self.assertTrue(resumed_report["checksums_unchanged"])
        full = torch.load(full_dir / "last.pt", weights_only=True)
        resumed = torch.load(resumed_dir / "last.pt", weights_only=True)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "epoch", "step"):
            self.assert_same(full[key], resumed[key])
        for a, b in zip(full_report["history"], resumed_report["history"]):
            for key in ("training", "validation", "probe_loss"):
                self.assert_same(a[key], b[key])
        _, evaluation = self.launch(action="evaluate", checkpoint=resumed_dir / "best.pt", split="test")
        self.assertEqual(evaluation["evaluation"]["split"], "test")
        self.assertEqual(evaluation["evaluation"]["groups_by_target_train_frequency"]["all"]["users"], 2)

    def test_smoke_config_cannot_evaluate_test(self):
        self.config.update(mode="smoke", train_users=2, eval_users=2)
        with self.assertRaisesRegex(ValueError, "cannot evaluate test"):
            self.launch(action="evaluate", checkpoint=self.root / "absent.pt", split="test")


if __name__ == "__main__":
    unittest.main()
