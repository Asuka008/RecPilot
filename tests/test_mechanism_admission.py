"""Standard-library checks of paired artifacts and diagnostic boundaries."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from diagnostics.tail_admission import (admit, source_hashes, validate_recipe,
                                       verify_payload, verify_unchanged)
from scripts.compare_title_validation import ROOT, digest
from scripts.evaluate_selected_title import file_digest
from tests import test_title_validation_comparison as comparison_fixture


class MechanismAdmissionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        fixture = comparison_fixture.TitleValidationComparisonTest()
        fixture.setUp()
        self.reports = fixture.reports
        self.recipe = json.loads((ROOT / "configs/recommendation-mechanism-audit-server.json").read_text())
        for folder in ("src", "content_hstu", "title_features", "data", "cache"):
            (self.root / folder).mkdir()
        for folder in ("src", "content_hstu", "title_features"):
            (self.root / folder / "model.py").write_text('"""Synthetic source."""\n')
        (self.root / "data/file").write_text("synthetic data")
        (self.root / "titles.csv").write_text("synthetic titles")
        (self.root / "cache/features.npy").write_bytes(b"synthetic cache; no numerical loading in file admission")
        inputs = {"file": file_digest(self.root / "data/file"), "titles_csv": file_digest(self.root / "titles.csv")}
        manifest = {"config": {"titles_csv": "titles.csv", "data_dir": "data"}, "input_sha256": inputs}
        (self.root / "cache/results.json").write_text(json.dumps(manifest))
        shared = source_hashes(self.root, ("src",))
        content_sources = source_hashes(self.root, ("content_hstu", "title_features"))
        content = {"manifest_sha256": file_digest(self.root / "cache/results.json"), "content_id": "content"}
        for name, report in self.reports.items():
            report["config"]["data_dir"] = str(self.root / "data")
            provenance = report["provenance"]
            provenance.update(data_sha256={"file": inputs["file"]}, source_sha256=shared, source_id=digest(shared))
            if name != "id":
                report["config"].update(reference_config=str(self.root / "id.json"), feature_cache=str(self.root / "cache"),
                                        features_sha256=file_digest(self.root / "cache/features.npy"))
                provenance.update(content_source_sha256=content_sources, shared_source_id=digest(shared),
                                  source_id=digest({"shared": shared, "content": content_sources}),
                                  content=content, reference_config=self.reports["id"]["config"],
                                  reference_config_sha256=digest(self.reports["id"]["config"]))
                report["content"] = content
            provenance["config_sha256"] = digest(report["config"])
            folder = self.root / name
            folder.mkdir()
            (folder / "best.pt").write_bytes(b"synthetic checkpoint")
            report["checkpoints"] = {"best.pt": {"sha256": file_digest(folder / "best.pt")}}
            (folder / "results.json").write_text(json.dumps(report))
            (self.root / f"{name}.json").write_text(json.dumps(report["config"]))
            self.recipe["reports"][name] = f"{name}/results.json"
            self.recipe["training_configs"][name] = f"{name}.json"
        bc = {"ks": [10, 20], "itemcf_neighbors": 100}
        baseline = {"config": bc, "split": "val", "data_sha256": {"file": inputs["file"]},
                    "models": {"itemcf": {"split": "val"}}}
        for key, value in (("baseline_config", bc), ("baseline_report", baseline)):
            self.recipe[key] = key + ".json"
            (self.root / self.recipe[key]).write_text(json.dumps(value))

    def test_admission_tracks_all_models_and_inputs(self):
        reports, _, checkpoints, tracked = admit(self.root, self.recipe)
        self.assertEqual(set(checkpoints), set(self.reports))
        self.assertEqual(reports["id"]["selection"]["best_epoch"], 1)
        verify_unchanged(tracked)
        (self.root / "titles.csv").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            verify_unchanged(tracked)

    def test_test_split_unknown_keys_and_undeclared_cutoffs_are_rejected(self):
        for changed in ({"split": "test"}, {"unknown": True}, {"rank_cutoffs": [10, 20]},
                        {"permutation_seed": True}, {"anchor_atol": float("nan")}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_recipe(self.recipe | changed)

    def test_corrupt_checkpoint_cache_source_or_current_config_is_rejected(self):
        for name in ("title-fusion/best.pt", "cache/features.npy", "src/model.py", "id.json", "data/file"):
            path = self.root / name
            before = path.read_bytes()
            path.write_text("{}")
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    admit(self.root, self.recipe)
            finally:
                path.write_bytes(before)

    def test_new_model_source_file_is_not_ignored(self):
        (self.root / "src/extra.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "source inventory"):
            admit(self.root, self.recipe)

    def test_baseline_test_report_is_rejected(self):
        path = self.root / self.recipe["baseline_report"]
        baseline = json.loads(path.read_text())
        baseline["split"] = "test"
        path.write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, "ItemCF"):
            admit(self.root, self.recipe)

    def test_payload_rejects_wrong_epoch_or_source_or_data(self):
        report = self.reports["id"]
        payload = {"schema_version": 1, "config": report["config"], "epoch": 1,
                   "provenance": copy.deepcopy(report["provenance"])}
        verify_payload(payload, report)
        for field in ("source_id", "data_id", "eval_users_sha256"):
            changed = copy.deepcopy(payload)
            changed["provenance"][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_payload(changed, report)
        with self.assertRaises(ValueError):
            verify_payload(payload | {"epoch": 2}, report)


if __name__ == "__main__":
    unittest.main()
