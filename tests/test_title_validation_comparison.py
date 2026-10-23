"""Pure-Python report checks with synthetic metrics and identities."""

import copy
import json
import unittest

from scripts.compare_title_validation import ROOT, AXES, compare_reports, digest


class TitleValidationComparisonTest(unittest.TestCase):
    def setUp(self):
        config = json.loads((ROOT / "configs/hstu-microlens-train-server.json").read_text())
        config["epochs"] = 2
        validation = {axis: {group: {"users": 10, "metrics": {"NDCG@10": 0.03, "Recall@20": 0.1}}
                             for group in groups} for axis, groups in AXES.items()}
        validation["split"] = "val"
        baseline = {"status": "PASS", "action": "train", "config": config, "checksums_unchanged": True,
                    "history": [{"epoch": e, "validation": copy.deepcopy(validation)} for e in (1, 2)],
                    "selection": {"metric": "NDCG@10", "best_score": 0.03, "best_epoch": 1, "split": "val"},
                    "runtime": {}, "data": {"catalog_items": 20, "train_users": 10, "evaluation_users": 10},
                    "provenance": {"config_sha256": digest(config), "data_id": "data", "data_sha256": {"file": "hash"},
                                   "train_users_sha256": "train", "eval_users_sha256": "val",
                                   "source_sha256": {"src/model.py": "source"}, "source_id": "shared"}}
        self.reports = {"id": baseline}
        for mode in ("title-init", "title-fusion"):
            report = copy.deepcopy(baseline)
            c = json.loads((ROOT / f"configs/{mode}-train-server.json").read_text())
            c["epochs"] = 2
            report.update(config=c, frozen_content_unchanged=True)
            report["provenance"].update(config_sha256=digest(c), reference_config=config,
                                         reference_config_sha256=digest(config), shared_source_id="shared",
                                         content_id="content", source_id="combined")
            self.reports[mode] = report

    def test_matched_reports_return_selected_validation(self):
        result = compare_reports(self.reports)
        self.assertEqual(result["title-init"]["selection"]["best_epoch"], 1)
        self.assertEqual(result["id"]["validation"]["split"], "val")

    def test_incomplete_or_failed_runs_are_rejected(self):
        for field, value in (("status", "FAIL"), ("action", "evaluate"), ("checksums_unchanged", False),
                             ("history", self.reports["id"]["history"][:1])):
            changed = copy.deepcopy(self.reports)
            changed["id"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_reports(changed)

    def test_wrong_best_epoch_and_test_selection_are_rejected(self):
        self.reports["title-init"]["selection"]["best_epoch"] = 2
        with self.assertRaisesRegex(ValueError, "earliest"):
            compare_reports(self.reports)
        self.reports["title-init"]["selection"]["best_epoch"] = 1
        self.reports["title-init"]["history"][0]["validation"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Only validation"):
            compare_reports(self.reports)

    def test_changed_pairing_or_projection_is_rejected(self):
        for field in ("data_sha256", "train_users_sha256", "source_sha256", "content_id"):
            changed = copy.deepcopy(self.reports)
            changed["title-fusion"]["provenance"][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_reports(changed)

    def test_mismatched_groups_and_nonfinite_metrics_are_rejected(self):
        for field, value in (("users", 9), ("metrics", {"NDCG@10": float("nan"), "Recall@20": 0.1})):
            changed = copy.deepcopy(self.reports)
            changed["title-init"]["history"][0]["validation"]["groups_by_target_train_frequency"]["1-5"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_reports(changed)


if __name__ == "__main__":
    unittest.main()
