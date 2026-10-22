"""Server CPU end-to-end check of frozen validation score audits on tiny files."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.diagnose_hstu_scores import run_audit
from src.run_hstu_microlens import run
from tests.test_microlens_training import ROOT, write_data


class ScoreAuditServerTest(unittest.TestCase):
    def test_completed_training_anchor_coverage_and_corrupted_reference_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_data(root)
            training = json.loads((ROOT / "configs/hstu-microlens-debug.json").read_text())
            training.update(mode="full", data_dir=str(root), output_dir=str(root / "training"),
                            train_users=None, eval_users=None, batch_size=2, eval_batch_size=2,
                            ks=[1, 2], selection_metric="NDCG@1")
            training["model"].update(sequence_length=2, dropout=0.2)
            config_path = root / "training.json"
            config_path.write_text(json.dumps(training))
            captured = io.StringIO()
            with redirect_stdout(captured):
                try:
                    run(SimpleNamespace(config=config_path, action="train", resume=None, stop_after_epoch=None))
                except SystemExit:
                    self.fail(captured.getvalue())
            training_dir = next((root / "training").iterdir())
            audit = {"training_config": str(config_path), "training_run": str(training_dir),
                     "output_dir": str(root / "audit"), "split": "val", "alphas": [0, 0.25],
                     "smoothing": 1.0, "anchor_atol": 1e-12}
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            with redirect_stdout(captured):
                try:
                    report, _ = run_audit(audit_path)
                except SystemExit:
                    self.fail(captured.getvalue())
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["checks"]["alpha_zero_reproduces_frozen_validation"])
            self.assertTrue(report["checks"]["inputs_unchanged"])
            self.assertEqual(len(report["variants"]), 2)
            for variant in report["variants"]:
                self.assertEqual(variant["recommendations"]["slots"], 4)
                self.assertEqual(sum(variant["recommendations"]["slots_by_history_frequency"].values()), 4)
                self.assertEqual(variant["evaluation"]["groups_by_target_train_frequency"]["all"]["users"], 2)
            reference_path = training_dir / "results.json"
            reference = json.loads(reference_path.read_text())
            best = next(row for row in reference["history"] if row["epoch"] == reference["selection"]["best_epoch"])
            best["validation"]["groups_by_history_length"]["all"]["metrics"]["Recall@1"] += 0.1
            reference_path.write_text(json.dumps(reference))
            with redirect_stdout(captured), self.assertRaises(SystemExit):
                run_audit(audit_path)
            failed_path = sorted((root / "audit").glob("*/results.json"))[-1]
            failed = json.loads(failed_path.read_text())
            self.assertEqual(failed["status"], "FAIL")
            self.assertIn("anchor", failed["error"]["message"].lower())


if __name__ == "__main__":
    unittest.main()
