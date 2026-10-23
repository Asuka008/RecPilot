"""Standard-library checks of frozen title checkpoint admission."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_selected_title import admit, file_digest


class SelectedTitleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = {"reports": {}, "selected_model": "title-init", "selected_epoch": 30,
                       "training_config": "training.json"}
        config = {"data_dir": "data", "reference_config": "reference.json"}
        (self.root / "training.json").write_text(json.dumps(config))
        (self.root / "reference.json").write_text('{}')
        (self.root / "model.py").write_text('source')
        (self.root / "data").mkdir()
        (self.root / "data/file").write_text('data')
        self.report = {"config": config, "selection": {"best_score": 0.04, "best_epoch": 30},
                       "provenance": {"reference_config": {}, "source_sha256": {"model.py": file_digest(self.root / "model.py")},
                                      "content_source_sha256": {}, "data_sha256": {"file": file_digest(self.root / "data/file")}},
                       "checkpoints": {"best.pt": {"sha256": ""}}}
        for name, score in (("id", 0.038), ("title-init", 0.04), ("title-fusion", 0.037)):
            directory = self.root / name
            directory.mkdir()
            (directory / "best.pt").write_bytes(b"synthetic checkpoint")
            report = copy.deepcopy(self.report)
            report["selection"]["best_score"] = score
            report["checkpoints"]["best.pt"]["sha256"] = file_digest(directory / "best.pt")
            (directory / "results.json").write_text(json.dumps(report))
            self.recipe["reports"][name] = f"{name}/results.json"
        # Report-level comparison has its own corruption tests; isolate file admission here.
        self.patcher = patch('scripts.evaluate_selected_title.compare_reports')
        self.comparator = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_matching_files_admit_exact_best_checkpoint(self):
        config, checkpoint, selection = admit(self.root, self.recipe)
        self.assertEqual(checkpoint, self.root / "title-init/best.pt")
        self.assertEqual(config, self.root / "training.json")
        self.assertEqual(selection["best_epoch"], 30)
        self.comparator.assert_called_once()

    def test_report_pairing_failure_is_propagated(self):
        self.comparator.side_effect = ValueError("pairing failed")
        with self.assertRaisesRegex(ValueError, "pairing failed"):
            admit(self.root, self.recipe)

    def test_wrong_winner_or_epoch_is_rejected(self):
        for key, value in (("selected_model", "title-fusion"), ("selected_epoch", 29)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                admit(self.root, {**self.recipe, key: value})

    def test_changed_checkpoint_source_data_or_config_is_rejected(self):
        for name in ("title-init/best.pt", "model.py", "data/file", "training.json", "reference.json"):
            path = self.root / name
            original = path.read_bytes()
            path.write_text('{"changed": true}')
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    admit(self.root, self.recipe)
            finally:
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
