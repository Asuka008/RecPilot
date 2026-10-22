"""Pure-Python corruption checks for frozen mean-control test admission."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_mean_control import digest, file_digest, validate_inputs


REPO = Path(__file__).resolve().parents[1]


class MeanFrozenTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = json.loads((REPO / "configs/mean-pooling-train-server.json").read_text())
        config["epochs"] = 2
        reference = {k: v for k, v in config.items() if k not in {"control", "reference_config"}}
        reference["output_dir"] = "runs/hstu"
        self.config_path = self.write("configs/mean.json", config)
        self.write(config["reference_config"], reference)
        source = self.write("src/example.py", "shared")
        control = self.write("controls/example.py", "control")
        data = self.write("data/processed/metadata.json", {})
        checkpoint = self.write("runs/mean/best.pt", "checkpoint bytes")
        shared = {"src/example.py": file_digest(source)}
        controls = {"controls/example.py": file_digest(control)}
        identity = {"data_sha256": {"metadata.json": file_digest(data)},
                    "train_users_sha256": "train", "eval_users_sha256": "eval"}
        hp = {**identity, "data_id": digest(identity), "source_sha256": shared,
              "source_id": digest(shared), "config_sha256": digest(reference)}
        mp = {**hp, "config_sha256": digest(config), "shared_source_id": digest(shared),
              "source_id": digest({"shared": shared, "control": controls}),
              "control_source_sha256": controls, "reference_config": reference,
              "reference_config_sha256": digest(reference)}
        history = [{"epoch": epoch, "validation": {"split": "val",
                    "groups_by_target_train_frequency": {"all": {"metrics": {"NDCG@10": .1}}}}}
                   for epoch in (1, 2)]
        base = {"status": "PASS", "action": "train", "checksums_unchanged": True, "history": history}
        self.mean = {**base, "config": config, "provenance": mp,
                     "selection": {"metric": "NDCG@10", "split": "val", "best_epoch": 1, "best_score": .1},
                     "checkpoints": {"best.pt": {"sha256": file_digest(checkpoint)}}}
        self.hstu = {**base, "config": reference, "provenance": hp}
        self.mean_path = self.write("runs/mean/results.json", self.mean)
        self.hstu_path = self.write("runs/hstu/results.json", self.hstu)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def check(self):
        return validate_inputs(self.root, self.mean_path, self.hstu_path, self.config_path)

    def test_accepts_matched_reports_and_earliest_validation_tie(self):
        checkpoint, selection = self.check()
        self.assertEqual(checkpoint, self.mean_path.parent / "best.pt")
        self.assertEqual(selection["best_epoch"], 1)

    def test_rejects_later_tie_or_test_selection_or_partial_training(self):
        for change in ("tie", "test", "partial"):
            report = copy.deepcopy(self.mean)
            if change == "tie":
                report["selection"]["best_epoch"] = 2
            elif change == "test":
                report["history"][0]["validation"]["split"] = "test"
            else:
                report["history"].pop()
            self.write("runs/mean/results.json", report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check()

    def test_rejects_different_user_identity(self):
        self.hstu["provenance"]["eval_users_sha256"] = "different"
        self.write("runs/hstu/results.json", self.hstu)
        with self.assertRaisesRegex(ValueError, "eval_users_sha256"):
            self.check()

    def test_rejects_changed_frozen_config(self):
        config = copy.deepcopy(self.mean["config"])
        config["output_dir"] = "runs/another"
        self.write("configs/mean.json", config)
        with self.assertRaisesRegex(ValueError, "frozen configurations"):
            self.check()

    def test_rejects_modified_source_data_or_checkpoint(self):
        for relative in ("src/example.py", "data/processed/metadata.json", "runs/mean/best.pt"):
            path = self.root / relative
            original = path.read_bytes()
            path.write_text("changed")
            with self.subTest(path=relative), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(original)

    def test_rejects_new_shared_source_file(self):
        self.write("src/new.py", "new module")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.check()


if __name__ == "__main__":
    unittest.main()
