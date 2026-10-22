"""Standard-library checks of the matched mean-pooling experiment contract."""

import copy
import json
from pathlib import Path
import unittest

from controls.config import resolved, validate_control_config


ROOT = Path(__file__).resolve().parents[1]


class MeanConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/mean-pooling-train-server.json").read_text())
        self.reference = json.loads(resolved(self.config["reference_config"]).read_text())

    def test_shipped_configs_match_their_hstu_settings(self):
        for path in (ROOT / "configs").glob("mean-pooling-*.json"):
            config = json.loads(path.read_text())
            with self.subTest(path=path):
                validate_control_config(config, json.loads(resolved(config["reference_config"]).read_text()))

    def test_rejects_different_training_budget_or_hyperparameters(self):
        for key, value in (("epochs", 31), ("seed", 43), ("temperature", 0.1),
                           ("learning_rate", 0.002), ("batch_size", 64)):
            changed = {**self.config, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differ from HSTU"):
                validate_control_config(changed, self.reference)
        changed = copy.deepcopy(self.config)
        changed["model"]["dropout"] = 0
        with self.assertRaisesRegex(ValueError, "differ from HSTU"):
            validate_control_config(changed, self.reference)

    def test_output_directory_cannot_alias_hstu_output(self):
        self.config["output_dir"] = self.reference["output_dir"] + "/../" + Path(self.reference["output_dir"]).name
        with self.assertRaisesRegex(ValueError, "separate directories"):
            validate_control_config(self.config, self.reference)

    def test_unknown_controls_and_score_corrections_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_control_config({**self.config, "control": "hstu"}, self.reference)
        with self.assertRaises(ValueError):
            validate_control_config({**self.config, "alpha": 0.5}, self.reference)

    def test_validation_does_not_mutate_either_configuration(self):
        expected_config, expected_reference = copy.deepcopy(self.config), copy.deepcopy(self.reference)
        validate_control_config(self.config, self.reference)
        self.assertEqual(self.config, expected_config)
        self.assertEqual(self.reference, expected_reference)


if __name__ == "__main__":
    unittest.main()
