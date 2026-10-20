"""Pure-Python checks; importing these tests does not load torch or numpy."""

import copy
import json
from pathlib import Path
import unittest

from src.toy_training_config import validate_toy_config


class ToyConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] /
                                  "configs/hstu-toy-fit-server.json").read_text())

    def test_server_config_is_valid(self):
        validate_toy_config(self.config)

    def test_invalid_data_and_training_settings_are_rejected(self):
        for key, value in (("sequences", [[0, 2]]), ("sequences", [[1]]),
                           ("sequences", [[1] * 99]), ("sequences", [[1, 999]]),
                           ("num_negatives", 24), ("steps", 0), ("temperature", 0),
                           ("learning_rate", float("nan")), ("minimum_loss_reduction", 1),
                           ("resume_atol", -1), ("seed", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_toy_config({**self.config, key: value})

    def test_invalid_model_and_dropout_are_rejected(self):
        for key, value in (("num_blocks", 0), ("sequence_length", 1),
                           ("num_items", True), ("dropout", 0.2)):
            config = copy.deepcopy(self.config)
            config["model"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_toy_config(config)


if __name__ == "__main__":
    unittest.main()
