"""Inexpensive pure-Python checks; no dataset, NumPy or PyTorch imports."""

import copy
import json
from pathlib import Path
import random
import unittest

from src.microlens_config import select_indices, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/hstu-microlens-smoke-server.json").read_text())

    def test_all_shipped_configs(self):
        for path in (ROOT / "configs").glob("hstu-microlens-*.json"):
            with self.subTest(path=path):
                validate_config(json.loads(path.read_text()))

    def test_rejects_invalid_settings(self):
        for key, value in (("temperature", 0), ("temperature", float("nan")),
                           ("learning_rate", True), ("num_workers", -1),
                           ("batch_size", 0), ("ks", [10, 10]),
                           ("selection_metric", "NDCG@50"), ("train_users", None),
                           ("eval_users", 0), ("device", "mps"), ("seed", -1)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_full_mode_requires_all_users(self):
        self.config["mode"] = "full"
        with self.assertRaisesRegex(ValueError, "cannot limit users"):
            validate_config(self.config)
        self.config.update(train_users=None, eval_users=None)
        validate_config(self.config)

    def test_catalog_size_is_not_hardcoded(self):
        self.config["model"]["num_items"] = 19220
        with self.assertRaisesRegex(ValueError, "inferred from the mapping"):
            validate_config(self.config)

    def test_selection_is_stable_sorted_and_does_not_consume_global_rng(self):
        state = random.getstate()
        selected = select_indices(50, 8, 42)
        self.assertEqual(selected, select_indices(50, 8, 42))
        self.assertEqual(selected, sorted(set(selected)))
        self.assertNotEqual(selected, select_indices(50, 8, 43))
        self.assertEqual(state, random.getstate())
        self.assertEqual(select_indices(3, 20, 42), [0, 1, 2])
        self.assertEqual(select_indices(3, None, 42), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
