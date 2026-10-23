"""Pure-Python guards for matched title experiments."""

import json
import unittest

from content_hstu.config import ROOT, validate_content_config


class ContentConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/title-init-smoke-server.json").read_text())
        self.reference = json.loads((ROOT / self.config["reference_config"]).read_text())

    def test_all_shipped_configs_match_reference(self):
        for path in (ROOT / "configs").glob("title-*-*.json"):
            if path.name.startswith("title-features"):
                continue
            config = json.loads(path.read_text())
            validate_content_config(config, json.loads((ROOT / config["reference_config"]).read_text()))

    def test_budget_or_scoring_changes_are_rejected(self):
        for key, value in (("epochs", 4), ("temperature", 0.5), ("alpha", 0.5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_content_config({**self.config, key: value}, self.reference)

    def test_unsafe_or_unpinned_settings_are_rejected(self):
        for key, value in (("output_dir", self.reference["output_dir"]), ("output_dir", "data/processed/run"),
                           ("output_dir", self.config["feature_cache"]), ("features_sha256", "latest"),
                           ("projection", "all-item-pca"), ("fusion_weight", 0.5), ("fusion_weight", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_content_config({**self.config, key: value}, self.reference)


if __name__ == "__main__":
    unittest.main()
