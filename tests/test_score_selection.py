"""NumPy-only checks of a frozen validation-selected correction recipe."""

import copy
import json
from pathlib import Path
import unittest

from scripts.evaluate_hstu_score_selection import selected_recipe, validate_test_config


ROOT = Path(__file__).resolve().parents[1]


def audit_fixture(values=(0.3, 0.4)):
    """Build a minimal two-variant selection artifact without model execution."""
    config = json.loads((ROOT / "configs/hstu-score-audit-server.json").read_text())
    config["alphas"] = [0, 0.5]
    variants = [{"alpha": alpha, "evaluation": {"split": "val", "groups_by_target_train_frequency": {
        "all": {"users": 2, "metrics": {"NDCG@10": value}}}}} for alpha, value in zip(config["alphas"], values)]
    index = max(range(len(values)), key=lambda i: values[i])
    return {"status": "PASS", "split": "val", "config": config, "variants": variants,
            "checks": {"alpha_zero_reproduces_frozen_validation": True, "inputs_unchanged": True},
            "provenance": {"training_config": {"selection_metric": "NDCG@10"}},
            "selection": {"alpha": config["alphas"][index], "metric": "NDCG@10", "value": values[index]}}


class ScoreSelectionTest(unittest.TestCase):
    def test_reconstructs_validation_winner_and_smallest_alpha_tie(self):
        self.assertEqual(selected_recipe(audit_fixture())["alpha"], 0.5)
        self.assertEqual(selected_recipe(audit_fixture((0.4, 0.4)))["alpha"], 0)
        changed = audit_fixture()
        changed["selection"]["alpha"] = 0
        with self.assertRaisesRegex(ValueError, "selection"):
            selected_recipe(changed)

    def test_rejects_failed_nonvalidation_or_unverified_artifacts(self):
        for key, value in (("status", "FAIL"), ("status", "RUNNING"), ("split", "test")):
            changed = audit_fixture()
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                selected_recipe(changed)
        changed = audit_fixture()
        changed["checks"]["alpha_zero_reproduces_frozen_validation"] = False
        with self.assertRaises(ValueError):
            selected_recipe(changed)
        changed = audit_fixture()
        changed["variants"][1]["alpha"] = 1
        with self.assertRaises(ValueError):
            selected_recipe(changed)

    def test_config_cannot_override_alpha_or_search_test_parameters(self):
        config = json.loads((ROOT / "configs/hstu-score-selected-test-server.json").read_text())
        validate_test_config(config)
        changed = copy.deepcopy(config)
        changed["alpha"] = 1
        with self.assertRaisesRegex(ValueError, "no alpha override"):
            validate_test_config(changed)


if __name__ == "__main__":
    unittest.main()
