"""Tiny NumPy-only tests for validation scoring diagnostics."""

from collections import Counter
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.diagnose_hstu_scores import corrected_scores, training_counts, validate_audit_config, verify_anchor
from src.evaluation import RankingAccumulator, rank_candidates


ROOT = Path(__file__).resolve().parents[1]


class ScoreAuditTest(unittest.TestCase):
    def test_zero_alpha_is_exact_and_ranking_keeps_history_and_ties(self):
        scores = np.array([999, 0.5, 0.5, 0.5, 0.2], dtype=np.float32)
        result = corrected_scores(scores, np.array([0, 100, 5, 1, 0]), 0.2, 0, 1)
        np.testing.assert_array_equal(result, scores)
        self.assertEqual(rank_candidates(result, [1], 20), [2, 3, 4])

    def test_formula_matches_logit_scale_and_penalizes_frequent_candidates(self):
        scores = np.array([0, 0.5, 0.5, 0.5])
        counts = np.array([0, 100, 1, 0])
        result = corrected_scores(scores, counts, 0.2, 0.5, 1)
        np.testing.assert_allclose(result[1:] / 0.2, scores[1:] / 0.2 - 0.5 * np.log(counts[1:] + 1))
        self.assertEqual(rank_candidates(result, [], 3), [3, 2, 1])
        self.assertEqual(result[0], scores[0])
        with self.assertRaises(ValueError):
            corrected_scores(scores, counts, 0, 0.5, 1)

    def test_counts_match_shift_and_do_not_read_held_out_labels(self):
        records = [{"train": [1, 2, 3, 4], "val": 7, "test": 8}, {"train": [1, 6], "val": 8}]
        history, positive = training_counts(records, 8, 2)
        np.testing.assert_array_equal(history, [0, 2, 1, 1, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(positive, [0, 0, 0, 1, 1, 0, 1, 0, 0])
        records[0].update(val="unread", test=None)
        again = training_counts(records, 8, 2)
        np.testing.assert_array_equal(again[0], history)
        np.testing.assert_array_equal(again[1], positive)

    def test_anchor_checks_metrics_denominators_and_null_groups(self):
        accumulator = RankingAccumulator([1, 2], Counter({2: 1}))
        accumulator.add(np.zeros(5), [1], 2)
        reference = accumulator.result("val")
        verify_anchor(reference, reference, 0)
        changed = copy.deepcopy(reference)
        changed["groups_by_target_train_frequency"]["all"]["metrics"]["NDCG@1"] += 0.01
        with self.assertRaisesRegex(AssertionError, "Uncorrected anchor"):
            verify_anchor(changed, reference, 1e-12)
        changed = copy.deepcopy(reference)
        changed["groups_by_history_length"]["all"]["users"] += 1
        with self.assertRaises(AssertionError):
            verify_anchor(changed, reference, 1e-12)

    def test_config_rejects_test_split_and_invalid_grids(self):
        config = json.loads((ROOT / "configs/hstu-score-audit-server.json").read_text())
        validate_audit_config(config)
        for key, value in (("split", "test"), ("alphas", [0.5]), ("alphas", [0, 0]),
                           ("alphas", [0, float("nan")]), ("smoothing", 0),
                           ("anchor_atol", 1e-2)):
            changed = {**config, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_audit_config(changed)


if __name__ == "__main__":
    unittest.main()
