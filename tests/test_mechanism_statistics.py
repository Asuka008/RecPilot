"""Tiny NumPy checks of diagnostic ranks, controls and same-user accounting."""

import unittest

import numpy as np

from diagnostics.tail_statistics import (bucket, complementarity, summarize_ranks,
                                         target_position, title_permutation)


class MechanismStatisticsTest(unittest.TestCase):
    def test_exact_ranks_match_full_sort_with_ties_and_complete_history_filter(self):
        rng = np.random.default_rng(8)
        for _ in range(12):
            scores = rng.integers(-2, 3, size=41).astype(float)
            history = [1, 3, 8, 20, 1]
            ordered = sorted(set(range(1, 41)) - set(history), key=lambda i: (-scores[i], i))
            for target in ordered:
                rank, margin, effective = target_position(scores, history, target)
                self.assertEqual(rank, ordered.index(target) + 1)
                self.assertEqual(margin, scores[target] - scores[ordered[19]])
                self.assertEqual(effective, 20)

    def test_zero_margin_can_miss_and_small_catalog_uses_actual_boundary(self):
        rank, margin, effective = target_position(np.zeros(25), [], 24)
        self.assertEqual((rank, margin, effective), (24, 0, 20))
        self.assertEqual(target_position([0, 2, 1, 3], [3], 2), (2, 0, 2))

    def test_invalid_scores_or_filtered_targets_fail(self):
        for scores, history, target in (([0, float("nan"), 1], [], 2), ([0, 1, 2], [2], 2),
                                         ([0, 1], [], 0), ([0, 1], [], 2)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                target_position(scores, history, target)

    def test_permutation_is_repeatable_bijective_and_preserves_frequency_buckets(self):
        frequency = np.array([0] + [0, 1, 5, 6, 20, 21, 50] * 5)
        p = title_permutation(frequency, 20261023)
        np.testing.assert_array_equal(p, title_permutation(frequency, 20261023))
        np.testing.assert_array_equal(np.sort(p), np.arange(len(frequency)))
        self.assertEqual(p[0], 0)
        self.assertGreater(np.count_nonzero(p != np.arange(len(p))), 0)
        for item in range(1, len(p)):
            self.assertEqual(bucket(frequency[item]), bucket(frequency[p[item]]))

    def test_rank_summary_keeps_empty_groups_and_exact_hit_counts(self):
        result = summarize_ranks([1, 20, 21, 501], [1, 0, -.1, -.9], [20]*4, [0, 2, 2, 30], [20, 100, 500, 1000])
        self.assertEqual(result["1-5"]["hits"]["Recall@20"], 1)
        self.assertEqual(result["all"]["recall"]["Recall@1000"], 1)
        self.assertIsNone(result["6-20"]["rank"])
        self.assertIsNone(result["6-20"]["recall"]["Recall@20"])

    def test_complementarity_counts_same_users_without_claiming_union_as_top20(self):
        result = complementarity([1, 3, 22, 50], [2, 40, 5, 99], [1, 1, 1, 1])["groups"]["all"]
        self.assertEqual([result[k] for k in ("both", "hstu_only", "itemcf_only", "neither")], [1]*4)
        self.assertEqual(result["oracle_union_hits"], 3)
        self.assertEqual(result["oracle_union_recall"], .75)
        with self.assertRaises(ValueError):
            complementarity([1], [1, 2], [1])


if __name__ == "__main__":
    unittest.main()
