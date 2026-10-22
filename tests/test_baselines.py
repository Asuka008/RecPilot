"""Small deterministic checks; never load the real MicroLens dataset."""

from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.baselines import ItemCF, Popularity
from src.evaluation import RankingAccumulator, evaluate, rank_candidates, single_target_metrics
from src.run_baselines import load_data


class RankingTest(unittest.TestCase):
    def test_group_boundaries_and_empty_denominators(self):
        frequency = Counter({41: 0, 42: 1, 43: 5, 44: 6, 45: 20, 46: 21})
        accumulator = RankingAccumulator([50], frequency)
        for target, length in zip(range(41, 47), (5, 6, 10, 11, 20, 21)):
            accumulator.add(np.zeros(51), list(range(1, length + 1)), target)
        result = accumulator.result("val")
        frequencies = result["groups_by_target_train_frequency"]
        lengths = result["groups_by_history_length"]
        self.assertEqual({name: group["users"] for name, group in frequencies.items()},
                         {"all": 6, "0": 1, "1-5": 2, "6-20": 2, "21+": 1})
        self.assertEqual({name: group["users"] for name, group in lengths.items()},
                         {"all": 6, "1-5": 1, "6-10": 2, "11-20": 2, "21+": 1})
        for groups in (frequencies, lengths):
            self.assertTrue(all(group["metrics"]["Recall@50"] == 1 for group in groups.values()))
        empty = RankingAccumulator([10], Counter()).result("val")
        self.assertIsNone(empty["groups_by_history_length"]["all"]["metrics"]["Recall@10"])

    def test_hand_computed_metrics(self):
        metrics = single_target_metrics([3, 8, 6], 8, [1, 2, 10])
        self.assertEqual(metrics["Recall@1"], 0)
        self.assertEqual(metrics["Recall@2"], 1)
        self.assertAlmostEqual(metrics["NDCG@2"], 1 / math.log2(3))
        self.assertEqual(single_target_metrics([3, 8], 9, [10])["NDCG@10"], 0)
        self.assertEqual(single_target_metrics([8], 8, [10])["NDCG@10"], 1)

    def test_filter_padding_history_and_boundary_ties(self):
        scores = np.array([100, 9, 8, 8, 8, 10, 0])
        self.assertEqual(rank_candidates(scores, [5], 3), [1, 2, 3])
        self.assertEqual(rank_candidates(np.zeros(6), [1, 3], 20), [2, 4, 5])
        self.assertEqual(rank_candidates(np.zeros(3), [1, 2], 20), [])

    def test_ranking_matches_exhaustive_sort(self):
        generator = np.random.default_rng(42)
        for _ in range(30):
            scores = generator.integers(-3, 4, size=30)
            history = [2, 4, 11]
            expected = sorted((i for i in range(1, 30) if i not in history),
                              key=lambda i: (-scores[i], i))
            for k in [1, 10, 20, 50]:
                self.assertEqual(rank_candidates(scores, history, k), expected[:k])

    def test_rejects_nonfinite_scores_and_duplicate_predictions(self):
        with self.assertRaises(ValueError):
            rank_candidates(np.array([0, np.nan]), [], 1)
        with self.assertRaises(ValueError):
            single_target_metrics([1, 1], 1, [10])


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"user_id": 1, "train": [1, 2, 1], "val": 3, "test": 4},
            {"user_id": 2, "train": [1, 3], "val": 4, "test": 5},
        ]

    def test_popularity_counts_train_events_only(self):
        model = Popularity(self.records, 5)
        self.assertEqual(model.score([]).tolist(), [0, 3, 1, 1, 0, 0])
        changed = [{**r, "val": 5, "test": 5} for r in self.records]
        np.testing.assert_array_equal(model.score([]), Popularity(changed, 5).score([]))

    def test_itemcf_binary_cosine_hand_computation(self):
        model = ItemCF(self.records, 5, neighbors=5)
        expected = 1 / math.sqrt(2)
        self.assertAlmostEqual(model.score([1])[2], expected)
        self.assertAlmostEqual(model.score([1])[3], expected)
        self.assertEqual(model.score([1])[4], 0)
        self.assertAlmostEqual(model.score([2, 3])[1], 2 * expected)
        np.testing.assert_array_equal(model.score([1, 1]), model.score([1]))
        changed = [{**r, "val": 5, "test": 5} for r in self.records]
        self.assertEqual(model.neighbors, ItemCF(changed, 5, 5).neighbors)

    def test_itemcf_neighbor_ties_and_zero_score_fallback(self):
        model = ItemCF(self.records, 5, neighbors=1)
        self.assertEqual([item for item, _ in model.neighbors[1]], [2])
        self.assertEqual(rank_candidates(model.score([5]), [5], 3), [1, 2, 3])

    def test_evaluation_histories_and_zero_frequency_denominator(self):
        class Recorder:
            def __init__(self):
                self.histories = []

            def score(self, history):
                self.histories.append(list(history))
                return np.zeros(6)

        model = Recorder()
        freq = Counter(i for r in self.records for i in r["train"])
        validation = evaluate(model, self.records, "val", [1], freq)
        self.assertEqual(model.histories, [[1, 2, 1], [1, 3]])
        groups = validation["groups_by_target_train_frequency"]
        self.assertEqual(groups["all"]["users"], 2)
        self.assertEqual(groups["0"]["users"], 1)
        self.assertEqual(groups["all"]["metrics"]["Recall@1"], 0.5)
        self.assertIsNone(groups["21+"]["metrics"]["Recall@1"])
        model.histories.clear()
        evaluate(model, self.records, "test", [1], freq)
        self.assertEqual(model.histories, [[1, 2, 1, 3], [1, 3, 4]])

    def test_repeat_target_fails_instead_of_dropping_user(self):
        invalid = [{"user_id": 1, "train": [1, 2], "val": 1, "test": 3}]
        with self.assertRaisesRegex(ValueError, "target is in filtered history"):
            evaluate(Popularity(invalid, 3), invalid, "val", [1], Counter())


class RunnerTest(unittest.TestCase):
    def test_tiny_end_to_end_run_and_metadata_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"user_id": 1, "train": [1, 2, 1], "val": 3, "test": 4},
                {"user_id": 2, "train": [1, 3], "val": 4, "test": 5},
            ]
            (root / "sequences.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records))
            (root / "item_mapping.json").write_text(json.dumps({
                "padding_idx": 0, "item2idx": {str(i): i for i in range(1, 6)},
                "idx2item": [None, 1, 2, 3, 4, 5],
            }))
            metadata = {"counts": {"users": 2, "items": 5, "interactions": 9},
                        "config": {"min_seq_len": 4}}
            (root / "metadata.json").write_text(json.dumps(metadata))
            config = {"data_dir": str(root), "output_dir": str(root / "runs"),
                      "models": ["popularity", "itemcf"], "ks": [1, 2],
                      "itemcf_neighbors": 2, "seed": 42}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            for split in ("val", "test"):
                completed = subprocess.run(
                    [sys.executable, "-m", "src.run_baselines", "--config", str(config_path),
                     "--split", split], capture_output=True, text=True, check=True)
                self.assertIn("Saved:", completed.stdout)
            outputs = list((root / "runs").glob("*/results.json"))
            self.assertEqual(len(outputs), 2)
            result = json.loads(outputs[0].read_text())
            self.assertEqual(set(result["models"]), {"popularity", "itemcf"})
            self.assertEqual(len(result["data_sha256"]), 3)
            metadata["counts"]["users"] = 3
            (root / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "metadata counts mismatch"):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
