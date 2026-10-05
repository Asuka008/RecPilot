"""Tests for SASRec sequence construction and evaluation datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    SASRecEvalDataset,
    SASRecTrainDataset,
    build_training_example,
    create_dataloader,
)


class BuildTrainingExampleTest(unittest.TestCase):
    def test_left_pads_a_short_sequence(self) -> None:
        input_ids, target_ids, padding_mask = build_training_example(
            [12, 31, 9, 71], max_seq_len=5
        )

        self.assertEqual(input_ids, [0, 0, 12, 31, 9])
        self.assertEqual(target_ids, [0, 0, 31, 9, 71])
        self.assertEqual(padding_mask, [False, False, True, True, True])

    def test_keeps_the_most_recent_items_when_truncating(self) -> None:
        input_ids, target_ids, padding_mask = build_training_example(
            [1, 2, 3, 4, 5], max_seq_len=3
        )

        self.assertEqual(input_ids, [2, 3, 4])
        self.assertEqual(target_ids, [3, 4, 5])
        self.assertEqual(padding_mask, [True, True, True])

    def test_rejects_a_sequence_without_a_next_item(self) -> None:
        with self.assertRaises(ValueError):
            build_training_example([12], max_seq_len=5)

    def test_rejects_padding_inside_an_unpadded_sequence(self) -> None:
        with self.assertRaises(ValueError):
            build_training_example([12, 0, 31], max_seq_len=5)


class EvaluationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sequence_path = Path(self.temp_dir.name) / "sequences.jsonl"
        record = {
            "user_id": 7,
            "train": [12, 31, 9],
            "val": 71,
            "test": 82,
        }
        self.sequence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validation_uses_only_training_history(self) -> None:
        dataset = SASRecEvalDataset(self.sequence_path, max_seq_len=5, split="val")
        example = dataset[0]

        self.assertTrue(torch.equal(example["input_ids"], torch.tensor([0, 0, 12, 31, 9])))
        self.assertEqual(example["target_id"].item(), 71)

    def test_test_split_adds_validation_item_to_history(self) -> None:
        dataset = SASRecEvalDataset(self.sequence_path, max_seq_len=5, split="test")
        example = dataset[0]

        self.assertTrue(torch.equal(example["input_ids"], torch.tensor([0, 12, 31, 9, 71])))
        self.assertEqual(example["target_id"].item(), 82)

    def test_dataloader_stacks_examples_into_a_batch(self) -> None:
        dataset = SASRecEvalDataset(self.sequence_path, max_seq_len=5, split="val")
        dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(batch["padding_mask"].shape), (1, 5))
        self.assertEqual(tuple(batch["target_id"].shape), (1,))

    def test_training_dataset_returns_shifted_tensor_sequences(self) -> None:
        dataset = SASRecTrainDataset(self.sequence_path, max_seq_len=4)
        example = dataset[0]

        self.assertTrue(
            torch.equal(example["input_ids"], torch.tensor([0, 0, 12, 31]))
        )
        self.assertTrue(
            torch.equal(example["target_ids"], torch.tensor([0, 0, 31, 9]))
        )
        self.assertTrue(
            torch.equal(
                example["padding_mask"],
                torch.tensor([False, False, True, True]),
            )
        )


if __name__ == "__main__":
    unittest.main()
