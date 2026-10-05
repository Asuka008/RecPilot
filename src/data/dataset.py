"""PyTorch datasets for the processed MicroLens user sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset


PADDING_INDEX = 0


class UserSequenceRecord(TypedDict):
    """One processed user sequence and its chronological evaluation targets."""

    user_id: int
    train: list[int]
    val: int
    test: int


def build_training_example(
    sequence: list[int], max_seq_len: int
) -> tuple[list[int], list[int], list[bool]]:
    """Create shifted SASRec inputs, targets, and a padding mask.

    Example:
        sequence = [12, 31, 9, 71]
        max_seq_len = 5

        input_ids = [0, 0, 12, 31, 9]
        target_ids = [0, 0, 31, 9, 71]
        padding_mask = [False, False, True, True, True]

    Hints:
        1. A sequence needs at least two items to create one input-target pair.
        2. Keep at most ``max_seq_len + 1`` recent items before shifting.
        3. Build inputs from every retained item except the last one.
        4. Build targets from every retained item except the first one.
        5. Add ``PADDING_INDEX`` values on the left until both lists have
           exactly ``max_seq_len`` elements.
        6. The mask is True where the final input is not padding.

    Your task is intentionally limited to this function. The datasets,
    DataLoader integration, evaluation logic, and tests are already provided.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if len(sequence) < 2:
        raise ValueError("A training sequence must contain at least two items")

    # TODO(user): Follow hints 2-6 above and return the three Python lists.
    raise NotImplementedError("Complete build_training_example using the hints above")


def load_sequence_records(path: Path) -> list[UserSequenceRecord]:
    """Load and minimally validate processed sequence records from JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")

    records: list[UserSequenceRecord] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            record = json.loads(line)
            missing_fields = {"user_id", "train", "val", "test"} - record.keys()
            if missing_fields:
                raise ValueError(
                    f"Line {line_number} is missing fields: {sorted(missing_fields)}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"No sequence records found in {path}")
    return records


def _left_pad_history(
    history: list[int], max_seq_len: int
) -> tuple[list[int], list[bool]]:
    """Keep recent history and left-pad it to a fixed length."""
    history = history[-max_seq_len:]
    padding_size = max_seq_len - len(history)
    input_ids = [PADDING_INDEX] * padding_size + history
    padding_mask = [item != PADDING_INDEX for item in input_ids]
    return input_ids, padding_mask


class SASRecTrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Return shifted next-item training examples for SASRec."""

    def __init__(self, path: Path, max_seq_len: int) -> None:
        self.records = load_sequence_records(path)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        input_ids, target_ids, padding_mask = build_training_example(
            record["train"], self.max_seq_len
        )
        return {
            "user_id": torch.tensor(record["user_id"], dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }


class SASRecEvalDataset(Dataset[dict[str, torch.Tensor]]):
    """Return leakage-free histories and one target for validation or testing."""

    def __init__(
        self,
        path: Path,
        max_seq_len: int,
        split: Literal["val", "test"],
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if split not in {"val", "test"}:
            raise ValueError("split must be either 'val' or 'test'")

        self.records = load_sequence_records(path)
        self.max_seq_len = max_seq_len
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if self.split == "val":
            history = record["train"]
            target_id = record["val"]
        else:
            history = [*record["train"], record["val"]]
            target_id = record["test"]

        input_ids, padding_mask = _left_pad_history(history, self.max_seq_len)
        return {
            "user_id": torch.tensor(record["user_id"], dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }


def create_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a deterministic single-process DataLoader."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
