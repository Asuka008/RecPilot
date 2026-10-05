"""Build deterministic sequential-recommendation data from MicroLens interactions.

Run from the project root with::

    conda run -n agenticrec python -m src.preprocess
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "MicroLens-50k_pairs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_COLUMNS = ("user", "item", "timestamp")
PADDING_INDEX = 0


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration values that affect the processed dataset."""

    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_seq_len: int = 5
    sample_size: int = 10


def parse_args() -> PreprocessConfig:
    """Parse command-line arguments into an immutable configuration."""
    parser = argparse.ArgumentParser(
        description="Preprocess MicroLens interactions for sequential recommendation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    if args.min_seq_len < 3:
        parser.error("--min-seq-len must be at least 3 for leave-one-out splitting")
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")

    return PreprocessConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        min_seq_len=args.min_seq_len,
        sample_size=args.sample_size,
    )


def load_interactions(path: Path) -> pd.DataFrame:
    """Load the interaction log and reject invalid schemas or missing values."""
    if not path.is_file():
        raise FileNotFoundError(f"Interaction file not found: {path}")

    interactions = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(interactions.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns {missing_columns}; "
            f"found {interactions.columns.tolist()}"
        )

    interactions = interactions.loc[:, list(REQUIRED_COLUMNS)].copy()
    non_numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(interactions[column])
    ]
    if non_numeric_columns:
        raise TypeError(f"Expected numeric columns: {non_numeric_columns}")

    missing_values = interactions.isna().sum()
    if int(missing_values.sum()) > 0:
        raise ValueError(f"Missing values detected:\n{missing_values.to_string()}")

    return interactions


def print_data_report(interactions: pd.DataFrame, input_path: Path) -> None:
    """Print a data-quality report before transforming the interactions."""
    sequence_lengths = interactions.groupby("user", sort=False).size()

    print(f"Data file: {input_path}")
    print(f"Columns: {interactions.columns.tolist()}")
    print(f"Interactions: {len(interactions):,}")
    print(f"Users: {interactions['user'].nunique():,}")
    print(f"Items: {interactions['item'].nunique():,}")
    print("\nMissing values:")
    print(interactions.isna().sum().to_string())
    print(f"\nDuplicate rows: {interactions.duplicated().sum():,}")
    print("\nInteractions per user:")
    print(sequence_lengths.describe().to_string())


def prepare_interactions(
    interactions: pd.DataFrame, min_seq_len: int
) -> tuple[pd.DataFrame, int, int]:
    """Remove exact duplicates, filter short histories, and sort chronologically."""
    original_interaction_count = len(interactions)
    original_user_count = interactions["user"].nunique()
    interactions = interactions.drop_duplicates().copy()
    duplicates_removed = original_interaction_count - len(interactions)

    sequence_lengths = interactions.groupby("user")["item"].transform("size")
    interactions = interactions.loc[sequence_lengths >= min_seq_len].copy()
    users_removed = int(original_user_count - interactions["user"].nunique())

    if interactions.empty:
        raise ValueError(
            f"No users remain after applying min_seq_len={min_seq_len}"
        )

    interactions["_source_order"] = range(len(interactions))
    interactions = interactions.sort_values(
        ["user", "timestamp", "_source_order"], kind="stable"
    ).drop(columns="_source_order")
    interactions = interactions.reset_index(drop=True)

    return interactions, duplicates_removed, users_removed


def build_item_mapping(
    interactions: pd.DataFrame,
) -> tuple[dict[int, int], list[int | None]]:
    """Map original item IDs to contiguous indices while reserving zero."""
    original_items = sorted(int(item) for item in interactions["item"].unique())
    item2idx = {item: index for index, item in enumerate(original_items, start=1)}
    idx2item: list[int | None] = [None, *original_items]
    return item2idx, idx2item


def build_user_records(
    interactions: pd.DataFrame, item2idx: dict[int, int]
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    """Create mapped user sequences and apply a chronological leave-one-out split."""
    mapped_interactions = interactions.copy()
    mapped_interactions["item_idx"] = mapped_interactions["item"].map(item2idx)

    records: list[dict[str, Any]] = []
    full_sequences: dict[int, list[int]] = {}
    for user_id, group in mapped_interactions.groupby("user", sort=True):
        sequence = [int(item) for item in group["item_idx"].tolist()]
        user_id = int(user_id)
        full_sequences[user_id] = sequence
        records.append(
            {
                "user_id": user_id,
                "train": sequence[:-2],
                "val": sequence[-2],
                "test": sequence[-1],
            }
        )

    return records, full_sequences


def calculate_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(value, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
        file_handle.write("\n")


def save_outputs(
    config: PreprocessConfig,
    interactions: pd.DataFrame,
    records: list[dict[str, Any]],
    item2idx: dict[int, int],
    idx2item: list[int | None],
    duplicates_removed: int,
    users_removed: int,
) -> None:
    """Save model-ready sequences, reversible mappings, and provenance metadata."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_path = config.input_path.resolve()
    try:
        source_file = str(resolved_input_path.relative_to(PROJECT_ROOT))
    except ValueError:
        source_file = str(resolved_input_path)

    sequences_path = config.output_dir / "sequences.jsonl"
    with sequences_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, separators=(",", ":")))
            file_handle.write("\n")

    mapping = {
        "padding_idx": PADDING_INDEX,
        "item2idx": {str(item): index for item, index in item2idx.items()},
        "idx2item": idx2item,
    }
    write_json(config.output_dir / "item_mapping.json", mapping)

    lengths = pd.Series([len(record["train"]) + 2 for record in records])
    metadata = {
        "source_file": source_file,
        "source_sha256": calculate_file_sha256(config.input_path),
        "config": {
            "min_seq_len": config.min_seq_len,
            "padding_idx": PADDING_INDEX,
            "split": "leave-one-out",
        },
        "counts": {
            "users": len(records),
            "items": len(item2idx),
            "interactions": len(interactions),
            "duplicates_removed": duplicates_removed,
            "users_removed": users_removed,
        },
        "sequence_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "schema": {
            "user_id": "original user ID",
            "train": "mapped chronological item history",
            "val": "mapped penultimate item",
            "test": "mapped final item",
        },
    }
    write_json(config.output_dir / "metadata.json", metadata)


def print_samples(
    full_sequences: dict[int, list[int]], sample_size: int
) -> None:
    """Print a deterministic sample for manual split inspection."""
    if sample_size == 0 or not full_sequences:
        return

    users = pd.Series(sorted(full_sequences), dtype="int64")
    sampled_users = users.sample(
        n=min(sample_size, len(users)), random_state=42
    ).sort_values()

    print("\nSampled user splits:")
    for user_id in sampled_users:
        sequence = full_sequences[int(user_id)]
        print(f"User {user_id}")
        print(f"  Full:  {sequence}")
        print(f"  Train: {sequence[:-2]}")
        print(f"  Val:   {sequence[-2]}")
        print(f"  Test:  {sequence[-1]}")


def main() -> None:
    """Run the complete preprocessing pipeline."""
    config = parse_args()
    interactions = load_interactions(config.input_path)
    print_data_report(interactions, config.input_path)

    prepared, duplicates_removed, users_removed = prepare_interactions(
        interactions, config.min_seq_len
    )
    item2idx, idx2item = build_item_mapping(prepared)
    records, full_sequences = build_user_records(prepared, item2idx)
    save_outputs(
        config,
        prepared,
        records,
        item2idx,
        idx2item,
        duplicates_removed,
        users_removed,
    )
    print_samples(full_sequences, config.sample_size)

    print(f"\nSaved {len(records):,} user records to {config.output_dir}")
    print(f"Mapped {len(item2idx):,} items; index 0 is reserved for padding")


if __name__ == "__main__":
    main()
