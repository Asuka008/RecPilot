"""Audit full-data shifted targets without importing PyTorch or reading test labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def audit_supervision(records, num_items, sequence_length):
    """Distinguish history exposure counts from retained positive target counts."""
    if type(sequence_length) is not int or sequence_length < 2 or num_items < 2:
        raise ValueError("Invalid sequence length or catalog size")
    history = Counter()
    supervised = Counter()
    for record in records:
        sequence = record["train"]
        if len(sequence) < 2 or any(type(i) is not int or not 1 <= i <= num_items for i in sequence):
            raise ValueError("Invalid training history")
        if type(record["val"]) is not int or not 1 <= record["val"] <= num_items:
            raise ValueError("Invalid validation target")
        history.update(sequence)
        # Match build_training_example: retain L+1 events, then shift once.
        supervised.update(sequence[-(sequence_length + 1):][1:])
    groups = {name: {"validation_users": 0, "zero_supervised_target_users": 0}
              for name in ("0", "1-5", "6-20", "21+")}
    for record in records:
        count = history[record["val"]]
        bucket = "0" if count == 0 else "1-5" if count <= 5 else "6-20" if count <= 20 else "21+"
        groups[bucket]["validation_users"] += 1
        groups[bucket]["zero_supervised_target_users"] += int(supervised[record["val"]] == 0)
    return {"sequence_length": sequence_length, "users": len(records),
            "train_history_events": sum(history.values()), "supervised_positions": sum(supervised.values()),
            "train_history_items": len(history), "positive_target_items": len(supervised),
            "catalog_items_without_positive_target": num_items - len(supervised),
            "history_observed_items_without_positive_target": sum(supervised[item] == 0 for item in history),
            "validation_by_target_history_frequency": groups}


def main():
    """Print counts with source hashes; leave raw and processed files unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-microlens-train-server.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["mode"] != "full" or config["train_users"] is not None or config["eval_users"] is not None:
        raise ValueError("This audit requires a full-data configuration without user limits")
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    paths = {name: data_dir / name for name in ("sequences.jsonl", "item_mapping.json", "metadata.json")}
    checksums = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    mapping = json.loads(paths["item_mapping.json"].read_text())
    records = [json.loads(line) for line in paths["sequences.jsonl"].read_text().splitlines()]
    result = audit_supervision(records, len(mapping["item2idx"]), config["model"]["sequence_length"])
    result.update(data_sha256=checksums, scope="full training histories; validation grouping only; no test labels")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
