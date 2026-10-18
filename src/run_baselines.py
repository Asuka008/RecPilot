"""Run deterministic full-data baselines and write experiment provenance."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from src.baselines import ItemCF, Popularity
from src.evaluation import evaluate


def file_sha256(path: Path) -> str:
    """Hash input artifacts without loading all bytes at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path) -> tuple[list[dict], int, dict]:
    """Validate record IDs, reversible mappings and metadata counts before fitting."""
    paths = {name: data_dir / name for name in (
        "sequences.jsonl", "item_mapping.json", "metadata.json"
    )}
    mapping = json.loads(paths["item_mapping.json"].read_text())
    metadata = json.loads(paths["metadata.json"].read_text())
    n = len(mapping["item2idx"])
    if (mapping["padding_idx"] != 0 or len(mapping["idx2item"]) != n + 1
            or mapping["idx2item"][0] is not None
            or set(mapping["item2idx"].values()) != set(range(1, n + 1))):
        raise ValueError("invalid contiguous item mapping")
    for original, index in mapping["item2idx"].items():
        if str(mapping["idx2item"][index]) != original:
            raise ValueError("item mapping is not reversible")
    records = []
    users = set()
    with paths["sequences.jsonl"].open() as handle:
        for line in handle:
            record = json.loads(line)
            if type(record["user_id"]) is not int or record["user_id"] in users:
                raise ValueError("user IDs must be unique integers")
            users.add(record["user_id"])
            if not isinstance(record["train"], list) or len(record["train"]) < 2:
                raise ValueError("training history must contain at least two items")
            items = record["train"] + [record["val"], record["test"]]
            if any(type(item) is not int or not 1 <= item <= n for item in items):
                raise ValueError("item IDs must be positive integers within the catalog")
            if len(items) < metadata["config"]["min_seq_len"]:
                raise ValueError("sequence violates metadata minimum length")
            records.append(record)
    observed = (len(records), n, sum(len(r["train"]) + 2 for r in records))
    expected = tuple(metadata["counts"][key] for key in ("users", "items", "interactions"))
    if not records or observed != expected:
        raise ValueError(f"metadata counts mismatch: observed={observed}, expected={expected}")
    return records, n, {name: file_sha256(path) for name, path in paths.items()}


def git_state() -> dict:
    """Record the exact code revision and whether uncommitted files exist."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    """Fit training histories once per model, then evaluate the requested split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines-server.json"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.data_dir:
        config["data_dir"] = str(args.data_dir)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if (not config["ks"] or any(type(k) is not int or k < 1 for k in config["ks"])
            or not config["models"] or len(set(config["models"])) != len(config["models"])
            or set(config["models"]) - {"popularity", "itemcf"}
            or type(config["itemcf_neighbors"]) is not int or config["itemcf_neighbors"] < 1):
        raise ValueError("invalid baseline configuration")
    records, num_items, checksums = load_data(Path(config["data_dir"]))
    frequency = Counter(item for record in records for item in record["train"])
    now = datetime.now(timezone.utc)
    run_dir = Path(config["output_dir"]) / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(), "git": git_state(),
        "config": config, "split": args.split, "data_sha256": checksums,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "device": "cpu", "score_dtype": "float64",
                    "packages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                            for distribution in importlib.metadata.distributions()
                                            if distribution.metadata["Name"]))},
        "protocol": {
            "name": "per-user-leave-one-out-full-catalog-v1",
            "catalog_items": num_items, "catalog_assumed_known": True,
            "filter": "padding and full visible history",
            "ties": "ascending mapped item ID", "fit_data": "train only",
            "repeat_targets": "fail explicitly", "zero_frequency_targets": "retained",
            "itemcf": "binary user-item cosine; top neighbors per source; distinct history sum",
            "fallback": "zero scores tie by ascending mapped item ID",
            "randomness": "none; seed recorded for experiment consistency",
        }, "models": {},
    }
    output_path = run_dir / "results.json"
    for name in config["models"]:
        print(f"Fitting {name} on {len(records)} training histories...", flush=True)
        start = time.perf_counter()
        model = (Popularity(records, num_items) if name == "popularity" else
                 ItemCF(records, num_items, config["itemcf_neighbors"]))
        fit_seconds = time.perf_counter() - start
        print(f"Evaluating {name}: {args.split}...", flush=True)
        start = time.perf_counter()
        metrics = evaluate(model, records, args.split, config["ks"], frequency)
        result["models"][name] = {
            "fit_seconds": fit_seconds,
            "evaluate_seconds": time.perf_counter() - start, **metrics,
        }
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"model": name, **metrics["groups_by_target_train_frequency"]["all"]}), flush=True)
        del model
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
