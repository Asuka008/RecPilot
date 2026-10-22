"""Validate configuration and align immutable title text to mapped catalog IDs."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TRANSFORMERS_VERSION = "4.57.1"


def file_sha256(path):
    """Hash inputs without retaining large files in memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_digest(value):
    """Hash a canonical JSON representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_config(config):
    """Keep this first feature recipe explicit and reproducible."""
    required = {"titles_csv", "data_dir", "output_dir", "model_id", "revision",
                "transformers_version", "feature_dim", "max_length", "batch_size",
                "smoke_items", "device", "seed"}
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    for key, expected in (("model_id", MODEL_ID), ("revision", REVISION),
                          ("transformers_version", TRANSFORMERS_VERSION), ("feature_dim", 384)):
        if config[key] != expected:
            raise ValueError(f"Frozen encoder recipe differs: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Encoder revision must be a full commit SHA")
    for key in ("max_length", "batch_size", "smoke_items"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Expected positive integer: {key}")
    if not 3 <= config["max_length"] <= 256:
        raise ValueError("max_length must include special tokens and be at most 256")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit the NumPy seed range")
    if config["device"] not in {"cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", config["device"]):
        raise ValueError("Expected cpu, cuda or cuda:<index>")
    for key in ("titles_csv", "data_dir", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Expected nonempty path: {key}")


def align_titles(titles_path, mapping_path):
    """Return [None, title_1, ...] in embedding-table order, never CSV row order."""
    mapping = json.loads(Path(mapping_path).read_text())
    forward, reverse = mapping["item2idx"], mapping["idx2item"]
    if (mapping["padding_idx"] != 0 or not forward or len(reverse) != len(forward) + 1
            or reverse[0] is not None or set(forward.values()) != set(range(1, len(reverse)))
            or any(type(index) is not int for index in forward.values())
            or any(forward.get(str(original)) != index for index, original in enumerate(reverse[1:], 1))):
        raise ValueError("Item mapping must be contiguous, reversible, and reserve padding 0")
    rows = {}
    with Path(titles_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != {"item", "title"} or len(reader.fieldnames) != 2:
            raise ValueError("Title CSV must contain exactly item,title columns")
        for line, row in enumerate(reader, 2):
            if None in row or row["item"] is None or row["title"] is None:
                raise ValueError(f"Malformed title CSV row: {line}")
            original = row["item"].strip()
            if not original or original in rows:
                raise ValueError(f"Empty or duplicate title item ID at row {line}")
            rows[original] = " ".join(row["title"].split())
    missing = [str(original) for original in reverse[1:] if str(original) not in rows]
    if missing:
        raise ValueError(f"Missing mapped titles: {len(missing)}; examples={missing[:5]}")
    aligned = [None] + [rows[str(original)] for original in reverse[1:]]
    blank = [index for index, title in enumerate(aligned[1:], 1) if not title]
    if blank:
        raise ValueError(f"Blank mapped titles: {len(blank)}; mapped examples={blank[:5]}")
    counts = Counter(aligned[1:])
    lengths = sorted(map(len, aligned[1:]))
    report = {"catalog_items": len(forward), "csv_rows": len(rows),
              "extra_unmapped_rows": len(set(rows) - set(forward)),
              "missing_titles": 0, "blank_titles": 0,
              "duplicate_text_excess_rows": sum(count - 1 for count in counts.values()),
              "titles_with_non_ascii": sum(any(ord(c) > 127 for c in title) for title in aligned[1:]),
              "titles_with_cjk": sum(any("\u4e00" <= c <= "\u9fff" for c in title) for title in aligned[1:]),
              "characters": {"min": lengths[0], "median": statistics.median(lengths), "max": lengths[-1]},
              "aligned_titles_sha256": json_digest(aligned),
              "normalization": "collapse Unicode whitespace; preserve case, punctuation and hashtags",
              "examples": [{"mapped_id": i, "original_id": reverse[i], "title": aligned[i]}
                           for i in sorted({1, len(forward) // 2 + 1, len(forward)})]}
    return aligned, report


def input_hashes(root, config):
    """Bind titles to the unchanged processed-data snapshot without reading labels."""
    paths = {"titles_csv": root / config["titles_csv"]}
    paths.update({name: root / config["data_dir"] / name
                  for name in ("item_mapping.json", "metadata.json", "sequences.jsonl")})
    return {name: file_sha256(path) for name, path in paths.items()}
