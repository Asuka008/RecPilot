"""Validate title experiments against an unchanged ID-only training recipe."""

import math
from pathlib import Path
import re

from src.microlens_config import validate_config

ROOT = Path(__file__).resolve().parents[1]
EXTRA = {"representation", "reference_config", "feature_cache", "features_sha256",
         "projection", "fusion_weight"}


def resolved(path):
    """Resolve experiment paths relative to the checkout."""
    return (ROOT / path).resolve()


def validate_content_config(config, reference):
    """Keep budgets identical and reject undeclared scoring adjustments."""
    if not EXTRA <= config.keys():
        raise ValueError("Missing title experiment settings")
    validate_config({k: v for k, v in config.items() if k not in EXTRA})
    validate_config(reference)
    if config["representation"] not in {"title-init", "title-fusion"}:
        raise ValueError("Unknown title representation")
    if config["projection"] != "train-item-pca-v1":
        raise ValueError("Unknown projection recipe")
    weight = config["fusion_weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("fusion_weight must be finite and nonnegative")
    if config["representation"] == "title-init" and weight != 0:
        raise ValueError("Initialization requires fusion_weight=0")
    for key in ("reference_config", "feature_cache"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Invalid path: {key}")
    if not isinstance(config["features_sha256"], str) or not re.fullmatch("[0-9a-f]{64}", config["features_sha256"]):
        raise ValueError("Expected a pinned features_sha256")
    if config["model"]["embedding_dim"] > 384:
        raise ValueError("PCA dimension cannot exceed the frozen feature dimension")
    differences = [k for k in reference if k != "output_dir" and config[k] != reference[k]]
    if differences:
        raise ValueError(f"Settings differ from HSTU reference: {differences}")
    output = resolved(config["output_dir"])
    protected = [resolved(config["data_dir"]), resolved(config["feature_cache"]), ROOT / "data"]
    if output == resolved(reference["output_dir"]) or any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("Output directory overlaps reference output or input artifacts")
