"""Pure-Python configuration and deterministic user selection for MicroLens runs."""

import math
import random


def validate_config(config):
    """Keep short integration checks distinct from full-data model selection."""
    required = {"mode", "data_dir", "output_dir", "device", "seed", "model", "epochs",
                "batch_size", "eval_batch_size", "num_workers", "learning_rate", "weight_decay",
                "temperature", "gradient_clip_norm", "ks", "selection_metric", "train_users", "eval_users"}
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    if config["mode"] not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    model = config["model"]
    dimensions = {"sequence_length", "embedding_dim", "attention_dim", "linear_dim", "num_heads", "num_blocks"}
    if set(model) != dimensions | {"dropout"}:
        raise ValueError("Model keys differ; catalog size is inferred from the mapping")
    for key in dimensions:
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Invalid model dimension: {key}")
    if model["sequence_length"] < 2:
        raise ValueError("sequence_length must be at least two")
    if type(model["dropout"]) not in (int, float) or not 0 <= model["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    for key in ("epochs", "batch_size", "eval_batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit the NumPy seed range")
    for key in ("learning_rate", "temperature", "gradient_clip_norm", "weight_decay"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid finite nonnegative number: {key}")
        if key != "weight_decay" and value == 0:
            raise ValueError(f"{key} must be positive")
    if not config["ks"] or any(type(k) is not int or k < 1 for k in config["ks"]):
        raise ValueError("ks must contain positive integers")
    if len(set(config["ks"])) != len(config["ks"]):
        raise ValueError("ks must be unique")
    if config["selection_metric"] not in {f"{m}@{k}" for m in ("Recall", "NDCG") for k in config["ks"]}:
        raise ValueError("selection_metric must be one of the reported validation metrics")
    for key in ("train_users", "eval_users"):
        value = config[key]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be null or a positive integer")
        if config["mode"] == "full" and value is not None:
            raise ValueError("Full runs cannot limit users")
        if config["mode"] == "smoke" and value is None:
            raise ValueError("Smoke runs require explicit user limits")
    for key in ("data_dir", "output_dir", "device"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if config["device"] != "cpu" and config["device"] != "cuda" and not (
        config["device"].startswith("cuda:") and config["device"][5:].isdigit()
    ):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")


def select_indices(count, limit, seed):
    """Select users without reading targets or consuming training RNG state."""
    if limit is None:
        return list(range(count))
    return sorted(random.Random(seed).sample(range(count), min(count, limit)))
