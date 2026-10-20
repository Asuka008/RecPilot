"""Pure-Python configuration validation for the synthetic learning gate."""

import math


def validate_toy_config(config):
    """Reject invalid dimensions, sequences and fitting thresholds early."""
    model = config["model"]
    for key in ("num_items", "sequence_length", "embedding_dim", "attention_dim", "linear_dim", "num_heads", "num_blocks"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Invalid model dimension: {key}")
    if model["num_items"] < 2 or model["sequence_length"] < 2 or model["dropout"] != 0:
        raise ValueError("Toy fitting requires at least two items/positions and dropout=0")
    if not config["sequences"]:
        raise ValueError("At least one synthetic sequence is required")
    for sequence in config["sequences"]:
        if not isinstance(sequence, list) or not 2 <= len(sequence) <= model["sequence_length"] + 1:
            raise ValueError("Synthetic sequence length must be between 2 and L+1")
        if any(type(item) is not int or not 1 <= item <= model["num_items"] for item in sequence):
            raise ValueError("Synthetic item IDs must be in the positive catalog")
    for key in ("steps", "log_every", "num_negatives"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["num_negatives"] >= model["num_items"]:
        raise ValueError("Without-replacement negatives must fit in the non-target catalog")
    for key in ("learning_rate", "temperature"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid positive number: {key}")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 < config["minimum_loss_reduction"] < 1:
        raise ValueError("minimum_loss_reduction must be in (0, 1)")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit the NumPy seed range")
    for key in ("resume_atol", "resume_rtol"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid resume tolerance: {key}")
