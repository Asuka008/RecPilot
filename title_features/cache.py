"""Validate full title-feature caches before recommendation experiments consume them."""

import json
from pathlib import Path

from title_features.data import file_sha256, json_digest, validate_config


def validate_array(array, num_items, feature_dim):
    """Require finite FP32 unit rows and an exactly zero padding row."""
    import numpy as np

    if array.shape != (num_items + 1, feature_dim) or array.dtype != np.dtype("float32"):
        raise ValueError("Feature shape or dtype differs from the mapped catalog")
    if not np.isfinite(array).all() or np.any(array[0] != 0):
        raise ValueError("Features must be finite with an exactly zero padding row")
    if not np.allclose(np.linalg.norm(array[1:], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Every catalog feature must have unit norm")


def load_cache(directory, expected_inputs):
    """Refuse smoke, partial, corrupt or differently aligned feature artifacts."""
    import numpy as np

    directory = Path(directory)
    report = json.loads((directory / "results.json").read_text())
    validate_config(report["config"])
    if report["status"] != "PASS" or report["action"] != "build":
        raise ValueError("Only successful full builds can be used as feature caches")
    if report["input_sha256"] != expected_inputs or not report["checksums_unchanged"]:
        raise ValueError("Feature input identity differs")
    if json_digest(report["config"]) != report["config_sha256"]:
        raise ValueError("Feature configuration identity differs")
    if (report["encoded_items"] != report["audit"]["catalog_items"]
            or report["encoder"]["snapshot_revision"] != report["config"]["revision"]
            or report["encoder"]["trainable_encoder_parameters"] != 0):
        raise ValueError("Feature cache must cover the catalog with the pinned frozen encoder")
    path = directory / "features.npy"
    if file_sha256(path) != report["features_sha256"]:
        raise ValueError("Feature file checksum differs")
    features = np.load(path, allow_pickle=False, mmap_mode="r")
    validate_array(features, report["audit"]["catalog_items"], report["config"]["feature_dim"])
    return features, report
