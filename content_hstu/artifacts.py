"""Bind training data, the verified title cache and the frozen projection."""

import json

from content_hstu.config import ROOT, resolved
from content_hstu.projection import fit_projection
from title_features.cache import load_cache
from title_features.data import file_sha256, input_hashes, json_digest


def prepare_content(config, checksums, num_items, training_records):
    """Validate inputs before fitting a label-free training-item projection."""
    directory = resolved(config["feature_cache"])
    manifest = directory / "results.json"
    manifest_hash = file_sha256(manifest)
    cached = json.loads(manifest.read_text())
    inputs = input_hashes(ROOT, cached["config"])
    features, cached = load_cache(directory, inputs)
    if cached["features_sha256"] != config["features_sha256"]:
        raise ValueError("Pinned feature checksum differs")
    if len(features) != num_items + 1 or any(inputs.get(k) != v for k, v in checksums.items()):
        raise ValueError("Title cache and training data identity differ")
    table, arrays, projection = fit_projection(features, training_records, config["model"]["embedding_dim"])
    identity = {"manifest_sha256": manifest_hash, "features_sha256": cached["features_sha256"],
                "input_sha256": inputs, "encoder_revision": cached["encoder"]["snapshot_revision"],
                "projection": projection}
    return table, arrays, {**identity, "content_id": json_digest(identity)}, cached["config"]


def verify_content(config, identity, feature_config):
    """Reject artifacts modified during an experiment."""
    directory = resolved(config["feature_cache"])
    if (file_sha256(directory / "results.json") != identity["manifest_sha256"]
            or file_sha256(directory / "features.npy") != identity["features_sha256"]
            or input_hashes(ROOT, feature_config) != identity["input_sha256"]):
        raise RuntimeError("Title cache or its inputs changed during execution")
