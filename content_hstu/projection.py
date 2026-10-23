"""Fit a frozen content projection using training-history item membership only."""

import hashlib

import numpy as np


def array_sha256(array):
    """Hash canonical contiguous array bytes; shape and dtype are recorded separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit_projection(features, training_records, dimension):
    """Return scaled PCA features, basis and an auditable training-only identity.

    Each distinct training item has equal weight. Held-out targets never select
    fitting rows. Padding is excluded from fitting and remains exactly zero.
    """
    ids = np.array(sorted({item for row in training_records for item in row["train"]}), dtype=np.int64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite feature matrix")
    if type(dimension) is not int or not 1 <= dimension <= features.shape[1]:
        raise ValueError("Invalid projection dimension")
    if len(ids) < 2 or ids[0] <= 0 or ids[-1] >= len(features):
        raise ValueError("Projection requires at least two valid training items")
    observed = np.asarray(features[ids], dtype=np.float64)
    mean = observed.mean(axis=0)
    centered = observed - mean
    covariance = centered.T @ centered / len(ids)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    total = float(np.maximum(eigenvalues, 0).sum())
    if total <= 1e-12:
        raise ValueError("Training title features have no usable variance")
    order = np.argsort(-eigenvalues, kind="stable")[:dimension]
    components = vectors[:, order].copy()
    pivots = np.argmax(np.abs(components), axis=0)
    components *= np.where(components[pivots, np.arange(dimension)] < 0, -1, 1)
    projected = (np.asarray(features[1:], dtype=np.float64) - mean) @ components
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A catalog title has a degenerate projected vector")
    scale = 0.02 * np.sqrt(dimension)
    table = np.zeros((len(features), dimension), dtype=np.float32)
    table[1:] = (projected / norms * scale).astype(np.float32)
    arrays = {"mean": mean, "components": components, "training_item_ids": ids, "table": table}
    metadata = {"recipe": "train-item-pca-v1", "fit": "distinct items in selected training histories",
                "fit_items": len(ids), "input_dim": features.shape[1], "output_dim": dimension,
                "row_norm": float(scale), "explained_variance_ratio": float(np.maximum(eigenvalues[order], 0).sum() / total),
                "array_sha256": {name: array_sha256(value) for name, value in arrays.items()}}
    return table, arrays, metadata
