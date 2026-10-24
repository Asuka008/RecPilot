"""Reuse audited cache and PCA identities without modifying historical sources."""

from content_hstu.artifacts import prepare_content as prepare_original, verify_content
from content_hstu.config import resolved
from title_features.cache import load_cache


def prepare_content(config, checksums, num_items, training_records):
    """Return full 384-dimensional features and the training-only PCA initializer."""
    _, arrays, identity, feature_config = prepare_original(config, checksums, num_items, training_records)
    features, _ = load_cache(resolved(config['feature_cache']), identity['input_sha256'])
    return features, arrays, identity, feature_config
