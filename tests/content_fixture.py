"""Create a tiny full-cache manifest without loading a text encoder."""

import json

import numpy as np

from content_hstu.config import ROOT
from title_features.data import file_sha256, input_hashes, json_digest


def write_cache(root, num_items=6, data_dir=None):
    """Use only synthetic titles and features in a temporary test directory."""
    title_path = root / "titles.csv"
    title_path.write_text("item,title\n" + "".join(f"{i},title {i}\n" for i in range(1, num_items + 1)))
    cache = root / "cache"
    cache.mkdir()
    config = json.loads((ROOT / "configs/title-features-minilm-server.json").read_text())
    config.update(titles_csv=str(title_path), data_dir=str(data_dir or root), output_dir=str(root / "unused"), device="cpu")
    features = np.random.default_rng(12).normal(size=(num_items + 1, 384)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    features[0] = 0
    np.save(cache / "features.npy", features, allow_pickle=False)
    inputs = input_hashes(ROOT, config)
    digest = file_sha256(cache / "features.npy")
    report = {"status": "PASS", "action": "build", "config": config,
              "config_sha256": json_digest(config), "input_sha256": inputs, "checksums_unchanged": True,
              "audit": {"catalog_items": num_items}, "encoded_items": num_items,
              "encoder": {"snapshot_revision": config["revision"], "trainable_encoder_parameters": 0},
              "features_sha256": digest}
    (cache / "results.json").write_text(json.dumps(report))
    return cache, digest, {k: v for k, v in inputs.items() if k != "titles_csv"}
