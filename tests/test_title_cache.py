"""Tiny NumPy checks that prevent partial or misaligned feature caches from loading."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from title_features.cache import load_cache, validate_array
from title_features.data import file_sha256, json_digest


ROOT = Path(__file__).resolve().parents[1]


class TitleCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.features = np.zeros((3, 384), dtype=np.float32)
        self.features[1, 0] = self.features[2, 1] = 1
        np.save(self.directory / "features.npy", self.features, allow_pickle=False)
        config = json.loads((ROOT / "configs/title-features-minilm-server.json").read_text())
        self.inputs = {"titles_csv": "raw", "item_mapping.json": "mapping"}
        self.report = {"status": "PASS", "action": "build", "config": config,
                       "config_sha256": json_digest(config), "input_sha256": self.inputs,
                       "checksums_unchanged": True, "audit": {"catalog_items": 2}, "encoded_items": 2,
                       "encoder": {"snapshot_revision": config["revision"], "trainable_encoder_parameters": 0},
                       "features_sha256": file_sha256(self.directory / "features.npy")}
        self.write_report()

    def write_report(self):
        (self.directory / "results.json").write_text(json.dumps(self.report))

    def test_full_cache_load_is_exact_and_read_only(self):
        features, report = load_cache(self.directory, self.inputs)
        np.testing.assert_array_equal(features, self.features)
        self.assertFalse(features.flags.writeable)
        self.assertEqual(report, self.report)

    def test_smoke_and_unaligned_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            load_cache(self.directory, {**self.inputs, "item_mapping.json": "other"})
        self.report["action"] = "smoke"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "full builds"):
            load_cache(self.directory, self.inputs)

    def test_bad_arrays_are_rejected(self):
        wrong_dtype = self.features.astype(np.float64)
        padding = self.features.copy()
        padding[0, 0] = 1
        zero = self.features.copy()
        zero[1] = 0
        nonfinite = self.features.copy()
        nonfinite[1, 0] = np.nan
        for array in (wrong_dtype, self.features[:2], padding, zero, nonfinite):
            with self.subTest(shape=array.shape), self.assertRaises(ValueError):
                validate_array(array, 2, 384)

    def test_corrupt_file_and_wrong_encoder_are_rejected(self):
        self.report["encoder"]["snapshot_revision"] = "other"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "pinned frozen"):
            load_cache(self.directory, self.inputs)
        self.report["encoder"]["snapshot_revision"] = self.report["config"]["revision"]
        self.write_report()
        with (self.directory / "features.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_cache(self.directory, self.inputs)


if __name__ == "__main__":
    unittest.main()
