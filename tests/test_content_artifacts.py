"""Tiny NumPy checks of cross-artifact identity before training."""

from pathlib import Path
import tempfile
import unittest

from content_hstu.artifacts import prepare_content, verify_content
from tests.content_fixture import write_cache


class ContentArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("sequences.jsonl", "item_mapping.json", "metadata.json"):
            (self.root / name).write_text("synthetic")
        self.cache, digest, self.checksums = write_cache(self.root)
        self.config = {"feature_cache": str(self.cache), "features_sha256": digest, "model": {"embedding_dim": 4}}
        self.records = [{"train": [1, 2, 3, 4]}]

    def test_consumer_identity_and_modification_guard(self):
        table, _, identity, config = prepare_content(self.config, self.checksums, 6, self.records)
        self.assertEqual(table.shape, (7, 4))
        verify_content(self.config, identity, config)
        with (self.cache / "features.npy").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            verify_content(self.config, identity, config)

    def test_different_training_mapping_or_pinned_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "training data identity"):
            prepare_content(self.config, {**self.checksums, "item_mapping.json": "other"}, 6, self.records)
        with self.assertRaisesRegex(ValueError, "Pinned feature"):
            prepare_content({**self.config, "features_sha256": "0" * 64}, self.checksums, 6, self.records)


if __name__ == "__main__":
    unittest.main()
