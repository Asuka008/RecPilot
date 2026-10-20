"""Pure-Python checks for public-data configuration and artifact boundaries."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.hstu_public_config import (
    DATA_FILES, audit_sequences, bindings, sha256, validate_config, verify_data,
)
from src.run_hstu_public import prepare_data

ROOT = Path(__file__).resolve().parents[1]


class PublicRecipeConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/hstu-ml1m-smoke-server.json').read_text())

    def test_smoke_keeps_architecture_and_objective_in_official_recipe(self):
        validate_config(self.config)
        overrides = bindings(self.config)
        self.assertIn('train_fn.num_epochs = 1', overrides)
        self.assertIn('train_fn.full_eval_every_n = 1', overrides)
        self.assertFalse(any('hstu_encoder.' in line or 'num_negatives' in line
                             or 'temperature' in line for line in overrides))

    def test_rejects_zero_workers_with_official_prefetch_and_invalid_settings(self):
        for key, value in [('num_workers', 0), ('batch_size', 129), ('master_port', 65536),
                           ('seed', True), ('prefetch_factor', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**self.config, key: value})

    def test_audit_detects_temporal_and_alignment_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sequences.csv'
            def write(ids, timestamps):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        'user_id', 'sequence_item_ids', 'sequence_timestamps', 'sequence_ratings'])
                    writer.writeheader()
                    writer.writerow({'user_id': 1, 'sequence_item_ids': ids,
                                     'sequence_timestamps': timestamps, 'sequence_ratings': '5,4,3'})
            write('2,4,6', '1,1,3')
            self.assertEqual(audit_sequences(path),
                             {'users': 1, 'items': 3, 'interactions': 3, 'max_item_id': 6})
            for ids, times in [('2,4,6', '3,1,2'), ('2,4', '1,2,3'), ('0,4,6', '1,2,3')]:
                write(ids, times)
                with self.assertRaises(ValueError):
                    audit_sequences(path)

    def test_cached_data_is_verified_without_download_or_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            hashes = {}
            for name in DATA_FILES:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic fixture\n')
                hashes[name] = sha256(path)
            manifest = {'upstream_revision': 'revision', 'seed': 42, 'files_sha256': hashes}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            with patch('src.run_hstu_public.urlretrieve') as download:
                self.assertEqual(prepare_data(directory, {'revision': 'revision'}, 42), manifest)
                download.assert_not_called()
            with self.assertRaises(ValueError):
                verify_data(directory, 'revision', 43)
            (directory / DATA_FILES[-1]).write_text('modified\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_data(directory, 'revision', 42)


if __name__ == '__main__':
    unittest.main()
