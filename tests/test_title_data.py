"""Standard-library title alignment and frozen-recipe contract checks."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from title_features.data import align_titles, validate_config


ROOT = Path(__file__).resolve().parents[1]


class TitleDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.mapping_path = self.directory / "mapping.json"
        self.csv_path = self.directory / "titles.csv"
        self.mapping = {"padding_idx": 0, "item2idx": {"99": 1, "7": 2}, "idx2item": [None, 99, 7]}
        self.mapping_path.write_text(json.dumps(self.mapping))
        self.rows = [(7, "  second\t title  "), (99, "first title"), (42, "extra")]
        self.write_rows(self.rows)

    def write_rows(self, rows):
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["item", "title"])
            writer.writerows(rows)

    def test_alignment_uses_reversible_ids_not_csv_positions(self):
        titles, audit = align_titles(self.csv_path, self.mapping_path)
        self.assertEqual(titles, [None, "first title", "second title"])
        self.assertEqual(audit["extra_unmapped_rows"], 1)
        self.write_rows(list(reversed(self.rows)))
        self.assertEqual(align_titles(self.csv_path, self.mapping_path), (titles, audit))

    def test_missing_blank_and_duplicate_item_ids_are_rejected(self):
        for rows in (self.rows[1:], [(99, "first"), (7, " \t")], self.rows + [(7, "conflict")]):
            self.write_rows(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                align_titles(self.csv_path, self.mapping_path)

    def test_duplicate_text_keeps_distinct_items(self):
        self.write_rows([(99, "same"), (7, "same")])
        titles, audit = align_titles(self.csv_path, self.mapping_path)
        self.assertEqual(len(titles), 3)
        self.assertEqual(audit["duplicate_text_excess_rows"], 1)

    def test_invalid_mapping_and_malformed_csv_are_rejected(self):
        self.mapping["idx2item"][1] = 7
        self.mapping_path.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(ValueError, "mapping"):
            align_titles(self.csv_path, self.mapping_path)
        self.mapping["idx2item"][1] = 99
        self.mapping_path.write_text(json.dumps(self.mapping))
        self.csv_path.write_text("item,title\n99,title,unexpected\n")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            align_titles(self.csv_path, self.mapping_path)

    def test_frozen_config_rejects_revision_dimension_and_budget_errors(self):
        config = json.loads((ROOT / "configs/title-features-minilm-server.json").read_text())
        expected = copy.deepcopy(config)
        validate_config(config)
        self.assertEqual(config, expected)
        for key, value in (("revision", "main"), ("feature_dim", 64), ("batch_size", 0),
                           ("max_length", 512), ("transformers_version", "latest"), ("unknown", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**config, key: value})


if __name__ == "__main__":
    unittest.main()
