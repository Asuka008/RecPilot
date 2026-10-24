"""Lightweight checks of the predeclared shared-mapping experiment."""

import json
import unittest
from pathlib import Path
from mapped_hstu.config import ROOT, validate_content_config


class MappedConfigTest(unittest.TestCase):
    def test_all_configs_match_reference_budgets(self):
        for path in sorted((ROOT / 'configs').glob('map-*.json')):
            c = json.loads(path.read_text())
            ref = json.loads((ROOT / c['reference_config']).read_text())
            validate_content_config(c, ref)

    def test_paired_configs_differ_only_in_mode_and_output(self):
        for size in ('debug', 'smoke-server', 'train-server'):
            pair = [json.loads((ROOT / f'configs/map-{mode}-{size}.json').read_text()) for mode in ('frozen', 'learned')]
            self.assertEqual({k for k in pair[0] if pair[0][k] != pair[1][k]}, {'representation', 'output_dir'})

    def test_rejects_fusion_and_budget_changes(self):
        c = json.loads((ROOT / 'configs/map-learned-train-server.json').read_text())
        ref = json.loads((ROOT / c['reference_config']).read_text())
        for change in ({'fusion_weight': .5}, {'epochs': 31}, {'representation': 'title-init'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_content_config(dict(c, **change), ref)


if __name__ == '__main__':
    unittest.main()
