"""Pure-Python regressions for epoch completeness and fixed final-epoch reporting."""

import json
from pathlib import Path
import unittest

from src.hstu_public_config import (
    REFERENCE_TAGS, bindings, compare_reference, epoch_history, validate_action,
)

ROOT = Path(__file__).resolve().parents[1]


class PublicTrainingReportTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/hstu-ml1m-train-server.json').read_text())
        self.reference = json.loads((ROOT / 'configs/hstu-ml1m-reference-metrics.json').read_text())

    def test_formal_and_smoke_actions_cannot_be_confused(self):
        validate_action(self.config, 'train')
        self.assertIn('train_fn.num_epochs = 101', bindings(self.config))
        with self.assertRaises(ValueError):
            validate_action(self.config, 'smoke')
        for overrides in ({'batch_size': 32}, {'num_epochs': 1}, {'eval_batch_size': 32}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_action({**self.config, **overrides}, 'train')
        smoke = {**self.config, 'num_epochs': 1}
        validate_action(smoke, 'smoke')

    def make_scalars(self):
        return {f'eval_epoch/{tag}': [{'step': epoch, 'value': 0.5 if epoch == 0 else 0.1}
                                    for epoch in range(101)] for tag in REFERENCE_TAGS}

    def test_final_epoch_is_reported_even_when_an_earlier_score_is_higher(self):
        history = epoch_history(self.make_scalars(), 101)
        self.assertEqual(len(history), 101)
        self.assertEqual(history[-1]['epoch'], 100)
        self.assertEqual(history[-1]['hr@10'], 0.1)
        self.assertEqual(compare_reference(history[-1], self.reference)['status'], 'REVIEW_REQUIRED')
        match = compare_reference(self.reference['metrics'], self.reference)
        self.assertEqual(match['status'], 'WITHIN_TOLERANCE')
        self.assertTrue(all(row['signed_gap'] == 0 for row in match['metrics'].values()))

    def test_incomplete_duplicate_or_nonfinite_epoch_evidence_fails(self):
        tag = 'eval_epoch/hr@10'
        for mode in ('missing', 'duplicate', 'nonfinite', 'out_of_range'):
            scalars = self.make_scalars()
            if mode == 'missing':
                scalars[tag].pop()
            elif mode == 'duplicate':
                scalars[tag][-1]['step'] = 99
            elif mode == 'nonfinite':
                scalars[tag][-1]['value'] = float('nan')
            else:
                scalars[tag][-1]['value'] = 1.1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                epoch_history(scalars, 101)

    def test_comparison_requires_all_metrics_and_flags_each_gap(self):
        metrics = dict(self.reference['metrics'])
        metrics['hr@10'] -= 0.03
        comparison = compare_reference(metrics, self.reference)
        self.assertFalse(comparison['metrics']['hr@10']['within_tolerance'])
        self.assertTrue(comparison['metrics']['ndcg@10']['within_tolerance'])
        self.assertEqual(comparison['status'], 'REVIEW_REQUIRED')
        for bad in (float('nan'), -0.01, 1.01):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_reference({**metrics, 'hr@10': bad}, self.reference)


if __name__ == '__main__':
    unittest.main()
