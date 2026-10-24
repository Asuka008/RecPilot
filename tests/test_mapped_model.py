"""Server CPU checks of shared mapping, causal reuse and checkpoint continuation."""

import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from mapped_hstu.model import TitleHSTU
from mapped_hstu.run import run
from tests import test_content_runner as runner_fixture


class MappedModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.features = F.normalize(torch.randn(8, 12), dim=-1)
        self.features[0] = 0
        self.mean = self.features[1:5].mean(0)
        self.components = torch.randn(12, 8)
        self.kwargs = dict(num_items=7, sequence_length=4, embedding_dim=8, attention_dim=4,
                           linear_dim=4, num_heads=1, num_blocks=1, dropout=0.)
        self.ids = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    def model(self, mode):
        torch.manual_seed(42)
        return TitleHSTU(semantic=self.features, mean=self.mean, components=self.components,
                         representation=mode, **self.kwargs)

    def test_pair_has_identical_initial_outputs_and_no_independent_id_parameters(self):
        frozen, learned = self.model('map-frozen'), self.model('map-learned')
        torch.testing.assert_close(frozen(self.ids), learned(self.ids), atol=0, rtol=0)
        self.assertNotIn('item_embedding.weight', dict(learned.named_parameters()))
        self.assertFalse(frozen.item_embedding.projector.weight.requires_grad)
        self.assertTrue(learned.item_embedding.projector.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in learned.parameters() if p.requires_grad) -
                         sum(p.numel() for p in frozen.parameters() if p.requires_grad), 12*8)

    def test_shared_formula_padding_and_causality(self):
        model = self.model('map-learned')
        table = F.normalize((self.features - self.mean) @ self.components, dim=-1) * (.02 * 8**.5)
        table[0] = 0
        torch.testing.assert_close(model.item_embedding.weight, table)
        query = model(self.ids)[:, -1]
        scores = model.score_candidates(query, torch.arange(1, 8))
        torch.testing.assert_close(scores, query @ F.normalize(table[1:], dim=-1).T)
        changed = self.ids.clone()
        changed[:, -1] = 7
        torch.testing.assert_close(model(self.ids)[:, :-1], model(changed)[:, :-1], atol=0, rtol=0)
        torch.testing.assert_close(model.item_embedding(torch.tensor([0])), torch.zeros(1, 8), atol=0, rtol=0)

    def test_gradients_update_shared_map_without_changing_cached_features(self):
        for mode in ('map-frozen', 'map-learned'):
            model = self.model(mode)
            before = model.item_embedding.projector.weight.detach().clone()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
            F.cross_entropy(model.score_candidates(model(self.ids)[:, -1], torch.arange(1, 8)) / .2,
                            torch.tensor([4, 5])).backward()
            for p in model.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
            self.assertEqual(torch.equal(before, model.item_embedding.projector.weight), mode == 'map-frozen')
            torch.testing.assert_close(model.item_embedding.semantic, self.features, atol=0, rtol=0)

    def test_full_runner_resume_and_frozen_control(self):
        fixture = runner_fixture.ContentRunnerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config.update(representation='map-learned', fusion_weight=0)
        with patch.object(runner_fixture, 'run', run):
            full_dir, full = fixture.launch()
            partial_dir, _ = fixture.launch(stop=1)
            resumed_dir, resumed = fixture.launch(resume=partial_dir / 'last.pt')
            self.assertEqual(full['selection'], resumed['selection'])
            full_payload = torch.load(full_dir / 'last.pt', weights_only=True)
            resumed_payload = torch.load(resumed_dir / 'last.pt', weights_only=True)
            fixture.assert_same(full_payload['model_state_dict'], resumed_payload['model_state_dict'])
            fixture.assert_same(full_payload['optimizer_state_dict'], resumed_payload['optimizer_state_dict'])
            _, evaluated = fixture.launch(action='evaluate', checkpoint=full_dir / 'best.pt')
            best = full['history'][full['selection']['best_epoch'] - 1]['validation']
            fixture.assert_same(evaluated['evaluation'], best)
            fixture.config['representation'] = 'map-frozen'
            _, frozen = fixture.launch()
            self.assertEqual(frozen['status'], 'PASS')
            self.assertEqual(frozen['mapping']['independent_id_parameters'], 0)
            with self.assertRaisesRegex(ValueError, 'validation only'):
                fixture.launch(action='evaluate', checkpoint=full_dir / 'best.pt', split='test')


if __name__ == '__main__':
    unittest.main()
