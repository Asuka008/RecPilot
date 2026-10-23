"""Server CPU checks for shared history/candidate content representations."""

import unittest

import torch
from torch.nn import functional as F

from content_hstu.model import TitleHSTU
from src.models.id_hstu import IDHSTU


class ContentModelTest(unittest.TestCase):
    def setUp(self):
        self.settings = dict(num_items=7, sequence_length=4, embedding_dim=8, attention_dim=4,
                             linear_dim=4, num_heads=1, num_blocks=1, dropout=0.0)
        generator = torch.Generator().manual_seed(80)
        self.semantic = F.normalize(torch.randn(8, 8, generator=generator), dim=-1) * (0.02 * 8**0.5)
        self.semantic[0] = 0
        self.ids = torch.tensor([[0, 0, 1, 2], [0, 3, 2, 4]])

    def model(self, mode="title-fusion", weight=0.5):
        torch.manual_seed(42)
        return TitleHSTU(semantic=self.semantic, representation=mode, fusion_weight=weight, **self.settings)

    def test_zero_fusion_matches_id_model_outputs_parameters_and_rng(self):
        torch.manual_seed(42)
        baseline = IDHSTU(**self.settings).eval()
        rng = torch.get_rng_state().clone()
        model = self.model(weight=0).eval()
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name], parameter, atol=0, rtol=0)
        a, b = baseline(self.ids), model(self.ids)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        candidates = torch.arange(1, 8)
        torch.testing.assert_close(baseline.score_candidates(a[:, -1], candidates), model.score_candidates(b[:, -1], candidates), atol=0, rtol=0)

    def test_initialization_changes_only_item_weights(self):
        baseline = self.model(weight=0)
        model = self.model("title-init", 0)
        torch.testing.assert_close(model.item_embedding.weight, self.semantic, atol=0, rtol=0)
        for name, parameter in baseline.named_parameters():
            if name != "item_embedding.weight":
                torch.testing.assert_close(dict(model.named_parameters())[name], parameter, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), model.reference_parameter_count)

    def test_fusion_formula_shared_and_per_query_scoring(self):
        model = self.model()
        ids = torch.arange(1, 8)
        expected = model.item_embedding.weight[ids] + 0.5 * self.semantic[ids]
        torch.testing.assert_close(model.item_embedding(ids), expected, atol=0, rtol=0)
        queries = model(self.ids)[:, -1]
        scores = queries @ F.normalize(expected, dim=-1).T
        torch.testing.assert_close(model.score_candidates(queries, ids), scores)
        torch.testing.assert_close(model.score_candidates(queries, ids.expand(2, -1)), scores)
        torch.testing.assert_close(model.item_embedding(torch.tensor([0])), torch.zeros(1, 8), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model.score_candidates(queries, torch.tensor([0]))

    def test_causality_and_padding_layout(self):
        model = self.model().eval()
        left = torch.tensor([[0, 1, 2, 3]])
        right = torch.tensor([[1, 2, 3, 0]])
        torch.testing.assert_close(model(left)[:, 1:], model(right)[:, :3], atol=0, rtol=0)
        changed = torch.tensor([[0, 1, 2, 7]])
        torch.testing.assert_close(model(left)[:, :3], model(changed)[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(model(left)[:, 0], torch.zeros(1, 8), atol=0, rtol=0)

    def test_optimizer_updates_trainable_parameters_but_not_semantic_buffer(self):
        for mode, weight in (("title-init", 0), ("title-fusion", 0.5)):
            model = self.model(mode, weight)
            before = model.item_embedding.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scores = model.score_candidates(model(self.ids)[:, -1], torch.arange(1, 8))
            F.cross_entropy(scores / 0.2, torch.tensor([4, 5])).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(before, model.item_embedding.weight))
            self.assertFalse(model.item_embedding.semantic.requires_grad)
            torch.testing.assert_close(model.item_embedding.semantic, self.semantic, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
