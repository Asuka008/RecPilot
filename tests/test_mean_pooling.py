"""Run on server CPU: hand-checked prefix, padding, scoring and gradient contracts."""

import math
import unittest

import torch

from controls.mean_pooling import PrefixMean
from src.microlens_training import full_catalog_loss, optimization_step
from src.models.id_hstu import IDHSTU


MODEL = {"num_items": 4, "sequence_length": 4, "embedding_dim": 2, "attention_dim": 1,
         "linear_dim": 1, "num_heads": 1, "num_blocks": 1, "dropout": 0.0}


def manual_model():
    """Use tiny noncollinear ID vectors for transparent expected prefix means."""
    model = PrefixMean(**MODEL)
    with torch.no_grad():
        model.item_embedding.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [1., 1.]]))
    model.eval()
    return model


class MeanPoolingTest(unittest.TestCase):
    def test_initial_lookup_rng_and_candidate_scoring_match_hstu_exactly(self):
        torch.manual_seed(42)
        reference = IDHSTU(**MODEL)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        control = PrefixMean(**MODEL)
        torch.testing.assert_close(control.item_embedding.weight, reference.item_embedding.weight, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), reference_rng, atol=0, rtol=0)
        self.assertEqual(list(dict(control.named_parameters())), ["item_embedding.weight"])
        self.assertEqual(list(control.state_dict()), ["item_embedding.weight"])
        self.assertLess(sum(p.numel() for p in control.parameters()), control.reference_parameter_count)
        queries = torch.tensor([[1., 0.], [0., 1.]])
        for candidates in (torch.tensor([1, 2, 3]), torch.tensor([[1, 2], [3, 4]])):
            torch.testing.assert_close(control.score_candidates(queries, candidates),
                                       reference.score_candidates(queries, candidates), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            control.score_candidates(queries, torch.tensor([0]))

    def test_hand_computed_prefix_means_padding_and_layout(self):
        model = manual_model()
        left = torch.tensor([[0, 1, 2, 3]])
        actual = model(left)
        expected = torch.tensor([[[0., 0.], [1., 0.], [1/math.sqrt(2), 1/math.sqrt(2)], [0., 1.]]])
        torch.testing.assert_close(actual, expected)
        right = torch.tensor([[1, 2, 3, 0]])
        torch.testing.assert_close(model(right)[:, :3], actual[:, 1:], atol=0, rtol=0)
        torch.testing.assert_close(model(torch.zeros_like(left)), torch.zeros_like(actual), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            model(left, torch.ones_like(left, dtype=torch.bool))

    def test_future_items_do_not_change_or_receive_gradient_from_earlier_prefix(self):
        model = manual_model()
        original = torch.tensor([[0, 1, 2, 3]])
        changed = torch.tensor([[0, 1, 4, 4]])
        torch.testing.assert_close(model(original)[:, :2], model(changed)[:, :2], atol=0, rtol=0)
        model(original)[0, 1, 1].backward()
        self.assertGreater(model.item_embedding.weight.grad[1].abs().sum().item(), 0)
        torch.testing.assert_close(model.item_embedding.weight.grad[2:], torch.zeros_like(model.item_embedding.weight.grad[2:]), atol=0, rtol=0)

    def test_final_prefix_is_order_invariant_and_counts_repeated_events(self):
        model = manual_model()
        first = model(torch.tensor([[0, 1, 2, 4]]))[:, -1]
        permuted = model(torch.tensor([[0, 2, 1, 4]]))[:, -1]
        torch.testing.assert_close(first, permuted)
        repeated = model(torch.tensor([[0, 1, 1, 2]]))[:, -1]
        torch.testing.assert_close(repeated, torch.tensor([[2/math.sqrt(5), 1/math.sqrt(5)]]))

    def test_shared_objective_ignores_padding_and_updates_finite_embedding(self):
        model = PrefixMean(**MODEL)
        batch = {"input_ids": torch.tensor([[0, 0, 1, 2]]),
                 "padding_mask": torch.tensor([[False, False, True, True]]),
                 "target_ids": torch.tensor([[0, 0, 2, 3]])}
        before = model.item_embedding.weight.detach().clone()
        loss = full_catalog_loss(model, batch, 0.2)
        batch["target_ids"][~batch["padding_mask"]] = 999
        torch.testing.assert_close(full_catalog_loss(model, batch, 0.2), loss, atol=0, rtol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        result, norm = optimization_step(model, optimizer, batch, {"temperature": 0.2, "gradient_clip_norm": 1.0})
        self.assertTrue(math.isfinite(result) and math.isfinite(norm))
        self.assertGreater((model.item_embedding.weight.detach()-before).abs().max().item(), 0)
        torch.testing.assert_close(model.item_embedding.weight[0], torch.zeros(2), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
