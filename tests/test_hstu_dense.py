"""Run on the server: tiny CPU contract checks for the portable HSTU block."""

import unittest

import torch

from src.models.hstu import DenseHSTUBlock, RelativeBias


class DenseHSTUTest(unittest.TestCase):
    def test_position_bias_matches_hand_computed_indices(self):
        bias = RelativeBias(3, "position_only")
        with torch.no_grad():
            bias._w.copy_(torch.arange(5))
        expected = torch.tensor([[[2., 3., 4.], [1., 2., 3.], [0., 1., 2.]]])
        torch.testing.assert_close(bias(), expected, rtol=0, atol=0)

    def test_time_bias_uses_next_query_time_and_log_buckets(self):
        bias = RelativeBias(3, "time_and_position")
        with torch.no_grad():
            bias._pos_w.zero_()
            bias._ts_w.copy_(torch.arange(129))
        timestamps = torch.tensor([[0, 1, 3]])
        expected = torch.tensor([[[0., 0., 2.], [3., 2., 0.], [3., 2., 0.]]])
        torch.testing.assert_close(bias(timestamps), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            bias()

    def test_padding_empty_row_and_finite_backward(self):
        torch.manual_seed(42)
        block = DenseHSTUBlock(6, 8, 4, 3, 2).eval()
        lengths = torch.tensor([5, 2, 0])
        valid = torch.arange(6)[None, :] < lengths[:, None]
        x = torch.randn(3, 6, 8, requires_grad=True)
        output = block(x, valid)
        self.assertEqual(tuple(output.shape), (3, 6, 8))
        torch.testing.assert_close(output[~valid], torch.zeros_like(output[~valid]), rtol=0, atol=0)
        output[valid].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        torch.testing.assert_close(x.grad[~valid], torch.zeros_like(x.grad[~valid]), rtol=0, atol=0)
        with torch.no_grad():
            poisoned = x.detach().clone()
            poisoned[~valid] = float("nan")
            torch.testing.assert_close(block(poisoned, valid), output.detach(), rtol=0, atol=0)

    def test_future_embedding_does_not_change_prefix(self):
        torch.manual_seed(123)
        block = DenseHSTUBlock(6, 8, 4, 3, 2).eval()
        x = torch.randn(2, 6, 8)
        valid = torch.tensor([[True] * 5 + [False], [True] * 3 + [False] * 3])
        with torch.no_grad():
            original = block(x, valid)
            changed = x.clone()
            changed[0, 4] += torch.linspace(-10, 10, 8)
            actual = block(changed, valid)
        torch.testing.assert_close(actual[0, :4], original[0, :4], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], original[1], rtol=0, atol=0)

    def test_invalid_layout_shape_and_mask_are_rejected(self):
        block = DenseHSTUBlock(6, 8, 4, 3, 2)
        x = torch.randn(1, 6, 8)
        left_padded = torch.tensor([[False, False, True, True, True, True]])
        with self.assertRaisesRegex(ValueError, "right-padded"):
            block(x, left_padded)
        with self.assertRaises(ValueError):
            block(x[:, :5], torch.ones(1, 5, dtype=torch.bool))
        with self.assertRaises(ValueError):
            block(x, torch.ones(1, 6, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
