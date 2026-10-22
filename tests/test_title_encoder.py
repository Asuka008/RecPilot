"""Server CPU numerical checks of masked token pooling; no model download required."""

import unittest

import torch

from title_features.encoder import masked_mean_pool


class TitleEncoderTest(unittest.TestCase):
    def test_hand_computed_pooling_ignores_padding(self):
        hidden = torch.tensor([[[1., 0.], [0., 1.], [100., 100.]]])
        pooled = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2 ** -.5, 2 ** -.5]]))
        torch.testing.assert_close(pooled, masked_mean_pool(hidden[:, :2], torch.ones((1, 2), dtype=torch.long)))

    def test_empty_nonbinary_or_wrong_shape_masks_are_rejected(self):
        hidden = torch.ones(1, 2, 3)
        for mask in (torch.tensor([[0, 0]]), torch.tensor([[1, 2]]), torch.ones(1, 3)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                masked_mean_pool(hidden, mask)

    def test_nonfinite_and_zero_features_are_rejected(self):
        for hidden in (torch.zeros(1, 2, 3), torch.full((1, 2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                masked_mean_pool(hidden, torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
