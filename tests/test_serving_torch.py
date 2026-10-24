"""Server-only tiny CPU parity check for the portable neural adapter."""
import unittest
import numpy as np
import torch

from serving.runtime import TorchScorer
from src.models.id_hstu import IDHSTU


class ServingTorchTest(unittest.TestCase):
    def test_padding_truncation_and_repeatable_inference(self):
        torch.manual_seed(42)
        model = IDHSTU(num_items=8, sequence_length=4, embedding_dim=4,
                       attention_dim=2, linear_dim=2, num_heads=1, num_blocks=1, dropout=.1).eval()
        scorer = TorchScorer(model, torch.device('cpu'))
        for history, padded in [([1], [0, 0, 0, 1]), ([1, 2, 3, 4, 5], [2, 3, 4, 5])]:
            with torch.inference_mode():
                ids = torch.tensor([padded])
                expected = model.score_candidates(model(ids, ids != 0)[:, -1], torch.arange(1, 9))[0].numpy()
            actual = scorer(history)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, scorer(history))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
