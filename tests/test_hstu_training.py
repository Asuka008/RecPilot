"""Run on the server: tiny CPU adapter, objective and checkpoint contract checks."""

import copy
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.checkpoints import load_checkpoint, save_checkpoint
from src.models.id_hstu import IDHSTU, compact_ids
from src.toy_training import make_toy_batch, sampled_loss, train_step


CONFIG = {
    "seed": 42, "model": {"num_items": 10, "sequence_length": 4, "embedding_dim": 8,
                          "attention_dim": 4, "linear_dim": 4, "num_heads": 1,
                          "num_blocks": 1, "dropout": 0.0},
    "sequences": [[1, 2, 3, 4], [5, 6, 7]], "num_negatives": 3,
}


class TrainingContractTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def create_pair(self):
        model = IDHSTU(**CONFIG["model"])
        return model, torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)

    def test_compaction_preserves_order_and_inverse_layout(self):
        ids = torch.tensor([[0, 0, 3, 1], [4, 2, 0, 0]])
        packed, inverse = compact_ids(ids, ids != 0)
        torch.testing.assert_close(packed, torch.tensor([[3, 1, 0, 0], [4, 2, 0, 0]]))
        torch.testing.assert_close(packed.gather(1, inverse), ids)
        with self.assertRaises(ValueError):
            compact_ids(ids, torch.ones_like(ids, dtype=torch.bool))

    def test_model_preserves_left_and_right_layout_equivalence(self):
        model, _ = self.create_pair()
        model.eval()
        left = torch.tensor([[0, 0, 1, 2]])
        right = torch.tensor([[1, 2, 0, 0]])
        with torch.no_grad():
            left_output, right_output = model(left), model(right)
        torch.testing.assert_close(left_output[:, 2:], right_output[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(left_output[:, :2], torch.zeros_like(left_output[:, :2]), rtol=0, atol=0)

    def test_fixed_negatives_and_dataset_targets_are_aligned(self):
        batch, identity = make_toy_batch(CONFIG, torch.device("cpu"))
        again, identity_again = make_toy_batch(CONFIG, torch.device("cpu"))
        self.assertEqual(identity, identity_again)
        torch.testing.assert_close(batch["input_ids"][0], torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(batch["target_ids"][0], torch.tensor([0, 2, 3, 4]))
        torch.testing.assert_close(batch["negative_ids"], again["negative_ids"])
        valid = batch["valid_mask"]
        self.assertTrue((batch["negative_ids"][valid] > 0).all())
        self.assertFalse((batch["negative_ids"][valid] == batch["target_ids"][valid][:, None]).any())

    def test_padding_labels_are_ignored_and_invalid_negatives_rejected(self):
        model, _ = self.create_pair()
        batch, _ = make_toy_batch(CONFIG, torch.device("cpu"))
        changed = {name: value.clone() for name, value in batch.items()}
        changed["target_ids"][~batch["valid_mask"]] = 9
        changed["negative_ids"][~batch["valid_mask"]] = 8
        torch.testing.assert_close(sampled_loss(model, batch, 0.2), sampled_loss(model, changed, 0.2), rtol=0, atol=0)
        changed["negative_ids"][0, 1, 0] = batch["target_ids"][0, 1]
        with self.assertRaises(ValueError):
            sampled_loss(model, changed, 0.2)

    def test_checkpoint_restores_output_optimizer_and_random_generators(self):
        batch, identity = make_toy_batch(CONFIG, torch.device("cpu"))
        model, optimizer = self.create_pair()
        train_step(model, optimizer, batch, 0.2)
        model.eval()
        with torch.no_grad():
            before = model(batch["input_ids"]).clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, config=CONFIG, epoch=1, step=1,
                            metrics={"loss": 1.0}, provenance=identity)
            expected_random = (random.random(), np.random.random(), torch.rand(3))
            expected_loss = train_step(model, optimizer, batch, 0.2)
            restored, restored_optimizer = self.create_pair()
            payload = load_checkpoint(path, restored, restored_optimizer, expected_config=CONFIG,
                                      expected_data_id=identity["data_id"])
            self.assertEqual(payload["step"], 1)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(batch["input_ids"]), before, rtol=0, atol=0)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.random(), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
            self.assertEqual(train_step(restored, restored_optimizer, batch, 0.2), expected_loss)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            for parameter, state in optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(restored_optimizer.state_dict()["state"][parameter][name],
                                               value, rtol=0, atol=0)
            wrong = copy.deepcopy(CONFIG)
            wrong["seed"] = 123
            with self.assertRaises(ValueError):
                load_checkpoint(path, restored, restored_optimizer, expected_config=wrong,
                                expected_data_id=identity["data_id"])
            with self.assertRaises(ValueError):
                load_checkpoint(path, restored, restored_optimizer, expected_config=CONFIG,
                                expected_data_id="different-data")


if __name__ == "__main__":
    unittest.main()
