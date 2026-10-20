"""Small synthetic batches and training operations for correctness checks."""

import hashlib
import json

import torch
from torch.nn import functional as F

from src.data.dataset import build_training_example


def json_digest(value):
    """Identify synthetic records and sampled negatives independently of formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_toy_batch(config, device):
    """Reuse the Dataset's shifted, left-padded pair construction without real data."""
    examples = [build_training_example(sequence, config["model"]["sequence_length"])
                for sequence in config["sequences"]]
    inputs = torch.tensor([example[0] for example in examples], dtype=torch.long)
    targets = torch.tensor([example[1] for example in examples], dtype=torch.long)
    valid = torch.tensor([example[2] for example in examples], dtype=torch.bool)
    negative_ids = torch.zeros(*inputs.shape, config["num_negatives"], dtype=torch.long)
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    catalog = torch.arange(1, config["model"]["num_items"] + 1)
    for user, position in valid.nonzero().tolist():
        candidates = catalog[catalog != targets[user, position]]
        selected = torch.randperm(len(candidates), generator=generator)[:config["num_negatives"]]
        negative_ids[user, position] = candidates[selected]
    records_hash = json_digest(config["sequences"])
    negatives_hash = json_digest(negative_ids.tolist())
    identity = {"synthetic_data_sha256": records_hash, "negative_ids_sha256": negatives_hash}
    batch = {"input_ids": inputs.to(device), "target_ids": targets.to(device),
             "valid_mask": valid.to(device), "negative_ids": negative_ids.to(device)}
    return batch, {**identity, "data_id": json_digest(identity), "data_version": "synthetic-id-sequences-v1"}


def sampled_loss(model, batch, temperature):
    """Compute positive-column cross entropy only at observed next-item positions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = batch["valid_mask"]
    if not valid.any():
        raise ValueError("Batch contains no supervised positions")
    targets = batch["target_ids"][valid]
    negatives = batch["negative_ids"][valid]
    if (targets <= 0).any() or (negatives <= 0).any() or (negatives == targets[:, None]).any():
        raise ValueError("Padding or the positive target cannot be used as a negative")
    hidden = model(batch["input_ids"], valid)
    candidates = torch.cat((targets[:, None], negatives), dim=1)
    logits = model.score_candidates(hidden[valid], candidates) / temperature
    return F.cross_entropy(logits, torch.zeros(len(targets), device=targets.device, dtype=torch.long))


def train_step(model, optimizer, batch, temperature):
    """Shuffle complete users together; do not shuffle items within a sequence."""
    model.train()
    order = torch.randperm(len(batch["input_ids"]), device=batch["input_ids"].device)
    shuffled = {name: value[order] for name, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = sampled_loss(model, shuffled, temperature)
    if not torch.isfinite(loss):
        raise AssertionError("Nonfinite training loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or nonfinite gradient: {name}")
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise AssertionError(f"Nonfinite updated parameter: {name}")
    return loss.item()


@torch.no_grad()
def evaluate_loss(model, batch, temperature):
    """Measure the same fixed training task without dropout or gradients."""
    model.eval()
    loss = sampled_loss(model, batch, temperature)
    if not torch.isfinite(loss):
        raise AssertionError("Nonfinite evaluation loss")
    return loss.item()
