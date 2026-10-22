"""Full-catalog training and batched evaluation of the portable pure-ID HSTU."""

from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.evaluation import RankingAccumulator


def make_loader(dataset, indices, batch_size, workers, seed, device, shuffle=False):
    """Seed each epoch's order independently, enabling epoch-boundary continuation."""
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


def full_catalog_loss(model, batch, temperature):
    """Supervise observed shifted positions against all positive catalog IDs.

    Earlier history items are valid output classes. No held-out values are used
    to alter the denominator; unobserved classes are not exposure negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = batch["padding_mask"]
    if not mask.any():
        raise ValueError("No supervised positions")
    targets = batch["target_ids"][mask]
    if (targets <= 0).any() or (targets > model.num_items).any():
        raise ValueError("Training targets must exclude padding and remain inside the catalog")
    queries = model(batch["input_ids"], mask)[mask]
    candidates = torch.arange(1, model.num_items + 1, device=queries.device)
    logits = model.score_candidates(queries, candidates) / temperature
    return F.cross_entropy(logits, targets - 1)


def move_batch(batch, device):
    """Keep user IDs on the host; move model tensors onto the configured device."""
    return {key: value.to(device) if key != "user_id" else value for key, value in batch.items()}


def optimization_step(model, optimizer, batch, config):
    """Reject nonfinite losses/gradients and clip before updating parameters."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = full_catalog_loss(model, batch, config["temperature"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing or nonfinite gradient: {name}")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameter")
    return loss.item(), norm.item()


def train_epoch(model, optimizer, loader, config, device):
    """Average losses by supervised positions rather than uneven batch sizes."""
    total, positions, steps, max_norm = 0.0, 0, 0, 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        count = int(batch["padding_mask"].sum())
        loss, norm = optimization_step(model, optimizer, batch, config)
        total += loss * count
        positions += count
        steps += 1
        max_norm = max(max_norm, norm)
    if positions == 0:
        raise ValueError("Training loader is empty")
    return {"loss": total / positions, "supervised_positions": positions,
            "steps": steps, "max_gradient_norm_before_clip": max_norm}


@torch.no_grad()
def fixed_batch_loss(model, batch, temperature):
    """Measure a fixed training-only probe without dropout or parameter updates."""
    model.eval()
    loss = full_catalog_loss(model, batch, temperature)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite probe loss")
    return loss.item()


@torch.no_grad()
def evaluate_hstu(model, dataset, indices, config, device, training_records):
    """Use shared CPU ranking rules after batched exact neural catalog scoring."""
    model.eval()
    frequency = Counter(item for record in training_records for item in record["train"])
    accumulator = RankingAccumulator(config["ks"], frequency)
    records = {record["user_id"]: record for record in dataset.records}
    candidates = torch.arange(1, model.num_items + 1, device=device)
    loader = make_loader(dataset, indices, config["eval_batch_size"], config["num_workers"],
                         config["seed"], device)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["padding_mask"].to(device)
        if not mask[:, -1].all():
            raise ValueError("Evaluation requires nonempty left-padded histories")
        queries = model(ids, mask)[:, -1]
        values = model.score_candidates(queries, candidates).cpu().numpy()
        for user_id, row in zip(batch["user_id"].tolist(), values):
            record = records[user_id]
            history = record["train"] + ([record["val"]] if dataset.split == "test" else [])
            scores = np.concatenate(([0.0], row))
            accumulator.add(scores, history, record[dataset.split])
    return accumulator.result(dataset.split)
