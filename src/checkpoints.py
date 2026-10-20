"""State-dict checkpoints with explicit configuration, provenance and RNG state."""

from pathlib import Path
import random

import numpy as np
import torch


def capture_rng_state():
    """Encode NumPy state as primitives so weights_only loading remains usable."""
    state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "numpy": {"name": state[0], "keys": state[1].tolist(), "position": state[2],
                      "has_gauss": state[3], "cached_gaussian": state[4]}}


def restore_rng_state(state):
    """Restore after constructing/loading models, since initialization consumes RNG."""
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state["name"], np.array(np_state["keys"], dtype=np.uint32),
                         np_state["position"], np_state["has_gauss"], np_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG restoration requires the same visible device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_checkpoint(path, model, optimizer, *, config, epoch, step, metrics, provenance):
    """Save state dictionaries atomically; never serialize a live model object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "config": config,
               "epoch": epoch, "step": step, "metrics": metrics, "provenance": provenance,
               "rng_state": capture_rng_state()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, *, expected_config, expected_data_id):
    """Validate experiment identity before loading parameters and RNG state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1 or payload["config"] != expected_config:
        raise ValueError("Checkpoint schema or configuration does not match")
    if payload["provenance"]["data_id"] != expected_data_id:
        raise ValueError("Checkpoint data identity does not match")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload
