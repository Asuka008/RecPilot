"""Encode frozen titles with the pinned MiniLM snapshot and masked mean pooling."""

import importlib.metadata
from pathlib import Path

from title_features.data import file_sha256


def masked_mean_pool(hidden, attention_mask):
    """Normalize [B,T,F] token states into [B,F], excluding padded tokens."""
    import torch
    from torch.nn import functional as F

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [B,T,F] and attention_mask [B,T]")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)) or not torch.all(attention_mask.sum(1) > 0):
        raise ValueError("Attention masks must be binary and nonempty")
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    if not torch.isfinite(pooled).all() or not torch.all(pooled.norm(dim=-1) > 0):
        raise ValueError("Encoder produced a nonfinite or zero feature")
    return F.normalize(pooled, dim=-1)


def encode_titles(config, titles):
    """Return only requested rows; callers distinguish smoke from full artifacts."""
    import numpy as np
    import torch

    if importlib.metadata.version("transformers") != config["transformers_version"]:
        raise ValueError("Install the pinned transformers version after reviewing server preflight")
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    required_files = ["config.json", "model.safetensors", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    snapshot = Path(snapshot_download(config["model_id"], revision=config["revision"], allow_patterns=required_files))
    if snapshot.name != config["revision"]:
        raise ValueError("Downloaded snapshot does not match the frozen revision")
    files = {name: file_sha256(snapshot / name) for name in required_files}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, dtype=torch.float32, attn_implementation="eager")
    model.eval().requires_grad_(False).to(device)
    if model.config.hidden_size != config["feature_dim"]:
        raise ValueError("Encoder dimension differs from configured features")
    result = np.zeros((len(titles), config["feature_dim"]), dtype=np.float32)
    truncated = 0
    repeat_checked = padding_checked = False

    def encode(batch):
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return masked_mean_pool(model(**encoded).last_hidden_state, encoded["attention_mask"])

    with torch.inference_mode():
        for start in range(1, len(titles), config["batch_size"]):
            batch = titles[start:start + config["batch_size"]]
            untruncated = tokenizer(batch, padding=False, truncation=False, verbose=False)["input_ids"]
            truncated += sum(len(tokens) > config["max_length"] for tokens in untruncated)
            vectors = encode(batch)
            if not repeat_checked:
                torch.testing.assert_close(vectors, encode(batch), atol=0, rtol=0)
                torch.testing.assert_close(vectors[:1], encode(batch[:1]), atol=1e-5, rtol=0)
                repeat_checked = padding_checked = True
            result[start:start + len(batch)] = vectors.cpu().numpy()
            print(f"Encoded titles {start + len(batch) - 1}/{len(titles) - 1}", flush=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Text encoder must remain frozen")
    after = {name: file_sha256(snapshot / name) for name in files}
    if after != files:
        raise RuntimeError("Encoder snapshot changed during inference")
    return result, {"snapshot_revision": snapshot.name, "snapshot_files_sha256": files,
                    "trainable_encoder_parameters": 0, "truncated_titles": truncated,
                    "repeat_output_exact": repeat_checked, "batch_padding_consistent": padding_checked,
                    "pooling": "attention-masked token mean, including special tokens; L2 normalized",
                    "attention": "eager", "precision": "float32; TF32 disabled"}
