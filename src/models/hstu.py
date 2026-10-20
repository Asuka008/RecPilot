# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 AgenticRec contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
# Adapted to dense tensors from meta-recsys/generative-recommenders,
# revision 10c4950343b98bacec9ffd5b8cd879f811fe2326, research/modeling/sequential/hstu.py.

"""Dense PyTorch HSTU reference blocks, with no FBGEMM or upstream imports."""

import torch
from torch import nn
from torch.nn import functional as F


class RelativeBias(nn.Module):
    """Match upstream relative position and optional next-event time buckets.

    Time mode exists for reference parity. It requires next-event timestamps and
    must not be treated as an online-safe history-only feature definition.
    """

    def __init__(self, sequence_length: int, mode: str, num_buckets: int = 128):
        super().__init__()
        if sequence_length < 2 or num_buckets < 1:
            raise ValueError("sequence_length >= 2 and num_buckets >= 1 are required")
        if mode not in {"position_only", "time_and_position"}:
            raise ValueError("Unknown relative bias mode")
        self.sequence_length = sequence_length
        self.mode = mode
        self.num_buckets = num_buckets
        # Preserve upstream parameter names for strict state_dict transfers.
        if mode == "position_only":
            self._w = nn.Parameter(torch.empty(2 * sequence_length - 1).normal_(std=0.02))
        else:
            self._pos_w = nn.Parameter(torch.empty(2 * sequence_length - 1).normal_(std=0.02))
            self._ts_w = nn.Parameter(torch.empty(num_buckets + 1).normal_(std=0.02))

    def forward(self, timestamps=None):
        """Return [1, L, L] position bias or [B, L, L] combined bias."""
        weights = self._w if self.mode == "position_only" else self._pos_w
        positions = torch.arange(self.sequence_length, device=weights.device)
        indices = self.sequence_length - 1 + positions[None, :] - positions[:, None]
        bias = weights[indices].unsqueeze(0)
        if self.mode == "position_only":
            return bias
        if (timestamps is None or timestamps.ndim != 2
                or timestamps.shape[1] != self.sequence_length):
            raise ValueError("Time bias requires timestamps with shape [B, L]")
        if timestamps.device != weights.device:
            raise ValueError("Timestamps and parameters must be on the same device")
        if timestamps.dtype not in (torch.int32, torch.int64):
            raise ValueError("Timestamps must use integer units matching the reference data")
        next_times = torch.cat((timestamps[:, 1:], timestamps[:, -1:]), dim=1)
        differences = next_times[:, :, None] - timestamps[:, None, :]
        buckets = (torch.log(differences.abs().clamp(min=1)) / 0.301).long()
        buckets = buckets.clamp(min=0, max=self.num_buckets)
        return bias + self._ts_w[buckets]


class DenseHSTUBlock(nn.Module):
    """Reference UVQK/SiLU/gating block on right-padded [B, L, D] inputs.

    True in valid_mask denotes an observed item. Sequence length L is fixed:
    upstream attention scales by L, rather than each user's observed length.
    This implements the rel_bias, uvqk, silu, concat_ua=False path only.
    """

    def __init__(self, sequence_length: int, embedding_dim: int, attention_dim: int,
                 linear_dim: int, num_heads: int, bias_mode: str = "position_only",
                 dropout: float = 0.0, epsilon: float = 1e-6):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in
               (sequence_length, embedding_dim, attention_dim, linear_dim, num_heads)):
            raise ValueError("All dimensions must be positive integers")
        if not 0 <= dropout < 1 or epsilon <= 0:
            raise ValueError("Invalid dropout or epsilon")
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self._rel_attn_bias = RelativeBias(sequence_length, bias_mode)
        width = num_heads * (2 * linear_dim + 2 * attention_dim)
        self._uvqk = nn.Parameter(torch.empty(embedding_dim, width).normal_(std=0.02))
        self._o = nn.Linear(num_heads * linear_dim, embedding_dim)
        nn.init.xavier_uniform_(self._o.weight)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, timestamps=None):
        """Return [B, L, D], with exact zeros at all padding positions."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.sequence_length, self.embedding_dim):
            raise ValueError("Input must have shape [B, configured L, configured D]")
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean [B, L] tensor")
        if valid_mask.device != x.device:
            raise ValueError("Input and mask must be on the same device")
        if (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
            raise ValueError("Inputs must be right-padded; compact left-padded histories first")
        if timestamps is not None and timestamps.shape != x.shape[:2]:
            raise ValueError("timestamps must have shape [B, L]")
        batch, length, _ = x.shape
        present = valid_mask.unsqueeze(-1)
        clean_x = x.masked_fill(~present, 0)
        normalized = F.layer_norm(clean_x, (self.embedding_dim,), eps=self.epsilon)
        projected = F.silu(normalized.reshape(-1, self.embedding_dim) @ self._uvqk)
        projected = projected.view(batch, length, -1).masked_fill(~present, 0)
        u, v, q, k = torch.split(projected, [self.linear_dim * self.num_heads,
                                           self.linear_dim * self.num_heads,
                                           self.attention_dim * self.num_heads,
                                           self.attention_dim * self.num_heads], dim=-1)
        q = q.reshape(batch, length, self.num_heads, self.attention_dim)
        k = k.reshape(batch, length, self.num_heads, self.attention_dim)
        v = v.reshape(batch, length, self.num_heads, self.linear_dim)
        attention = torch.einsum("bnhd,bmhd->bhnm", q, k)
        attention = F.silu(attention + self._rel_attn_bias(timestamps).unsqueeze(1)) / length
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        attention = attention.masked_fill(~allowed, 0)
        aggregate = torch.einsum("bhnm,bmhd->bnhd", attention, v)
        aggregate = aggregate.reshape(batch, length, self.num_heads * self.linear_dim)
        gate_input = u * F.layer_norm(aggregate, (self.num_heads * self.linear_dim,), eps=self.epsilon)
        output = self._o(F.dropout(gate_input, p=self.dropout, training=self.training)) + clean_x
        return output.masked_fill(~present, 0)
