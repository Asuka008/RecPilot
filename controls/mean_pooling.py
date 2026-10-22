"""Causal prefix-mean encoder with the frozen HSTU ID initialization and scoring."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.id_hstu import IDHSTU, compact_ids


class PrefixMean(nn.Module):
    """Replace learned HSTU encoding with means of observed prefixes only.

    Construct the reference once on CPU to match its exact embedding initializer
    and initial RNG consumption. Only its ID table and input dropout are retained;
    no attention blocks or learned positions are parameters of this control.
    """

    def __init__(self, num_items, sequence_length, embedding_dim, attention_dim,
                 linear_dim, num_heads, num_blocks, dropout=0.0, epsilon=1e-6):
        super().__init__()
        reference = IDHSTU(num_items, sequence_length, embedding_dim, attention_dim,
                           linear_dim, num_heads, num_blocks, dropout, epsilon)
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.reference_parameter_count = sum(parameter.numel() for parameter in reference.parameters())
        self.item_embedding = reference.item_embedding
        self.input_dropout = reference.input_dropout
        # Dense prefix weights avoid CUDA floating-point cumsum under strict
        # deterministic mode in the pinned PyTorch 2.10 environment.
        self.register_buffer("prefix_weights", torch.ones(sequence_length, sequence_length).tril(), persistent=False)
        self.register_buffer("prefix_counts", torch.arange(1, sequence_length + 1).float()[:, None], persistent=False)

    def forward(self, input_ids, valid_mask=None):
        """Return normalized [B,L,D] prefix means in the caller's original layout."""
        if input_ids.ndim != 2 or input_ids.shape[1] != self.sequence_length:
            raise ValueError("ID history must have the configured sequence length")
        if (input_ids < 0).any() or (input_ids > self.num_items).any():
            raise ValueError("Item ID outside the configured catalog")
        valid_mask = input_ids != 0 if valid_mask is None else valid_mask
        packed, inverse = compact_ids(input_ids, valid_mask)
        present = packed != 0
        x = self.input_dropout(self.item_embedding(packed) * math.sqrt(self.embedding_dim))
        x = x.masked_fill(~present.unsqueeze(-1), 0)
        # Valid packed positions have exactly position+1 observed predecessors,
        # including themselves. Padded outputs are discarded after normalization.
        means = torch.matmul(self.prefix_weights, x) / self.prefix_counts
        hidden = F.normalize(means, dim=-1, eps=self.epsilon)
        hidden = hidden.gather(1, inverse.unsqueeze(-1).expand(-1, -1, self.embedding_dim))
        return hidden.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def score_candidates(self, queries, candidate_ids):
        """Delegate to the identical frozen normalized, tied ID scoring head."""
        return IDHSTU.score_candidates(self, queries, candidate_ids)
