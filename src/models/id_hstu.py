"""A small pure-ID HSTU stack with a portable sequence-layout adapter."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.hstu import DenseHSTUBlock


def compact_ids(input_ids, valid_mask):
    """Pack valid IDs in order, returning an inverse map to the original layout."""
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be an int64 [B, L] tensor")
    if valid_mask.shape != input_ids.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean [B, L] tensor")
    if valid_mask.device != input_ids.device or not torch.equal(valid_mask, input_ids != 0):
        raise ValueError("Mask must agree with nonzero IDs on the same device")
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device).expand_as(input_ids)
    permutation = torch.argsort((~valid_mask).long() * length + positions, dim=1)
    return input_ids.gather(1, permutation), torch.argsort(permutation, dim=1)


class IDHSTU(nn.Module):
    """ID lookup, learned positions, position-only HSTU blocks and tied scoring.

    Hidden states retain the caller's layout so existing left-padded Dataset
    targets remain aligned. No future timestamps or content features are used.
    """

    def __init__(self, num_items, sequence_length, embedding_dim, attention_dim,
                 linear_dim, num_heads, num_blocks, dropout=0.0, epsilon=1e-6):
        super().__init__()
        if type(num_items) is not int or num_items < 2 or type(num_blocks) is not int or num_blocks < 1:
            raise ValueError("At least two items and one block are required")
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding.weight, std=1 / math.sqrt(embedding_dim))
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DenseHSTUBlock(sequence_length, embedding_dim, attention_dim, linear_dim,
                           num_heads, bias_mode="position_only", dropout=dropout, epsilon=epsilon)
            for _ in range(num_blocks)
        ])

    def forward(self, input_ids, valid_mask=None):
        """Return normalized [B, L, D] states aligned to the original IDs."""
        if input_ids.ndim != 2 or input_ids.shape[1] != self.sequence_length:
            raise ValueError("ID history must have the configured sequence length")
        if (input_ids < 0).any() or (input_ids > self.num_items).any():
            raise ValueError("Item ID outside the configured catalog")
        valid_mask = input_ids != 0 if valid_mask is None else valid_mask
        packed, inverse = compact_ids(input_ids, valid_mask)
        present = packed != 0
        positions = torch.arange(self.sequence_length, device=input_ids.device)
        x = self.item_embedding(packed) * math.sqrt(self.embedding_dim)
        x = self.input_dropout(x + self.position_embedding(positions)[None])
        x = x.masked_fill(~present.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, present)
        x = F.normalize(x, dim=-1, eps=self.epsilon)
        x = x.gather(1, inverse.unsqueeze(-1).expand(-1, -1, self.embedding_dim))
        return x.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def score_candidates(self, queries, candidate_ids):
        """Score [N,D] queries against shared [K] or per-query [N,K] IDs."""
        if queries.ndim != 2 or queries.shape[1] != self.embedding_dim:
            raise ValueError("Queries must have shape [N, D]")
        if candidate_ids.dtype != torch.long or candidate_ids.ndim not in (1, 2):
            raise ValueError("Candidate IDs must be int64 [K] or [N, K]")
        if candidate_ids.ndim == 2 and candidate_ids.shape[0] != queries.shape[0]:
            raise ValueError("Per-query candidate batch size does not match queries")
        if (candidate_ids <= 0).any() or (candidate_ids > self.num_items).any():
            raise ValueError("Candidates must exclude padding and stay within the catalog")
        embeddings = F.normalize(self.item_embedding(candidate_ids), dim=-1, eps=self.epsilon)
        return (queries @ embeddings.T if candidate_ids.ndim == 1 else
                torch.einsum("nd,nkd->nk", queries, embeddings))
