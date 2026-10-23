"""Use one title-aware item representation for histories and candidate scores."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.id_hstu import IDHSTU


class ContentLookup(nn.Module):
    """Retain the original ID parameter and a checkpointed, frozen semantic table."""

    def __init__(self, embedding, semantic, representation, fusion_weight):
        super().__init__()
        if representation not in {"title-init", "title-fusion"}:
            raise ValueError("Unknown title representation")
        if semantic.shape != embedding.weight.shape or semantic.dtype != torch.float32:
            raise ValueError("Semantic table must match the FP32 ID embedding")
        if not torch.isfinite(semantic).all() or torch.any(semantic[0] != 0):
            raise ValueError("Semantic table must be finite with zero padding")
        if fusion_weight < 0 or not torch.isfinite(torch.tensor(fusion_weight)):
            raise ValueError("Invalid fusion weight")
        if representation == "title-init" and fusion_weight != 0:
            raise ValueError("Initialization requires zero fusion weight")
        self.weight = embedding.weight
        self.register_buffer("semantic", semantic.detach().clone())
        self.fusion_weight = float(fusion_weight)
        if representation == "title-init":
            with torch.no_grad():
                self.weight.copy_(semantic)

    def forward(self, ids):
        """Lookup the same effective table on both sides of tied scoring."""
        result = F.embedding(ids, self.weight, padding_idx=0)
        if self.fusion_weight:
            result = result + self.fusion_weight * F.embedding(ids, self.semantic, padding_idx=0)
        return result


class TitleHSTU(IDHSTU):
    """Preserve HSTU initialization RNG and all non-item architecture settings."""

    def __init__(self, *, semantic, representation, fusion_weight, **kwargs):
        super().__init__(**kwargs)
        self.reference_parameter_count = sum(p.numel() for p in self.parameters())
        self.item_embedding = ContentLookup(self.item_embedding, semantic, representation, fusion_weight)
