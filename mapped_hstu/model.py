"""Use shared content parameters instead of a free per-item ID table."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.id_hstu import IDHSTU


class SharedProjection(nn.Module):
    """Register fixed weights as buffers and learned weights as parameters."""

    def __init__(self, components, learned):
        super().__init__()
        weight = components.T.float().detach().clone()
        if learned:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer('weight', weight)

    def forward(self, values):
        """Use the same linear operation for both experiment arms."""
        return F.linear(values, self.weight)


class SharedLookup(nn.Module):
    """Map frozen titles through a shared centered linear projection on both sides."""

    def __init__(self, semantic, mean, components, learned):
        super().__init__()
        if (semantic.ndim != 2 or mean.shape != (semantic.shape[1],)
                or components.shape[0] != semantic.shape[1]
                or semantic.dtype != torch.float32 or torch.any(semantic[0] != 0)
                or not all(torch.isfinite(x).all() for x in (semantic, mean, components))):
            raise ValueError('Invalid aligned features or projection')
        self.register_buffer('semantic', semantic.detach().clone())
        self.register_buffer('center', mean.float().detach().clone())
        self.projector = SharedProjection(components, learned)
        self.scale = .02 * math.sqrt(components.shape[1])

    def forward(self, ids):
        """Center, project and normalize; padding remains exactly zero."""
        content = F.embedding(ids, self.semantic) - self.center
        values = F.normalize(self.projector(content), dim=-1, eps=1e-12) * self.scale
        return values.masked_fill((ids == 0).unsqueeze(-1), 0)

    @property
    def weight(self):
        """Expose the effective table for diagnostics, never as an ID parameter."""
        return self(torch.arange(len(self.semantic), device=self.semantic.device))


class TitleHSTU(IDHSTU):
    """Preserve HSTU initialization while removing the independent item parameters."""

    def __init__(self, *, semantic, mean, components, representation, fusion_weight=0, **kwargs):
        if representation not in {'map-frozen', 'map-learned'} or fusion_weight != 0:
            raise ValueError('Invalid shared mapping recipe')
        super().__init__(**kwargs)
        self.reference_parameter_count = sum(p.numel() for p in self.parameters())
        if semantic.shape[0] != self.num_items + 1 or components.shape[1] != self.embedding_dim:
            raise ValueError('Feature catalog or output dimension mismatch')
        # Constructor randomness must not shift the matched backbone/dropout stream.
        state = torch.get_rng_state()
        self.item_embedding = SharedLookup(semantic, mean, components, representation == 'map-learned')
        torch.set_rng_state(state)
