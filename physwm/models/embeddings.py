"""Shared embedding utilities: sinusoidal time + learned positional embeddings."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a (B,) continuous scalar (e.g. diffusion/flow time)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """sinusoidal(t) -> MLP -> (B, d). Returns zeros when t is None (drifting)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor | None, batch: int, device) -> torch.Tensor:
        if t is None:
            return torch.zeros(batch, self.d_model, device=device)
        return self.mlp(sinusoidal_embedding(t, self.d_model))
