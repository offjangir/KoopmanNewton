"""Method interface: a learning paradigm wrapping a network.

Every method exposes ``loss(batch)`` and ``sample(context)`` over the same
conditional next-state task, so AR / diffusion / flow / drifting are directly
comparable. The **stability** inductive bias (noisy-context training) is applied
uniformly here via :meth:`_noisy`.

``batch`` is a dict with ``context (B,H,F)`` and ``target (B,F)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models import ARTransformer, DiTBackbone


def build_network(model_cfg, state_dim: int) -> nn.Module:
    if model_cfg.kind == "ar_transformer":
        return ARTransformer(model_cfg, state_dim)
    if model_cfg.kind == "dit":
        return DiTBackbone(model_cfg, state_dim)
    raise ValueError(f"unknown model kind {model_cfg.kind}")


class Method(nn.Module):
    #: which model kind this method expects ("ar" or "dit")
    model_kind = "dit"

    def __init__(self, model_cfg, method_cfg, state_dim: int):
        super().__init__()
        self.mcfg = method_cfg
        self.state_dim = state_dim
        self.net = build_network(model_cfg, state_dim)

    def _noisy(self, context: torch.Tensor) -> torch.Tensor:
        s = self.mcfg.ctx_noise_sigma
        if s > 0 and self.training:
            return context + s * torch.randn_like(context)
        return context

    def loss(self, batch) -> torch.Tensor:                      # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:    # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def hidden_states(self, context: torch.Tensor) -> torch.Tensor:
        return self.net.hidden_states(context)
