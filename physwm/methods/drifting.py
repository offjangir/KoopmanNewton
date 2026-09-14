"""Generative Modeling via Drifting (arXiv 2602.04770), conditional variant.

A single network ``f_theta(eps, context)`` maps noise directly to the next state
in one forward pass (no sampler, no time input). Training uses the kernel drift
field ``V = V+ - V-``: attraction toward the data target and repulsion among the
model's own samples, with the stop-gradient MSE objective
``E || f - stopgrad(f + V) ||^2``. At equilibrium the drift vanishes.

For the (near-)deterministic conditional physics targets we draw ``K`` particles
per context: the positive is the single data target (attraction collapses the
particle cloud onto it) and the negatives are the other particles (repulsion
keeps the cloud from degenerating during training). Kernel: ``exp(-||.||/tau)``.
"""

from __future__ import annotations

import torch

from .base import Method


class DriftingMethod(Method):
    model_kind = "dit"

    def _drift(self, x: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
        """x (B,K,F) model particles, y_pos (B,F) data target -> drift (B,K,F)."""
        tau = self.mcfg.drift_tau
        # attraction toward the single positive (kernel normalization cancels)
        V_pos = y_pos.unsqueeze(1) - x                       # (B,K,F)
        # repulsion among particles
        xi_minus_xj = x.unsqueeze(2) - x.unsqueeze(1)        # (B,K,K,F): x_i - x_j
        dist = xi_minus_xj.norm(dim=-1)                      # (B,K,K)
        kmat = torch.exp(-dist / tau)                        # (B,K,K)
        xj_minus_xi = -xi_minus_xj
        num = (kmat.unsqueeze(-1) * xj_minus_xi).sum(dim=2)  # (B,K,F)
        denom = kmat.sum(dim=2, keepdim=True).clamp_min(1e-8)
        V_neg = num / denom
        return V_pos - V_neg

    def loss(self, batch) -> torch.Tensor:
        context = self._noisy(batch["context"])
        y = batch["target"]
        B, H, F = context.shape
        K = self.mcfg.drift_particles
        eps = torch.randn(B, K, F, device=y.device)
        ctx = context.unsqueeze(1).expand(B, K, H, F).reshape(B * K, H, F)
        x = self.net(eps.reshape(B * K, F), ctx, None).view(B, K, F)
        V = self._drift(x, y)
        target = (x + V).detach()
        return ((x - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(context.shape[0], self.state_dim, device=context.device)
        return self.net(eps, context, None)
