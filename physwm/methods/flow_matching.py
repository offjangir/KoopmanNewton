"""Conditional flow matching / rectified flow over the next state.

Linear path ``x_t = (1-t) x0 + t x1`` (x0 ~ N(0,I), x1 = target), constant
velocity target ``x1 - x0``. Training regresses ``v_theta(x_t, t, context)`` onto
that target; sampling integrates the ODE ``dx = v dt`` with Euler steps.
"""

from __future__ import annotations

import torch

from .base import Method


class FlowMatchingMethod(Method):
    model_kind = "dit"

    def loss(self, batch) -> torch.Tensor:
        context = self._noisy(batch["context"])
        x1 = batch["target"]
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        v_target = x1 - x0
        pred = self.net(x_t, context, t)
        return ((pred - v_target) ** 2).mean()

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        n = self.mcfg.flow_sample_steps
        x = torch.randn(B, self.state_dim, device=context.device)
        for i in range(n):
            t = torch.full((B,), i / n, device=context.device)
            x = x + (1.0 / n) * self.net(x, context, t)
        return x
