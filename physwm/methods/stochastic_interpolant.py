"""Stochastic Interpolants (Albergo, Boffi, Vanden-Eijnden) -- optional method.

Latent-noise interpolant ``x_t = (1-t)x0 + t x1 + gamma(t) z`` with
``gamma(t) = t(1-t)`` (so endpoints are clean). Two networks are learned:

* drift ``b(x,t)`` regressed onto ``(x1 - x0) + gammadot(t) z`` ;
* denoiser ``eta(x,t) ~ E[z | x_t]`` regressed onto ``z`` (giving score
  ``s = -eta / gamma``).

Sampling can use the probability-flow ODE ``dx = b dt`` (default, deterministic)
or the SDE ``dx = [b + e*s] dt + sqrt(2e) dW`` for stochastic trajectory
generation without changing the marginals.
"""

from __future__ import annotations

import torch

from .base import Method, build_network


class StochasticInterpolantMethod(Method):
    model_kind = "dit"

    def __init__(self, model_cfg, method_cfg, state_dim: int):
        super().__init__(model_cfg, method_cfg, state_dim)   # self.net == drift b
        self.score_net = build_network(model_cfg, state_dim)  # denoiser eta

    @staticmethod
    def _gamma(t):
        return t * (1 - t)

    @staticmethod
    def _gammadot(t):
        return 1 - 2 * t

    def loss(self, batch) -> torch.Tensor:
        context = self._noisy(batch["context"])
        x1 = batch["target"]
        B, H, F = context.shape
        x0 = torch.randn_like(x1)
        z = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device).clamp(0.02, 0.98)
        g = self._gamma(t).unsqueeze(-1)
        gd = self._gammadot(t).unsqueeze(-1)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1 + g * z
        b_target = (x1 - x0) + gd * z
        loss_b = ((self.net(x_t, context, t) - b_target) ** 2).mean()
        loss_z = ((self.score_net(x_t, context, t) - z) ** 2).mean()
        return loss_b + loss_z

    @torch.no_grad()
    def sample(self, context: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        B = context.shape[0]
        n = self.mcfg.flow_sample_steps
        x = torch.randn(B, self.state_dim, device=context.device)
        for i in range(n):
            tval = i / n
            t = torch.full((B,), tval, device=context.device)
            b = self.net(x, context, t)
            dx = b
            if epsilon > 0 and 0 < tval < 1:
                g = self._gamma(torch.tensor(tval))
                s = -self.score_net(x, context, t) / g.clamp_min(1e-3)
                dx = b + epsilon * s
            x = x + (1.0 / n) * dx
            if epsilon > 0 and i < n - 1:
                x = x + (2 * epsilon / n) ** 0.5 * torch.randn_like(x)
        return x
