"""Conditional DDPM diffusion over the next state, with a DiT backbone.

epsilon-prediction: ``x_t = sqrt(acp) x0 + sqrt(1-acp) eps``, network predicts
``eps`` conditioned on the context and the (normalized) diffusion time; ancestral
sampling generates the next state from Gaussian noise.
"""

from __future__ import annotations

import torch

from .base import Method


class DiffusionMethod(Method):
    model_kind = "dit"

    def __init__(self, model_cfg, method_cfg, state_dim: int):
        super().__init__(model_cfg, method_cfg, state_dim)
        T = method_cfg.n_diffusion_steps
        betas = torch.linspace(method_cfg.beta_start, method_cfg.beta_end, T)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", acp)
        self.T = T

    def loss(self, batch) -> torch.Tensor:
        context = self._noisy(batch["context"])
        x0 = batch["target"]
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        acp = self.acp[t].unsqueeze(-1)
        x_t = acp.sqrt() * x0 + (1 - acp).sqrt() * eps
        pred = self.net(x_t, context, t.float() / self.T)
        return ((pred - eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        x = torch.randn(B, self.state_dim, device=context.device)
        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=context.device)
            eps = self.net(x, context, t.float() / self.T)
            alpha = self.alphas[ti]
            acp = self.acp[ti]
            mean = (x - (1 - alpha) / (1 - acp).sqrt() * eps) / alpha.sqrt()
            if ti > 0:
                x = mean + self.betas[ti].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
