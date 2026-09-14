"""Autoregressive next-state prediction (the paper's learning paradigm)."""

from __future__ import annotations

import torch

from .base import Method


class ARMethod(Method):
    model_kind = "ar"

    def loss(self, batch) -> torch.Tensor:
        context = self._noisy(batch["context"])
        return self.net.loss(context, batch["target"])

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        return self.net.predict(context)
