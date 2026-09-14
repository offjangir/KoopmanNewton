"""Output heads for the AR transformer.

* :class:`RegressionHead` -- continuous next-state (spatial-smoothness bias ON).
* :class:`ClassificationHead` -- per-dimension binning over a fixed range, the
  discretized setup used as the "no spatial smoothness" ablation. Predicts a
  categorical over ``n_bins`` per output dimension; the point estimate is the
  softmax-weighted bin center.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(d_model, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h)

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(h), target)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, n_bins: int, bin_range: float):
        super().__init__()
        self.out_dim = out_dim
        self.n_bins = n_bins
        self.lin = nn.Linear(d_model, out_dim * n_bins)
        centers = torch.linspace(-bin_range, bin_range, n_bins)
        self.register_buffer("centers", centers)
        self.bin_range = bin_range

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h).view(*h.shape[:-1], self.out_dim, self.n_bins)

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self._logits(h)                     # (..., out_dim, n_bins)
        # nearest-bin index for each target coordinate
        idx = torch.bucketize(target.clamp(-self.bin_range, self.bin_range), self.centers)
        idx = idx.clamp(0, self.n_bins - 1)
        return F.cross_entropy(logits.reshape(-1, self.n_bins), idx.reshape(-1))

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self._logits(h), dim=-1)
        return (probs * self.centers).sum(-1)
