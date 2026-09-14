"""Autoregressive multi-step rollout for any method (AR or generative).

Every method exposes ``sample(context) -> next_state``; we seed with the first
``context`` ground-truth states and feed predictions back in, sliding the window.
Returns predictions and ground truth in *physical* (denormalized) units.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_next(method, ctx_norm: np.ndarray, mean, std, device="cpu",
                 batch_size: int = 1024) -> np.ndarray:
    """One-step predictions for normalized contexts ``(N,H,F)``, in physical units."""
    method.eval()
    ctx_norm = ctx_norm.astype(np.float32)
    preds = [method.sample(torch.from_numpy(ctx_norm[i:i + batch_size]).to(device)).cpu().numpy()
             for i in range(0, len(ctx_norm), batch_size)]
    return np.concatenate(preds, 0) * np.asarray(std, float) + np.asarray(mean, float)


@torch.no_grad()
def autoregressive_rollout(method, dataset, split: str, context: int, steps: int,
                           n_traj: int = 64, device="cpu") -> dict:
    method.eval()
    states = dataset.splits[split].states[:n_traj]           # (n,T,F) physical
    norm = dataset.normalize(states).astype(np.float32)
    T = norm.shape[1]
    steps = min(steps, T - context)
    cur = torch.from_numpy(norm[:, :context]).to(device)     # (n,H,F)
    preds = []
    for _ in range(steps):
        nxt = method.sample(cur)                             # (n,F)
        preds.append(nxt.cpu().numpy())
        cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], dim=1)
    pred_norm = np.stack(preds, axis=1)                      # (n,steps,F)
    gt_norm = norm[:, context:context + steps]
    pred = dataset.denormalize(pred_norm)
    gt = dataset.denormalize(gt_norm)
    return dict(pred=pred, gt=gt, pred_norm=pred_norm, gt_norm=gt_norm)
