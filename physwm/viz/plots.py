"""Plotting helpers (matplotlib, Agg backend). Clean, theme-neutral figures."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PALETTE = ["#3B6FB6", "#D9822B", "#4A9B7F", "#B0524B", "#8562A8", "#7A7A7A"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list, path: str):
    steps = [h["step"] for h in history if "loss" in h]
    loss = [h["loss"] for h in history if "loss" in h]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(steps, loss, color=_PALETTE[0])
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_rollout_error(errors: dict, path: str, ylabel="RMSE"):
    """errors: label -> per-step array."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (label, arr) in enumerate(errors.items()):
        ax.plot(np.arange(1, len(arr) + 1), arr, label=label, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xlabel("rollout step"); ax.set_ylabel(ylabel)
    ax.set_title("Autoregressive rollout error"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_probe_bars(results: dict, path: str):
    """results: method -> {target: r2}. Grouped bars over targets."""
    methods = list(results.keys())
    targets = list(next(iter(results.values())).keys())
    x = np.arange(len(targets))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(targets)), 3.4))
    for i, m in enumerate(methods):
        vals = [results[m].get(t, np.nan) for t in targets]
        ax.bar(x + i * w, vals, w, label=m, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("probe $R^2$"); ax.set_ylim(0, 1.02)
    ax.set_title("Linear probe: force vs invariants"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def plot_probe_with_controls(model_r2: dict, raw_r2: dict, untrained_r2: dict,
                             path: str, title="Probe vs controls"):
    """Model probe R^2 against both control floors -- the only honest view.

    Raw R^2 saturates near 1, so the controls must be drawn alongside: the
    visible gap above the taller control bar is what training actually bought.
    """
    targets = [t for t in model_r2 if t in raw_r2 or t in untrained_r2]
    x = np.arange(len(targets))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(targets)), 3.6))
    ax.bar(x - w, [raw_r2.get(t, np.nan) for t in targets], w,
           label="control: raw context", color="#C9CDD2")
    ax.bar(x, [untrained_r2.get(t, np.nan) for t in targets], w,
           label="control: untrained net", color="#8A9199")
    ax.bar(x + w, [model_r2.get(t, np.nan) for t in targets], w,
           label="trained model", color=_PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("inv_", "") for t in targets], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("probe $R^2$"); ax.set_ylim(min(0, min(
        [v for v in list(raw_r2.values()) + list(untrained_r2.values()) if np.isfinite(v)] or [0])) - 0.05, 1.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def plot_trajectory_2d(pred, gt, path: str, title="Rollout (x,y)"):
    """pred/gt: (K,F) single trajectory; plots first two dims."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(gt[:, 0], gt[:, 1], "-", color=_PALETTE[5], label="truth", lw=2)
    ax.plot(pred[:, 0], pred[:, 1], "--", color=_PALETTE[1], label="model", lw=2)
    ax.scatter([gt[0, 0]], [gt[0, 1]], color="k", s=20, zorder=5)
    ax.set_aspect("equal"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, path)
