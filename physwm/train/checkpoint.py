"""Rebuild a trained run (config + dataset + model) from its output directory."""

from __future__ import annotations

import os

import torch

from .config import ExperimentCfg


def load_run_config(run_dir: str) -> ExperimentCfg:
    return ExperimentCfg.load(os.path.join(run_dir, "config.yaml"))


def load_method(cfg: ExperimentCfg, state_dim: int, ckpt_path: str, device="cpu"):
    """Instantiate the run's method and load a checkpoint into it (eval mode).

    Returns ``(method, step)``.
    """
    from ..methods import make_method

    method = make_method(cfg.model, cfg.method, state_dim)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    method.load_state_dict(blob["model"])
    return method.to(device).eval(), blob.get("step")


def load_run(run_dir: str, ckpt: str = "final.pt", device="cpu"):
    """``(cfg, dataset, method)`` for a run. The dataset is regenerated from the
    saved data config, which is seeded, so the arrays match the training run."""
    from ..data import build_dataset

    cfg = load_run_config(run_dir)
    ds = build_dataset(cfg.data)
    method, _ = load_method(cfg, ds.state_dim, os.path.join(run_dir, ckpt), device)
    return cfg, ds, method
