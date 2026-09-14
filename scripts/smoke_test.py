#!/usr/bin/env python
"""End-to-end smoke test (CPU-friendly): for each (domain, method) combo, build a
tiny dataset, train a few steps, roll out, probe, and emit metrics + plots.

Proves the whole pipeline wires together. NOT a science run -- data and models
are deliberately tiny. Use ``--full`` to sweep all domains, else a fast subset.

    python scripts/smoke_test.py            # fast subset
    python scripts/smoke_test.py --full     # all domains x all methods
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physwm.train.config import ExperimentCfg, DataCfg, ModelCfg, MethodCfg, TrainCfg
from physwm.data import build_dataset, make_loader
from physwm.methods import make_method, ALL_METHODS
from physwm.train.trainer import Trainer
from physwm.eval import autoregressive_rollout, rmse_per_step, probe_representations, invariant_drift
from physwm.physics import make_simulator
from physwm import viz


def tiny_cfg(domain, method, steps, context, out):
    return ExperimentCfg(
        name=f"{domain}-{method}",
        data=DataCfg(domain=domain, n_train=48, n_val=12, n_test=24,
                     n_steps=40, context=context, seed=0),
        model=ModelCfg(d_model=64, n_layers=3, n_heads=4, d_ff=128),
        method=MethodCfg(kind=method, ctx_noise_sigma=0.05,
                         n_diffusion_steps=20, flow_sample_steps=8, drift_particles=4),
        train=TrainCfg(steps=steps, batch_size=64, lr=1e-3, device="cpu",
                       log_every=20, eval_every=0, ckpt_every=0, amp=False, out_dir=out),
    )


def run_combo(domain, method, steps, context, root):
    out = os.path.join(root, f"{domain}__{method}")
    cfg = tiny_cfg(domain, method, steps, context, out)
    torch.manual_seed(0)
    ds = build_dataset(cfg.data)
    train_loader = make_loader(ds, "train", cfg.data.context, cfg.train.batch_size)
    val_loader = make_loader(ds, "val", cfg.data.context, cfg.train.batch_size, shuffle=False)
    method_obj = make_method(cfg.model, cfg.method, ds.state_dim)
    n_params = sum(p.numel() for p in method_obj.parameters())

    tr = Trainer(cfg, method_obj, train_loader, val_loader)
    hist = tr.fit()
    loss0 = hist[0]["loss"]
    lossf = [h["loss"] for h in hist if "loss" in h][-1]

    roll = autoregressive_rollout(method_obj, ds, "test", cfg.data.context,
                                  steps=20, n_traj=16, device=tr.device)
    rmse = rmse_per_step(roll["pred_norm"], roll["gt_norm"])
    probe = probe_representations(method_obj, ds, fit_split="train", eval_split="test",
                                  context=cfg.data.context, device=tr.device,
                                  n_traj=24, max_samples=1500)
    sim = make_simulator(domain)
    drift = invariant_drift(sim, roll["pred"])

    # plots
    viz.plot_loss(hist, os.path.join(out, "loss.png"))
    viz.plot_probe_bars({method: probe.r2}, os.path.join(out, "probe.png"))
    if ds.state_dim >= 2:
        viz.plot_trajectory_2d(roll["pred"][0], roll["gt"][0], os.path.join(out, "traj.png"),
                               title=f"{domain} / {method}")

    return dict(
        domain=domain, method=method, n_params=n_params,
        loss0=round(loss0, 4), lossf=round(lossf, 4),
        rmse_step1=round(float(rmse[0]), 4), rmse_step20=round(float(rmse[-1]), 4),
        force_r2=round(probe.r2["force"], 3),
        best_inv=max((k for k in probe.r2 if k.startswith("inv_")),
                     key=lambda k: probe.r2[k], default="-"),
        best_inv_r2=round(max([probe.r2[k] for k in probe.r2 if k.startswith("inv_")] + [0.0]), 3),
        inv_drift=round(float(np.mean(list(drift.values()))) if drift else -1, 4),
        status="ok",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="all domains")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--context", type=int, default=6)
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--out", default="runs/smoke")
    args = ap.parse_args()

    if args.full:
        domains = ["kepler", "gr_kepler", "nbody3", "driven_pendulum", "double_pendulum",
                   "rabi", "ehrenfest", "wavepacket", "rel_lorentz", "schwarzschild"]
    else:
        domains = ["kepler", "rabi", "rel_lorentz"]

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for domain in domains:
        for method in args.methods:
            tag = f"{domain:16s} {method:10s}"
            try:
                row = run_combo(domain, method, args.steps, args.context, args.out)
                print(f"[ok]   {tag} loss {row['loss0']:.3f}->{row['lossf']:.3f}  "
                      f"force_r2={row['force_r2']:.3f}  rmse20={row['rmse_step20']:.3f}")
            except Exception as e:  # noqa
                traceback.print_exc()
                row = dict(domain=domain, method=method, status=f"FAIL: {e}")
                print(f"[FAIL] {tag} {e}")
            rows.append(row)

    csv_path = os.path.join(args.out, "summary.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n{n_ok}/{len(rows)} combos OK -> {csv_path}")
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
