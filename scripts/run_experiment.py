#!/usr/bin/env python
"""Run one experiment (domain x method x biases) from a YAML config.

    python scripts/run_experiment.py --config configs/kepler_ar.yaml
    python scripts/run_experiment.py --config configs/kepler_flow.yaml --set train.steps=20000

Outputs to ``train.out_dir``: config.yaml, log.jsonl, checkpoints, metrics.json,
and plots (loss, rollout error, probe bars, sample trajectory). Evaluation
includes in-distribution + OOD rollout, the force-vs-invariant probe, the
counterfactual (OOD) force probe, and invariant-drift physics consistency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physwm.train.config import ExperimentCfg
from physwm.data import build_dataset, make_loader
from physwm.methods import make_method
from physwm.train.trainer import Trainer
from physwm.eval import autoregressive_rollout, rmse_per_step, probe_representations, invariant_drift
from physwm.eval.probe import probe_raw_context, excess_over_control, probe_transfer
from physwm.eval.metrics import position_mde
from physwm.physics import make_simulator
from physwm import viz


def _coerce(val: str, cur):
    """Cast a CLI override string to match the existing config value's type.

    When the current value is ``None`` (e.g. obs_dims, attn_window, dt) there is
    no type to match, so infer it -- otherwise the field silently stays a string
    and blows up deep in the pipeline.
    """
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, list):
        return [] if val in ("", "[]") else val.split(",")
    if isinstance(cur, str):
        return val
    # cur is None -> infer from the literal
    if val.lower() in ("none", "null"):
        return None
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def apply_overrides(cfg_dict, overrides):
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg_dict
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _coerce(val, node.get(parts[-1]))
    return cfg_dict


def evaluate(method, ds, cfg, device):
    metrics = {}
    H = cfg.data.context
    # in-distribution rollout
    roll = autoregressive_rollout(method, ds, "test", H, steps=30, n_traj=128, device=device)
    metrics["rmse_per_step"] = rmse_per_step(roll["pred_norm"], roll["gt_norm"]).tolist()
    metrics["mde_pos_per_step"] = position_mde(roll["pred"], roll["gt"]).tolist()
    sim = make_simulator(cfg.data.domain)
    metrics["invariant_drift"] = invariant_drift(sim, roll["pred"], dt=ds.dt)

    # OOD rollout (all held-out regimes)
    ood = [k for k in ds.splits if k.startswith("test_")]
    metrics["ood"] = {}
    for split in ood:
        r = autoregressive_rollout(method, ds, split, H, steps=30, n_traj=64, device=device)
        metrics["ood"][split] = dict(rmse_per_step=rmse_per_step(r["pred_norm"], r["gt_norm"]).tolist())

    # probing: fit on train trajectories, score on DISJOINT held-out trajectories
    probe = probe_representations(method, ds, fit_split="train", eval_split="test",
                                  context=H, device=device, n_traj=128)
    metrics["probe_r2"] = probe.r2
    metrics["probe_best_layer"] = probe.best_layer
    metrics["probe_n_traj"] = dict(fit=probe.n_fit_traj, eval=probe.n_eval_traj)
    # counterfactual: same readout scored on an unseen OOD regime.
    # Uses robust transfer metrics -- plain R^2 explodes when the ridge map
    # extrapolates off its training support.
    if ood:
        metrics["counterfactual"] = probe_transfer(
            method, ds, fit_split="train", eval_split=ood[0], context=H,
            device=device, n_traj=128)
        metrics["counterfactual_split"] = ood[0]

    # ---- CONTROLS: a high force-R^2 is only meaningful above these floors ----
    # 1) linear readout straight off the raw context (no network at all)
    metrics["control_raw_context_r2"] = probe_raw_context(
        ds, "train", "test", context=H, n_traj=128).r2
    # 2) an untrained (random-init) network of the same architecture
    untrained = make_method(cfg.model, cfg.method, ds.state_dim).to(device)
    metrics["control_untrained_r2"] = probe_representations(
        untrained, ds, fit_split="train", eval_split="test",
        context=H, device=device, n_traj=128).r2
    # PRIMARY (cross-method comparable): excess over the architecture-independent
    # raw-context control. Do NOT use max(raw, untrained) here -- the untrained
    # control is architecture-dependent (adaLN-Zero makes an untrained DiT
    # near-identity) and would penalize AR far more than DiT.
    metrics["excess_over_raw"] = excess_over_control(
        probe.r2, metrics["control_raw_context_r2"])
    # SECONDARY (within-architecture diagnostic only)
    metrics["excess_over_untrained"] = excess_over_control(
        probe.r2, metrics["control_untrained_r2"])
    return metrics, roll, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override e.g. train.steps=20000")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ExperimentCfg.from_dict(apply_overrides(cfg_dict, args.set))
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.out_dir, "config.yaml"))
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # data (cache keyed by everything that changes the generated arrays)
    d = cfg.data
    ood_tag = "-".join(d.ood_regimes) if d.ood_regimes else "auto"
    cache = os.path.join(
        d.data_root,
        f"{d.domain}_n{d.n_train}-{d.n_val}-{d.n_test}_T{d.n_steps}_dt{d.dt}"
        f"_obs{d.obs_dims}_norm{int(d.normalize)}_ood{ood_tag}_seed{d.seed}.npz",
    )
    if os.path.exists(cache):
        from physwm.data import PhysicsDataset
        ds = PhysicsDataset.load(cache)
    else:
        ds = build_dataset(cfg.data)
        ds.save(cache)

    # Training stream: online (fresh trajectories every batch) or the fixed split.
    # Eval/probe always use the fixed splits so metrics stay comparable.
    if cfg.data.online:
        from physwm.data import make_online_loader
        train_loader = make_online_loader(
            cfg.data.domain, cfg.data.context, ds.mean, ds.std, cfg.train.batch_size,
            horizon=cfg.data.horizon, n_steps=cfg.data.n_steps, dt=cfg.data.dt,
            obs_dims=cfg.data.obs_dims, seed=cfg.data.seed,
            num_workers=cfg.data.online_workers)
        print(f"[{cfg.name}] ONLINE data stream ({cfg.data.online_workers} workers)")
    else:
        train_loader = make_loader(ds, "train", cfg.data.context, cfg.train.batch_size)
    val_loader = make_loader(ds, "val", cfg.data.context, cfg.train.batch_size, shuffle=False)
    method = make_method(cfg.model, cfg.method, ds.state_dim)
    print(f"[{cfg.name}] params={sum(p.numel() for p in method.parameters()):,} state_dim={ds.state_dim}")

    trainer = Trainer(cfg, method, train_loader, val_loader)
    hist = trainer.fit()

    metrics, roll, probe = evaluate(method, ds, cfg, trainer.device)

    # --- overfitting diagnostics -------------------------------------------
    val = trainer.validate(max_batches=40)
    tr_eval = trainer.train_loss_eval(max_batches=40)
    metrics["final_train_loss_eval"] = tr_eval
    metrics.update(val or {})
    if val:
        metrics["generalization_gap"] = val["val_loss"] - tr_eval
        metrics["generalization_gap_rel"] = (val["val_loss"] - tr_eval) / (abs(tr_eval) + 1e-12)
    if cfg.data.online:
        metrics["epochs_over_train_set"] = None   # infinite stream: no repeats
    else:
        n_windows = cfg.data.n_train * ((cfg.data.n_steps or 100) - cfg.data.context)
        metrics["epochs_over_train_set"] = cfg.train.steps * cfg.train.batch_size / max(n_windows, 1)
    with open(os.path.join(cfg.train.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    viz.plot_loss(hist, os.path.join(cfg.train.out_dir, "loss.png"))
    viz.plot_probe_with_controls(
        probe.r2, metrics["control_raw_context_r2"], metrics["control_untrained_r2"],
        os.path.join(cfg.train.out_dir, "probe.png"),
        title=f"{cfg.data.domain}/{cfg.method.kind} ctx={cfg.data.context}")
    viz.plot_rollout_error({"test": np.array(metrics["rmse_per_step"])},
                           os.path.join(cfg.train.out_dir, "rollout.png"))
    if ds.state_dim >= 2:
        viz.plot_trajectory_2d(roll["pred"][0], roll["gt"][0],
                               os.path.join(cfg.train.out_dir, "traj.png"),
                               title=f"{cfg.data.domain}/{cfg.method.kind}")
    print(f"[{cfg.name}] done. force_r2={probe.r2['force']:.3f}  "
          f"rmse@30={metrics['rmse_per_step'][-1]:.3f}  -> {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
