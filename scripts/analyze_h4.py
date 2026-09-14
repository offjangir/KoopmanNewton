#!/usr/bin/env python
"""H4 analysis: does force-use decay and invariant-conservation take over as
context grows (generator -> Koopman), and does the high-sigma/long-context
"Koopman cell" recover accuracy through invariants rather than force?

For each runs/h4_kepler/* checkpoint, report vs (method, context, sigma):
  force_r2     -- on-trajectory Verlet force-term R^2 (force USE)
  rollout_rmse -- 30-step autoregressive prediction error
  E_drift      -- energy drift under rollout (Koopman-side conservation)
  probe_excess -- the confounded metric (should stay ~flat)

    python scripts/analyze_h4.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physwm.train.checkpoint import load_run_config, load_method
from physwm.data import build_dataset
from physwm.eval.probe import probe_representations, probe_raw_context
from physwm.eval.behavioral import force_term_r2_ontraj
from physwm.eval.rollout import autoregressive_rollout
from physwm.eval.metrics import invariant_drift, rmse_per_step
from physwm.physics import make_simulator


def main():
    runs = sorted(glob.glob("runs/h4_kepler/*/final.pt"))
    rows = []
    sim = make_simulator("kepler")
    cache = {}
    for cp in runs:
        run = os.path.dirname(cp)
        cfg = load_run_config(run)
        H = cfg.data.context
        # the generated arrays depend on the data config but not on the context window
        key = json.dumps({k: v for k, v in asdict(cfg.data).items() if k != "context"},
                         sort_keys=True)
        if key not in cache:
            cache[key] = build_dataset(cfg.data)
        ds = cache[key]
        m, _ = load_method(cfg, ds.state_dim, cp)

        fr = force_term_r2_ontraj(m, ds, H, n_traj=128)
        roll = autoregressive_rollout(m, ds, "test", H, steps=30, n_traj=96)
        rmse30 = float(rmse_per_step(roll["pred_norm"], roll["gt_norm"])[-1])
        Ed = invariant_drift(sim, roll["pred"], dt=ds.dt).get("energy", float("nan"))
        raw = probe_raw_context(ds, "train", "test", context=H, n_traj=96)
        pr = probe_representations(m, ds, "train", "test", context=H, n_traj=96)
        cv = raw.r2["force"]
        excess = (pr.r2["force"] - cv) / (1 - cv) if cv < 1 else float("nan")

        name = os.path.basename(run)
        meth = name.split("_")[0]
        ctx = int(re.search(r"ctx(\d+)", name).group(1))
        sig = float(re.search(r"s([\d.]+)$", name).group(1))
        rows.append(dict(method=meth, ctx=ctx, sigma=sig, force_r2=fr["force_term_r2"],
                         rollout_rmse=rmse30, E_drift=Ed, probe_excess=excess))
        print(f"{name:22s} force_r2={fr['force_term_r2']:+.3f} rmse30={rmse30:.3f} "
              f"E_drift={Ed:.3g} probe_excess={excess:+.3f}", flush=True)

    json.dump(rows, open("runs/h4_kepler/analysis.json", "w"), indent=2)
    # pivot: force_r2 vs context, per (method,sigma)
    print("\n=== force-term R^2  vs  context ===")
    ctxs = sorted({r["ctx"] for r in rows})
    print(f"{'method/sigma':16s}" + "".join(f"ctx{c:>7d}" for c in ctxs))
    for meth in sorted({r["method"] for r in rows}):
        for sig in sorted({r["sigma"] for r in rows}):
            cells = {r["ctx"]: r["force_r2"] for r in rows if r["method"] == meth and r["sigma"] == sig}
            if cells:
                print(f"{meth+' s='+str(sig):16s}" + "".join(f"{cells.get(c, float('nan')):>10.3f}" for c in ctxs))
    print("\n=== rollout rmse@30  vs  context (lower=better; should improve with ctx) ===")
    for meth in sorted({r["method"] for r in rows}):
        for sig in sorted({r["sigma"] for r in rows}):
            cells = {r["ctx"]: r["rollout_rmse"] for r in rows if r["method"] == meth and r["sigma"] == sig}
            if cells:
                print(f"{meth+' s='+str(sig):16s}" + "".join(f"{cells.get(c, float('nan')):>10.3f}" for c in ctxs))
    print("\n-> runs/h4_kepler/analysis.json")


if __name__ == "__main__":
    main()
