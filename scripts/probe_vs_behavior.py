#!/usr/bin/env python
"""THE decisive experiment: does the force probe track actual force USE?

For each checkpoint of a run, measure side by side

  * ``excess_force``   -- hidden-state linear probe, excess over the
    architecture-independent raw-context control (what prior work reports), and
  * ``force_term_r2``  -- the behavioural test: how much of the true ``dt^2 F``
    the model's own prediction residual reproduces, relative to the linear null
    ``2 r_t - r_{t-1}``. Zero means the model behaves as if it never learned the
    force. No ``dt^2`` amplification, so a null result is meaningful.

If probe excess stays high and flat while ``force_term_r2`` moves, the probe does
not measure force use.

    python scripts/probe_vs_behavior.py runs/longrun_ctx2 [runs/other ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physwm.train.checkpoint import load_run_config, load_method
from physwm.data import build_dataset
from physwm.eval.probe import probe_representations, probe_raw_context
from physwm.eval.behavioral import behavioral_score


def ckpt_step(path):
    m = re.search(r"step(\d+)\.pt$", path)
    if m:
        return int(m.group(1))
    return None  # final.pt -> resolved from the checkpoint itself


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--n-traj", type=int, default=96)
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        cfg = load_run_config(run)
        ds = build_dataset(cfg.data)
        H = cfg.data.context
        raw = probe_raw_context(ds, "train", "test", context=H, n_traj=args.n_traj)

        ckpts = sorted(glob.glob(os.path.join(run, "step*.pt")),
                       key=lambda p: ckpt_step(p) or 0)
        ckpts += glob.glob(os.path.join(run, "final.pt"))
        for cp in ckpts:
            method, saved_step = load_method(cfg, ds.state_dim, cp)
            step = ckpt_step(cp) or saved_step

            pr = probe_representations(method, ds, fit_split="train", eval_split="test",
                                       context=H, n_traj=args.n_traj)
            cv = raw.r2["force"]
            excess = (pr.r2["force"] - cv) / (1 - cv) if cv < 1 else float("nan")
            beh = behavioral_score(method, ds, cfg.data.domain, H, device="cpu", n=args.n)
            scalar = {k: v for k, v in beh.items() if isinstance(v, (int, float, str))}
            rows.append(dict(run=os.path.basename(run.rstrip("/")), step=step,
                             ctx=H, domain=cfg.data.domain, probe_force_r2=pr.r2["force"],
                             raw_ctl=cv, excess_force=excess, behavior=beh["value"], **scalar))
            if beh["metric"] == "force_term_r2":
                tail = f"force_term_R2={beh['value']:+.4f}  model/linear={beh.get('ratio', float('nan')):.4f}"
            else:
                tail = (f"conservation={beh['value']:+.3f}  drift={beh.get('model_drift', float('nan')):.3g}"
                        f" (floor {beh.get('floor_drift', float('nan')):.3g})")
            print(f"{rows[-1]['run']:20s} step={step:>7} probe_excess={excess:+.3f}  {tail}")

    if rows:
        json.dump(rows, open("runs/probe_vs_behavior.json", "w"), indent=2)
        print("\n" + "=" * 84)
        print(f"{'run':22s}{'ctx':>5s}{'probe_excess':>14s}{'behaviour':>12s}{'metric':>22s}")
        for r in rows:
            print(f"{r['run']:22s}{r['ctx']:>5d}{r['excess_force']:>+14.3f}"
                  f"{r['behavior']:>+12.3f}{r.get('metric',''):>22s}")
        print("\n-> runs/probe_vs_behavior.json")


if __name__ == "__main__":
    main()
