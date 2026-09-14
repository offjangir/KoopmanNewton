#!/usr/bin/env python
"""dt-sweep analysis: does force-term R^2 / force recovery track the
identifiability bound  eps << dt^2 |F| ?

For each runs/dt_kepler/* (Kepler, sigma=0, ctx=2, varying dt):
  force_term_r2 -- behavioral force use (no dt^2 amplification; robust)
  pred_rmse     -- one-step error eps
  signal        -- dt^2 |F|  (the sliver the force occupies)
  eps/signal    -- < 1 required for the force to be recoverable from behaviour
  SNR, exponent -- isotropy-averaged force-law exponent (should -> -2 only when conclusive)

    python scripts/analyze_dt.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physwm.train.checkpoint import load_run
from physwm.eval.force_field import counterfactual_force_field, behavioral_force_test


def main():
    rows = []
    for cp in sorted(glob.glob("runs/dt_kepler/*/final.pt")):
        run = os.path.dirname(cp)
        cfg, ds, m = load_run(run)
        dt = ds.dt
        beh = behavioral_force_test(m, ds.mean, ds.std, dt, (0.5, 2.0), (0.0, 0.6), n=6000)
        cf = counterfactual_force_field(m, ds.mean, ds.std, dt, (0.5, 2.0), (0.0, 0.6), n=6000)
        q = cf["resolution"]
        signal = dt ** 2 * q["force_scale"]
        name = os.path.basename(run)
        meth = name.split("_")[0]
        rows.append(dict(method=meth, dt=dt, force_term_r2=beh["force_term_r2"],
                         pred_rmse=q["pred_rmse"], signal=signal,
                         eps_over_signal=q["pred_rmse"] / signal, snr=q["snr"],
                         conclusive=q["conclusive"], exponent=cf["model_profile"]["exponent"]))
        print(f"{name:20s} dt={dt:.2f} force_r2={beh['force_term_r2']:+.3f} "
              f"eps={q['pred_rmse']:.4f} signal={signal:.4f} eps/sig={q['pred_rmse']/signal:5.2f} "
              f"SNR={q['snr']:5.1f} p={cf['model_profile']['exponent']:+.2f} "
              f"[{'CONCL' if q['conclusive'] else 'noise-lim'}]", flush=True)

    json.dump(rows, open("runs/dt_kepler/analysis.json", "w"),
              default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print("\n=== force-term R^2 (behavioural) vs exponent recovery, by dt ===")
    print(f"{'method':10s}{'dt':>6s}{'force_r2':>10s}{'eps/signal':>12s}{'exponent':>10s}{'conclusive':>12s}")
    for r in sorted(rows, key=lambda r: (r["method"], r["dt"])):
        print(f"{r['method']:10s}{r['dt']:>6.2f}{r['force_term_r2']:>+10.3f}"
              f"{r['eps_over_signal']:>12.2f}{r['exponent']:>+10.2f}{str(r['conclusive']):>12s}")
    print("\n-> runs/dt_kepler/analysis.json")


if __name__ == "__main__":
    main()
