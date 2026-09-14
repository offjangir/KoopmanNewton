#!/usr/bin/env python
"""Extract the force law a trained model implicitly computes, and test it.

    python scripts/extract_force_law.py runs/sweep_kepler/kepler_ar_ctx2 [more runs...]

For each run: rebuild the model from its checkpoint, then
 1. map the implied force field over IN-RANGE states,
 2. map it over EXTRAPOLATED radii no training orbit reached,
 3. test velocity-independence (the true force depends on position only),
 4. fit log|a| vs log|r| to see whether the inverse-square exponent (-2) emerges.

Everything is compared against the *extraction floor* -- the same Verlet
operator applied to the exact ground-truth next position.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physwm.train.checkpoint import load_run
from physwm.eval.force_field import counterfactual_force_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"{'run':30s}{'regime':>10s}{'field_R2':>10s}{'dir_cos':>9s}{'exponent':>10s}{'law_R2':>8s}")
    results = {}
    for run in args.runs:
        cfg, ds, method = load_run(run, device=args.device)
        H = cfg.data.context
        dt = ds.dt
        name = os.path.basename(run.rstrip("/"))
        if H != 2:
            print(f"{name:30s}  -- skipped: grid probe needs context=2 "
                  f"(got {H}); see trajectory-based probe instead")
            continue
        res = {}
        # orbit families: the training window vs a shifted (counterfactual) one
        families = {
            "in-dist":   ((0.5, 2.0), (0.0, 0.6)),    # training range
            "ecc-OOD":   ((0.5, 2.0), (0.6, 0.9)),    # unseen eccentricities
            "axis-OOD":  ((2.0, 3.0), (0.0, 0.6)),    # unseen semi-major axes
        }
        for tag, (a_rng, e_rng) in families.items():
            out = counterfactual_force_field(method, ds.mean, ds.std, dt, a_rng, e_rng,
                                             n=args.n, device=args.device)
            res[tag] = out
            m, f = out["model"], out["floor"]
            print(f"{name:30s}{tag:>10s}{m['field_r2']:10.3f}{m['direction_cos']:9.3f}"
                  f"{m['power_law_exponent']:10.3f}{m['power_law_r2']:8.3f}")
            print(f"{'  (extraction floor)':30s}{'':>10s}{f['field_r2']:10.3f}"
                  f"{f['direction_cos']:9.3f}{f['power_law_exponent']:10.3f}{f['power_law_r2']:8.3f}")
            q = out["resolution"]
            verdict = "CONCLUSIVE" if q["conclusive"] else "NOISE-LIMITED -> inconclusive"
            print(f"    resolution: pred_rmse={q['pred_rmse']:.4f} -> accel_noise={q['implied_accel_noise']:.2f} "
                  f"vs |F|={q['force_scale']:.2f}  SNR={q['snr']:.2f}  [{verdict}]")
            print(f"    need pred_rmse < {q['pred_rmse_needed_for_snr10']:.4f} for SNR 10")
            mp, fp = out["model_profile"], out["floor_profile"]
            print(f"    isotropy-averaged power law:  model p = {mp['exponent']:+.3f} "
                  f"+/- {mp['exponent_se']:.3f}   (truth -2;  floor {fp['exponent']:+.3f} "
                  f"+/- {fp['exponent_se']:.3f})")
            print(f"    fraction attractive: model {mp['frac_attractive']:.2f}  "
                  f"floor {fp['frac_attractive']:.2f}")
        print(f"{'  v-sensitivity (0=ideal)':30s}{res['in-dist']['v_sensitivity']:>10.3f}")
        results[name] = res

    if results:
        out_path = "runs/force_law_extraction.json"
        json.dump(results, open(out_path, "w"), indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
