#!/usr/bin/env python
"""Run a domain x method x context sweep sequentially on one GPU, then aggregate.

This is the headline experiment: does the Newtonian-vs-Keplerian phase transition
(driven by the temporal-locality / context knob) survive across learning
paradigms, not just autoregressive?

    python scripts/sweep_local.py --domains kepler --contexts 2 8 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=["kepler"])
    ap.add_argument("--methods", nargs="*", default=["ar", "diffusion", "flow", "drifting"])
    ap.add_argument("--contexts", nargs="*", type=int, default=[2, 8, 32])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--obs-dims", type=int, default=2,
                    help="observed dims (2 = position-only, as in the paper; "
                         "including velocity makes the force probe trivial)")
    ap.add_argument("--online", action="store_true", help="stream fresh training data")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="runs/sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(d, m, c) for d in args.domains for m in args.methods for c in args.contexts]
    print(f"{len(jobs)} runs queued")

    for i, (domain, method, ctx) in enumerate(jobs, 1):
        tag = f"{domain}_{method}_ctx{ctx}"
        out_dir = os.path.join(args.out, tag)
        if os.path.exists(os.path.join(out_dir, "metrics.json")):
            print(f"[{i}/{len(jobs)}] {tag} already done, skipping")
            continue
        overrides = [
            f"data.domain={domain}", f"data.n_train={args.n_train}",
            "data.n_val=400", f"data.n_test={args.n_test}",
            f"data.context={ctx}", f"method.kind={method}",
            f"data.obs_dims={args.obs_dims}",
            f"model.attn_window={ctx}" if method == "ar" else "model.dropout=0.0",
            f"train.steps={args.steps}", "train.device=cuda",
            f"train.eval_every={max(args.steps // 10, 1)}", "train.ckpt_every=0",
            f"train.batch_size={args.batch}",
            f"model.d_model={args.d_model}", f"model.n_layers={args.layers}",
            f"data.online={'true' if args.online else 'false'}",
            f"train.out_dir={out_dir}", f"name={tag}",
        ]
        cmd = [sys.executable, os.path.join(HERE, "run_experiment.py"),
               "--config", os.path.join(ROOT, "configs", "kepler_ar.yaml"),
               "--set"] + overrides
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        else:
            print(f"   {r.stdout.strip().splitlines()[-1]}  ({time.time()-t0:.0f}s)")

    collect(args.out)


def collect(root):
    rows = []
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        cf = m.get("counterfactual_probe_r2", {})
        invs = {k: v for k, v in m["probe_r2"].items() if k.startswith("inv_")}
        best_inv = max(invs, key=invs.get) if invs else "-"
        raw = m.get("control_raw_context_r2", {})
        unt = m.get("control_untrained_r2", {})

        # Always recompute excess against the ARCHITECTURE-INDEPENDENT raw control
        # so methods are comparable (works for older metrics.json too).
        def excess(target):
            mv, cv = m["probe_r2"].get(target), raw.get(target)
            if mv is None or cv is None or cv >= 1.0:
                return float("nan")
            return (mv - cv) / (1.0 - cv)

        cfd = m.get("counterfactual", {}).get("force", {})
        rows.append(dict(
            run=d,
            force_r2=round(m["probe_r2"]["force"], 3),
            ctl_raw=round(raw.get("force", float("nan")), 3),
            ctl_untrained=round(unt.get("force", float("nan")), 3),
            excess_force=round(excess("force"), 3),
            best_inv=best_inv.replace("inv_", ""),
            excess_inv=round(excess(best_inv), 3),
            cf_r2=round(cfd.get("r2", cf.get("force", float("nan"))), 2),
            cf_cal=round(cfd.get("r2_calibrated", float("nan")), 3),
            cf_corr=round(cfd.get("corr", float("nan")), 3),
            rmse30=round(m["rmse_per_step"][-1], 3),
        ))
    if not rows:
        print("no results yet")
        return
    import csv
    path = os.path.join(root, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    hdr = list(rows[0])
    print("\n" + "  ".join(f"{h:>14s}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r[h]):>14s}" for h in hdr))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
