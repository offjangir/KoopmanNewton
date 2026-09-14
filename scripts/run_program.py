#!/usr/bin/env python
"""Launch the experiment program: named sets of ``run_experiment.py`` jobs, packed
over the local GPUs (see gpu_pool.py). Runs whose ``final.pt`` exists are skipped,
so this is safe to re-run.

    python scripts/run_program.py --sets h4 dt --dry
    python scripts/run_program.py --sets all --per-gpu 4

Sets (all use online data, position-only obs for 2nd-order laws, 256x6 models):
  rel_lorentz  relativity contrast, flow + diffusion (AR from run_contrast.sh)
  h4           context dial on Kepler: ar/flow x ctx 2..32 x sigma 0/0.1
  sigma        sigma-threshold fill (0.01, 0.05) for kepler/pendulum/rabi/rel_lorentz
  dt           dt sweep on Kepler (identifiability bound), sigma=0
  universal    fixed-constant Rabi / relativity contrasts (first-order force test)
  scale        scale-invariance ladder on Kepler (25M, 90M)
  sigma_high   high-sigma threshold for the first-order universal laws
  seeds        seeds 1, 2 on the headline Kepler contrast (seed 0 is runs/h5b)
"""

from __future__ import annotations

import argparse
import os
import sys

from gpu_pool import ROOT, run_pool

BASE = ["data.online=true", "data.online_workers=1", "data.n_train=300", "data.n_val=200",
        "data.n_test=600", "model.d_model=256", "model.n_layers=6", "train.batch_size=512",
        "train.device=cuda", "train.eval_every=0", "train.ckpt_every=0", "train.steps=10000"]
METHODS = ("ar", "flow", "diffusion")
SIGMAS = ("0.0", "0.1")


def obs_of(dom):
    """Position-only for 2nd-order laws (Verlet force test); full state otherwise."""
    return "2" if dom in ("kepler", "pendulum", "gr_kepler") else "none"


def job(out, dom, method, sigma, ctx=2, name=None, extra=()):
    return out, [f"data.domain={dom}", f"data.obs_dims={obs_of(dom)}", f"data.context={ctx}",
                 f"model.attn_window={ctx}", f"method.kind={method}",
                 f"method.ctx_noise_sigma={sigma}", *extra,
                 f"train.out_dir={out}", f"name={name or os.path.basename(out)}"]


def set_rel_lorentz():
    return [job(f"runs/contrast_rel_lorentz/{m}_s{s}", "rel_lorentz", m, s)
            for m in ("flow", "diffusion") for s in SIGMAS]


def set_h4():
    return [job(f"runs/h4_kepler/{m}_ctx{c}_s{s}", "kepler", m, s, ctx=c)
            for m in ("ar", "flow") for c in (2, 4, 8, 16, 32) for s in SIGMAS]


def set_sigma():
    return [job(f"runs/sigma_{d}/{m}_s{s}", d, m, s)
            for d in ("kepler", "pendulum", "rabi", "rel_lorentz") for m in METHODS
            for s in ("0.01", "0.05")]


def set_dt():
    return [job(f"runs/dt_kepler/{m}_dt{dt}", "kepler", m, "0.0", extra=[f"data.dt={dt}"])
            for m in METHODS for dt in ("0.05", "0.1", "0.2", "0.4")]


def set_universal():
    return [job(f"runs/contrast_{d}/{m}_s{s}", d, m, s, name=f"{d}_{m}_s{s}")
            for d in ("rabi_u", "rel_lorentz_u") for m in METHODS for s in SIGMAS]


def set_scale():
    jobs = []
    for tag, d, L, H, ff, steps in [("25M", 512, 8, 8, 2048, 12000), ("90M", 768, 12, 12, 3072, 15000)]:
        for m in (("ar", "flow") if tag == "25M" else ("ar",)):
            for s in SIGMAS:
                jobs.append(job(f"runs/scale_kepler/{m}_{tag}_s{s}", "kepler", m, s,
                                extra=[f"model.d_model={d}", f"model.n_layers={L}",
                                       f"model.n_heads={H}", f"model.d_ff={ff}",
                                       f"train.steps={steps}"]))
    return jobs


def set_sigma_high():
    return [job(f"runs/sigma_{d}/{m}_s{s}", d, m, s, name=f"{d}_{m}_s{s}")
            for d in ("rabi_u", "rel_lorentz_u") for m in METHODS for s in ("0.2", "0.5", "1.0")]


def set_seeds():
    return [job(f"runs/seeds_kepler/{m}_s{s}_seed{seed}", "kepler", m, s,
                extra=[f"data.seed={seed}", f"train.seed={seed}"])
            for seed in (1, 2) for m in METHODS for s in SIGMAS]


SETS = {name[4:]: fn for name, fn in globals().items() if name.startswith("set_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, choices=[*SETS, "all"])
    ap.add_argument("--n-gpu", type=int, default=4)
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps for every job")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    names = list(SETS) if "all" in args.sets else args.sets
    all_jobs = [j for n in names for j in SETS[n]()]
    todo = [(o, ov) for o, ov in all_jobs if not os.path.exists(os.path.join(ROOT, o, "final.pt"))]
    print(f"{len(all_jobs)} jobs, {len(all_jobs) - len(todo)} already done, {len(todo)} to run")
    if args.dry:
        for o, _ in todo:
            print("  ", o)
        return

    steps = [f"train.steps={args.steps}"] if args.steps else []
    cmds = [([sys.executable, "scripts/run_experiment.py", "--config", "configs/kepler_ar.yaml",
              "--set", *BASE, *ov, *steps], os.path.join(ROOT, o, "train.log")) for o, ov in todo]
    run_pool(cmds, n_gpu=args.n_gpu, per_gpu=args.per_gpu)


if __name__ == "__main__":
    main()
