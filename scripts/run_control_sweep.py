#!/usr/bin/env python
"""Pack the control-task probe sweep (control_probe.py) over the local GPUs.

paradigm x context x sigma x seed. The sigma axis is the causal test: at sigma=0 the
velocity is exactly recoverable from two clean states, so any context-dependence that
survives cannot be noise-averaging.

    N_GPU=4 PER_GPU=3 python scripts/run_control_sweep.py
"""
import itertools
import os
import sys

from gpu_pool import ROOT, run_pool

OUT = os.path.join(ROOT, "results_control")
PARADIGMS = ["ar", "flow", "diffusion"]
CTXS = [2, 5, 10, 20, 50, 90]
SIGMAS = [0.0, 0.1]
SEEDS = [0, 1, 2]
STEPS = 6000

# seed-major order: the full grid completes at seed 0 before any replicate starts, so
# a truncated run still yields a complete (if unreplicated) grid rather than a ragged one
jobs = []
for sd, p, c, s in itertools.product(SEEDS, PARADIGMS, CTXS, SIGMAS):
    tag = f"{p}_ctx{c}_sig{s}_s{sd}"
    if not os.path.exists(f"{OUT}/{tag}.json"):
        jobs.append(([sys.executable, "scripts/control_probe.py", p, str(c), str(s), str(sd),
                      str(STEPS)], f"{OUT}/{tag}.log"))

run_pool(jobs, n_gpu=int(os.environ.get("N_GPU", "4")), per_gpu=int(os.environ.get("PER_GPU", "3")))
