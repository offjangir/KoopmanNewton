#!/usr/bin/env python
"""Launch the Newton->Kepler context transition for MULTIPLE laws, packed over GPUs.

For each (domain, paradigm, context) it runs scripts/universal_transition.py, which
trains a small next-state model at that context on that law's trajectories and probes
the hidden state for the local force (Newton) and the conserved invariants (Kepler),
against an untrained-init control. Comparing harmonic vs kepler asks: is the transition
specific to the inverse-square law, or general to closed-orbit dynamics?

    N_GPU=2 python scripts/run_law_transition.py
"""
import itertools
import os
import sys

from gpu_pool import ROOT, run_pool

OUT = os.path.join(ROOT, "results_law")
DOMAINS = ["harmonic", "kepler"]
PARADIGMS = ["ar", "flow", "diffusion"]
CTXS = [2, 5, 10, 20, 50]
STEPS = 6000

jobs = [([sys.executable, "scripts/universal_transition.py", p, str(c), d, str(STEPS)],
         f"{OUT}/{d}_{p}_ctx{c}.log")
        for d, p, c in itertools.product(DOMAINS, PARADIGMS, CTXS)
        if not os.path.exists(f"{OUT}/{d}_{p}_ctx{c}.json")]

run_pool(jobs, n_gpu=int(os.environ.get("N_GPU", "2")), per_gpu=2)
