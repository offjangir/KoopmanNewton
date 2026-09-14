#!/usr/bin/env python
"""Generate the experiment grid: domain x method x (context/bias) -> YAML configs.

    python scripts/gen_configs.py --out configs/generated

Each config is a full ExperimentCfg. The generative methods use the DiT backbone;
AR uses the causal transformer (the factory sets model.kind automatically). The
temporal-locality sweep varies data.context (and attn_window for AR).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physwm.train.config import ExperimentCfg, DataCfg, ModelCfg, MethodCfg, TrainCfg
from physwm.methods import ALL_METHODS

DOMAINS = ["kepler", "gr_kepler", "nbody3", "driven_pendulum", "double_pendulum",
           "rabi", "ehrenfest", "wavepacket", "rel_lorentz", "schwarzschild"]
CONTEXTS = [2, 8, 32]        # temporal-locality sweep


def make_cfg(domain, method, context):
    name = f"{domain}_{method}_ctx{context}"
    model = ModelCfg(
        kind="ar_transformer" if method == "ar" else "dit",
        d_model=128, n_layers=4, n_heads=4, d_ff=256,
        attn_window=context if method == "ar" else None,
        head="regression",
    )
    return ExperimentCfg(
        name=name,
        data=DataCfg(domain=domain, n_train=4000, n_val=400, n_test=1000,
                     context=context, seed=0),
        model=model,
        method=MethodCfg(kind=method, ctx_noise_sigma=0.1),
        train=TrainCfg(steps=20000, batch_size=256, lr=1e-3, device="cuda",
                       out_dir=f"runs/{name}"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/generated")
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--contexts", nargs="*", type=int, default=CONTEXTS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for domain in args.domains:
        for method in args.methods:
            for ctx in args.contexts:
                cfg = make_cfg(domain, method, ctx)
                cfg.save(os.path.join(args.out, f"{cfg.name}.yaml"))
                n += 1
    print(f"wrote {n} configs to {args.out}")


if __name__ == "__main__":
    main()
