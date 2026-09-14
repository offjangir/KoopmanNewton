#!/usr/bin/env python
"""Parameter counts and training throughput per method/model size.

Used to plan compute budgets: prints params and steps/s so wall-clock for a
target step count can be read off directly.

    python scripts/benchmark.py --state-dim 2 --batch 512 --context 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physwm.train.config import ModelCfg, MethodCfg
from physwm.methods import make_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dim", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--methods", nargs="*", default=["ar", "diffusion", "flow", "drifting"])
    ap.add_argument("--target-steps", type=int, default=6000)
    ap.add_argument("--sizes", nargs="*", default=["128x4", "256x6", "384x8", "768x12"])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hdr = ("method", "d_model", "layers", "params", "step/s", f"{args.target_steps} steps")
    print(f"{hdr[0]:10s}{hdr[1]:>8s}{hdr[2]:>7s}{hdr[3]:>13s}{hdr[4]:>9s}{hdr[5]:>14s}")
    for kind in args.methods:
        for size in args.sizes:
            d, L = (int(v) for v in size.split("x"))
            mc = ModelCfg(d_model=d, n_layers=L, n_heads=8, d_ff=4 * d,
                          attn_window=args.context if kind == "ar" else None)
            m = make_method(mc, MethodCfg(kind=kind, drift_particles=4), args.state_dim).to(dev)
            n = sum(p.numel() for p in m.parameters())
            b = {"context": torch.randn(args.batch, args.context, args.state_dim, device=dev),
                 "target": torch.randn(args.batch, args.state_dim, device=dev)}
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for _ in range(5):
                opt.zero_grad(); m.loss(b).backward(); opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(30):
                opt.zero_grad(); m.loss(b).backward(); opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            sps = 30 / (time.time() - t0)
            print(f"{kind:10s}{d:8d}{L:7d}{n:13,d}{sps:9.1f}{args.target_steps/sps/60:12.1f}m")
            del m, opt
            if dev == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
