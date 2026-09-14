#!/usr/bin/env python
"""THE demonstration: the prior work's Newton->Kepler transition is a probe artifact.

We reproduce their setup: linear-probe the hidden states for the "Newton" variable
(the force) and the "Kepler" variables (orbital elements: energy->a, Lz, LRL vector),
as a function of context length -- on the trained models. Then we run the SAME probe
on (i) the raw input context (no model at all) and (ii) an untrained network. If the
transition reappears in the controls, it is information-availability, not learning:
the force is decodable from one point; orbital elements need velocity, so they only
become decodable as context grows.

Uses the existing runs/h4_kepler/ar_ctx{2,4,8,16,32}_s0.0 checkpoints. CPU only.
"""
from __future__ import annotations
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physwm.train.checkpoint import load_run
from physwm.methods import make_method
from physwm.eval.probe import probe_representations, probe_raw_context

CTXS = [2, 4, 8, 16, 32]
KEP = ["inv_energy", "inv_Lz", "inv_LRL_x", "inv_LRL_y"]   # orbital elements = "Kepler"


def kepler_r2(r2dict):
    return max(r2dict.get(k, -9) for k in KEP)


def main():
    rows = []
    for ctx in CTXS:
        run = f"runs/h4_kepler/ar_ctx{ctx}_s0.0"
        if not os.path.exists(os.path.join(run, "final.pt")):
            print("missing", run); continue
        cfg, ds, m = load_run(run)
        pr = probe_representations(m, ds, "train", "test", context=ctx, n_traj=96)
        # untrained network (random init, same arch)
        torch.manual_seed(0)
        un = make_method(cfg.model, cfg.method, ds.state_dim); un.eval()
        pu = probe_representations(un, ds, "train", "test", context=ctx, n_traj=96)
        # raw input control (no network)
        rw = probe_raw_context(ds, "train", "test", context=ctx, n_traj=96)

        rows.append(dict(ctx=ctx,
                         model_newton=pr.r2["force"], model_kepler=kepler_r2(pr.r2),
                         untr_newton=pu.r2["force"],  untr_kepler=kepler_r2(pu.r2),
                         raw_newton=rw.r2["force"],    raw_kepler=kepler_r2(rw.r2)))
        r = rows[-1]
        print(f"ctx={ctx:2d} | MODEL N={r['model_newton']:.3f} K={r['model_kepler']:.3f} | "
              f"UNTRAINED N={r['untr_newton']:.3f} K={r['untr_kepler']:.3f} | "
              f"RAW N={r['raw_newton']:.3f} K={r['raw_kepler']:.3f}", flush=True)

    json.dump(rows, open("runs/probe_artifact.json", "w"), indent=2)
    print("\n=== the transition: Newton(force) vs Kepler(orbital elements) probe R^2 vs context ===")
    print("If Newton falls & Kepler rises for the MODEL *and* the controls reproduce it,")
    print("the 'Newton->Kepler transition' is an information-availability artifact.\n")
    print(f"{'ctx':>4}  {'MODEL N/K':>14}  {'UNTRAINED N/K':>16}  {'RAW-INPUT N/K':>16}")
    for r in rows:
        print(f"{r['ctx']:>4}  {r['model_newton']:.2f}/{r['model_kepler']:.2f}      "
              f"{r['untr_newton']:.2f}/{r['untr_kepler']:.2f}        "
              f"{r['raw_newton']:.2f}/{r['raw_kepler']:.2f}")
    print("\n-> runs/probe_artifact.json")


if __name__ == "__main__":
    main()
