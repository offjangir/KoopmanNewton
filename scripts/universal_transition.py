#!/usr/bin/env python
"""Their block-size Newton->Kepler transition, for OUR paradigms (AR / flow / diffusion),
on THEIR data (or one of our domains), with a force + invariant probe against an
untrained-init control.

  python universal_transition.py <paradigm> <context> [domain=kepler_theirs] [steps=6000]

Trains a small model (matched to their scale) at a given context length, then probes
hidden states for the "Newton" variable (force) and the "Kepler" variables (orbital
elements a,e,LRL for their data; the domain's invariants otherwise).
``kepler_theirs`` reads $NEWTON_KEPLER_DATA (see physwm/data/newton_kepler.py).
"""
import sys, os, json
from itertools import cycle
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physwm.data import build_dataset
from physwm.data.dataset import make_loader
from physwm.data.newton_kepler import load_newton_kepler, KEPLER_ELEMENTS
from physwm.methods import make_method
from physwm.train.config import DataCfg, ModelCfg, MethodCfg
from physwm.eval.probe import probe_representations


def main():
    paradigm = sys.argv[1]; ctx = int(sys.argv[2])
    domain = sys.argv[3] if len(sys.argv) > 3 else "kepler_theirs"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if domain == "kepler_theirs":
        ds = load_newton_kepler(); kep_keys = KEPLER_ELEMENTS
    else:
        ds = build_dataset(DataCfg(domain=domain, n_train=8000, n_val=200, n_test=2000,
                                   obs_dims=2, seed=0))
        kep_keys = ds.invariant_names
    sd = ds.state_dim
    def nk(method):
        pr = probe_representations(method, ds, "train", "test", context=ctx, n_traj=128, device=dev)
        return pr.r2["force"], max(pr.r2[f"inv_{k}"] for k in kep_keys)
    mcfg = ModelCfg(d_model=32, n_layers=2, n_heads=2, d_ff=64, attn_window=ctx)
    mkm = lambda: make_method(mcfg, MethodCfg(kind=paradigm, ctx_noise_sigma=0.1,
                                              flow_sample_steps=10, n_diffusion_steps=20), sd).to(dev)
    un = mkm(); un.eval()
    un_N, un_K = nk(un)
    method = mkm(); method.train()
    loader = make_loader(ds, "train", ctx, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
    it = cycle(loader)
    for _ in range(steps):
        b = {k: v.to(dev) for k, v in next(it).items()}
        opt.zero_grad(); loss = method.loss(b); loss.backward(); opt.step()
    method.eval()
    tr_N, tr_K = nk(method)
    out = dict(paradigm=paradigm, domain=domain, context=ctx, trained_newton=tr_N,
               trained_kepler=tr_K, untrained_newton=un_N, untrained_kepler=un_K)
    out_dir = os.path.join(ROOT, "results_law")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, f"{domain}_{paradigm}_ctx{ctx}.json"), "w"), indent=2)
    print(f"{domain} {paradigm} ctx={ctx}: TRAINED N={tr_N:.3f} K={tr_K:.3f} | UNTR N={un_N:.3f} K={un_K:.3f}", flush=True)


if __name__ == "__main__":
    main()
