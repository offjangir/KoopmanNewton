#!/usr/bin/env python
"""Control-task probing for the claimed Newton->Kepler world-model transition.

  python control_probe.py <paradigm> <ctx> <sigma> <seed> [steps]

The original claim is that as context grows, the representation stops encoding the
local Newtonian force and starts encoding the global Keplerian orbital elements --
read as the emergence of a global world model.

Orbital elements are an exact deterministic function of (r, v), and v is an exact
function of any two consecutive clean states. So at ctx=2 the information is already
complete, and *no* rise from ctx=2 to ctx=90 can be information becoming available.
This script measures, under one protocol, what else the rise could be:

  force    -- -r/|r|^3, a pointwise function of the CURRENT position alone
  vel      -- v_t, needs >=2 states, NOT conserved
  kepler   -- a, e, LRL_x, LRL_y: conserved, a nonlinear function of (r, v)
  sham     -- arbitrary nonlinear functions of (r, v), NOT conserved, no physics

plus three baselines: the trained net, an untrained net at the same init (random
nonlinear features), and a linear readout of the raw context (architecture-free
information+linearity floor).

If `kepler` rises with context but `vel` and `sham` rise identically, the transition
is velocity decodability, not an emergent world model. If the rise also survives at
sigma=0, noise-averaging is ruled out; if it vanishes, it is the denoising objective.
"""
import sys, os, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np, torch

torch.set_num_threads(2)          # the node is shared by ~12 of these; without this
torch.set_num_interop_threads(1)  # each process grabs all 32 cores and the box thrashes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physwm.data.dataset import make_loader
from physwm.data.newton_kepler import load_newton_kepler, KEPLER_ELEMENTS, VELOCITY, SHAM
from physwm.methods import make_method
from physwm.train.config import ModelCfg, MethodCfg
from physwm.eval.probe import probe_representations, probe_raw_context

GROUPS = {"kepler": KEPLER_ELEMENTS, "vel": VELOCITY, "sham": SHAM}


def build(n_test=2000):
    """Their trajectories, with velocity / sham targets added alongside the elements.
    Data dir: $NEWTON_KEPLER_DATA (see physwm/data/newton_kepler.py)."""
    return load_newton_kepler(n_test=n_test, controls=True)


def group_r2(r2: dict) -> dict:
    """Collapse per-column R^2 into the four named targets (max within a group)."""
    out = {"force": r2["force"]}
    for g, keys in GROUPS.items():
        out[g] = max(r2[f"inv_{k}"] for k in keys)
    return out


@torch.no_grad()
def pred_rmse(method, ds, ctx, dev, n=64):
    """1-step prediction error on held-out trajectories, in normalized units."""
    st = ds.splits["test"].states[:n]
    nm = ds.normalize(st).astype(np.float32)
    X = np.stack([nm[i, t - ctx: t] for i in range(len(nm)) for t in range(ctx, nm.shape[1])])
    Y = np.stack([nm[i, t] for i in range(len(nm)) for t in range(ctx, nm.shape[1])])
    se, cnt = 0.0, 0
    for i in range(0, len(X), 1024):
        p = method.sample(torch.from_numpy(X[i:i + 1024]).to(dev)).cpu().numpy()
        se += float(((p - Y[i:i + 1024]) ** 2).sum()); cnt += p.size
    return float(np.sqrt(se / cnt))


def main():
    paradigm, ctx = sys.argv[1], int(sys.argv[2])
    sigma, seed = float(sys.argv[3]), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{paradigm}_ctx{ctx}_sig{sigma}_s{seed}"
    out_path = os.path.join(ROOT, "results_control", f"{tag}.json")
    if os.path.exists(out_path):
        print(f"skip {tag}"); return

    ds = build()
    torch.manual_seed(seed); np.random.seed(seed)
    mcfg = ModelCfg(d_model=32, n_layers=2, n_heads=2, d_ff=64, attn_window=ctx)
    mk = lambda: make_method(mcfg, MethodCfg(kind=paradigm, ctx_noise_sigma=sigma,
                                             flow_sample_steps=10, n_diffusion_steps=20),
                             ds.state_dim).to(dev)
    probe = lambda net: group_r2(probe_representations(
        net, ds, "train", "test", context=ctx, n_traj=128, device=dev, seed=seed).r2)

    untrained = mk(); untrained.eval()
    r2_untrained = probe(untrained)

    torch.manual_seed(seed)
    method = mk(); method.train()
    loader = make_loader(ds, "train", ctx, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
    from itertools import cycle
    it = cycle(loader)
    for _ in range(steps):
        b = {k: v.to(dev) for k, v in next(it).items()}
        opt.zero_grad(); loss = method.loss(b); loss.backward(); opt.step()
    method.eval()
    r2_trained = probe(method)

    # architecture-independent floor: linear readout of the raw normalized context
    r2_raw = group_r2(probe_raw_context(ds, "train", "test", context=ctx,
                                        n_traj=128, seed=seed).r2)

    exc = lambda t, c: {k: float((t[k] - c[k]) / (1.0 - c[k])) if c[k] < 1.0 else float("nan")
                        for k in t}
    out = dict(paradigm=paradigm, context=ctx, sigma=sigma, seed=seed, steps=steps,
               trained=r2_trained, untrained=r2_untrained, raw_context=r2_raw,
               excess_over_untrained=exc(r2_trained, r2_untrained),
               excess_over_raw=exc(r2_trained, r2_raw),
               final_loss=float(loss.item()), pred_rmse=pred_rmse(method, ds, ctx, dev))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    f = lambda d: " ".join(f"{k[:3]}={d[k]:+.3f}" for k in ["force", "vel", "kepler", "sham"])
    print(f"{tag}\n  trained  {f(r2_trained)}\n  untrain  {f(r2_untrained)}\n"
          f"  rawctx   {f(r2_raw)}\n  rmse={out['pred_rmse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
