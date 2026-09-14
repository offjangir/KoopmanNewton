#!/usr/bin/env python
"""Is there ONE criterion that says when a model learns a physical law?

  python generality.py <domain> <paradigm> <sigma_ratio> <seed> [steps]

Every system here is first order on the state dims its law drives:

    kepler / harmonic   dv/dt = a(r)            law acts on (vx, vy)
    pendulum            d(thetadot)/dt = -sin   law acts on (thetadot,)
    rabi_u              dR/dt = b x R           law acts on (Rx, Ry, Rz)
    rel_lorentz_u       du/dt = q(E + v x B)    law acts on (ux, uy, uz)

so a single behavioral test covers all of them: compare the model's predicted
increment on those dims to the true `dt * law`. Persistence (pred = s_t) scores 0,
so the null is meaningful and no dt-division amplifies error.

The signal carried by the law in the units the noise is applied in (normalized) is

    sigma_star = mean |dt * law| / std

which is a prediction with no free parameters: learning should succeed while
sigma << sigma_star and fail above it. We sweep sigma as a RATIO of sigma_star, so
if the criterion is right every law collapses onto one curve in sigma/sigma_star.
"""
import sys, os, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np, torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physwm.data import build_dataset
from physwm.data.dataset import make_loader
from physwm.methods import make_method
from physwm.train.config import ModelCfg, MethodCfg, DataCfg

# state dims each law drives, and the ODE order of the underlying system
LAW_DIMS = {"kepler": [2, 3], "harmonic": [2, 3], "pendulum": [2],
            "rabi_u": [0, 1, 2], "rel_lorentz_u": [3, 4, 5]}
ORDER = {"kepler": 2, "harmonic": 2, "pendulum": 2, "rabi_u": 1, "rel_lorentz_u": 1}


def sigma_star(ds, domain):
    """Law signal per step, in the normalized units the context noise lives in."""
    dims = LAW_DIMS[domain]
    law = ds.splits["train"].law                       # (N,T,L) analytic RHS
    std = np.asarray(ds.std, float)[dims]
    sig = np.abs(ds.dt * law) / std                    # per-dim signal / unit noise
    return float(sig.mean()), float(np.sqrt((sig ** 2).mean()))


@torch.no_grad()
def increment_r2(method, ds, domain, ctx, device, n_traj=128, max_samples=6000, seed=0):
    """R^2 of the model's predicted increment against dt*law on the law-carrying dims."""
    dims = LAW_DIMS[domain]
    rng = np.random.default_rng(seed)
    st = ds.splits["test"].states[:n_traj]
    law = ds.splits["test"].law[:n_traj]
    nm = ds.normalize(st).astype(np.float32)
    N, T, _ = nm.shape
    idx = [(n, t) for n in range(N) for t in range(ctx - 1, T - 1)]
    if len(idx) > max_samples:
        idx = [idx[i] for i in rng.choice(len(idx), max_samples, replace=False)]
    C = np.stack([nm[n, t - ctx + 1: t + 1] for n, t in idx]).astype(np.float32)
    s_t = np.stack([st[n, t] for n, t in idx])
    s_nx = np.stack([st[n, t + 1] for n, t in idx])
    L_t = np.stack([law[n, t] for n, t in idx])

    method.eval()
    mean, std = np.asarray(ds.mean, float), np.asarray(ds.std, float)
    preds = []
    for i in range(0, len(C), 1024):
        preds.append(method.sample(torch.from_numpy(C[i:i + 1024]).to(device)).cpu().numpy())
    pred = np.concatenate(preds, 0) * std + mean

    target = ds.dt * L_t                               # true increment on those dims
    model_inc = pred[:, dims] - s_t[:, dims]
    true_inc = s_nx[:, dims] - s_t[:, dims]
    ss = lambda a: float((a ** 2).sum())
    r2 = 1 - ss(model_inc - target) / ss(target - target.mean(0))
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    return {"increment_r2": float(r2),
            "persistence_r2": float(1 - ss(0 - target) / ss(target - target.mean(0))),
            "increment_r2_vs_observed": float(1 - ss(model_inc - true_inc)
                                              / ss(true_inc - true_inc.mean(0))),
            "model_rmse": rmse(pred, s_nx),
            "persist_rmse": rmse(s_t, s_nx)}


def main():
    domain, paradigm = sys.argv[1], sys.argv[2]
    ratio, seed = float(sys.argv[3]), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    ctx = int(os.environ.get("CTX", "8"))
    dev = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{domain}_{paradigm}_r{ratio}_s{seed}"
    out_p = os.path.join(ROOT, "results_gen", f"{tag}.json")
    if os.path.exists(out_p):
        print(f"skip {tag}"); return

    ds = build_dataset(DataCfg(domain=domain, n_train=8000, n_val=200, n_test=1000, seed=0))
    s_mean, s_rms = sigma_star(ds, domain)
    sigma = ratio * s_mean

    torch.manual_seed(seed); np.random.seed(seed)
    mcfg = ModelCfg(d_model=128, n_layers=4, n_heads=4, d_ff=256, attn_window=ctx)
    method = make_method(mcfg, MethodCfg(kind=paradigm, ctx_noise_sigma=sigma,
                                         flow_sample_steps=10, n_diffusion_steps=20),
                         ds.state_dim).to(dev)
    method.train()
    loader = make_loader(ds, "train", ctx, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
    from itertools import cycle
    it = cycle(loader)
    for _ in range(steps):
        b = {k: v.to(dev) for k, v in next(it).items()}
        opt.zero_grad(); loss = method.loss(b); loss.backward(); opt.step()

    res = increment_r2(method, ds, domain, ctx, dev, seed=seed)
    out = dict(domain=domain, paradigm=paradigm, order=ORDER[domain], context=ctx,
               sigma_ratio=ratio, sigma=sigma, sigma_star=s_mean, sigma_star_rms=s_rms,
               seed=seed, steps=steps, dt=ds.dt, final_loss=float(loss.item()), **res)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(out, open(out_p, "w"), indent=2)
    print(f"{tag}: sigma*={s_mean:.4f} sigma={sigma:.4f} "
          f"inc_R2={res['increment_r2']:+.4f} persist={res['persistence_r2']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
