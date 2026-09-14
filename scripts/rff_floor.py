#!/usr/bin/env python
"""The information floor the probe should have been measured against.

`probe_raw_context` fits a LINEAR ridge on the raw context, so it reports ~0 for the
orbital elements and looks like a clean floor. It is not: the elements are a smooth
NONLINEAR function of (r, v), and v is a linear function of two frames. A random-
Fourier-feature ridge on the identical raw context -- no network, no training, no
physics -- recovers them almost perfectly.

That makes the RFF readout the right control: it is architecture-free (unlike the
untrained net, whose score depends on whether the backbone zero-inits its residual
gates) and nonlinearity-matched (unlike the linear readout). Any claim that a trained
model "encodes the Keplerian world model" has to clear this line.

  python rff_floor.py [n_features]
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from control_probe import build, GROUPS
from physwm.eval.probe import ridge_r2

CTXS = [2, 5, 10, 20, 50, 90]
D_RFF = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_TRAJ, MAX_S = 128, 4000


def gather(ds, split, ctx, rng):
    st = ds.splits[split].states[:N_TRAJ]
    law = ds.splits[split].law[:N_TRAJ]
    inv = ds.splits[split].invariants[:N_TRAJ]
    nm = ds.normalize(st).astype(np.float64)
    N, T, _ = nm.shape
    idx = [(n, t) for n in range(N) for t in range(ctx - 1, T)]
    if len(idx) > MAX_S:
        idx = [idx[i] for i in rng.choice(len(idx), MAX_S, replace=False)]
    X = np.stack([nm[n, t - ctx + 1: t + 1].ravel() for n, t in idx])
    return X, np.stack([law[n, t] for n, t in idx]), np.stack([inv[n, t] for n, t in idx])


def main():
    ds = build()
    names = ds.invariant_names
    out = {}
    for ctx in CTXS:
        rng = np.random.default_rng(0)
        Xf, lf, vf = gather(ds, "train", ctx, rng)
        Xe, le, ve = gather(ds, "test", ctx, rng)
        # RBF random Fourier features on the raw context. Bandwidth from the median
        # pairwise-distance heuristic so the floor stays comparable as ctx (and hence
        # the input dimension) grows -- a fixed bandwidth silently degrades at long
        # context and would understate the floor exactly where we need it honest.
        d = Xf.shape[1]
        sub = Xf[rng.choice(len(Xf), min(800, len(Xf)), replace=False)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.sqrt(np.median(d2[d2 > 0]))
        W = rng.normal(0.0, 1.0 / max(med, 1e-9), size=(d, D_RFF))
        b = rng.uniform(0, 2 * np.pi, size=D_RFF)
        phi = lambda X: np.sqrt(2.0 / D_RFF) * np.cos(X @ W + b)
        Pf, Pe = phi(Xf), phi(Xe)

        tgt_f = {"force": lf}; tgt_e = {"force": le}
        for j, nm_ in enumerate(names):
            tgt_f[f"inv_{nm_}"] = vf[:, j:j + 1]; tgt_e[f"inv_{nm_}"] = ve[:, j:j + 1]

        # ridge strength chosen per target on a split of the FIT set only
        cut = int(0.8 * len(Pf))
        alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
        def tuned(Ptr, Pte, k):
            best = max(alphas, key=lambda a: ridge_r2(Ptr[:cut], tgt_f[k][:cut],
                                                      Ptr[cut:], tgt_f[k][cut:], a))
            return ridge_r2(Ptr, tgt_f[k], Pte, tgt_e[k], best)
        lin = {k: ridge_r2(Xf, tgt_f[k], Xe, tgt_e[k], 1e-2) for k in tgt_f}
        rff = {k: tuned(Pf, Pe, k) for k in tgt_f}
        grp = lambda r: {"force": r["force"],
                         **{g: max(r[f"inv_{k}"] for k in ks) for g, ks in GROUPS.items()}}
        out[str(ctx)] = {"linear": grp(lin), "rff": grp(rff)}
        L, R = out[str(ctx)]["linear"], out[str(ctx)]["rff"]
        print(f"ctx={ctx:3d}  " + "  ".join(
            f"{t}: lin={L[t]:+.3f} rff={R[t]:+.3f}" for t in ["force", "vel", "kepler", "sham"]),
            flush=True)

    json.dump({"d_rff": D_RFF, "results": out}, open(f"{ROOT}/runs/rff_floor.json", "w"), indent=2)
    print("wrote runs/rff_floor.json")


if __name__ == "__main__":
    main()
