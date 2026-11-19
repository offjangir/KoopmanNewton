"""Tier 1: test one Tier 0 model (paper code, arXiv 2602.06923) on the orbit sets. No training.

Usage: python tier1_eval.py <block_size> <data_dir> <ckpt_dir> <out_dir>

Probe steps (1000 windows, last position, as the paper; R2 of every target and layer):
  reference: fit and score on the same windows of the paper training orbits (the paper way)
  step 1: fit on training-orbit windows, score with no refit on A, BmA, C_ecc, C_far, C_all
  step 2: fit and score on the same windows of BmA, C_ecc, C_far, C_all, with the paper functions
  step 3: split by orbit inside BmA, C_ecc, C_far, C_all (fit on one half of the orbits, score on the other)
Errors on train, A, BmA, C_ecc, C_far, C_all: next-step error, and the paper rollout error (50 known points, 50 predicted points, 2000 orbits).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PAPER = "/weka/home/jhu/ycai71/newton-kepler"
sys.path.insert(0, PAPER)
import kepler_cv_blocksize as K  # noqa: E402

b = int(sys.argv[1])
data_dir, ckpt_dir, out_dir = sys.argv[2:5]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
torch.set_grad_enabled(False)

FORCE = ['F_magnitude', 'F_direction_x', 'F_direction_y', 'Fx', 'Fy',
         'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y']
GEOM = ['e', 'a', 'b', 'c', 'average_radius', 'LRL_x', 'LRL_y', 'LRL_magnitude', 'LRL_angle', 'n_x', 'n_y',
        '1/a', '1/a^2', '1/b', '1/b^2']
PKEYS = ['e', 'a', 'b', 'c', 'average_radius', 'LRL_x', 'LRL_y', 'LRL_magnitude', 'LRL_angle', 'n_x', 'n_y']

ck = torch.load(f"{ckpt_dir}/model_block_size_{b}.pt", weights_only=False)
cfg = ck["config"]
model = K.setup_model(block_size=cfg["block_size"], n_layer=cfg["n_layer"], n_embd=cfg["n_embd"], device=K.device)
model.load_state_dict(ck["state_dict"])
model.eval()
hooks, act = K.setup_activation_hooks(model)


def load(name):
    if name == "C_all":
        x, y = load("C_ecc"), load("C_far")
        return {k: np.concatenate([x[k], y[k]]) for k in x}
    if name == "train":
        # the paper's training orbits: data_cv trajectories 0-9999 (chunk 0)
        import importlib.util
        trajs = torch.load(f"{PAPER}/data_cv/trajectories_chunk_000000.pt").numpy()
        spec = importlib.util.spec_from_file_location("p0", f"{PAPER}/data_cv/orbital_params_chunk_000000.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = {k: np.array([p[k] for p in mod.orbital_params], dtype=np.float64) for k in PKEYS}
        out["trajectories"] = trajs
        return out
    d = np.load(f"{data_dir}/{name}.npz")
    return {k: d[k] for k in d.files}


def inputs_of(trajs, seed):
    if b < K.num_points_per_trajectory:
        inp, tgt = K.chop_trajectories_into_sequences(trajs, b, seed=seed)
    else:
        inp, tgt = trajs[:, :-1], trajs[:, 1:]
    return inp.astype(np.float32), tgt.astype(np.float32)


def sample(trajs, n, seed, seq_seed):
    inp, tgt = inputs_of(trajs, seq_seed)
    per = inp.shape[0] // trajs.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(inp.shape[0], size=min(n, inp.shape[0]), replace=False)
    return inp, tgt, idx, idx // per


def activations(x):
    for k in list(act.keys()):
        del act[k]
    model.forward(torch.from_numpy(x).to(K.device), None)
    return {k: v.cpu() for k, v in act.items()}


def targets_last(x, P, tidx):
    g = K.compute_gravitational_force(x)
    B, T = x.shape[:2]
    out = {t: g[t].reshape(B, T)[:, -1] for t in FORCE}
    for t in PKEYS:
        out[t] = P[t][tidx]
    out['1/a'] = 1.0 / P['a'][tidx]
    out['1/a^2'] = 1.0 / P['a'][tidx] ** 2
    out['1/b'] = 1.0 / P['b'][tidx]
    out['1/b^2'] = 1.0 / P['b'][tidx] ** 2
    return out


def last_acts(x):
    return {l: v[:, -1, :].numpy() for l, v in activations(x).items()}


def fit(a_last, y):
    return {l: {t: LinearRegression().fit(a, y[t]) for t in FORCE + GEOM} for l, a in a_last.items()}


def score(pr, a_last, y):
    return {l: {t: float(r2_score(y[t], pr[l][t].predict(a))) for t in FORCE + GEOM} for l, a in a_last.items()}


def windows(D, orbits, seed, seq_seed):
    """1000 windows (last position) from the given orbits of a set, sampled as the paper does."""
    trajs = D["trajectories"][orbits]
    P = {k: D[k][orbits] for k in PKEYS}
    inp, _, idx, tidx = sample(trajs, 1000, seed=seed, seq_seed=seq_seed)
    return inp[idx], P, tidx


NEW_SETS = ["BmA", "C_ecc", "C_far", "C_all"]
data = {s: load(s) for s in ["train", "A"] + NEW_SETS}
res = {"block_size": b, "probe_steps": {}, "errors": {}}
ALL = lambda s: np.arange(data[s]["trajectories"].shape[0])  # noqa: E731

# Reference: fit and score on the same 1000 windows of the training orbits (the paper's chop seed 1).
xT, PT, tT = windows(data["train"], ALL("train"), seed=12, seq_seed=1)
aT, yT = last_acts(xT), targets_last(xT, PT, tT)
pT = fit(aT, yT)
res["probe_steps"]["reference_in_sample_training"] = score(pT, aT, yT)

# Step 1: the training-orbit probes, scored with no refit on other sets.
for s in ["A"] + NEW_SETS:
    x, P, t = windows(data[s], ALL(s), seed=12, seq_seed=22)
    res["probe_steps"][f"step1_fit_training_score_{s}"] = score(pT, last_acts(x), targets_last(x, P, t))

for s in NEW_SETS:
    # Step 2: in-sample, with the paper's own functions (fit and score on the same windows).
    x, P, t = windows(data[s], ALL(s), seed=12, seq_seed=22)
    acts = activations(x)
    grav = K.compute_gravitational_force(x)
    plist = [{k: float(P[k][i]) for k in PKEYS} for i in range(len(P['e']))]
    rf = K.run_linear_probes(acts, grav)
    rg = K.run_geometry_probes(acts, plist, t)
    res["probe_steps"][f"step2_in_sample_{s}"] = {
        l: {**{q: float(rf[l][q]['r2_last']) for q in FORCE}, **{q: float(rg[l][q]['r2_last']) for q in GEOM}} for l in rf}
    # Step 3: split by orbit. Fit on the first half of the orbits, score on the second half.
    # For C_all, each half holds half of C_ecc and half of C_far.
    n = data[s]["trajectories"].shape[0]
    if s == "C_all":
        h = n // 4
        o1 = np.r_[0:h, 2 * h:3 * h]
        o2 = np.r_[h:2 * h, 3 * h:4 * h]
    else:
        o1, o2 = np.arange(n // 2), np.arange(n // 2, n)
    x1, P1, t1 = windows(data[s], o1, seed=17, seq_seed=24)
    x2, P2, t2 = windows(data[s], o2, seed=18, seq_seed=25)
    a1, y1 = last_acts(x1), targets_last(x1, P1, t1)
    a2, y2 = last_acts(x2), targets_last(x2, P2, t2)
    res["probe_steps"][f"step3_split_by_orbit_{s}"] = score(fit(a1, y1), a2, y2)
print(f"[tier1] block {b} probes done at {time.time() - t0:.0f} s", flush=True)

# Next-step error and the paper's 50-step rollout error.
for s in ["train", "A"] + NEW_SETS:
    trajs = data[s]["trajectories"]
    inp, tgt = inputs_of(trajs, {"train": 1}.get(s, 22))
    rng = np.random.default_rng(14)
    j = rng.choice(inp.shape[0], size=min(10000, inp.shape[0]), replace=False)
    errs = []
    for k in range(0, len(j), 2000):
        jj = j[k:k + 2000]
        pred, _ = model.forward(torch.from_numpy(inp[jj]).to(K.device), None)
        errs.append(np.linalg.norm(pred[:, -1, :].cpu().numpy() - tgt[jj][:, -1, :], axis=-1))
    ri = np.random.default_rng(15).choice(trajs.shape[0], size=min(2000, trajs.shape[0]), replace=False)
    tr = trajs[ri]
    st = K.generate_trajectory_and_compute_error(model, torch.from_numpy(tr[:, :50]).float().to(K.device), tr, 50,
                                                 block_size=b)
    res["errors"][s] = {"next_step_error": float(np.mean(np.concatenate(errs))),
                        "rollout_error_mean": float(np.mean(st["position_errors"])),
                        "rollout_error_curve": [float(v) for v in st["position_errors"]]}
    print(f"[tier1] block {b} errors {s} done at {time.time() - t0:.0f} s", flush=True)

# Example rollouts on the held-out far orbits (3 of C_ecc and 3 of C_far), for the page figure.
ex = np.concatenate([data["C_ecc"]["trajectories"][:3], data["C_far"]["trajectories"][:3]])
seq = torch.from_numpy(ex[:, :50]).float().to(K.device)
for _ in range(50):
    cur = seq[:, -b:] if b < K.num_points_per_trajectory else seq[:, -model.config.block_size:]
    p, _ = model.forward(cur, None)
    seq = torch.cat([seq, p[:, -1:, :]], dim=1)
np.savez(f"{out_dir}/example_rollouts_far.npz", true=ex, pred=seq.cpu().numpy())

for h in hooks:
    h.remove()
res["seconds"] = time.time() - t0
json.dump(res, open(f"{out_dir}/results.json", "w"))
print(f"[tier1] block {b} finished in {res['seconds']:.0f} s", flush=True)
