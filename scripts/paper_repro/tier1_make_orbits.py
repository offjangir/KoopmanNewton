"""Make the orbit sets for the Tier 1 test of the paper models (arXiv 2602.06923, repo d1e37bd).

All orbits use the paper's own generate_kepler_trajectory (GM 1, dt 0.2, 100 points, solve_ivp rtol/atol 1e-8).
The orbital parameters use the same formulas as generate_trajectories() in generate_kepler_cv.py
(that code is inline there, so it is repeated here without change).

Sets (file names keep short names):
  A      the paper's own test split: trajectories 10000-19999 of data_cv (e 0-0.8, a 0.5-2)
  BmA    e uniform 0-0.9, a uniform 0.5-3, points inside A's box (e < 0.8 and a < 2) rejected
  C_ecc  e uniform 0.9-0.95, a uniform 0.5-3
  C_far  e uniform 0-0.9, a uniform 3-4
The new sets have 10000 orbits (BmA) and 5000 + 5000 orbits (C_ecc + C_far = 10000), the size of the paper test split.

Usage: python tier1_make_orbits.py <set name> <out_dir>
"""
import importlib.util
import os
import sys

import numpy as np
import torch

PAPER = "/weka/home/jhu/ycai71/newton-kepler"
sys.path.insert(0, PAPER)
import generate_kepler_cv as G  # noqa: E402

name, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)


def params_of(e, a, angle, traj, vel):
    """Same formulas as generate_trajectories() in generate_kepler_cv.py."""
    b = a * np.sqrt(1 - e ** 2)
    c = e * a
    average_radius = np.sqrt(a * b)
    x, y = traj[0]
    vx, vy = vel[0]
    r = np.sqrt(x ** 2 + y ** 2)
    Lz = x * vy - y * vx
    GM = 1.0
    Ax = vy * Lz - GM * x / r
    Ay = -vx * Lz - GM * y / r
    return {'e': float(e), 'a': float(a), 'b': float(b), 'c': float(c), 'average_radius': float(average_radius),
            'LRL_x': float(Ax), 'LRL_y': float(Ay), 'LRL_magnitude': float(np.sqrt(Ax ** 2 + Ay ** 2)),
            'LRL_angle': float(np.arctan2(Ay, Ax)), 'n_x': np.cos(angle), 'n_y': np.sin(angle)}


if name == "A":
    trajs = torch.load(f"{PAPER}/data_cv/trajectories_chunk_000001.pt").numpy()
    spec = importlib.util.spec_from_file_location("p", f"{PAPER}/data_cv/orbital_params_chunk_000001.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.orbital_params
else:
    rng = np.random.default_rng({"BmA": 101, "C_ecc": 102, "C_far": 103}[name])
    n = {"BmA": 10000, "C_ecc": 5000, "C_far": 5000}[name]
    trajs, params = [], []
    while len(trajs) < n:
        if name == "BmA":
            e, a = rng.uniform(0.0, 0.9), rng.uniform(0.5, 3.0)
            if e < 0.8 and a < 2.0:
                continue
        elif name == "C_ecc":
            e, a = rng.uniform(0.9, 0.95), rng.uniform(0.5, 3.0)
        else:
            e, a = rng.uniform(0.0, 0.9), rng.uniform(3.0, 4.0)
        angle = rng.uniform(0, 2 * np.pi)
        traj, vel = G.generate_kepler_trajectory(eccentricity=e, semi_major_axis=a, angle=angle,
                                                 dt=G.dt, num_points=G.num_points_per_trajectory)
        trajs.append(traj)
        params.append(params_of(e, a, angle, traj, vel))
    trajs = np.array(trajs, dtype=np.float32)

keys = list(params[0].keys())
P = {k: np.array([p[k] for p in params], dtype=np.float64) for k in keys}
np.savez(f"{out_dir}/{name}.npz", trajectories=trajs, **P)
r = np.linalg.norm(trajs, axis=-1)
print(name, "shape", trajs.shape, "finite", bool(np.isfinite(trajs).all()))
print(" e", P['e'].min(), P['e'].max(), " a", P['a'].min(), P['a'].max())
print(" x", trajs[..., 0].min(), trajs[..., 0].max(), " y", trajs[..., 1].min(), trajs[..., 1].max(),
      " r", r.min(), r.max())
if name == "BmA":
    print(" points inside A box:", int(((P['e'] < 0.8) & (P['a'] < 2.0)).sum()))
