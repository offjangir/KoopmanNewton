"""Load the Kepler trajectories released with the newton-kepler paper repo.

Those trajectories (``data_cv/`` produced by ``generate_kepler_cv.py`` in
https://github.com/KindXiaoming/newton-kepler) are the data behind the original
Newton->Kepler block-size result, so experiments that re-examine that result run
on them rather than on our own simulator.

The directory is taken from ``$NEWTON_KEPLER_DATA``; by default a
``newton-kepler`` checkout next to this repo is assumed.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from .build_dataset import PhysicsDataset, Split

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(os.path.dirname(_REPO), "newton-kepler", "data_cv")

KEPLER_ELEMENTS = ["a", "e", "LRL_x", "LRL_y"]
VELOCITY = ["vx", "vy"]
SHAM = ["sham1", "sham2", "sham3", "sham4"]
DT = 0.2


def data_dir() -> str:
    return os.environ.get("NEWTON_KEPLER_DATA", DEFAULT_DIR)


def load_newton_kepler(n_test: int = 2000, controls: bool = False,
                       path: str | None = None) -> PhysicsDataset:
    """Their trajectories as a :class:`PhysicsDataset`.

    ``law`` is the Newtonian force ``-r/|r|^3``; ``invariants`` are the orbital
    elements (per-trajectory constants). With ``controls=True`` two control target
    groups are appended: the velocity (needs >=2 states, not conserved) and four
    sham nonlinear functions of (r, v) with no dynamical meaning.
    """
    import torch

    path = path or data_dir()
    tr = np.asarray(torch.load(os.path.join(path, "trajectories_chunk_000000.pt"),
                               weights_only=False), dtype=np.float64)
    spec = importlib.util.spec_from_file_location(
        "orbital_params", os.path.join(path, "orbital_params_chunk_000000.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    op = mod.orbital_params
    N, T, _ = tr.shape

    r = np.linalg.norm(tr, axis=-1, keepdims=True)
    law = -tr / r ** 3
    kep = np.array([[op[i][k] for k in KEPLER_ELEMENTS] for i in range(N)])
    inv = np.repeat(kep[:, None, :], T, axis=1)
    names = list(KEPLER_ELEMENTS)

    if controls:
        # central difference on the clean trajectory; O(dt^2), endpoints one-sided
        v = np.empty_like(tr)
        v[:, 1:-1] = (tr[:, 2:] - tr[:, :-2]) / (2 * DT)
        v[:, 0] = (tr[:, 1] - tr[:, 0]) / DT
        v[:, -1] = (tr[:, -1] - tr[:, -2]) / DT
        x, y = tr[..., 0], tr[..., 1]
        vx, vy = v[..., 0], v[..., 1]
        # matched to the orbital elements in the inputs they need, not in physics
        sham = np.stack([np.sin(2.0 * x) * np.cos(3.0 * vy),
                         x * vy - 0.5 * y * vx * vx,
                         np.tanh(1.5 * vx) * (x ** 2 - y ** 2),
                         np.cos(2.5 * y + vx)], axis=-1)
        inv = np.concatenate([inv, v, sham], axis=-1)
        names += VELOCITY + SHAM

    ntr = N - n_test

    def split(sl):
        return Split(tr[sl], law[sl], inv[sl], "x")

    splits = {"train": split(slice(0, ntr)), "val": split(slice(ntr, ntr + 200)),
              "test": split(slice(ntr, N))}
    flat = tr[:ntr].reshape(-1, 2)
    return PhysicsDataset("kepler_theirs", splits, flat.mean(0), flat.std(0) + 1e-6,
                          names, dt=DT)
