"""Generate physics-trajectory datasets with train/val/test + OOD regime splits.

Produces a :class:`PhysicsDataset` holding, per split, arrays of shape
``states (N,T,F)``, ``law (N,T,L)``, ``invariants (N,T,I)``. Normalization stats
(per-feature mean/std of states) are computed on the train split only. Everything
is seeded for reproducibility and can be cached to ``.npz`` (see ``PhysicsDataset.save``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..physics import make_simulator
from .splits import default_ood_regimes


@dataclass
class Split:
    states: np.ndarray
    law: np.ndarray
    invariants: np.ndarray
    regime: str


def _generate(sim, n, regime, n_steps, dt, rng) -> Split:
    S, La, Iv = [], [], []
    for _ in range(n):
        tr = sim.rollout(rng, regime=regime, n_steps=n_steps, dt=dt)
        S.append(tr.states)
        La.append(tr.law)
        Iv.append(tr.invariants)
    return Split(np.stack(S), np.stack(La), np.stack(Iv), regime)


class PhysicsDataset:
    def __init__(self, domain, splits, mean, std, invariant_names, dt):
        self.domain = domain
        self.splits: dict[str, Split] = splits
        self.mean = mean            # (F,)
        self.std = std              # (F,)
        self.invariant_names = invariant_names
        self.dt = dt
        self.state_dim = mean.shape[0]
        self.law_dim = splits["train"].law.shape[-1]

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    # -- caching ------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = dict(domain=self.domain, mean=self.mean, std=self.std, dt=self.dt,
                    invariant_names=np.array(self.invariant_names, dtype=object))
        for name, sp in self.splits.items():
            blob[f"{name}__states"] = sp.states
            blob[f"{name}__law"] = sp.law
            blob[f"{name}__inv"] = sp.invariants
        np.savez_compressed(path, **blob)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        names = {k.split("__")[0] for k in z.files if "__" in k}
        splits = {n: Split(z[f"{n}__states"], z[f"{n}__law"], z[f"{n}__inv"], n) for n in names}
        return cls(str(z["domain"]), splits, z["mean"], z["std"],
                   list(z["invariant_names"]), float(z["dt"]))


def build_dataset(cfg) -> PhysicsDataset:
    """Build (and optionally cache) a dataset from a :class:`DataCfg`."""
    sim = make_simulator(cfg.domain)
    rng = np.random.default_rng(cfg.seed)
    splits: dict[str, Split] = {}
    splits["train"] = _generate(sim, cfg.n_train, "default", cfg.n_steps, cfg.dt, rng)
    splits["val"] = _generate(sim, cfg.n_val, "default", cfg.n_steps, cfg.dt, rng)
    splits["test"] = _generate(sim, cfg.n_test, "default", cfg.n_steps, cfg.dt, rng)

    ood = cfg.ood_regimes or default_ood_regimes(cfg.domain)
    for regime in ood:
        splits[f"test_{regime}"] = _generate(sim, cfg.n_test, regime, cfg.n_steps, cfg.dt, rng)

    # Restrict the *observation* (what the model sees/predicts) if requested.
    # law/invariants targets are untouched -- they remain the probe ground truth.
    obs = cfg.obs_dims
    if obs is not None:
        if not 0 < obs <= sim.state_dim:
            raise ValueError(f"obs_dims={obs} outside 1..{sim.state_dim} for {cfg.domain}")
        for sp in splits.values():
            sp.states = sp.states[..., :obs]
    state_dim = obs if obs is not None else sim.state_dim

    tr = splits["train"].states.reshape(-1, state_dim)
    mean = tr.mean(0) if cfg.normalize else np.zeros(state_dim)
    std = (tr.std(0) + 1e-6) if cfg.normalize else np.ones(state_dim)
    dt = cfg.dt or sim.default_dt
    return PhysicsDataset(cfg.domain, splits, mean.astype(np.float64), std.astype(np.float64),
                          sim.invariant_names, dt)
