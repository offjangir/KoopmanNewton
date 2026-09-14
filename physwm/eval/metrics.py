"""Evaluation metrics: rollout error and physics-consistency (invariant drift)."""

from __future__ import annotations

import numpy as np


def rmse_per_step(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Full-state RMSE per rollout step. pred/gt: (N, K, F) -> (K,)."""
    return np.sqrt(((pred - gt) ** 2).mean(axis=(0, 2)))


def position_mde(pred: np.ndarray, gt: np.ndarray, pos_dims: int = 2) -> np.ndarray:
    """Mean distance error over the first ``pos_dims`` coordinates. (N,K,F)->(K,)."""
    d = pred[..., :pos_dims] - gt[..., :pos_dims]
    return np.linalg.norm(d, axis=-1).mean(axis=0)


def invariant_drift(sim, states: np.ndarray, dt: float | None = None) -> dict:
    """Mean |invariant(t) - invariant(0)| along a model rollout, per invariant.

    ``states`` is (N, K, F) in *physical* (denormalized) units. Uses the
    simulator's parameter-free ``invariants_from_state``; returns {} if the
    domain does not expose one.

    If only positions are observed (``obs_dims``) but the simulator uses a
    [positions, velocities] layout, velocities are rebuilt by central finite
    difference so the physics-consistency metric survives the position-only
    protocol.
    """
    probe = sim.invariants_from_state(states[0, 0])
    if probe is None and dt and getattr(sim, "pos_vel_layout", False) \
            and 2 * states.shape[-1] == sim.state_dim:
        vel = np.gradient(states, dt, axis=1)
        states = np.concatenate([states, vel], axis=-1)
        probe = sim.invariants_from_state(states[0, 0])
    if probe is None:
        return {}
    keys = list(probe.keys())
    N, K, _ = states.shape
    out = {k: [] for k in keys}
    for n in range(N):
        vals = np.array([[sim.invariants_from_state(states[n, k])[key] for key in keys]
                         for k in range(K)])           # (K, n_inv)
        drift = np.abs(vals - vals[0]).mean(axis=0)     # (n_inv,)
        for j, key in enumerate(keys):
            out[key].append(drift[j])
    return {k: float(np.mean(v)) for k, v in out.items()}
