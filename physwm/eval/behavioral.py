"""Per-domain behavioral tests (the §5.1 instruments).

Each law gets the test its structure demands:

* Kepler / pendulum (2nd-order nonlinear) -> Verlet force-term R²
  (``eval.force_field`` for Kepler; :func:`pendulum_force_test` here).
* Rabi / relativity (1st-order, invariant-defined physicist signature) ->
  conservation under rollout via ``eval.metrics.invariant_drift`` (|R|, gamma).

The pendulum test mirrors the Kepler one but in angle space: the Verlet identity
``theta_{t+1} = 2 theta_t - theta_{t-1} + dt^2 * (-sin theta_t)`` holds, and the
learnable residual is the restoring torque ``-sin theta`` -- so a curve-fitter
that reproduces only the small-angle linear ``-theta`` is exposed at large angle.
"""

from __future__ import annotations

import numpy as np
import torch

from ..physics import make_simulator
from .metrics import invariant_drift
from .rollout import autoregressive_rollout, predict_next


def _pendulum_prop(theta, thd, dt, omega0_sq=1.0, nsub=32):
    """Integrate theta''=-omega0^2 sin theta for time dt (dt may be negative)."""
    th, w = np.array(theta, float), np.array(thd, float)
    h = dt / nsub
    for _ in range(nsub):
        k1 = (w, -omega0_sq * np.sin(th))
        k2 = (w + 0.5 * h * k1[1], -omega0_sq * np.sin(th + 0.5 * h * k1[0]))
        k3 = (w + 0.5 * h * k2[1], -omega0_sq * np.sin(th + 0.5 * h * k2[0]))
        k4 = (w + h * k3[1], -omega0_sq * np.sin(th + h * k3[0]))
        th = th + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return th, w


@torch.no_grad()
def pendulum_force_test(method, mean, std, dt, n=6000, device="cpu", seed=0,
                        theta_range=(-2.0, 2.0), thd_range=(-1.2, 1.2),
                        omega0_sq=1.0) -> dict:
    """Behavioral force test for the nonlinear pendulum (position-only obs).

    Returns force_term_r2 (fraction of the true ``dt^2(-sin theta)`` reproduced),
    the linear-null and model rmse, their ratio, and the large-amplitude subset
    R² (where linear -theta and nonlinear -sin theta diverge most).
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(*theta_range, n)
    thd = rng.uniform(*thd_range, n)
    th_prev, _ = _pendulum_prop(theta, thd, -dt, omega0_sq)
    th_next, _ = _pendulum_prop(theta, thd, +dt, omega0_sq)

    def enc(a):
        return np.stack([np.sin(a), np.cos(a)], -1)
    hist = np.stack([enc(th_prev), enc(theta)], axis=1).astype(np.float64)   # (n,2,2)

    mean, std = np.asarray(mean, float), np.asarray(std, float)
    pred = predict_next(method, (hist - mean) / std, mean, std, device)   # predicted (sin,cos)

    # recover a continuous predicted angle near the current theta
    th_hat = theta + np.arctan2(
        np.sin(np.arctan2(pred[:, 0], pred[:, 1]) - theta),
        np.cos(np.arctan2(pred[:, 0], pred[:, 1]) - theta))
    linear = 2 * theta - th_prev
    force_term = th_next - linear                                # == dt^2(-sin theta)+O(dt^4)
    model_res = th_hat - linear

    r2 = lambda num, y: float(1 - (num ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
    big = np.abs(theta) > 1.2
    return {
        "force_scale": float(np.abs(force_term).mean()),
        "linear_rmse": rmse(linear, th_next),
        "model_rmse": rmse(th_hat, th_next),
        "ratio": rmse(th_hat, th_next) / (rmse(linear, th_next) + 1e-12),
        "force_term_r2": r2(model_res - force_term, force_term),
        "force_term_r2_largeangle": (r2((model_res - force_term)[big], force_term[big])
                                     if big.sum() > 20 else float("nan")),
    }


@torch.no_grad()
def rollout_invariant_test(method, dataset, domain, context, steps=60, n_traj=64,
                           device="cpu") -> dict:
    """Behavioral test for first-order / invariant-defined laws (Rabi, relativity).

    A "physicist" model keeps the conserved quantity constant along a long
    autoregressive rollout; a curve-fitter with small one-step error still lets it
    drift as errors compound. We report the model's invariant drift, the
    discretization floor (same drift on the ground-truth trajectory), and the
    rollout RMSE (so a trivially-conserving-but-wrong model is caught).
    """
    sim = make_simulator(domain)
    roll = autoregressive_rollout(method, dataset, "test", context, steps=steps,
                                  n_traj=n_traj, device=device)
    model_drift = invariant_drift(sim, roll["pred"], dt=dataset.dt)
    floor_drift = invariant_drift(sim, roll["gt"], dt=dataset.dt)
    rmse = float(np.sqrt(((roll["pred_norm"] - roll["gt_norm"]) ** 2).mean()))
    key = next(iter(model_drift), None)
    md = model_drift.get(key, float("nan")) if key else float("nan")
    fd = floor_drift.get(key, float("nan")) if key else float("nan")
    # conservation score in ~[0,1]: 1 = drifts no more than the floor
    score = float(1.0 - (md - fd) / (abs(md) + 0.05)) if key else float("nan")
    return {"metric": "invariant_conservation", "invariant": key,
            "model_drift": md, "floor_drift": fd, "rollout_rmse": rmse,
            "value": score, "model_drift_all": model_drift, "floor_drift_all": floor_drift}


def behavioral_score(method, dataset, domain, context, device="cpu", n=6000) -> dict:
    """Dispatch to the structurally-correct behavioral instrument for a domain.

    * kepler / gr_kepler -> Verlet force-term R² (off-orbit force field)
    * pendulum           -> Verlet force-term R² in angle space
    * else (rabi, rel_lorentz, ...) -> invariant conservation under rollout
    """
    if domain in ("kepler", "gr_kepler"):
        from .force_field import behavioral_force_test
        r = behavioral_force_test(method, dataset.mean, dataset.std, dataset.dt,
                                  (0.5, 2.0), (0.0, 0.6), n=n, device=device)
        return {"metric": "force_term_r2", "value": r["force_term_r2"],
                "ratio": r["ratio"], **r}
    if domain == "pendulum":
        r = pendulum_force_test(method, dataset.mean, dataset.std, dataset.dt,
                                n=n, device=device)
        return {"metric": "force_term_r2", "value": r["force_term_r2"],
                "ratio": r["ratio"], **r}
    if domain in ("rabi_u", "rel_lorentz_u"):
        return first_order_force_test(method, dataset, domain, context, device=device)
    return rollout_invariant_test(method, dataset, domain, context, device=device)


@torch.no_grad()
def force_term_r2_ontraj(method, dataset, context, device="cpu", n_traj=128,
                         mu=1.0, max_samples=6000, seed=0):
    """On-trajectory Verlet force-term R^2 for ANY context length (H4).

    The 2-point counterfactual test can't feed a context>2 model. Here we take
    real test-trajectory windows of length ``context`` ending at t, predict
    s_{t+1}, and Verlet-decompose using the last two observed positions:

        model_residual = pred - (2 r_t - r_{t-1})   vs   dt^2 * F(r_t)

    F is the analytic law (Kepler: -mu r/|r|^3). Reports force_term_r2, the
    linear-null ratio, and rollout-free one-step rmse. Position-only (obs_dims=2).
    """
    rng = np.random.default_rng(seed)
    states = dataset.splits["test"].states[:n_traj]          # (N,T,2) physical
    norm = dataset.normalize(states).astype(np.float32)
    dt = dataset.dt
    N, T, F = states.shape
    idx = [(n, t) for n in range(N) for t in range(max(context - 1, 1), T - 1)]
    if len(idx) > max_samples:
        sel = rng.choice(len(idx), max_samples, replace=False)
        idx = [idx[i] for i in sel]
    ctx = np.stack([norm[n, t - context + 1: t + 1] for n, t in idx]).astype(np.float32)
    r_t = np.stack([states[n, t] for n, t in idx])
    r_prev = np.stack([states[n, t - 1] for n, t in idx])
    r_next = np.stack([states[n, t + 1] for n, t in idx])
    pred = predict_next(method, ctx, dataset.mean, dataset.std, device)

    rn = np.linalg.norm(r_t, axis=1, keepdims=True)
    F_true = -mu * r_t / rn ** 3
    force_term = dt ** 2 * F_true
    linear = 2 * r_t - r_prev
    model_res = pred - linear
    ss = lambda a: (a ** 2).sum()
    r2 = float(1 - ss(model_res - force_term) / ss(force_term - force_term.mean(0)))
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    return {"force_term_r2": r2, "linear_rmse": rmse(linear, r_next),
            "model_rmse": rmse(pred, r_next),
            "ratio": rmse(pred, r_next) / (rmse(linear, r_next) + 1e-12)}


@torch.no_grad()
def first_order_force_test(method, dataset, domain, context, device="cpu",
                           n_traj=128, max_samples=6000, seed=0):
    """Sharper behavioral test for first-order UNIVERSAL laws (rabi_u, rel_lorentz_u).

    For s_{t+1} = s_t + dt*f(s_t) + O(dt^2), the physics is the increment dt*f on
    the law-carrying state components (``sim.law_state_dims``). We compare the
    model's predicted increment ``pred - s_t`` on those dims to the true ``dt*f``
    (the stored instantaneous law). Persistence (pred=s_t) scores ~0, so unlike
    invariant-conservation this separates a model that got the generator RATE
    right from one that merely stays on the sphere. No dt-amplification.
    """
    sim = make_simulator(domain)
    ld = list(sim.law_state_dims)
    rng = np.random.default_rng(seed)
    states = dataset.splits["test"].states[:n_traj]
    law = dataset.splits["test"].law[:n_traj]
    norm = dataset.normalize(states).astype(np.float32)
    dt = dataset.dt
    N, T, F = states.shape
    idx = [(n, t) for n in range(N) for t in range(context - 1, T - 1)]
    if len(idx) > max_samples:
        sel = rng.choice(len(idx), max_samples, replace=False)
        idx = [idx[i] for i in sel]
    ctx = np.stack([norm[n, t - context + 1: t + 1] for n, t in idx]).astype(np.float32)
    s_t = np.stack([states[n, t] for n, t in idx])
    s_next = np.stack([states[n, t + 1] for n, t in idx])
    law_t = np.stack([law[n, t] for n, t in idx])
    pred = predict_next(method, ctx, dataset.mean, dataset.std, device)

    incr = pred[:, ld] - s_t[:, ld]
    target = dt * law_t
    ss = lambda a: (a ** 2).sum()
    r2 = float(1 - ss(incr - target) / ss(target - target.mean(0)))
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    return {"metric": "force_term_r2", "value": r2, "force_term_r2": r2,
            "ratio": rmse(pred, s_next) / (rmse(s_t, s_next) + 1e-12)}
