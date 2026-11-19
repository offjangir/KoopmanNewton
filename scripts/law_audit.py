#!/usr/bin/env python
"""Law audit: measure each simulator against five properties of a test law.

CPU only. Generic over every domain in the registry. Two modes:

    python scripts/law_audit.py --out runs/law_audit --domains kepler pendulum
        -> runs/law_audit/<domain>/result.json and figure.png, one per domain
    python scripts/law_audit.py --out runs/law_audit --summarize
        -> summary.csv, summary.md, audit.json and page/index.html

Per domain and per split (train, in-distribution test, each OOD regime) it
generates ``--n`` trajectories at the simulator's default dt and steps, then
measures:

1. which ``sample_params`` keys vary per trajectory (law parameters vs start state);
2. invariant drift (max over time of |I(t) - I(0)| / |I(0)|, per trajectory);
3. the one-step change in train-normalized observed units (s_{t+1}-s_t and the
   constant-velocity residual s_{t+1}-(2 s_t - s_{t-1}));
4. law-free baselines (constant velocity, ridge autoregression, nearest
   neighbour) for one-step and 20-step rollout error, in- and out-of-distribution;
5. wall time to generate each split.

The observation and normalization follow the project runs: ``obs_of`` from
``scripts/run_program.py`` and the train mean / std of ``build_dataset``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from physwm.physics import ALL_DOMAINS, make_simulator  # noqa: E402
from physwm.physics.base import _RejectTrajectory  # noqa: E402
from run_program import obs_of  # noqa: E402  (single source of the project observation rule)

CTX_NOISE = 0.1          # context-noise level the project uses
ROLL_H = 20              # rollout horizon
KS = (2, 8)              # window lengths for the fitted baselines
RIDGE_ALPHA = 1e-4

# --------------------------------------------------------------------------- #
# Facts read from the simulator code (not measured). The audit says so.
# --------------------------------------------------------------------------- #
#: sample_params keys that set only the start state (position, velocity, phase,
#: orientation, orbit shape under a fixed law). Every other varying key changes the law.
START_STATE_KEYS = {
    "kepler": {"a", "e", "omega", "M0"},
    "gr_kepler": {"a", "e", "omega"},
    "harmonic": {"Ax", "Ay", "phi_x", "phi_y"},
    "nbody3": {"pos", "vel"},
    "nbody4": {"pos", "vel"},
    "pendulum": {"theta0", "thetadot0"},
    "driven_pendulum": {"theta0", "thetadot0"},
    "double_pendulum": {"th1", "th2", "w1", "w2"},
    "rabi": {"R0"},
    "rabi_u": {"R0"},
    "ehrenfest": {"x0", "p0", "sigma"},
    "wavepacket": {"x0", "p0", "sigma"},
    "rel_lorentz": {"u0", "x0"},
    "rel_lorentz_u": {"u0", "x0"},
    "schwarzschild": {"p", "e", "omega"},
}

#: invariants that the code does not hold constant by design, with the reason.
EXPECTED_DRIFT = {
    "gr_kepler": {"LRL_x": "precession", "LRL_y": "precession", "peri_angle": "precession",
                  "energy": "Newtonian energy, not the energy of the corrected law"},
    "driven_pendulum": {"energy": "drive and damping", "theta_dot": "not a conserved quantity"},
    "rel_lorentz": {"gamma": "electric field does work", "speed": "electric field does work"},
}

#: notes on how an invariant is computed, read from the code.
CODE_NOTES = {
    "schwarzschild": "E and L are stored as constants, so their drift is zero by construction.",
    "nbody3": "The simulator rejects runs with energy drift above 15 percent.",
    "nbody4": "The simulator rejects runs with energy drift above 15 percent.",
    "kepler": "peri_angle is not defined for a circular orbit (e near 0).",
    "rel_lorentz": "Speed is the norm of u divided by gamma, so it changes when gamma changes.",
    "gr_kepler": "Some LRL components start near zero, so their relative drift is large. audit.json also gives the absolute drift.",
    "double_pendulum": "Energy drift does not get smaller with a 10 times smaller dt. The evidence is in logs/dp_check.out.",
}

BASELINES = ["cv", "ar2", "ar8", "nn2", "nn8"]
BASELINE_LABEL = {"cv": "constant velocity", "ar2": "linear fit, last 2 states", "ar8": "linear fit, last 8 states",
                  "nn2": "nearest neighbour, last 2 states", "nn8": "nearest neighbour, last 8 states"}
# reference categorical palette (dataviz skill), fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# --------------------------------------------------------------------------- #
# Plain-word labels for the page and the figures (read from the simulator code).
# --------------------------------------------------------------------------- #
SYSTEM_NAME = {
    "kepler": "Kepler orbit", "gr_kepler": "Kepler orbit with relativity correction",
    "harmonic": "2D spring (harmonic oscillator)", "nbody3": "three bodies under gravity",
    "nbody4": "four bodies under gravity", "pendulum": "simple pendulum",
    "driven_pendulum": "driven, damped pendulum", "double_pendulum": "double pendulum",
    "rabi": "two-level quantum system (Rabi)", "rabi_u": "two-level quantum system (Rabi), fixed field",
    "ehrenfest": "quantum packet averages (Ehrenfest)", "wavepacket": "quantum wavefunction on a grid",
    "rel_lorentz": "relativistic charge in electric and magnetic fields",
    "rel_lorentz_u": "relativistic charge in a fixed magnetic field",
    "schwarzschild": "orbit near a black hole (Schwarzschild)",
}
LAW_PLAIN = {
    "gr_kepler": "precession strength", "nbody3": "masses", "nbody4": "masses",
    "driven_pendulum": "natural frequency, damping, drive strength and drive frequency",
    "double_pendulum": "masses and arm lengths", "rabi": "detuning and Rabi frequency",
    "ehrenfest": "trap frequency and quartic strength", "wavepacket": "potential type and strength",
    "rel_lorentz": "charge sign, electric field and magnetic field",
}
REGIME_NAME = {
    "ecc_extrap": "more stretched orbits", "axis_extrap": "larger orbits",
    "strong_precession": "stronger precession", "amp_extrap": "larger amplitudes",
    "large_amplitude": "larger swings", "chaotic": "stronger drive",
    "detuning_extrap": "larger detuning", "anharmonic": "added quartic potential",
    "ultrarelativistic": "faster particles", "strong_field": "orbits closer to the black hole",
}
INVARIANT_NAME = {
    "energy": "energy", "Lz": "angular momentum", "LRL_x": "Laplace-Runge-Lenz vector, x",
    "LRL_y": "Laplace-Runge-Lenz vector, y", "peri_angle": "perihelion angle",
    "S_xx": "Fradkin tensor, xx", "S_yy": "Fradkin tensor, yy", "S_xy": "Fradkin tensor, xy",
    "px": "total momentum, x", "py": "total momentum, y", "theta_dot": "angular velocity",
    "purity": "Bloch vector length", "norm": "wavefunction norm", "gamma": "Lorentz factor",
    "speed": "speed", "E": "energy E", "L": "angular momentum L",
}
_XYV = ["x position", "y position", "x velocity", "y velocity"]
FEATURE_NAMES = {
    "kepler": _XYV, "gr_kepler": _XYV, "harmonic": _XYV, "schwarzschild": _XYV,
    "pendulum": ["sin θ", "cos θ", "angular velocity"],
    "driven_pendulum": ["sin θ", "cos θ", "angular velocity", "sin of drive phase", "cos of drive phase"],
    "double_pendulum": ["sin θ1", "cos θ1", "sin θ2", "cos θ2", "angular velocity 1", "angular velocity 2"],
    "rabi": ["Bloch x", "Bloch y", "Bloch z"], "rabi_u": ["Bloch x", "Bloch y", "Bloch z"],
    "ehrenfest": ["mean position", "mean momentum", "position variance", "momentum variance"],
    "rel_lorentz": ["x position", "y position", "z position", "x proper velocity", "y proper velocity", "z proper velocity"],
    "rel_lorentz_u": ["x position", "y position", "z position", "x proper velocity", "y proper velocity", "z proper velocity"],
}
#: possible problems found by the audit, shown in the row of the domain
PROBLEMS = {
    "double_pendulum": "Possible problem. The energy formula may have a sign error in the coupling terms. "
                       "In a separate check with 50 trajectories, the median relative energy drift is 0.357 for dt 0.01 and for dt 0.001.",
    "rabi_u": "Possible problem. The test outside the training range uses the training distribution, "
              "because the simulator ignores the regime.",
    "rel_lorentz_u": "Possible problem. The test outside the training range uses the training distribution, "
                     "because the simulator ignores the regime.",
    "schwarzschild": "Possible problem. The simulator stores E and L as constants, so the drift check tests nothing.",
}


def feature_names(domain, F):
    names = FEATURE_NAMES.get(domain)
    if names and len(names) >= F:
        return names[:F]
    if domain.startswith("nbody"):
        nb = F // 4
        return ([f"body {i + 1} {a} position" for i in range(nb) for a in "xy"]
                + [f"body {i + 1} {a} velocity" for i in range(nb) for a in "xy"])
    if domain == "wavepacket":
        h = F // 2
        return [f"real part, grid point {i + 1}" for i in range(h)] + [f"imaginary part, grid point {i + 1}" for i in range(h)]
    return [f"state value {i + 1}" for i in range(F)]


def regime_name(r):
    return REGIME_NAME.get(r, r.replace("_", " "))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(sim, n, regime, rng, max_retries=50):
    """Mirror Simulator.rollout, but count the draws the simulator rejects."""
    trajs, tries = [], 0
    t0 = time.perf_counter()
    for _ in range(n):
        for _attempt in range(max_retries):
            tries += 1
            params = sim.sample_params(rng, regime=regime)
            try:
                tr = sim.simulate(params, rng=rng)
            except _RejectTrajectory:
                continue
            if np.all(np.isfinite(tr.states)) and np.all(np.isfinite(tr.law)):
                trajs.append(tr)
                break
        else:
            raise RuntimeError(f"{sim.name}/{regime}: no stable trajectory after {max_retries} tries")
    return trajs, time.perf_counter() - t0, tries


def varying_keys(trajs):
    """Keys of params whose value differs between trajectories."""
    keys = list(trajs[0].params.keys())
    out = []
    for k in keys:
        vals = [np.asarray(tr.params[k]) for tr in trajs]
        if vals[0].dtype.kind in "USO":
            differs = len({str(v) for v in vals}) > 1
        else:
            differs = any(not np.array_equal(vals[0], v) for v in vals[1:])
        if differs:
            out.append(k)
    return out


# --------------------------------------------------------------------------- #
# conservation
# --------------------------------------------------------------------------- #
def drift_curves(trajs, names):
    """Per invariant: (N, T) drift curve and the measure used.

    relative: |I(t)-I(0)| / |I(0)|; angle: wrapped |I(t)-I(0)| in radians;
    absolute: |I(t)-I(0)| when the invariant is about zero (for example momentum
    in the centre-of-mass frame), where a ratio has no meaning.
    """
    inv = np.stack([tr.invariants for tr in trajs])           # (N, T, I)
    out = {}
    for j, name in enumerate(names):
        x = inv[:, :, j]
        d = x - x[:, :1]
        if "angle" in name:
            curve = np.abs(np.angle(np.exp(1j * d)))
            kind = "angle (rad)"
        elif np.median(np.abs(x[:, 0])) < 1e-8:
            curve = np.abs(d)
            kind = "absolute"
        else:
            curve = np.abs(d) / np.maximum(np.abs(x[:, :1]), 1e-300)
            kind = "relative"
        out[name] = (curve, kind)
    return out


# --------------------------------------------------------------------------- #
# baselines (all on train-normalized observed states)
# --------------------------------------------------------------------------- #
def windows(S, k):
    """S (N,T,F) -> X (M, k*F) windows of k states, Y (M,F) next state, last (M,F)."""
    N, T, F = S.shape
    X = np.stack([S[:, i:T - k + i] for i in range(k)], axis=2)   # (N, T-k, k, F)
    Y = S[:, k:]
    return X.reshape(-1, k * F), Y.reshape(-1, F), X[:, :, -1].reshape(-1, F)


class Ridge:
    def __init__(self, k, alpha=RIDGE_ALPHA):
        self.k, self.alpha = k, alpha

    def fit(self, S):
        X, Y, _ = windows(S, self.k)
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        A = Xb.T @ Xb + self.alpha * len(X) * np.eye(Xb.shape[1])
        A[-1, -1] -= self.alpha * len(X)                          # no penalty on the bias
        self.W = np.linalg.solve(A, Xb.T @ Y)
        return self

    def predict(self, Xw):
        return np.concatenate([Xw, np.ones((len(Xw), 1))], 1) @ self.W


class NearestNeighbour:
    """Copy the training step that follows the closest training window."""

    def __init__(self, k):
        self.k = k

    def fit(self, S):
        X, Y, last = windows(S, self.k)
        self.X, self.step = X, Y - last
        self.X2 = np.einsum("ij,ij->i", X, X)
        return self

    def predict(self, Xw, chunk=2048):
        F = self.step.shape[1]
        out = np.empty((len(Xw), F))
        for i in range(0, len(Xw), chunk):
            q = Xw[i:i + chunk]
            d2 = self.X2[None, :] - 2.0 * q @ self.X.T
            j = np.argmin(d2, axis=1)
            out[i:i + chunk] = q[:, -F:] + self.step[j]
        return out


def predict_window(name, models, Xw, F):
    """One-step prediction from a flattened window whose last 8 states are given."""
    if name == "cv":
        return 2 * Xw[:, -F:] - Xw[:, -2 * F:-F]
    k = int(name[2:])
    return models[name].predict(Xw[:, -k * F:])


def one_step_preds(S, models):
    """Predictions for every target t >= 8 (so every baseline sees the same targets)."""
    X, Y, _ = windows(S, 8)
    F = S.shape[2]
    return {b: predict_window(b, models, X, F) for b in BASELINES}, Y


def rollout_err(S, models):
    """20-step rollout RMS from starts t0 = 7, 27, ... (window ends at t0)."""
    N, T, F = S.shape
    starts = list(range(7, T - ROLL_H, ROLL_H))
    out = {}
    for b in BASELINES:
        errs = []
        for t0 in starts:
            ctx = S[:, t0 - 7:t0 + 1].reshape(N, -1).copy()     # (N, 8F)
            preds = []
            for _h in range(ROLL_H):
                p = predict_window(b, models, ctx, F)
                preds.append(p)
                ctx = np.concatenate([ctx[:, F:], p], 1)
            P = np.stack(preds, 1)                              # (N, H, F)
            errs.append(P - S[:, t0 + 1:t0 + 1 + ROLL_H])
        E = np.concatenate(errs, 0)
        out[b] = float(np.sqrt(np.mean(E ** 2))) if np.all(np.isfinite(E)) else float("inf")
    return out, len(starts)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# --------------------------------------------------------------------------- #
# one domain
# --------------------------------------------------------------------------- #
def audit_domain(domain, n, seed, outdir, fig_name="figure.png"):
    sim = make_simulator(domain)
    rng = np.random.default_rng(seed)
    ood = [r for r in sim.regimes if r != "default"]
    split_regime = {"train": "default", "test": "default", **{f"ood_{r}": r for r in ood}}

    data, timing, tries = {}, {}, {}
    for split, regime in split_regime.items():
        trajs, sec, nt = generate(sim, n, regime, rng)
        data[split], timing[split], tries[split] = trajs, sec, nt

    # 1. law fixed or not
    vary = varying_keys(data["train"] + sum((data[s] for s in data if s.startswith("ood_")), []))
    start_keys = START_STATE_KEYS.get(domain)
    law_keys = None if start_keys is None else [k for k in vary if k not in start_keys]
    law_class = ("unknown (no code reading entry)" if law_keys is None
                 else "start state only" if not law_keys else "law parameters vary too")

    # 2. conservation (train and each split)
    names = list(sim.invariant_names)
    cons, curves_train = {}, None
    for split, trajs in data.items():
        dc = drift_curves(trajs, names)
        if split == "train":
            curves_train = dc
        inv_all = np.stack([tr.invariants for tr in trajs])
        absd = np.abs(inv_all - inv_all[:, :1]).max(1)             # (N, I) absolute max drift
        cons[split] = {nm: {"kind": kind, "median": float(np.median(c.max(1))),
                            "max": float(np.max(c.max(1))),
                            "abs_median": float(np.median(absd[:, j])), "abs_max": float(np.max(absd[:, j])),
                            "median_abs_initial_value": float(np.median(np.abs(inv_all[:, 0, j])))}
                       for j, (nm, (c, kind)) in enumerate(dc.items())}

    # 3. observation and normalization
    obs = obs_of(domain)
    obs_dims = None if obs == "none" else int(obs)
    F = obs_dims or sim.state_dim
    raw = {s: np.stack([tr.states for tr in t]) for s, t in data.items()}
    obsd = {s: v[..., :F] for s, v in raw.items()}
    tr_flat = obsd["train"].reshape(-1, F)
    mean, std = tr_flat.mean(0), tr_flat.std(0) + 1e-6
    S = {s: (v - mean) / std for s, v in obsd.items()}

    def visibility(Sx):
        d1 = Sx[:, 1:] - Sx[:, :-1]
        d2 = Sx[:, 2:] - 2 * Sx[:, 1:-1] + Sx[:, :-2]
        return {"rms_step": rms(d1), "rms_cv_residual": rms(d2),
                "per_feature_rms_cv_residual": np.sqrt((d2 ** 2).mean((0, 1))).tolist()}

    vis = {s: visibility(v) for s, v in S.items()}
    pos_only = None
    if sim.pos_vel_layout:
        P = sim.state_dim // 2
        pr = {s: v[..., :P] for s, v in raw.items()}
        pm, ps = pr["train"].reshape(-1, P).mean(0), pr["train"].reshape(-1, P).std(0) + 1e-6
        pos_only = {s: visibility((v - pm) / ps) for s, v in pr.items()}

    # 4. baselines
    t_fit = time.perf_counter()
    models = {}
    for k in KS:
        models[f"ar{k}"] = Ridge(k).fit(S["train"])
        models[f"nn{k}"] = NearestNeighbour(k).fit(S["train"])
    eval_splits = [s for s in S if s != "train"]
    onestep, rollout, n_starts, example = {}, {}, None, {}
    for s in eval_splits:
        preds, Y = one_step_preds(S[s], models)
        onestep[s] = {b: rms(p - Y) for b, p in preds.items()}
        rollout[s], n_starts = rollout_err(S[s], models)
        T = S[s].shape[1]
        example[s] = {"true": Y[: T - 8], **{b: p[: T - 8] for b, p in preds.items()}}  # trajectory 0
    sec_baselines = time.perf_counter() - t_fit

    res = dict(
        domain=domain, ok=True, n_per_split=n, seed=seed,
        dt=sim.default_dt, steps=sim.default_steps, state_dim=sim.state_dim,
        obs_dims=F, observation="position only" if obs_dims and obs_dims < sim.state_dim else "full state",
        pos_vel_layout=bool(sim.pos_vel_layout), ood_regimes=ood,
        law=dict(varying_keys=vary, start_state_keys=sorted(start_keys or []), law_keys=law_keys,
                 classification=law_class, source="varying keys measured; start-state vs law split read from code"),
        conservation=cons, expected_drift=EXPECTED_DRIFT.get(domain, {}), code_note=CODE_NOTES.get(domain, ""),
        visibility=vis, visibility_position_only=pos_only, ctx_noise=CTX_NOISE,
        onestep_rms=onestep, rollout20_rms=rollout, rollout_starts_per_traj=n_starts,
        gen_seconds=timing, gen_tries=tries, baseline_seconds=sec_baselines,
        norm_mean=mean.tolist(), norm_std=std.tolist(),
    )
    os.makedirs(outdir, exist_ok=True)
    make_figure(domain, sim, res, curves_train, S["train"][0], example, os.path.join(outdir, fig_name))
    return res


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(domain, sim, res, curves, s0, example, path):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "text.color": "#0b0b0b", "axes.titlesize": 9,
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    names = list(curves.keys())
    fig = plt.figure(figsize=(13, 16), facecolor="#fcfcfb")
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2.2], hspace=0.38, top=0.93, bottom=0.07,
                             left=0.07, right=0.97)
    t = np.arange(s0.shape[0]) * sim.default_dt
    kind_label = {"relative": "relative drift", "angle (rad)": "drift (radians)", "absolute": "absolute drift"}

    # (a) drift of each invariant, 5 training trajectories
    ga = outer[0].subgridspec(1, max(len(names), 3), wspace=0.5)
    for j, nm in enumerate(names):
        ax = fig.add_subplot(ga[0, j])
        c, kind = curves[nm]
        for i in range(min(5, len(c))):
            ax.semilogy(t, np.maximum(c[i], 1e-17), color=COLORS[i], lw=1.4)
        title = "(a) " + INVARIANT_NAME.get(nm, nm)
        exp = res["expected_drift"].get(nm)
        if exp:
            title += ", drifts by design (" + exp + ")"
        ax.set_title("\n".join(textwrap.wrap(title, 30)), fontsize=8.5)
        ax.set_xlabel("time")
        ax.set_ylabel(kind_label.get(kind, kind))

    # (b) one observed training trajectory, normalized
    ax = fig.add_subplot(outer[1])
    F = s0.shape[1]
    fn = feature_names(domain, F)
    if F > 8:
        vmax = np.abs(s0).max()
        im = ax.imshow(s0.T, aspect="auto", cmap="RdBu_r", extent=[t[0], t[-1], F - 0.5, -0.5], vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label="value (training standard deviations)", pad=0.01)
        ticks = list(range(F)) if F <= 16 else [0, F // 2 - 1, F - 1]
        if F > 16:
            ax.axhline(F // 2 - 0.5, color="#52514e", lw=0.8, ls=":")
        ax.set_yticks(ticks)
        ax.set_yticklabels([fn[i] for i in ticks], fontsize=7)
        ax.grid(False)
    else:
        for f in range(F):
            ax.plot(t, s0[:, f], color=COLORS[f % 8], lw=1.6, label=fn[f])
        ax.legend(fontsize=8, ncol=min(F, 6), loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
        ax.set_ylabel("value (training\nstandard deviations)")
    ax.set_xlabel("time")
    ax.set_title(f"(b) one training trajectory, observed values ({res['observation']})")

    # (c) true next state vs each law-free predictor, one trajectory within and one outside the training range
    changes = regime_effect(sim)
    ood_key = next((k for k in example if k.startswith("ood_")), None)
    gc = outer[2].subgridspec(2, 2, hspace=0.45, wspace=0.2)
    styles = ["-", "--", "-.", ":", (0, (5, 1))]
    handles = None
    for col, key in enumerate(["test", ood_key]):
        axv, axe = fig.add_subplot(gc[0, col]), fig.add_subplot(gc[1, col])
        if key is None:
            for a in (axv, axe):
                a.text(0.5, 0.5, "This simulator defines no test\noutside the training range.",
                       ha="center", va="center", transform=a.transAxes)
                a.set_axis_off()
            continue
        ex = example[key]
        tt = (np.arange(len(ex["true"])) + 8) * sim.default_dt
        axv.plot(tt, ex["true"][:, 0], color="#0b0b0b", lw=2.4, label="true next state")
        for b, color, ls in zip(BASELINES, COLORS, styles):
            axv.plot(tt, ex[b][:, 0], color=color, lw=1.3, ls=ls, label=BASELINE_LABEL[b])
            err = np.sqrt(np.mean((ex[b] - ex["true"]) ** 2, axis=1))
            axe.semilogy(tt, np.maximum(err, 1e-17), color=color, lw=1.3, ls=ls)
        if key == "test":
            where = "within the training range"
        else:
            reg = key[4:]
            where = f"outside the training range ({regime_name(reg)})"
            if not changes.get(reg, True):
                where += ", but with\nthe sampled parameters of the training range"
        axv.set_title(f"(c) one-step prediction of {fn[0]},\n{where}")
        axv.set_ylabel("value (training\nstandard deviations)")
        axe.set_title(f"(c) one-step error over all observed values,\n{where}")
        axe.set_ylabel("RMS error (training\nstandard deviations)")
        axe.set_xlabel("time")
        axe.axhline(CTX_NOISE, color="#8a8984", lw=1, ls="--")
        axe.text(tt[0], CTX_NOISE, " noise level 0.1", va="bottom", ha="left", fontsize=7.5, color="#52514e")
        handles = axv.get_legend_handles_labels()
    if handles:
        fig.legend(*handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{SYSTEM_NAME.get(domain, domain)} ({domain})", fontsize=12, x=0.02, ha="left")
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)


def regime_effect(sim, n=20, seed=0):
    """Measured: does each OOD regime draw different parameters than the default regime with the same seed?"""
    out = {}
    for reg in [r for r in sim.regimes if r != "default"]:
        a = [sim.sample_params(np.random.default_rng(seed + i), "default") for i in range(n)]
        b = [sim.sample_params(np.random.default_rng(seed + i), reg) for i in range(n)]
        out[reg] = not all(str(x) == str(y) for x, y in zip(a, b))
    return out


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def g(x):
    return "n/a" if x is None else ("inf" if not np.isfinite(x) else f"{x:.2g}")


def per_regime(d, splits, key):
    vals = [(regime_name(s[4:]), d[s][key]) for s in splits if s.startswith("ood_")]
    return " / ".join(f"{g(v)} for {r}" for r, v in vals) if vals else "no test defined"


WITHIN, OUTSIDE = "within the training range", "outside the training range"
COLUMNS = ["system", "code name", "status", "observed values", "one law or many", "law parameters that vary",
           f"tests {OUTSIDE}", "the test changes the sampled parameters",
           "invariant drift, training data (median / maximum)", "drift by design", "note",
           "change per step (RMS)", "force effect per step (constant-velocity residual, RMS)",
           "force effect per step, positions only", "noise level",
           *[f"one-step error, {BASELINE_LABEL[b]}, {WITHIN}" for b in BASELINES],
           *[f"one-step error, {BASELINE_LABEL[b]}, {OUTSIDE}" for b in BASELINES],
           *[f"20-step error, {BASELINE_LABEL[b]}, {WITHIN}" for b in BASELINES[1:]],
           *[f"20-step error, {BASELINE_LABEL[b]}, {OUTSIDE}" for b in BASELINES[1:]],
           "seconds to make 200 trajectories (training / test within / each test outside)", "rejected draws"]
KIND_SHORT = {"relative": "", "angle (rad)": " (radians)", "absolute": " (absolute)"}
DESIGN_SHORT = {"gr_kepler": "precession", "driven_pendulum": "drive and damping",
                "rel_lorentz": "the electric field does work"}
CONSERVATION_PROBLEM = {"double_pendulum", "schwarzschild"}


def row_of(r):
    d = r["domain"]
    if not r.get("ok"):
        return {c: "" for c in COLUMNS} | {"system": SYSTEM_NAME.get(d, d), "code name": d,
                                           "status": "error: " + r.get("error", "?")}
    tr = r["conservation"]["train"]
    drift = " / ".join(f"{INVARIANT_NAME.get(k, k)} {g(v['median'])} and {g(v['max'])}{KIND_SHORT.get(v['kind'], '')}"
                       for k, v in tr.items())
    v = r["visibility"]["train"]
    po = r["visibility_position_only"]
    splits = list(r["onestep_rms"].keys())
    row = {
        "system": SYSTEM_NAME.get(d, d), "code name": d, "status": "ran",
        "observed values": f"{r['observation']}, {r['obs_dims']} of {r['state_dim']}",
        "one law or many": "one law" if r["law"]["classification"] == "start state only" else "many laws",
        "law parameters that vary": LAW_PLAIN.get(d, "none"),
        f"tests {OUTSIDE}": " / ".join(regime_name(x) for x in r["ood_regimes"]) or "none",
        "the test changes the sampled parameters": " / ".join(
            f"{'yes' if c else 'no'} for {regime_name(k)}" for k, c in r.get("regime_changes_sampling", {}).items())
        or "no test defined",
        "invariant drift, training data (median / maximum)": drift,
        "drift by design": " / ".join(f"{INVARIANT_NAME.get(k, k)} ({w})" for k, w in r["expected_drift"].items()) or "none",
        "note": " ".join(x for x in (PROBLEMS.get(d, ""), r["code_note"]) if x),
        "change per step (RMS)": g(v["rms_step"]),
        "force effect per step (constant-velocity residual, RMS)": g(v["rms_cv_residual"]),
        "force effect per step, positions only": g(po["train"]["rms_cv_residual"]) if po else "n/a",
        "noise level": g(r["ctx_noise"]),
    }
    for b in BASELINES:
        row[f"one-step error, {BASELINE_LABEL[b]}, {WITHIN}"] = g(r["onestep_rms"]["test"][b])
        row[f"one-step error, {BASELINE_LABEL[b]}, {OUTSIDE}"] = per_regime(r["onestep_rms"], splits, b)
    for b in BASELINES[1:]:
        row[f"20-step error, {BASELINE_LABEL[b]}, {WITHIN}"] = g(r["rollout20_rms"]["test"][b])
        row[f"20-step error, {BASELINE_LABEL[b]}, {OUTSIDE}"] = per_regime(r["rollout20_rms"], splits, b)
    gs = r["gen_seconds"]
    row["seconds to make 200 trajectories (training / test within / each test outside)"] = (
        f"{gs['train']:.2f} / {gs['test']:.2f}" + "".join(f" / {gs[s]:.2f}" for s in gs if s.startswith("ood_")))
    rej = {s: r["gen_tries"][s] - r["n_per_split"] for s in r["gen_tries"]}
    row["rejected draws"] = " / ".join(
        f"{n} for {regime_name(s[4:]) if s.startswith('ood_') else s}" for s, n in rej.items() if n) or "0"
    return row


SHORT_COLUMNS = ["system", "one law or many", "conservation",
                 "force effect per step (noise level 0.1)",
                 f"best law-free one-step error, {WITHIN}",
                 f"best law-free one-step error, {OUTSIDE}",
                 f"best law-free 20-step error, {OUTSIDE}"]


def best(d, keys):
    b = min(keys, key=lambda k: d[k])
    return f"{g(d[b])} ({BASELINE_LABEL[b]})"


def short_row(r):
    """Seven plain columns. Every value is taken from result.json; nothing is recomputed."""
    d = r["domain"]
    name = f"{SYSTEM_NAME.get(d, d)} ({d})"
    if not r.get("ok"):
        return {c: "" for c in SHORT_COLUMNS} | {"system": name, "one law or many": "the audit failed"}
    law = ("one law, only the start state varies" if r["law"]["classification"] == "start state only"
           else f"many laws, with different {LAW_PLAIN.get(d, ', '.join(r['law']['law_keys']))}")
    tr = r["conservation"]["train"]
    if d in CONSERVATION_PROBLEM:
        k, v = max(tr.items(), key=lambda kv: kv[1]["median"])
        cons = (f"looks wrong (median {INVARIANT_NAME.get(k, k)} drift {g(v['median'])})" if v["median"] > 0
                else "looks wrong (E and L are stored constants)")
    elif r["expected_drift"]:
        kept = {k: v for k, v in tr.items() if k not in r["expected_drift"]}
        cons = f"drifts by design ({DESIGN_SHORT.get(d, 'see full table')})"
        if kept:
            k, v = max(kept.items(), key=lambda kv: kv[1]["median"])
            cons += f", {INVARIANT_NAME.get(k, k)} median drift {g(v['median'])}"
    else:
        k, v = max(tr.items(), key=lambda kv: kv[1]["median"])
        cons = f"good, largest median drift {g(v['median'])}{KIND_SHORT.get(v['kind'], '')} ({INVARIANT_NAME.get(k, k)})"
    vis = g(r["visibility"]["train"]["rms_cv_residual"])
    po = r["visibility_position_only"]
    if po and r["observation"] == "full state":
        vis += f" (positions only {g(po['train']['rms_cv_residual'])})"
    one, roll = r["onestep_rms"], r["rollout20_rms"]
    ood = [s for s in one if s.startswith("ood_")]
    changes = r.get("regime_changes_sampling", {})

    def outside(dd, keys):
        if not ood:
            return "no test defined"
        parts = []
        for s in ood:
            txt = f"{best(dd[s], keys)} for {regime_name(s[4:])}"
            if not changes.get(s[4:], True):
                txt += " (same distribution as training)"
            parts.append(txt)
        return " / ".join(parts)

    return {"system": name, "one law or many": law, "conservation": cons,
            "force effect per step (noise level 0.1)": vis,
            f"best law-free one-step error, {WITHIN}": best(one["test"], BASELINES),
            f"best law-free one-step error, {OUTSIDE}": outside(one, BASELINES),
            f"best law-free 20-step error, {OUTSIDE}": outside(roll, BASELINES[1:])}


def md_table(cols, rows):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for row in rows:
        out += "| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |\n"
    return out


def summarize(out, domains):
    results = []
    for d in domains:
        p = os.path.join(out, d, "result.json")
        if os.path.exists(p):
            with open(p) as fh:
                r = json.load(fh)
            if r.get("ok"):
                r["code_note"] = CODE_NOTES.get(d, "")
                r["regime_changes_sampling"] = regime_effect(make_simulator(d))
            results.append(r)
        else:
            results.append({"domain": d, "ok": False, "error": "no result file"})
    rows = [row_of(r) for r in results]
    srows = [short_row(r) for r in results]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("# Simulator law audit\n\n## Short table\n\n" + md_table(SHORT_COLUMNS, srows))
        fh.write("\nPossible problems:\n\n" + "".join(
            f"- {SYSTEM_NAME[d]} ({d}). {PROBLEMS[d][len('Possible problem. '):]}\n" for d in PROBLEMS))
        fh.write("\n## Full table\n\n" + md_table(COLUMNS, rows))
    with open(os.path.join(out, "audit.json"), "w") as fh:
        json.dump({r["domain"]: r for r in results}, fh, indent=1)
    write_page(out, results, rows, srows)
    print(f"wrote summary for {len(rows)} domains to {out}")


def write_page(out, results, rows, srows):
    """Write page/index.html with figures and the CSV copied next to it. The top text comes from top.txt."""
    import shutil
    pdir = os.path.join(out, "page")
    os.makedirs(pdir, exist_ok=True)
    shutil.copy(os.path.join(out, "summary.csv"), os.path.join(pdir, "summary.csv"))
    top_path = os.path.join(out, "top.txt")
    top = open(top_path).read().strip() if os.path.exists(top_path) else "Top text is not written yet."
    esc = html.escape

    head = "".join(f"<th>{esc(c)}</th>" for c in SHORT_COLUMNS)
    body = ""
    for r, sr in zip(results, srows):
        d = r["domain"]
        cells = []
        for c in SHORT_COLUMNS:
            val = esc(str(sr[c])).replace(" / ", "<br>")
            if c == "system":
                val = f"{esc(SYSTEM_NAME.get(d, d))}<br><code>{esc(d)}</code>"
            cells.append(f"<td>{val}</td>")
        cls = " class=\"has-problem\"" if d in PROBLEMS else ""
        body += f"<tr{cls}>{''.join(cells)}</tr>"
        if d in PROBLEMS:
            body += (f"<tr class=\"problem\"><td colspan=\"{len(SHORT_COLUMNS)}\"><span class=\"tag\">Possible problem</span> "
                     f"{esc(PROBLEMS[d][len('Possible problem. '):])}</td></tr>")
    short = f"<div class=\"scroll\"><table class=\"short\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    fhead = "".join(f"<th>{esc(c)}</th>" for c in COLUMNS)
    fbody = "".join("<tr>" + "".join(f"<td>{esc(str(r[c]))}</td>" for c in COLUMNS) + "</tr>" for r in rows)
    full = (f"<details><summary>Full table, {len(COLUMNS)} columns</summary><div class=\"scroll\"><table>"
            f"<thead><tr>{fhead}</tr></thead><tbody>{fbody}</tbody></table></div></details>"
            f"<p class=\"note\"><a href=\"summary.csv\">Open the full table as a CSV file</a></p>")

    figs = []
    for r in results:
        d = r["domain"]
        src = os.path.join(out, d, "figure.png")
        name = SYSTEM_NAME.get(d, d)
        if r.get("ok") and os.path.exists(src):
            shutil.copy(src, os.path.join(pdir, f"{d}.png"))
            figs.append(f"<figure><img src=\"{d}.png\" alt=\"Audit figure for {esc(name)}\" loading=\"lazy\">"
                        f"<figcaption>{esc(caption(r))}</figcaption></figure>")
        else:
            figs.append(f"<figure><figcaption>{esc(name)}. The audit failed. {esc(r.get('error', ''))}</figcaption></figure>")
    paras = "".join(f"<p>{esc(p)}</p>" for p in top.split("\n") if p.strip())
    page = PAGE_TEMPLATE.format(top=paras, short=short, full=full, figs="".join(figs))
    with open(os.path.join(pdir, "index.html"), "w") as fh:
        fh.write(page)


def caption(r):
    d = r["domain"]
    return (f"{SYSTEM_NAME.get(d, d)}, with invariant drift in (a), one training trajectory in (b) "
            f"and law-free predictions in (c).")


# The publish step wraps the page in its own document skeleton, so the page starts with <title> and <style>.
PAGE_TEMPLATE = """<title>Simulator Law Audit</title>
<style>
/* Layout: one reading column; the short table first; wide tables scroll inside their own box; figures full width. */
:root {{
  --bg: #fbfbf9; --ink: #14161a; --ink-2: #4f5560; --line: #d9dcd6; --head: #e7e9e4;
  --flag-bg: #fbf1df; --flag-ink: #6b4300;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #17191c; --ink: #eceee9; --ink-2: #b9bec6; --line: #353a40; --head: #262a2f;
    --flag-bg: #3a2e1a; --flag-ink: #f2d29b;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #17191c; --ink: #eceee9; --ink-2: #b9bec6; --line: #353a40; --head: #262a2f;
  --flag-bg: #3a2e1a; --flag-ink: #f2d29b;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 var(--font-body); }}
main {{ max-width: 1200px; margin: 0 auto; padding-block: 24px 48px; padding-inline: 16px; display: grid; gap: 10px; }}
main > * {{ min-width: 0; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; text-wrap: balance; }}
h2 {{ font-size: 1.1rem; margin: 22px 0 0; text-wrap: balance; }}
.top p, p.note {{ margin: 0; max-width: 72ch; }}
p.note {{ color: var(--ink-2); font-size: 13.5px; }}
a {{ color: inherit; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 2px; }}
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; }}
table {{ border-collapse: collapse; font-size: 12px; line-height: 1.45; font-variant-numeric: tabular-nums; min-width: 100%; }}
table.short {{ font-size: 13px; }}
table.short td {{ min-width: 9rem; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: var(--head); color: var(--ink-2); font-weight: 600; }}
td:first-child {{ font-weight: 600; }}
code {{ font: 12px var(--font-data); color: var(--ink-2); font-weight: 400; }}
tr.has-problem td {{ border-bottom: 0; }}
tr.problem td {{ background: var(--flag-bg); color: var(--flag-ink); font-weight: 400; }}
.tag {{ font-weight: 600; margin-right: 6px; }}
details {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }}
details > summary {{ cursor: pointer; font-weight: 600; }}
details .scroll {{ margin-top: 8px; }}
figure {{ margin: 14px 0 0; display: grid; gap: 6px; }}
figure img {{ width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; background: #fcfcfb; }}
figcaption {{ color: var(--ink-2); font-size: 13.5px; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<h1>Simulator law audit</h1>
<div class="top">{top}</div>
<h2>Summary</h2>
<p class="note">The force effect per step is the gap between the next state and a straight-line guess from the last two states.
All values use the standard deviation of the training data as the unit.
A law-free error is the best of five simple predictors that use no physics.</p>
{short}
{full}
<h2>Figures</h2>
{figs}
</main>
"""


# --------------------------------------------------------------------------- #
def compare_numbers(new, old, path="", tol=1e-9, skip=("gen_seconds", "baseline_seconds", "total_seconds", "code_note")):
    """Largest difference between two result dicts, ignoring wall times. Returns (max_rel_diff, worst_path)."""
    worst = (0.0, "")
    if isinstance(old, dict):
        for k in old:
            if k in skip or k not in new:
                continue
            worst = max(worst, compare_numbers(new[k], old[k], f"{path}/{k}", tol, skip))
    elif isinstance(old, list):
        for i, (a, b) in enumerate(zip(new, old)):
            worst = max(worst, compare_numbers(a, b, f"{path}[{i}]", tol, skip))
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        a, b = float(new), float(old)
        if np.isfinite(a) or np.isfinite(b):
            diff = abs(a - b) / max(abs(b), 1e-300) if b != 0 else abs(a)
            worst = max(worst, (diff, path))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="trajectories per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", action="store_true", help="collect result.json files into tables and the page")
    ap.add_argument("--figures-only", action="store_true",
                    help="make the same data again, draw figure.png, and compare all numbers with result.json "
                         "(result.json is not written)")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.out, args.domains)
        return
    for d in args.domains:
        ddir = os.path.join(args.out, d)
        os.makedirs(ddir, exist_ok=True)
        t0 = time.perf_counter()
        if args.figures_only:
            with open(os.path.join(ddir, "result.json")) as fh:
                old = json.load(fh)
            fig = os.path.join(ddir, "figure.png")
            if os.path.exists(fig) and not os.path.exists(os.path.join(ddir, "figure_first_layout.png")):
                os.rename(fig, os.path.join(ddir, "figure_first_layout.png"))
            new = audit_domain(d, old["n_per_split"], old["seed"], ddir)
            diff, where = compare_numbers(json.loads(json.dumps(new)), old)
            print(f"{d}: figure redrawn, largest relative difference to result.json {diff:.3g} at {where or 'none'}"
                  f" {time.perf_counter() - t0:.1f}s", flush=True)
            continue
        try:
            res = audit_domain(d, args.n, args.seed, ddir)
        except Exception as ex:  # a failed domain still gets a row
            res = {"domain": d, "ok": False, "error": f"{type(ex).__name__}: {ex}",
                   "traceback": traceback.format_exc()}
            print(res["traceback"], file=sys.stderr)
        res["total_seconds"] = time.perf_counter() - t0
        with open(os.path.join(ddir, "result.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"{d}: ok={res['ok']} {res['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
