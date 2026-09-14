"""Direct extraction of the force law a model has *implicitly* learned.

Rationale
---------
Linear probing of hidden states is a weak and confounded test (random features
alone reach R^2 ~ 0.92). A far stronger test reads the model's **output**.

For a second-order ODE the Verlet identity is exact to O(dt^4):

    r_{t+1} = 2 r_t - r_{t-1} + dt^2 F(r_t)

so given a history ending at (r_{t-1}, r_t) and the model's prediction r_hat,
the model's *implied acceleration* is

    a_implied = (r_hat - 2 r_t + r_{t-1}) / dt^2

This needs no probe, no readout, and no control task -- it is what the model
actually does.

Crucially, at **context H = 2 any two points form a valid history**, so we can
place (r_{t-1}, r_t) anywhere in the plane -- including states no training
trajectory ever visited -- and map the model's force field over all space:

* a model that learned the *generator* (the law) reproduces -mu r/|r|^3 there;
* a model that learned the *solution manifold* (orbit fitting) cannot.

Two further discriminators fall out for free:

* **v-independence**: the true force depends on position only. Evaluate the same
  r with different velocities; a generator-learner's output must not move.
* **power law**: fit log|a| against log|r|; recovering slope ~ -2 is the model
  rediscovering the inverse-square law.

All comparisons are reported against an *extraction floor*: the same Verlet
extraction applied to the exact ground-truth next position, which quantifies the
O(dt^2) discretization error of the estimator itself.
"""

from __future__ import annotations

import numpy as np

from .rollout import predict_next

MU = 1.0


# --------------------------------------------------------------------------- #
# exact-ish two-body propagation (vectorized RK4 with substeps)
# --------------------------------------------------------------------------- #
def _accel(r):
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return -MU * r / rn ** 3


def propagate(r0, v0, dt, nsub=64):
    """Integrate the exact 2-body ODE for time ``dt`` (may be negative)."""
    r, v = np.array(r0, float), np.array(v0, float)
    h = dt / nsub
    for _ in range(nsub):
        k1r, k1v = v, _accel(r)
        k2r, k2v = v + 0.5 * h * k1v, _accel(r + 0.5 * h * k1r)
        k3r, k3v = v + 0.5 * h * k2v, _accel(r + 0.5 * h * k2r)
        k4r, k4v = v + h * k3v, _accel(r + h * k3r)
        r = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


def sample_states_from_orbits(n, a_lo, a_hi, e_lo, e_hi, seed=0):
    """Sample (r, v) that are genuine states of the Kepler orbit family.

    Sampling (r, v) freely puts most points on orbits the model never trained on
    (e.g. near-parabolic), so a poor prediction there measures distribution shift
    rather than whether the force law was learned. Drawing orbital elements in a
    chosen (a, e) window and then a random true anomaly keeps every state on a
    physically realizable orbit of a controlled family.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(a_lo, a_hi, n)
    e = rng.uniform(e_lo, e_hi, n)
    omega = rng.uniform(0, 2 * np.pi, n)
    nu = rng.uniform(0, 2 * np.pi, n)                      # true anomaly
    p = a * (1 - e ** 2)
    rad = p / (1 + e * np.cos(nu))
    # orbital-frame position / velocity
    rhat = np.stack([np.cos(nu), np.sin(nu)], 1)
    that = np.stack([-np.sin(nu), np.cos(nu)], 1)
    r_orb = rad[:, None] * rhat
    k = np.sqrt(MU / p)
    v_orb = (k * e * np.sin(nu))[:, None] * rhat + (k * (1 + e * np.cos(nu)))[:, None] * that
    c, s = np.cos(omega), np.sin(omega)
    rot = lambda z: np.stack([c * z[:, 0] - s * z[:, 1], s * z[:, 0] + c * z[:, 1]], 1)
    return rot(r_orb), rot(v_orb)


def sample_states(n, r_lo, r_hi, speed_lo=0.4, speed_hi=1.4, seed=0):
    """Sample (r, v): |r| log-uniform in [r_lo,r_hi]; |v| a multiple of circular speed."""
    rng = np.random.default_rng(seed)
    radius = np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), n))
    ang = rng.uniform(0, 2 * np.pi, n)
    r = np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1)
    v_circ = np.sqrt(MU / radius)
    speed = v_circ * rng.uniform(speed_lo, speed_hi, n)
    vang = ang + np.pi / 2 + rng.normal(0, 0.4, n)     # near-tangential, spread
    v = np.stack([speed * np.cos(vang), speed * np.sin(vang)], 1)
    return r, v


# --------------------------------------------------------------------------- #
# implied-acceleration extraction
# --------------------------------------------------------------------------- #
def _predict_next(method, mean, std, hist, device="cpu"):
    """Model's predicted next position in physical units. hist: (N,H,2) physical."""
    ctx = (hist - np.asarray(mean, float)) / np.asarray(std, float)
    return predict_next(method, ctx, mean, std, device)


def implied_acceleration(method, mean, std, hist, dt, device="cpu"):
    """hist: (N, H, 2) physical positions -> implied acceleration (N, 2).

    The model predicts the next position from the (normalized) history; the
    Verlet identity then converts that prediction into an acceleration.
    """
    pred = _predict_next(method, mean, std, hist, device)
    return (pred - 2 * hist[:, -1] + hist[:, -2]) / dt ** 2


def field_scores(a_est, r, a_true=None) -> dict:
    """Compare an estimated acceleration field against the true inverse-square law."""
    if a_true is None:
        a_true = _accel(r)
    ss_res = ((a_true - a_est) ** 2).sum()
    ss_tot = ((a_true - a_true.mean(0)) ** 2).sum() + 1e-12
    r2 = float(1 - ss_res / ss_tot)

    # direction: cosine to -r_hat (true force direction)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    fhat = -r / rn
    an = np.linalg.norm(a_est, axis=1, keepdims=True) + 1e-12
    cos = float(np.mean(np.sum(a_est / an * fhat, axis=1)))

    # power law: log|a| = c + p log|r|   ->  p should be -2
    mag = np.linalg.norm(a_est, axis=1)
    ok = (mag > 1e-9) & np.isfinite(mag)
    if ok.sum() > 10:
        A = np.stack([np.log(np.linalg.norm(r[ok], axis=1)), np.ones(ok.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, np.log(mag[ok]), rcond=None)
        expo = float(coef[0])
        pred = A @ coef
        resid = np.log(mag[ok]) - pred
        law_r2 = float(1 - (resid ** 2).sum() /
                       ((np.log(mag[ok]) - np.log(mag[ok]).mean()) ** 2).sum())
    else:
        expo, law_r2 = float("nan"), float("nan")
    return {"field_r2": r2, "direction_cos": cos,
            "power_law_exponent": expo, "power_law_r2": law_r2}


def behavioral_force_test(method, mean, std, dt, a_range, e_range, n=6000,
                          device="cpu", seed=0) -> dict:
    """Does the model's BEHAVIOUR use the force at all? (no dt^2 amplification)

    Decompose the Verlet identity into a part that is linear in the input and the
    part that carries the physics::

        r_{t+1} = (2 r_t - r_{t-1})  +  dt^2 F(r_t)
                   \\_ free, linear _/    \\_ the only thing to learn _/

    The linear term is obtainable with no knowledge of physics, so it is the
    correct null model. We report

    * ``linear_rmse``  -- error of the null model (ignores the force entirely)
    * ``model_rmse``   -- error of the model
    * ``ratio``        -- model/linear; >= 1 means the model is no better than
                          ignoring the force
    * ``force_term_r2``-- R^2 of the model's residual ``pred - linear`` against
                          the true ``dt^2 F``. This is the direct behavioural
                          measure of force-learning: 0 means none, 1 means exact.

    Unlike the implied-acceleration field, nothing here is divided by dt^2, so
    the measurement is not noise-amplified and a null result is meaningful.
    """
    r, v = sample_states_from_orbits(n, *a_range, *e_range, seed=seed)
    r_prev, _ = propagate(r, v, -dt)
    r_next, _ = propagate(r, v, +dt)
    hist = np.stack([r_prev, r], axis=1)

    pred = _predict_next(method, mean, std, hist, device)
    linear = 2 * r - r_prev
    force_term = r_next - linear                       # == dt^2 F + O(dt^4)

    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    model_res = pred - linear
    ss_res = ((model_res - force_term) ** 2).sum()
    ss_tot = ((force_term - force_term.mean(0)) ** 2).sum() + 1e-12
    return {
        "force_scale": float(np.linalg.norm(force_term, axis=1).mean()),
        "linear_rmse": rmse(linear, r_next),
        "model_rmse": rmse(pred, r_next),
        "ratio": rmse(pred, r_next) / (rmse(linear, r_next) + 1e-12),
        "force_term_r2": float(1 - ss_res / ss_tot),
    }


def radial_profile(a_est, r, n_bins=14):
    """Isotropy-averaged force law: |F|(|r|) from the radial component.

    The true central force depends on |r| only, so averaging the inward radial
    component over directions at fixed radius keeps the signal constant while
    beating independent prediction noise down as 1/sqrt(N_bin). This turns the
    noise-limited pointwise field test into a statistically powerful estimator
    of the *law* (the exponent), which is what we actually want to identify.

    Returns bin centres, mean inward acceleration per bin, its standard error,
    and the fitted power-law exponent with a standard error.
    """
    rad = np.linalg.norm(r, axis=1)
    rhat = r / rad[:, None]
    a_in = -np.sum(a_est * rhat, axis=1)          # inward (positive = attractive)
    edges = np.exp(np.linspace(np.log(rad.min()), np.log(rad.max()), n_bins + 1))
    centres, means, ses = [], [], []
    for i in range(n_bins):
        m = (rad >= edges[i]) & (rad < edges[i + 1])
        if m.sum() < 20:
            continue
        centres.append(np.sqrt(edges[i] * edges[i + 1]))
        means.append(a_in[m].mean())
        ses.append(a_in[m].std(ddof=1) / np.sqrt(m.sum()))
    centres, means, ses = map(np.asarray, (centres, means, ses))

    ok = means > 0                                 # log-fit needs attractive bins
    expo = se_expo = float("nan")
    if ok.sum() >= 4:
        X = np.stack([np.log(centres[ok]), np.ones(ok.sum())], 1)
        w = 1.0 / np.maximum(ses[ok] / np.maximum(means[ok], 1e-12), 1e-6)  # log-space weights
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, w * np.log(means[ok]), rcond=None)[0]
        expo = float(beta[0])
        resid = w * np.log(means[ok]) - (W @ X) @ beta
        dof = max(ok.sum() - 2, 1)
        cov = np.linalg.pinv((W @ X).T @ (W @ X)) * float((resid ** 2).sum() / dof)
        se_expo = float(np.sqrt(max(cov[0, 0], 0.0)))
    return {"r": centres, "a_in": means, "se": ses,
            "exponent": expo, "exponent_se": se_expo,
            "frac_attractive": float(np.mean(a_in > 0))}


def counterfactual_force_field(method, mean, std, dt, a_range, e_range, n=4000,
                               device="cpu", seed=0) -> dict:
    """H=2 probe: map the implied force field over states of a chosen orbit family.

    Builds a *physically exact* 2-point history by back-propagating the true
    dynamics one step, so the only thing under test is the model's prediction.
    Also returns the extraction floor (same operator on the true next position).
    ``a_range``/``e_range`` select the orbit family: the training window gives an
    in-distribution reading, a shifted window gives the counterfactual reading.
    """
    r, v = sample_states_from_orbits(n, *a_range, *e_range, seed=seed)
    r_prev, _ = propagate(r, v, -dt)
    r_next, _ = propagate(r, v, +dt)
    hist = np.stack([r_prev, r], axis=1)                    # (n,2,2)

    a_model = implied_acceleration(method, mean, std, hist, dt, device)
    a_floor = (r_next - 2 * r + r_prev) / dt ** 2           # ground-truth extraction
    out = {"model": field_scores(a_model, r), "floor": field_scores(a_floor, r)}
    out["model_profile"] = radial_profile(a_model, r)
    out["floor_profile"] = radial_profile(a_floor, r)

    # ---- measurement resolution -------------------------------------------
    # The Verlet operator divides by dt^2, so a 1-step prediction error eps
    # becomes an acceleration error eps/dt^2. This test can only certify
    # force-learning when that noise is well below the force scale:
    #        eps  <<  dt^2 * |F|
    # Report the SNR so a null result is never confused with a noise-limited one.
    pred = _predict_next(method, mean, std, hist, device)
    pred_err = float(np.sqrt(((pred - r_next) ** 2).sum(1).mean()))
    accel_noise = pred_err / dt ** 2
    force_scale = float(np.linalg.norm(_accel(r), axis=1).mean())
    out["resolution"] = {
        "pred_rmse": pred_err,
        "implied_accel_noise": accel_noise,
        "force_scale": force_scale,
        "snr": force_scale / (accel_noise + 1e-12),
        "pred_rmse_needed_for_snr10": dt ** 2 * force_scale / 10.0,
        "conclusive": bool(force_scale / (accel_noise + 1e-12) > 3.0),
    }

    # velocity-independence: same radii, different orbit shapes (=> different v)
    r2, v2 = sample_states_from_orbits(n, *a_range, e_range[0], e_range[0] + 0.05, seed=seed)
    r3, v3 = sample_states_from_orbits(n, *a_range, max(e_range[1] - 0.05, e_range[0]),
                                       e_range[1], seed=seed)
    h2 = np.stack([propagate(r2, v2, -dt)[0], r2], axis=1)
    h3 = np.stack([propagate(r3, v3, -dt)[0], r3], axis=1)
    a2 = implied_acceleration(method, mean, std, h2, dt, device)
    a3 = implied_acceleration(method, mean, std, h3, dt, device)
    denom = np.linalg.norm(a2, axis=1) + np.linalg.norm(a3, axis=1) + 1e-12
    out["v_sensitivity"] = float(np.mean(2 * np.linalg.norm(a2 - a3, axis=1) / denom))
    return out
