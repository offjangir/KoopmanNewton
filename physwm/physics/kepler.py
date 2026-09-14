"""Gravitation: Kepler two-body (exact) and a GR/post-Newtonian precessing variant.

Kepler baseline reproduces the original paper's data-generating law
``r'' = -GM r / |r|^3`` (GM = 1). We generate it *exactly* by solving Kepler's
equation ``M = E - e sin E`` so there is zero integration error to confound the
"does the model learn the law" question.

The GR variant adds the leading post-Newtonian perihelion term
``-3 GM L^2 / (c^2 r^4) r_hat``. This makes the *law itself* subtly different:
the Laplace-Runge-Lenz vector is no longer conserved but *precesses*, so a model
that has truly discovered the (corrected) force must track a rotating perihelion.
"""

from __future__ import annotations


import numpy as np

from .base import Simulator, Trajectory, _RejectTrajectory

MU = 1.0  # GM


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _lrl(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Laplace-Runge-Lenz (eccentricity) vector for mu=1, unit mass, in 2D."""
    speed2 = float(v @ v)
    rn = float(np.linalg.norm(r))
    rv = float(r @ v)
    return (speed2 - MU / rn) * r - rv * v


def _kepler_invariants(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    rn = float(np.linalg.norm(r))
    E = 0.5 * float(v @ v) - MU / rn          # specific energy
    L = float(r[0] * v[1] - r[1] * v[0])       # specific angular momentum (z)
    A = _lrl(r, v)
    peri = float(np.arctan2(A[1], A[0]))       # perihelion direction
    return np.array([E, L, A[0], A[1], peri])


INVARIANT_NAMES = ["energy", "Lz", "LRL_x", "LRL_y", "peri_angle"]


class KeplerSimulator(Simulator):
    """Exact Newtonian two-body orbit via Kepler's equation."""

    pos_vel_layout = True
    name = "kepler"
    state_dim = 4          # (x, y, vx, vy)
    law_dim = 2            # acceleration (ax, ay)
    invariant_names = INVARIANT_NAMES
    default_dt = 0.2
    default_steps = 100
    regimes = ("default", "ecc_extrap", "axis_extrap")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        if regime == "ecc_extrap":
            e = rng.uniform(0.6, 0.9)
            a = rng.uniform(0.5, 2.0)
        elif regime == "axis_extrap":
            e = rng.uniform(0.0, 0.6)
            a = rng.uniform(2.0, 3.0)
        else:
            e = rng.uniform(0.0, 0.6)
            a = rng.uniform(0.5, 2.0)
        return dict(
            a=float(a),
            e=float(e),
            omega=float(rng.uniform(0, 2 * np.pi)),   # argument of periapsis (orientation)
            M0=float(rng.uniform(0, 2 * np.pi)),       # initial mean anomaly (phase)
        )

    @staticmethod
    def _solve_E(M: np.ndarray, e: float, iters: int = 60) -> np.ndarray:
        """Newton solve of Kepler's equation M = E - e sin E (vectorized)."""
        E = M.copy()
        for _ in range(iters):
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            dE = f / fp
            E -= dE
            if np.max(np.abs(dE)) < 1e-13:
                break
        return E

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        a, e, omega, M0 = params["a"], params["e"], params["omega"], params["M0"]

        t = np.arange(n) * dt
        nmean = MU ** 0.5 * a ** (-1.5)          # mean motion
        M = M0 + nmean * t
        E = self._solve_E(M, e)
        b = a * np.sqrt(1 - e * e)
        Edot = nmean / (1 - e * np.cos(E))

        # orbital-plane coordinates (periapsis along +x)
        x_o = a * (np.cos(E) - e)
        y_o = b * np.sin(E)
        vx_o = -a * np.sin(E) * Edot
        vy_o = b * np.cos(E) * Edot

        R = _rot(omega)
        pos = R @ np.stack([x_o, y_o])           # (2, n)
        vel = R @ np.stack([vx_o, vy_o])
        states = np.concatenate([pos, vel], axis=0).T   # (n, 4)

        rn = np.linalg.norm(pos, axis=0)
        acc = -(MU / rn ** 3) * pos              # (2, n)
        law = acc.T                              # (n, 2)

        inv = np.array([_kepler_invariants(states[i, :2], states[i, 2:]) for i in range(n)])
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime=params.get("regime", "default"))

    def invariants_from_state(self, state):
        if np.shape(state)[0] < self.state_dim:
            return None      # observation truncated (e.g. position-only): velocities unavailable
        v = _kepler_invariants(state[:2], state[2:4])
        return dict(zip(self.invariant_names, v))


class GRKeplerSimulator(Simulator):
    """Post-Newtonian precessing two-body orbit (integrated).

    Law: ``a = -mu r/|r|^3 - 3 mu L^2/(c^2 |r|^4) r_hat`` with instantaneous L.
    The precession strength is controlled by ``c`` (smaller c => stronger GR).
    """

    pos_vel_layout = True
    name = "gr_kepler"
    state_dim = 4
    law_dim = 2
    invariant_names = INVARIANT_NAMES
    default_dt = 0.05
    default_steps = 200
    regimes = ("default", "strong_precession")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        a = rng.uniform(0.8, 2.0)
        e = rng.uniform(0.1, 0.6)
        # dimensionless precession knob eps = mu / (c^2 * a); larger eps -> more precession
        if regime == "strong_precession":
            eps = rng.uniform(0.03, 0.08)
        else:
            eps = rng.uniform(0.005, 0.03)
        c = float(np.sqrt(MU / (eps * a)))
        return dict(a=float(a), e=float(e), omega=float(rng.uniform(0, 2 * np.pi)), c=c, eps=float(eps))

    def _accel(self, r: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
        rn = float(np.linalg.norm(r))
        L = float(r[0] * v[1] - r[1] * v[0])
        newton = -(MU / rn ** 3) * r
        gr = -(3.0 * MU * L * L / (c * c * rn ** 5)) * r   # r_hat = r/rn folded in (rn^4 * rn)
        return newton + gr

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        from scipy.integrate import solve_ivp

        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        a, e, omega, c = params["a"], params["e"], params["omega"], params["c"]

        # start at periapsis with Newtonian periapsis speed, then let GR precess it
        r_p = a * (1 - e)
        v_p = np.sqrt(MU * (1 + e) / (a * (1 - e)))
        R = _rot(omega)
        r0 = R @ np.array([r_p, 0.0])
        v0 = R @ np.array([0.0, v_p])

        def rhs(_t, y):
            r = y[:2]
            v = y[2:]
            return np.concatenate([v, self._accel(r, v, c)])

        t = np.arange(n) * dt
        sol = solve_ivp(rhs, (0.0, t[-1]), np.concatenate([r0, v0]), t_eval=t,
                        method="DOP853", rtol=1e-10, atol=1e-10, max_step=dt)
        if not sol.success or sol.y.shape[1] != n:
            raise _RejectTrajectory()
        states = sol.y.T                          # (n, 4)
        rmin = np.linalg.norm(states[:, :2], axis=1).min()
        if rmin < 5.0 / (c * c):                  # too close to Schwarzschild radius ~ 2mu/c^2
            raise _RejectTrajectory()

        law = np.array([self._accel(states[i, :2], states[i, 2:], c) for i in range(n)])
        inv = np.array([_kepler_invariants(states[i, :2], states[i, 2:]) for i in range(n)])
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")
