"""Relativity as trajectory-prediction tasks.

* :class:`RelativisticLorentzSimulator` -- a charged particle in static E/B fields
  pushed by the relativistic Boris integrator. State is ``(x, u)`` with proper
  velocity ``u = gamma v`` (avoids the ``v -> c`` stiffness). The "law" is the
  relativistic force ``du/dt = q(E + v x B)``; the SR signature is that
  ``gamma = sqrt(1+|u|^2)`` is conserved in a pure magnetic field.
* :class:`SchwarzschildGeodesicSimulator` -- equatorial timelike geodesics; the
  orbit precesses (the ``3 M u^2`` term), with conserved energy ``E`` and angular
  momentum ``L`` per unit mass. Same "is the corrected law discovered" flavor as
  the GR-Kepler task, but from the full metric.

Units: c = 1, m = 1, and G = M-scaled as noted per class.
"""

from __future__ import annotations


import numpy as np

from .base import Simulator, Trajectory, _RejectTrajectory


class RelativisticLorentzSimulator(Simulator):
    name = "rel_lorentz"
    law_state_dims = (3, 4, 5)   # law = du/dt acts on the proper-velocity components
    state_dim = 6          # (x,y,z, ux,uy,uz), u = gamma v
    law_dim = 3            # du/dt
    invariant_names = ["gamma", "speed"]
    default_dt = 0.02
    default_steps = 120
    regimes = ("default", "ultrarelativistic")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        gamma0 = rng.uniform(2.0, 5.0) if regime == "ultrarelativistic" else rng.uniform(1.05, 2.0)
        # random initial direction for velocity
        v_dir = rng.normal(size=3)
        v_dir /= np.linalg.norm(v_dir)
        v0 = np.sqrt(1 - 1 / gamma0 ** 2)
        u0 = gamma0 * v0 * v_dir
        B = np.array([0.0, 0.0, rng.uniform(0.5, 2.0)]) + rng.normal(0, 0.1, 3)
        E = rng.uniform(0.0, 0.5) * rng.normal(size=3)
        return dict(q=float(rng.choice([-1.0, 1.0])), B=B, E=E, u0=u0,
                    x0=rng.uniform(-1, 1, 3))

    @staticmethod
    def _gamma(u):
        return np.sqrt(1.0 + u @ u)

    def _force(self, u, q, E, B):
        v = u / self._gamma(u)
        return q * (E + np.cross(v, B))

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        q, E, B = params["q"], np.asarray(params["E"], float), np.asarray(params["B"], float)
        x = np.asarray(params["x0"], float).copy()
        u = np.asarray(params["u0"], float).copy()

        states = np.zeros((n, 6))
        law = np.zeros((n, 3))
        inv = np.zeros((n, 2))
        t = np.arange(n) * dt
        for k in range(n):
            g = self._gamma(u)
            states[k] = np.concatenate([x, u])
            law[k] = self._force(u, q, E, B)
            inv[k] = [g, np.linalg.norm(u) / g]
            # relativistic Boris pusher
            u_minus = u + q * E * (dt / 2)
            g_minus = np.sqrt(1 + u_minus @ u_minus)
            tvec = q * B * (dt / 2) / g_minus
            s = 2 * tvec / (1 + tvec @ tvec)
            u_prime = u_minus + np.cross(u_minus, tvec)
            u_plus = u_minus + np.cross(u_prime, s)
            u = u_plus + q * E * (dt / 2)
            x = x + dt * u / self._gamma(u)
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")

    def invariants_from_state(self, state):
        if np.shape(state)[0] < self.state_dim:
            return None
        u = state[3:6]
        g = self._gamma(u)
        return {"gamma": float(g), "speed": float(np.linalg.norm(u) / g)}


class SchwarzschildGeodesicSimulator(Simulator):
    """Equatorial timelike geodesic in Schwarzschild (G=c=1, M=1)."""

    pos_vel_layout = True
    name = "schwarzschild"
    state_dim = 4          # (x, y, dx/dtau, dy/dtau)
    law_dim = 2            # (d2x/dtau2, d2y/dtau2)
    invariant_names = ["E", "L"]
    default_dt = 0.5
    default_steps = 150
    regimes = ("default", "strong_field")

    M = 1.0

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        if regime == "strong_field":
            p = rng.uniform(8.0, 12.0)
        else:
            p = rng.uniform(15.0, 40.0)
        e = rng.uniform(0.0, 0.5)
        return dict(p=float(p), e=float(e), omega=float(rng.uniform(0, 2 * np.pi)))

    def _EL(self, p, e):
        M = self.M
        denom = p - 3 * M - M * e * e
        if denom <= 0:
            raise _RejectTrajectory()
        E = np.sqrt((p - 2 * M - 2 * M * e) * (p - 2 * M + 2 * M * e) / (p * denom))
        L = p * np.sqrt(M) / np.sqrt(denom)
        return E, L

    def _cart(self, r, phi, pr, L):
        """Return Cartesian state (x,y,vx,vy) and acceleration (ax,ay) in proper time."""
        M = self.M
        rddot = -M / r ** 2 + L ** 2 / r ** 3 - 3 * M * L ** 2 / r ** 4
        phidot = L / r ** 2
        phiddot = -2 * L * pr / r ** 3
        c, s = np.cos(phi), np.sin(phi)
        x = r * c
        y = r * s
        vx = pr * c - r * phidot * s
        vy = pr * s + r * phidot * c
        ax = rddot * c - 2 * pr * phidot * s - r * phiddot * s - r * phidot ** 2 * c
        ay = rddot * s + 2 * pr * phidot * c + r * phiddot * c - r * phidot ** 2 * s
        return np.array([x, y, vx, vy]), np.array([ax, ay])

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        p, e, omega = params["p"], params["e"], params["omega"]
        E, L = self._EL(p, e)
        r = p / (1 + e)        # start at periapsis
        phi = omega
        pr = 0.0
        M = self.M

        def rddot(rr):
            return -M / rr ** 2 + L ** 2 / rr ** 3 - 3 * M * L ** 2 / rr ** 4

        states = np.zeros((n, 4))
        law = np.zeros((n, 2))
        inv = np.zeros((n, 2))
        t = np.arange(n) * dt
        y = np.array([r, phi, pr])   # (r, phi, pr)
        for k in range(n):
            st, ac = self._cart(y[0], y[1], y[2], L)
            states[k] = st
            law[k] = ac
            inv[k] = [E, L]

            def deriv(yy):
                return np.array([yy[2], L / yy[0] ** 2, rddot(yy[0])])
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if y[0] < 3 * M or y[0] > 1e3:   # plunged past photon sphere / escaped
                raise _RejectTrajectory()
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")


class UniversalLorentzSimulator(RelativisticLorentzSimulator):
    """Relativistic Lorentz in a FIXED pure magnetic field (E=0, B=(0,0,1), q=1).
    gamma is then exactly conserved and the generator is a single universal law,
    so the force test is well posed. Only the initial proper velocity varies."""

    name = "rel_lorentz_u"

    def sample_params(self, rng, regime="default"):
        gamma0 = rng.uniform(1.05, 2.0)
        v_dir = rng.normal(size=3); v_dir /= np.linalg.norm(v_dir)
        v0 = np.sqrt(1 - 1 / gamma0 ** 2)
        u0 = gamma0 * v0 * v_dir
        return dict(q=1.0, B=np.array([0.0, 0.0, 1.0]), E=np.zeros(3), u0=u0,
                    x0=rng.uniform(-1, 1, 3))
