"""Gravitation's closed-orbit sibling: the 2D isotropic harmonic oscillator.

``r'' = -omega^2 r`` with ``omega = 1`` (a single universal law; only the initial
conditions vary). By Bertrand's theorem this is the *only* central force other
than the Kepler inverse-square whose bound orbits close, so it is the natural
"other law" for testing whether the Newton->Kepler representation transition is
specific to gravity or general to closed-orbit dynamics.

The physics differs from Kepler on every axis that matters:

* the force is *linear* in position (``-r``), not ``-r/|r|^3``;
* orbits are ellipses *centred on the origin*, not with the origin at a focus;
* the conserved "orbital elements" are the **Demkov-Fradkin tensor**
  ``S_ij = 1/2 (v_i v_j + omega^2 r_i r_j)`` -- the harmonic analogue of the
  Laplace-Runge-Lenz vector -- which fixes the ellipse's axes and orientation.

Exact solution (no integration error, mirroring the Kepler generator): each
Cartesian coordinate is simple harmonic, ``x = A_x cos(t + phi_x)``.
"""

from __future__ import annotations

import numpy as np

from .base import Simulator, Trajectory

OMEGA2 = 1.0  # spring constant / mass, fixed -> single universal law


def _harmonic_invariants(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    E = 0.5 * float(v @ v) + 0.5 * OMEGA2 * float(r @ r)     # energy
    Lz = float(r[0] * v[1] - r[1] * v[0])                    # angular momentum
    Sxx = 0.5 * (v[0] * v[0] + OMEGA2 * r[0] * r[0])         # Fradkin tensor
    Syy = 0.5 * (v[1] * v[1] + OMEGA2 * r[1] * r[1])
    Sxy = 0.5 * (v[0] * v[1] + OMEGA2 * r[0] * r[1])
    return np.array([E, Lz, Sxx, Syy, Sxy])


INVARIANT_NAMES = ["energy", "Lz", "S_xx", "S_yy", "S_xy"]


class HarmonicSimulator(Simulator):
    """Exact 2D isotropic harmonic oscillator, ``r'' = -r``."""

    pos_vel_layout = True
    name = "harmonic"
    state_dim = 4          # (x, y, vx, vy)
    law_dim = 2            # acceleration (ax, ay) = -omega^2 (x, y)
    invariant_names = INVARIANT_NAMES
    default_dt = 0.2
    default_steps = 100
    regimes = ("default", "amp_extrap")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        if regime == "amp_extrap":
            ax, ay = rng.uniform(2.0, 3.0), rng.uniform(2.0, 3.0)
        else:
            ax, ay = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)
        return dict(
            Ax=float(ax),
            Ay=float(ay),
            phi_x=float(rng.uniform(0, 2 * np.pi)),
            phi_y=float(rng.uniform(0, 2 * np.pi)),
        )

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        w = OMEGA2 ** 0.5
        Ax, Ay, px, py = params["Ax"], params["Ay"], params["phi_x"], params["phi_y"]

        t = np.arange(n) * dt
        x = Ax * np.cos(w * t + px)
        y = Ay * np.cos(w * t + py)
        vx = -Ax * w * np.sin(w * t + px)
        vy = -Ay * w * np.sin(w * t + py)
        states = np.stack([x, y, vx, vy], axis=1)            # (n, 4)

        law = np.stack([-OMEGA2 * x, -OMEGA2 * y], axis=1)   # (n, 2)
        inv = np.array([_harmonic_invariants(states[i, :2], states[i, 2:]) for i in range(n)])
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime=params.get("regime", "default"))
