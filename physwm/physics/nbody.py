"""Newtonian gravitation: N-body (default three-body) in 2D.

Softened gravity ``a_i = G sum_j m_j (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^{3/2}``
integrated with a symplectic kick-drift-kick (velocity Verlet) leapfrog in the
center-of-mass frame. Softening keeps the law clean and learnable while avoiding
the ``1/r^2`` singularity at close encounters. Three-body is chaotic, so this is
primarily a short-horizon / distributional stress test.

Probe targets: per-body accelerations (the "law"); conserved total energy,
linear momentum (=0 in COM frame), and angular momentum (the "invariants").
"""

from __future__ import annotations


import numpy as np

from .base import Simulator, Trajectory, _RejectTrajectory

G = 1.0


class NBodySimulator(Simulator):
    def __init__(self, num_bodies: int = 3, softening: float = 0.1):
        self.num_bodies = int(num_bodies)
        self.softening = float(softening)
        self.name = f"nbody{self.num_bodies}"
        self.state_dim = 4 * self.num_bodies
        self.law_dim = 2 * self.num_bodies
        self.invariant_names = ["energy", "Lz", "px", "py"]

    default_dt = 0.005
    default_steps = 100
    regimes = ("default",)

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        N = self.num_bodies
        masses = rng.uniform(0.5, 2.0, size=N)
        pos = rng.uniform(-1.0, 1.0, size=(N, 2))
        vel = rng.uniform(-0.3, 0.3, size=(N, 2))
        return dict(masses=masses, pos=pos, vel=vel)

    def _accel(self, pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
        N = pos.shape[0]
        acc = np.zeros_like(pos)
        eps2 = self.softening ** 2
        for i in range(N):
            d = pos - pos[i]                       # (N,2)
            r2 = np.sum(d * d, axis=1) + eps2
            inv = np.where(np.arange(N) == i, 0.0, r2 ** -1.5)
            acc[i] = G * np.sum((masses * inv)[:, None] * d, axis=0)
        return acc

    def _energy(self, pos, vel, masses):
        ke = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))
        pe = 0.0
        N = pos.shape[0]
        eps2 = self.softening ** 2
        for i in range(N):
            for j in range(i + 1, N):
                r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2) + eps2)
                pe -= G * masses[i] * masses[j] / r
        return ke + pe

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        masses = np.asarray(params["masses"], float)
        pos = np.asarray(params["pos"], float).copy()
        vel = np.asarray(params["vel"], float).copy()
        N = pos.shape[0]

        # center-of-mass frame
        M = masses.sum()
        pos -= (masses[:, None] * pos).sum(0) / M
        vel -= (masses[:, None] * vel).sum(0) / M

        states = np.zeros((n, 4 * N))
        law = np.zeros((n, 2 * N))
        inv = np.zeros((n, 4))

        acc = self._accel(pos, masses)
        for k in range(n):
            states[k] = np.concatenate([pos.ravel(), vel.ravel()])
            law[k] = acc.ravel()
            E = self._energy(pos, vel, masses)
            Lz = np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
            p = (masses[:, None] * vel).sum(0)
            inv[k] = [E, Lz, p[0], p[1]]

            # velocity Verlet
            vel = vel + 0.5 * dt * acc
            pos = pos + dt * vel
            acc = self._accel(pos, masses)
            vel = vel + 0.5 * dt * acc

            if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 1e3:
                raise _RejectTrajectory()

        t = np.arange(n) * dt
        # reject runs whose energy drifts too much (numerically unreliable / hard close encounter)
        e0 = inv[0, 0]
        if np.max(np.abs(inv[:, 0] - e0)) > 0.15 * (abs(e0) + 1e-6):
            raise _RejectTrajectory()
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")
