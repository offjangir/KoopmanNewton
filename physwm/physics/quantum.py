"""Quantum mechanics as trajectory-prediction tasks (three difficulty tiers).

* :class:`RabiSimulator` -- two-level system; the Bloch vector precesses about a
  fixed axis ``Omega_eff = (Omega, 0, Delta)``: ``dR/dt = Omega_eff x R``. This is
  the cleanest analog of the Kepler task (a rotating unit vector). Exact.
* :class:`EhrenfestSimulator` -- true wavefunction evolved by split-step Fourier
  in an (an)harmonic potential; the low-dim state is the moment vector
  ``[<x>, <p>, var_x, var_p]`` with law ``[d<x>/dt, d<p>/dt] = [<p>, -<V'(x)>]``.
  For anharmonic V the quantum force ``<V'>`` differs from the classical ``V'(<x>)``.
* :class:`WavepacketSimulator` -- full-field PDE-emulation: state is ``[Re psi, Im psi]``
  on a small grid, law is ``d psi/dt = -i H psi``.

Invariants everywhere: norm (unitarity) and energy ``<H>``.
"""

from __future__ import annotations


import numpy as np

from .base import Simulator, Trajectory


# --------------------------------------------------------------------------- #
# Two-level Rabi system
# --------------------------------------------------------------------------- #
class RabiSimulator(Simulator):
    name = "rabi"
    law_state_dims = (0, 1, 2)   # law = dR/dt acts on the Bloch components
    state_dim = 3          # Bloch vector (Rx, Ry, Rz)
    law_dim = 3            # dR/dt
    invariant_names = ["purity", "energy"]
    default_dt = 0.05
    default_steps = 120
    regimes = ("default", "detuning_extrap")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        if regime == "detuning_extrap":
            Delta = rng.uniform(2.0, 4.0) * rng.choice([-1.0, 1.0])
        else:
            Delta = rng.uniform(-2.0, 2.0)
        theta = np.arccos(rng.uniform(-1, 1))
        phi = rng.uniform(0, 2 * np.pi)
        R0 = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        return dict(Delta=float(Delta), Omega=float(rng.uniform(0.5, 2.0)), R0=R0)

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        b = np.array([params["Omega"], 0.0, params["Delta"]])   # effective field
        bn = np.linalg.norm(b)
        khat = b / bn
        R = np.asarray(params["R0"], float)

        states = np.zeros((n, 3))
        law = np.zeros((n, 3))
        inv = np.zeros((n, 2))
        t = np.arange(n) * dt
        for k in range(n):
            states[k] = R
            law[k] = np.cross(b, R)
            inv[k] = [np.linalg.norm(R), 0.5 * (params["Omega"] * R[0] + params["Delta"] * R[2])]
            # exact rotation about khat by angle bn*dt (Rodrigues)
            ang = bn * dt
            R = (R * np.cos(ang)
                 + np.cross(khat, R) * np.sin(ang)
                 + khat * (khat @ R) * (1 - np.cos(ang)))
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")

    def invariants_from_state(self, state):
        # only purity |R| is parameter-free (energy needs Omega, Delta)
        if np.shape(state)[0] < self.state_dim:
            return None
        return {"purity": float(np.linalg.norm(state))}


# --------------------------------------------------------------------------- #
# Split-step Fourier propagator (shared by Ehrenfest & Wavepacket)
# --------------------------------------------------------------------------- #
def _splitstep_run(psi0, x, V, dt, n):
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)
    expV = np.exp(-1j * V * dt / 2)
    expK = np.exp(-1j * (k ** 2) / 2 * dt)
    psi = psi0.astype(np.complex128).copy()
    frames = np.empty((n, x.size), dtype=np.complex128)
    for i in range(n):
        frames[i] = psi
        psi = expV * psi
        psi = np.fft.ifft(expK * np.fft.fft(psi))
        psi = expV * psi
    return frames, k


def _gaussian_packet(x, x0, p0, sigma):
    psi = np.exp(-(x - x0) ** 2 / (4 * sigma ** 2) + 1j * p0 * x)
    dx = x[1] - x[0]
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi


class EhrenfestSimulator(Simulator):
    name = "ehrenfest"
    state_dim = 4          # [<x>, <p>, var_x, var_p]
    law_dim = 2            # [d<x>/dt, d<p>/dt]
    invariant_names = ["norm", "energy"]
    default_dt = 0.02
    default_steps = 150
    regimes = ("default", "anharmonic")

    def __init__(self, grid=256, L=20.0):
        self.grid = grid
        self.L = L

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        lam = rng.uniform(0.02, 0.1) if regime == "anharmonic" else 0.0
        return dict(
            omega=float(rng.uniform(0.3, 1.2)),
            lam=float(lam),
            x0=float(rng.uniform(-4, 4)),
            p0=float(rng.uniform(-2, 2)),
            sigma=float(rng.uniform(0.6, 1.5)),
        )

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        x = np.linspace(-self.L, self.L, self.grid, endpoint=False)
        dx = x[1] - x[0]
        w, lam = params["omega"], params["lam"]
        V = 0.5 * w ** 2 * x ** 2 + lam * x ** 4
        Vp = w ** 2 * x + 4 * lam * x ** 3
        psi0 = _gaussian_packet(x, params["x0"], params["p0"], params["sigma"])
        frames, k = _splitstep_run(psi0, x, V, dt, n)

        states = np.zeros((n, 4))
        law = np.zeros((n, 2))
        inv = np.zeros((n, 2))
        t = np.arange(n) * dt
        for i in range(n):
            psi = frames[i]
            dens = np.abs(psi) ** 2
            norm = np.sum(dens) * dx
            xm = np.sum(x * dens) * dx / norm
            x2 = np.sum(x ** 2 * dens) * dx / norm
            psik = np.fft.fft(psi) * dx
            densk = np.abs(psik) ** 2
            nk = np.sum(densk)
            pm = np.sum(k * densk) / nk
            p2 = np.sum(k ** 2 * densk) / nk
            Vmean = np.sum(V * dens) * dx / norm
            Vp_mean = np.sum(Vp * dens) * dx / norm
            states[i] = [xm, pm, x2 - xm ** 2, p2 - pm ** 2]
            law[i] = [pm, -Vp_mean]
            inv[i] = [norm, 0.5 * p2 + Vmean]
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")


class WavepacketSimulator(Simulator):
    """Full-field PDE emulation: predict the next wavefunction snapshot."""

    name = "wavepacket"
    law_dim = 0            # set in __init__ (= state_dim)
    invariant_names = ["norm", "energy"]
    default_dt = 0.02
    default_steps = 100
    regimes = ("default",)

    def __init__(self, grid=64, L=15.0):
        self.grid = grid
        self.L = L
        self.state_dim = 2 * grid
        self.law_dim = 2 * grid

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        vtype = rng.choice(["free", "harmonic", "barrier"])
        return dict(
            vtype=str(vtype),
            omega=float(rng.uniform(0.2, 0.8)),
            barrier=float(rng.uniform(1.0, 4.0)),
            x0=float(rng.uniform(-3, 3)),
            p0=float(rng.uniform(-2, 2)),
            sigma=float(rng.uniform(0.8, 1.5)),
        )

    def _potential(self, x, p):
        if p["vtype"] == "harmonic":
            return 0.5 * p["omega"] ** 2 * x ** 2
        if p["vtype"] == "barrier":
            return p["barrier"] * np.exp(-(x ** 2) / 0.5)
        return np.zeros_like(x)

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        x = np.linspace(-self.L, self.L, self.grid, endpoint=False)
        dx = x[1] - x[0]
        k = 2 * np.pi * np.fft.fftfreq(self.grid, d=dx)
        V = self._potential(x, params)
        psi0 = _gaussian_packet(x, params["x0"], params["p0"], params["sigma"])
        frames, _ = _splitstep_run(psi0, x, V, dt, n)

        states = np.zeros((n, 2 * self.grid))
        law = np.zeros((n, 2 * self.grid))
        inv = np.zeros((n, 2))
        t = np.arange(n) * dt
        for i in range(n):
            psi = frames[i]
            # dpsi/dt = -i H psi ; H psi = -1/2 psi'' + V psi  (spectral 2nd derivative)
            lap = np.fft.ifft(-(k ** 2) * np.fft.fft(psi))
            Hpsi = -0.5 * lap + V * psi
            dpsi = -1j * Hpsi
            states[i] = np.concatenate([psi.real, psi.imag])
            law[i] = np.concatenate([dpsi.real, dpsi.imag])
            dens = np.abs(psi) ** 2
            norm = np.sum(dens) * dx
            energy = np.real(np.sum(np.conj(psi) * Hpsi) * dx) / norm
            inv[i] = [norm, energy]
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")


class UniversalRabiSimulator(RabiSimulator):
    """Rabi with FIXED Delta, Omega -> a single universal generator (dR/dt = b x R,
    b = (1, 0, 1)). Only the initial Bloch vector varies, so "did it learn THE law"
    is well posed (unlike the parametric RabiSimulator, whose per-trajectory
    Omega/Delta must be inferred)."""

    name = "rabi_u"

    def sample_params(self, rng, regime="default"):
        theta = np.arccos(rng.uniform(-1, 1)); phi = rng.uniform(0, 2 * np.pi)
        R0 = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        return dict(Delta=1.0, Omega=1.0, R0=R0)
