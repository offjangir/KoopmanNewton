"""Newtonian mechanics: driven-damped pendulum and double pendulum.

* :class:`DrivenPendulumSimulator` -- a non-autonomous, dissipative system
  ``theta'' + b theta' + omega0^2 sin theta = A cos(omega_d t)``. The drive phase
  is carried in the state so the process is Markov. Rich regime structure
  (fixed point / limit cycle / chaos).
* :class:`DoublePendulumSimulator` -- the canonical chaotic conservative system;
  total energy is conserved and is the key invariant a "physicist" model should
  keep constant along rollouts.

Angles are encoded as ``(sin, cos)`` to avoid wrap discontinuities.
"""

from __future__ import annotations


import numpy as np

from .base import Simulator, Trajectory


class SimplePendulumSimulator(Simulator):
    """Undamped, undriven nonlinear pendulum: ``theta'' = -(g/L) sin theta``.

    The grid's "Newtonian" law: a clean second-order autonomous system whose
    generator is *nonlinear* (the ``sin``). Fixed ``g/L`` (=1) so it is a single
    universal law; only initial conditions vary. Observing position-only
    (``obs_dims=2`` -> (sin th, cos th)) makes it the direct pendulum analog of
    the Kepler position task, and the Verlet force-term test applies.
    """

    name = "pendulum"
    state_dim = 3          # (sin th, cos th, th_dot)
    law_dim = 1            # angular acceleration th_ddot = -sin th
    invariant_names = ["energy"]
    default_dt = 0.05
    default_steps = 150
    regimes = ("default", "large_amplitude")
    omega0_sq = 1.0        # g/L, fixed (universal law)

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        # keep energy below the separatrix (E < 2*omega0^2) so motion oscillates
        if regime == "large_amplitude":
            theta0 = rng.uniform(2.0, 3.0) * rng.choice([-1.0, 1.0])
            thd0 = rng.uniform(-0.5, 0.5)
        else:
            theta0 = rng.uniform(-2.0, 2.0)
            thd0 = rng.uniform(-1.2, 1.2)
        return dict(theta0=float(theta0), thetadot0=float(thd0))

    def _accel(self, theta):
        return -self.omega0_sq * np.sin(theta)

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        theta, thd = params["theta0"], params["thetadot0"]
        t = np.arange(n) * dt
        states = np.zeros((n, 3)); law = np.zeros((n, 1)); inv = np.zeros((n, 1))
        for k in range(n):
            states[k] = [np.sin(theta), np.cos(theta), thd]
            law[k] = self._accel(theta)
            inv[k] = 0.5 * thd ** 2 + self.omega0_sq * (1 - np.cos(theta))
            # RK4
            def f(a, b):
                return b, self._accel(a)
            k1 = f(theta, thd)
            k2 = f(theta + 0.5 * dt * k1[0], thd + 0.5 * dt * k1[1])
            k3 = f(theta + 0.5 * dt * k2[0], thd + 0.5 * dt * k2[1])
            k4 = f(theta + dt * k3[0], thd + dt * k3[1])
            theta += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            thd += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return Trajectory(states, law, inv, self.invariant_names, t, params, regime="default")


class DrivenPendulumSimulator(Simulator):
    name = "driven_pendulum"
    state_dim = 5          # (sin th, cos th, th_dot, sin phi, cos phi)  phi = drive phase
    law_dim = 1            # angular acceleration th_ddot
    invariant_names = ["energy", "theta_dot"]
    default_dt = 0.05
    default_steps = 200
    regimes = ("default", "chaotic")

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        if regime == "chaotic":
            A = rng.uniform(1.0, 1.5)
        else:
            A = rng.uniform(0.0, 1.0)
        return dict(
            omega0=float(rng.uniform(0.5, 2.0)),
            b=float(rng.uniform(0.0, 0.5)),
            A=float(A),
            omega_d=float(rng.uniform(0.5, 2.0)),
            theta0=float(rng.uniform(-np.pi, np.pi)),
            thetadot0=float(rng.uniform(-2.0, 2.0)),
        )

    def _accel(self, theta, thetadot, t, p):
        return -p["b"] * thetadot - p["omega0"] ** 2 * np.sin(theta) + p["A"] * np.cos(p["omega_d"] * t)

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        theta, thetadot = params["theta0"], params["thetadot0"]
        t_arr = np.arange(n) * dt
        states = np.zeros((n, 5))
        law = np.zeros((n, 1))
        inv = np.zeros((n, 2))
        for k in range(n):
            tk = t_arr[k]
            phi = params["omega_d"] * tk
            acc = self._accel(theta, thetadot, tk, params)
            states[k] = [np.sin(theta), np.cos(theta), thetadot, np.sin(phi), np.cos(phi)]
            law[k] = acc
            E = 0.5 * thetadot ** 2 + params["omega0"] ** 2 * (1 - np.cos(theta))
            inv[k] = [E, thetadot]
            # RK4 step (non-autonomous)
            def f(th, thd, tt):
                return thd, self._accel(th, thd, tt, params)
            k1 = f(theta, thetadot, tk)
            k2 = f(theta + 0.5 * dt * k1[0], thetadot + 0.5 * dt * k1[1], tk + 0.5 * dt)
            k3 = f(theta + 0.5 * dt * k2[0], thetadot + 0.5 * dt * k2[1], tk + 0.5 * dt)
            k4 = f(theta + dt * k3[0], thetadot + dt * k3[1], tk + dt)
            theta += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            thetadot += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return Trajectory(states, law, inv, self.invariant_names, t_arr, params, regime="default")


class DoublePendulumSimulator(Simulator):
    name = "double_pendulum"
    state_dim = 6          # (sin th1, cos th1, sin th2, cos th2, w1, w2)
    law_dim = 2            # (th1_ddot, th2_ddot)
    invariant_names = ["energy"]
    default_dt = 0.01
    default_steps = 150
    regimes = ("default",)

    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        return dict(
            m1=float(rng.uniform(0.5, 2.0)), m2=float(rng.uniform(0.5, 2.0)),
            L1=float(rng.uniform(0.5, 2.0)), L2=float(rng.uniform(0.5, 2.0)),
            g=1.0,
            th1=float(rng.uniform(-np.pi, np.pi)), th2=float(rng.uniform(-np.pi, np.pi)),
            w1=float(rng.uniform(-1.0, 1.0)), w2=float(rng.uniform(-1.0, 1.0)),
        )

    @staticmethod
    def _accel(th1, th2, w1, w2, p):
        m1, m2, L1, L2, g = p["m1"], p["m2"], p["L1"], p["L2"], p["g"]
        d = th1 - th2
        cd, sd = np.cos(d), np.sin(d)
        den = (m1 + m2) * L1 - m2 * L1 * cd * cd
        a1 = (m2 * L1 * w1 * w1 * sd * cd
              + m2 * g * np.sin(th2) * cd
              + m2 * L2 * w2 * w2 * sd
              - (m1 + m2) * g * np.sin(th1)) / den
        den2 = (L2 / L1) * den
        a2 = (-m2 * L2 * w2 * w2 * sd * cd
              + (m1 + m2) * (g * np.sin(th1) * cd - L1 * w1 * w1 * sd - g * np.sin(th2))) / den2
        return a1, a2

    @staticmethod
    def _energy(th1, th2, w1, w2, p):
        m1, m2, L1, L2, g = p["m1"], p["m2"], p["L1"], p["L2"], p["g"]
        ke = (0.5 * m1 * L1 ** 2 * w1 ** 2
              + 0.5 * m2 * (L1 ** 2 * w1 ** 2 + L2 ** 2 * w2 ** 2
                           + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)))
        pe = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)
        return ke + pe

    def simulate(self, params, n_steps=None, dt=None, rng=None) -> Trajectory:
        n = n_steps or self.default_steps
        dt = dt or self.default_dt
        th1, th2 = params["th1"], params["th2"]
        w1, w2 = params["w1"], params["w2"]
        t_arr = np.arange(n) * dt
        states = np.zeros((n, 6))
        law = np.zeros((n, 2))
        inv = np.zeros((n, 1))

        def deriv(y):
            a, b, c, e = y
            aa, bb = self._accel(a, b, c, e, params)
            return np.array([c, e, aa, bb])

        y = np.array([th1, th2, w1, w2])
        for k in range(n):
            a1, a2 = self._accel(y[0], y[1], y[2], y[3], params)
            states[k] = [np.sin(y[0]), np.cos(y[0]), np.sin(y[1]), np.cos(y[1]), y[2], y[3]]
            law[k] = [a1, a2]
            inv[k] = [self._energy(y[0], y[1], y[2], y[3], params)]
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Trajectory(states, law, inv, self.invariant_names, t_arr, params, regime="default")
