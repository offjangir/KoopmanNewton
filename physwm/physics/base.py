"""Base types for physics simulators.

Every domain implements :class:`Simulator`, which produces :class:`Trajectory`
objects. A trajectory carries three aligned per-timestep arrays:

* ``states``     -- the observable state ``s_t in R^F`` the model sees / predicts.
* ``law``        -- the *instantaneous law* target ``in R^L`` (e.g. force / accel /
                    precession rate). This is what a "physicist" model should encode.
* ``invariants`` -- conserved (or slowly varying) quantities ``in R^I`` (energy,
                    angular momentum, LRL vector, norm, gamma...). This is what a
                    "curve-fitter" tends to encode, and it is used both for probing
                    and for physics-consistency (invariant-drift) metrics.

The split between ``law`` and ``invariants`` is the direct generalization of the
Newtonian-force vs Keplerian-orbital-elements probe from the original paper.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    """A single simulated trajectory with probe targets.

    Shapes: ``states (T, F)``, ``law (T, L)``, ``invariants (T, I)``.
    """

    states: np.ndarray
    law: np.ndarray
    invariants: np.ndarray
    invariant_names: list[str]
    t: np.ndarray
    params: dict = field(default_factory=dict)
    regime: str = "default"

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float64)
        self.law = np.asarray(self.law, dtype=np.float64)
        self.invariants = np.asarray(self.invariants, dtype=np.float64)
        T = self.states.shape[0]
        assert self.states.ndim == 2, f"states must be (T,F), got {self.states.shape}"
        assert self.law.shape[0] == T, "law/states length mismatch"
        assert self.invariants.shape[0] == T, "invariants/states length mismatch"
        assert self.invariants.shape[1] == len(self.invariant_names)

    @property
    def n_steps(self) -> int:
        return self.states.shape[0]


class Simulator(abc.ABC):
    """Abstract base class for a physics-trajectory generator.

    Subclasses declare ``state_dim`` (F), ``law_dim`` (L), ``invariant_names``,
    and default integration settings, then implement :meth:`sample_params` and
    :meth:`simulate`.
    """

    name: str = "base"
    state_dim: int = 0
    law_dim: int = 0
    invariant_names: list[str] = []
    default_dt: float = 0.1
    default_steps: int = 100
    #: named OOD regimes (e.g. train-range vs extrapolation) understood by ``sample_params``
    regimes: tuple[str, ...] = ("default",)
    #: True when the state is laid out as [positions..., velocities...] with
    #: velocity = d(position)/d(time param). Lets the invariant-drift metric
    #: rebuild velocities by finite difference when only positions are observed.
    pos_vel_layout: bool = False

    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator, regime: str = "default") -> dict:
        """Sample physical parameters + initial conditions for one trajectory."""

    @abc.abstractmethod
    def simulate(
        self,
        params: dict,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Trajectory:
        """Integrate one trajectory from ``params``."""

    def invariants_from_state(self, state: np.ndarray) -> Optional[dict]:
        """Parameter-free invariants computable from a single state vector.

        Returns ``{name: value}`` for the invariants that need no trajectory
        parameters (used by the invariant-drift physics-consistency metric on
        model rollouts), or ``None`` if unavailable for this domain.
        """
        return None

    # -- convenience ---------------------------------------------------------
    def rollout(
        self,
        rng: np.random.Generator,
        regime: str = "default",
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        max_retries: int = 50,
    ) -> Trajectory:
        """Sample params and simulate, retrying on rejected (unstable) draws."""
        for _ in range(max_retries):
            params = self.sample_params(rng, regime=regime)
            try:
                traj = self.simulate(params, n_steps=n_steps, dt=dt, rng=rng)
            except _RejectTrajectory:
                continue
            if np.all(np.isfinite(traj.states)) and np.all(np.isfinite(traj.law)):
                return traj
        raise RuntimeError(f"{self.name}: could not produce a stable trajectory after {max_retries} tries")


class _RejectTrajectory(Exception):
    """Raised internally by a simulator to reject an unstable parameter draw."""


def angle_features(theta: np.ndarray) -> np.ndarray:
    """Encode an angle as (sin, cos) to avoid the 2*pi wrap discontinuity."""
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)
