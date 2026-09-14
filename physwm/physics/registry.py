"""Registry mapping domain names -> Simulator factory.

Grouped by the four physics families in the paper. Use :func:`make_simulator`
to instantiate by name, and :data:`DOMAIN_GROUPS` for sweeps.
"""

from __future__ import annotations

from .base import Simulator
from .kepler import KeplerSimulator, GRKeplerSimulator
from .harmonic import HarmonicSimulator
from .nbody import NBodySimulator
from .pendulum import SimplePendulumSimulator, DrivenPendulumSimulator, DoublePendulumSimulator
from .quantum import RabiSimulator, UniversalRabiSimulator, EhrenfestSimulator, WavepacketSimulator
from .relativity import RelativisticLorentzSimulator, UniversalLorentzSimulator, SchwarzschildGeodesicSimulator

_FACTORIES = {
    # gravitation
    "kepler": KeplerSimulator,
    "gr_kepler": GRKeplerSimulator,
    "harmonic": HarmonicSimulator,
    "nbody3": lambda: NBodySimulator(num_bodies=3),
    "nbody4": lambda: NBodySimulator(num_bodies=4),
    # newtonian
    "pendulum": SimplePendulumSimulator,
    "driven_pendulum": DrivenPendulumSimulator,
    "double_pendulum": DoublePendulumSimulator,
    # quantum
    "rabi": RabiSimulator,
    "rabi_u": UniversalRabiSimulator,
    "ehrenfest": EhrenfestSimulator,
    "wavepacket": WavepacketSimulator,
    # relativity
    "rel_lorentz": RelativisticLorentzSimulator,
    "rel_lorentz_u": UniversalLorentzSimulator,
    "schwarzschild": SchwarzschildGeodesicSimulator,
}

DOMAIN_GROUPS = {
    "gravitation": ["kepler", "gr_kepler", "nbody3"],
    "newtonian": ["pendulum", "driven_pendulum", "double_pendulum"],
    "quantum": ["rabi", "ehrenfest", "wavepacket"],
    "relativity": ["rel_lorentz", "schwarzschild"],
}

ALL_DOMAINS = list(_FACTORIES.keys())


def make_simulator(name: str) -> Simulator:
    if name not in _FACTORIES:
        raise KeyError(f"unknown domain '{name}'. Available: {ALL_DOMAINS}")
    return _FACTORIES[name]()
