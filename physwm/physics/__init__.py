"""Physics-trajectory simulators for four domains."""

from .base import Simulator, Trajectory
from .registry import make_simulator, DOMAIN_GROUPS, ALL_DOMAINS

__all__ = ["Simulator", "Trajectory", "make_simulator", "DOMAIN_GROUPS", "ALL_DOMAINS"]
