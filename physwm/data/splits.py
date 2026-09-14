"""OOD test-split design.

Beyond a random train/test split we hold out entire *regimes* so the test set
probes learned-law generalization, not interpolation. Each simulator declares its
regimes; the non-"default" ones are natural OOD axes (parameter extrapolation,
stronger nonlinearity, chaotic bands). This directly targets the original paper's
"tiny test set proves nothing" weakness.
"""

from __future__ import annotations

from ..physics import make_simulator


def default_ood_regimes(domain: str) -> list[str]:
    """All non-default regimes a simulator supports (used as held-out OOD tests)."""
    sim = make_simulator(domain)
    return [r for r in sim.regimes if r != "default"]
