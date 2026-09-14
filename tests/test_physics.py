"""Physics correctness: shapes, finiteness, and conserved invariants."""

import numpy as np
import pytest

from physwm.physics import make_simulator, ALL_DOMAINS


@pytest.mark.parametrize("name", ALL_DOMAINS)
def test_shapes_and_finite(name):
    rng = np.random.default_rng(0)
    sim = make_simulator(name)
    tr = sim.rollout(rng)
    assert tr.states.shape == (tr.n_steps, sim.state_dim)
    assert tr.law.shape == (tr.n_steps, sim.law_dim)
    assert tr.invariants.shape[1] == len(sim.invariant_names)
    assert np.all(np.isfinite(tr.states))
    assert np.all(np.isfinite(tr.law))


def test_kepler_energy_exact():
    rng = np.random.default_rng(1)
    sim = make_simulator("kepler")
    p = sim.sample_params(rng)
    tr = sim.simulate(p)
    E = tr.invariants[:, 0]
    assert np.ptp(E) < 1e-10                       # energy conserved to machine precision
    assert abs(E.mean() + 1 / (2 * p["a"])) < 1e-8  # equals -1/(2a)
    assert np.ptp(tr.invariants[:, 1]) < 1e-10      # angular momentum conserved


def test_rabi_norm_conserved():
    rng = np.random.default_rng(2)
    tr = make_simulator("rabi").rollout(rng)
    Rn = np.linalg.norm(tr.states, axis=1)
    assert np.ptp(Rn) < 1e-10


def test_lorentz_gamma_conserved_pure_B():
    rng = np.random.default_rng(3)
    sim = make_simulator("rel_lorentz")
    p = sim.sample_params(rng)
    p["E"] = np.zeros(3)                            # pure magnetic field
    tr = sim.simulate(p)
    assert np.ptp(tr.invariants[:, 0]) < 1e-8       # gamma conserved (Boris pusher)


def test_gr_kepler_precesses():
    rng = np.random.default_rng(4)
    tr = make_simulator("gr_kepler").rollout(rng)
    assert np.ptp(tr.invariants[:, 1]) < 1e-8       # L conserved
    assert np.ptp(tr.invariants[:, 4]) > 0.05       # perihelion angle precesses


def test_schwarzschild_geodesic_constraint():
    """Non-trivial check that the RK4 integration stays on the geodesic.

    E and L are stored as constants, so asserting they don't drift proves
    nothing. The real invariant is the timelike first integral
    ``(dr/dtau)^2 = E^2 - (1-2M/r)(1 + L^2/r^2)``, equivalent to the 4-velocity
    normalization ``g_uv u^u u^v = -1``. We reconstruct r and dr/dtau from the
    Cartesian state and verify the residual stays ~0 along the trajectory.
    """
    rng = np.random.default_rng(5)
    sim = make_simulator("schwarzschild")
    tr = sim.rollout(rng)
    M = sim.M
    x, y, vx, vy = tr.states[:, 0], tr.states[:, 1], tr.states[:, 2], tr.states[:, 3]
    r = np.hypot(x, y)
    pr = (x * vx + y * vy) / r                      # dr/dtau
    Lz = x * vy - y * vx                            # r^2 dphi/dtau
    E = tr.invariants[0, 0]
    residual = pr ** 2 - (E ** 2 - (1 - 2 * M / r) * (1 + Lz ** 2 / r ** 2))
    assert np.max(np.abs(residual)) < 1e-6, f"geodesic constraint violated: {np.max(np.abs(residual))}"
    assert np.allclose(Lz, Lz[0], rtol=1e-6)        # angular momentum actually conserved


def test_invariants_from_state():
    rng = np.random.default_rng(6)
    for name in ["kepler", "rabi", "rel_lorentz"]:
        sim = make_simulator(name)
        tr = sim.rollout(rng)
        d = sim.invariants_from_state(tr.states[0])
        assert isinstance(d, dict) and len(d) >= 1
        assert all(np.isfinite(v) for v in d.values())
