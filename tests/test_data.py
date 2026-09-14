"""Data pipeline: build, splits, normalization, windowing."""

import numpy as np

from physwm.train.config import DataCfg
from physwm.data import build_dataset, make_loader


def _cfg(**kw):
    d = dict(domain="kepler", n_train=20, n_val=5, n_test=8, n_steps=30, context=5, seed=0)
    d.update(kw)
    return DataCfg(**d)


def test_build_and_splits():
    ds = build_dataset(_cfg())
    assert ds.state_dim == 4
    assert ds.splits["train"].states.shape == (20, 30, 4)
    # kepler exposes OOD regimes -> extra test splits generated
    assert any(k.startswith("test_") for k in ds.splits)


def test_normalization_stats():
    ds = build_dataset(_cfg())
    z = ds.normalize(ds.splits["train"].states.reshape(-1, ds.state_dim))
    assert np.allclose(z.mean(0), 0, atol=1e-5)
    assert np.allclose(z.std(0), 1, atol=1e-2)


def test_window_loader_shapes():
    ds = build_dataset(_cfg())
    loader = make_loader(ds, "train", context=5, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    assert batch["context"].shape == (16, 5, 4)
    assert batch["target"].shape == (16, 4)


def test_obs_dims_truncates_observation_only():
    """obs_dims restricts what the model sees; probe targets stay full."""
    ds = build_dataset(_cfg(obs_dims=2))
    assert ds.state_dim == 2
    assert ds.splits["train"].states.shape[-1] == 2
    assert ds.splits["train"].law.shape[-1] == 2            # force target untouched
    assert ds.splits["train"].invariants.shape[-1] == 5      # invariants untouched


def test_velocity_in_state_makes_force_probe_trivial():
    """Regression test for a real confound found during development.

    If velocity is observed, two consecutive states linearly determine the
    acceleration via (v_t - v_{t-1})/dt, so a linear readout of the RAW context
    recovers the 'Newtonian force' almost perfectly -- with no model at all.
    Position-only observation removes most of that shortcut. Any force-probe
    result must therefore be reported against this control.
    """
    from physwm.eval.probe import probe_raw_context

    full = build_dataset(_cfg(n_train=60, n_test=40, n_steps=60))
    pos = build_dataset(_cfg(n_train=60, n_test=40, n_steps=60, obs_dims=2))
    r_full = probe_raw_context(full, "train", "test", context=2, n_traj=60).r2["force"]
    r_pos = probe_raw_context(pos, "train", "test", context=2, n_traj=60).r2["force"]
    assert r_full > 0.95, f"expected near-trivial force decoding with velocities, got {r_full}"
    assert r_pos < r_full - 0.1, f"position-only should be much harder: {r_pos} vs {r_full}"


def test_save_load_roundtrip(tmp_path):
    ds = build_dataset(_cfg())
    p = str(tmp_path / "d.npz")
    ds.save(p)
    ds2 = build_dataset.__globals__["PhysicsDataset"].load(p)
    assert ds2.state_dim == ds.state_dim
    assert np.allclose(ds2.mean, ds.mean)
    assert set(ds2.splits) == set(ds.splits)


def test_online_stream_shapes_and_freshness():
    """Online loader yields correct shapes and never repeats a fixed sample set."""
    import torch
    from physwm.data import make_online_loader, estimate_norm_stats

    mean, std = estimate_norm_stats("kepler", n_pilot=20, n_steps=30, obs_dims=2)
    assert mean.shape == (2,)
    loader = make_online_loader("kepler", context=4, mean=mean, std=std, batch_size=16,
                                n_steps=30, obs_dims=2, num_workers=0)
    it = iter(loader)
    b1, b2 = next(it), next(it)
    assert b1["context"].shape == (16, 4, 2)
    assert b1["target"].shape == (16, 2)
    assert torch.isfinite(b1["context"]).all()
    # fresh stream: two consecutive batches must not be identical
    assert not torch.allclose(b1["context"], b2["context"])
