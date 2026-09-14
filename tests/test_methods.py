"""Methods: loss is finite & differentiable, sampling shapes, probe sanity."""

import numpy as np
import torch

from physwm.train.config import ModelCfg, MethodCfg
from physwm.methods import make_method, ALL_METHODS
from physwm.eval.probe import ridge_r2


def _batch(B=32, H=6, F=4):
    return {"context": torch.randn(B, H, F), "target": torch.randn(B, F)}


def _cfg():
    return ModelCfg(d_model=32, n_layers=2, n_heads=4, d_ff=64)


import pytest


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_loss_finite_and_backprops(kind):
    torch.manual_seed(0)
    mcfg = MethodCfg(kind=kind, n_diffusion_steps=10, flow_sample_steps=4, drift_particles=3)
    m = make_method(_cfg(), mcfg, state_dim=4)
    m.train()
    loss = m.loss(_batch())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_sample_and_hidden_shapes(kind):
    torch.manual_seed(0)
    mcfg = MethodCfg(kind=kind, n_diffusion_steps=10, flow_sample_steps=4, drift_particles=3)
    m = make_method(_cfg(), mcfg, state_dim=4)
    ctx = torch.randn(8, 6, 4)
    out = m.sample(ctx)
    assert out.shape == (8, 4)
    h = m.hidden_states(ctx)
    assert h.shape[0] == 8 and h.shape[1] == _cfg().n_layers + 1


def test_drifting_reduces_drift_on_toy():
    """Drifting should shrink the drift target on a fixed conditional map."""
    torch.manual_seed(0)
    mcfg = MethodCfg(kind="drifting", drift_particles=4, drift_tau=0.5)
    m = make_method(_cfg(), mcfg, state_dim=4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    # deterministic target = linear function of context-mean (a learnable map)
    ctx = torch.randn(64, 6, 4)
    tgt = ctx.mean(1) * 0.5
    batch = {"context": ctx, "target": tgt}
    losses = []
    for _ in range(150):
        opt.zero_grad(); l = m.loss(batch); l.backward(); opt.step()
        losses.append(l.item())
    assert losses[-1] < losses[0]                 # drift target loss decreases
    with torch.no_grad():
        err = ((m.sample(ctx) - tgt) ** 2).mean().item()
    assert err < 0.5                              # samples approach the target map


def test_probe_rejects_same_split_leakage():
    """Probing must be scored on trajectories disjoint from those it was fit on."""
    from physwm.train.config import DataCfg
    from physwm.data import build_dataset
    from physwm.eval.probe import probe_representations

    ds = build_dataset(DataCfg(domain="kepler", n_train=8, n_val=4, n_test=8,
                               n_steps=20, context=4, seed=0))
    m = make_method(_cfg(), MethodCfg(kind="ar"), state_dim=ds.state_dim)
    with pytest.raises(ValueError, match="must differ"):
        probe_representations(m, ds, fit_split="train", eval_split="train", context=4)


def test_probe_reports_disjoint_trajectory_counts():
    from physwm.train.config import DataCfg
    from physwm.data import build_dataset
    from physwm.eval.probe import probe_representations

    ds = build_dataset(DataCfg(domain="kepler", n_train=8, n_val=4, n_test=6,
                               n_steps=20, context=4, seed=0))
    m = make_method(_cfg(), MethodCfg(kind="ar"), state_dim=ds.state_dim)
    res = probe_representations(m, ds, fit_split="train", eval_split="test",
                                context=4, n_traj=8, max_samples=500)
    assert res.n_fit_traj > 0 and res.n_eval_traj > 0
    assert "force" in res.r2


def test_ridge_r2_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    W = rng.normal(size=(8, 2))
    Y = X @ W + 0.01 * rng.normal(size=(400, 2))
    r2 = ridge_r2(X[:300], Y[:300], X[300:], Y[300:], alpha=1e-3)
    assert r2 > 0.99
