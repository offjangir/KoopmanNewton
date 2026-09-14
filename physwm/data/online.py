"""Online (streaming) data generation -- every batch is freshly simulated.

Physics data is *procedural*, so there is no reason to train on a fixed finite
sample. Streaming freshly generated trajectories makes overfitting structurally
impossible and removes the data ceiling entirely, converting "data-hungry"
into "compute-hungry" (which is measurable and fair across methods).

Generation is cheap relative to training: Kepler runs ~1160 traj/s per core and
each trajectory yields ~T windows, so a handful of dataloader workers saturates
any single-GPU training loop.

Evaluation still uses a *fixed* dataset (val/test/OOD) so metrics and probes stay
reproducible and comparable across runs -- only the training stream is online.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from ..physics import make_simulator


def estimate_norm_stats(domain: str, n_pilot: int = 500, n_steps=None, dt=None,
                        obs_dims: int | None = None, seed: int = 0):
    """Per-feature mean/std from a pilot batch of trajectories."""
    sim = make_simulator(domain)
    rng = np.random.default_rng(seed)
    S = []
    for _ in range(n_pilot):
        tr = sim.rollout(rng, n_steps=n_steps, dt=dt)
        s = tr.states[:, :obs_dims] if obs_dims else tr.states
        S.append(s)
    S = np.concatenate(S, 0)
    return S.mean(0), S.std(0) + 1e-6


class OnlineWindowDataset(IterableDataset):
    """Infinite stream of (context, target) windows from freshly simulated trajectories."""

    def __init__(self, domain: str, context: int, mean, std, horizon: int = 1,
                 n_steps=None, dt=None, regime: str = "default",
                 obs_dims: int | None = None, seed: int = 0, shuffle_buffer: int = 4096):
        super().__init__()
        self.domain = domain
        self.context = context
        self.horizon = horizon
        self.n_steps = n_steps
        self.dt = dt
        self.regime = regime
        self.obs_dims = obs_dims
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 9973 * (wid + 1))
        sim = make_simulator(self.domain)
        buf: list = []
        while True:
            tr = sim.rollout(rng, regime=self.regime, n_steps=self.n_steps, dt=self.dt)
            s = tr.states[:, :self.obs_dims] if self.obs_dims else tr.states
            s = ((s - self.mean) / self.std).astype(np.float32)
            for t in range(self.context - 1, len(s) - self.horizon):
                buf.append((s[t - self.context + 1: t + 1], s[t + self.horizon]))
            # decorrelate windows across trajectories before emitting
            if len(buf) >= self.shuffle_buffer:
                rng.shuffle(buf)
                for ctx, tgt in buf:
                    yield {"context": torch.from_numpy(ctx.copy()),
                           "target": torch.from_numpy(tgt.copy())}
                buf = []


def make_online_loader(domain: str, context: int, mean, std, batch_size: int,
                       horizon: int = 1, n_steps=None, dt=None, obs_dims=None,
                       seed: int = 0, num_workers: int = 4) -> DataLoader:
    if num_workers > 0:
        # An infinite IterableDataset streams tensors from workers for the whole
        # run; the default 'file_descriptor' sharing strategy leaks fds until the
        # loader dies with "Too many open files".
        torch.multiprocessing.set_sharing_strategy("file_system")
    ds = OnlineWindowDataset(domain, context, mean, std, horizon=horizon,
                             n_steps=n_steps, dt=dt, obs_dims=obs_dims, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
