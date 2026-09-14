"""Windowed torch Dataset over trajectories: (context H, next target)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    """Sliding (context, target) windows from a split's trajectories.

    ``states`` are normalized with the dataset stats. Each item is
    ``{"context": (H,F), "target": (F,)}``. The context ends at time ``t`` and
    the target is ``t + horizon``.
    """

    def __init__(self, dataset, split: str, context: int, horizon: int = 1):
        self.ds = dataset
        sp = dataset.splits[split]
        self.states = dataset.normalize(sp.states).astype(np.float32)   # (N,T,F)
        self.H = context
        self.horizon = horizon
        N, T, _ = self.states.shape
        self.index = [
            (n, t)
            for n in range(N)
            for t in range(context - 1, T - horizon)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i]
        ctx = self.states[n, t - self.H + 1 : t + 1]
        tgt = self.states[n, t + self.horizon]
        return {"context": torch.from_numpy(ctx), "target": torch.from_numpy(tgt)}


def make_loader(dataset, split: str, context: int, batch_size: int,
                shuffle: bool = True, horizon: int = 1, num_workers: int = 0) -> DataLoader:
    ds = WindowDataset(dataset, split, context, horizon)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle)
