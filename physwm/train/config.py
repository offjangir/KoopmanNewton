"""Config dataclasses with YAML load/save.

An :class:`ExperimentCfg` bundles data / model / method / train settings and is
the single object passed around the pipeline. The three inductive biases from the
paper are exposed as flags: ``ModelCfg.attn_window`` (temporal locality),
``MethodCfg.ctx_noise_sigma`` (stability), ``ModelCfg.head`` (spatial smoothness).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class DataCfg:
    domain: str = "kepler"
    n_train: int = 2000            # number of trajectories
    n_val: int = 200
    n_test: int = 500
    n_steps: Optional[int] = None  # None -> simulator default
    dt: Optional[float] = None
    context: int = 8               # history window H fed to the model
    horizon: int = 1               # steps predicted per training example (1 = next-state)
    # Observed state dimensions. None = full state (includes velocities).
    # IMPORTANT: if velocities are observed, two consecutive states linearly
    # determine the acceleration via (v_t - v_{t-1})/dt, making the force probe
    # nearly trivial (R^2 ~ 0.996 at context 2). Set obs_dims=2 for the
    # position-only setup used by the original paper.
    obs_dims: int | None = None
    ood_regimes: list[str] = field(default_factory=list)  # extra held-out OOD test regimes
    normalize: bool = True
    seed: int = 0
    data_root: str = "data_cache"
    # Stream freshly simulated trajectories for TRAINING (eval/probe stay on the
    # fixed splits). Removes the data ceiling and makes overfitting impossible.
    online: bool = False
    online_workers: int = 4


@dataclass
class ModelCfg:
    kind: str = "ar_transformer"   # or "dit"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    attn_window: Optional[int] = None   # temporal-locality bias (None = full causal)
    head: str = "regression"            # "regression" | "classification"
    n_bins: int = 512                   # classification-head bins per dim
    bin_range: float = 8.0              # coord range for binning (normalized units)


@dataclass
class MethodCfg:
    kind: str = "ar"               # "ar" | "diffusion" | "flow" | "drifting" | "si"
    ctx_noise_sigma: float = 0.0   # stability bias (noisy-context training)
    # diffusion
    n_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # flow matching
    flow_sample_steps: int = 20
    # drifting
    drift_tau: float = 0.5
    drift_particles: int = 4
    drift_cfg_gamma: float = 0.0


@dataclass
class TrainCfg:
    steps: int = 5000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup: int = 100
    amp: bool = True
    log_every: int = 100
    eval_every: int = 1000
    ckpt_every: int = 1000
    device: str = "cuda"
    seed: int = 0
    out_dir: str = "runs/exp"


@dataclass
class ExperimentCfg:
    name: str = "exp"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    method: MethodCfg = field(default_factory=MethodCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # ---- (de)serialization ------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentCfg":
        return cls(
            name=d.get("name", "exp"),
            data=DataCfg(**d.get("data", {})),
            model=ModelCfg(**d.get("model", {})),
            method=MethodCfg(**d.get("method", {})),
            train=TrainCfg(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str) -> "ExperimentCfg":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
