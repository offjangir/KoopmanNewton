"""Learning-paradigm factory."""

from __future__ import annotations

from .base import Method
from .ar import ARMethod
from .diffusion import DiffusionMethod
from .flow_matching import FlowMatchingMethod
from .drifting import DriftingMethod
from .stochastic_interpolant import StochasticInterpolantMethod

_METHODS = {
    "ar": ARMethod,
    "diffusion": DiffusionMethod,
    "flow": FlowMatchingMethod,
    "drifting": DriftingMethod,
    "si": StochasticInterpolantMethod,
}

ALL_METHODS = list(_METHODS.keys())


def make_method(model_cfg, method_cfg, state_dim: int) -> Method:
    if method_cfg.kind not in _METHODS:
        raise KeyError(f"unknown method '{method_cfg.kind}'. Available: {ALL_METHODS}")
    cls = _METHODS[method_cfg.kind]
    # ensure the backbone matches the paradigm (AR uses the causal transformer)
    model_cfg.kind = "ar_transformer" if cls.model_kind == "ar" else "dit"
    return cls(model_cfg, method_cfg, state_dim)


__all__ = ["Method", "make_method", "ALL_METHODS"]
