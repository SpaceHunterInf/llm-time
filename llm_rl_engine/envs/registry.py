from __future__ import annotations

from typing import Any

from llm_rl_engine.envs.base import Env
from llm_rl_engine.envs.bandits import (
    DriftingBandit,
    PeriodicBandit,
    ReversalBandit,
    StationaryBandit,
)
from llm_rl_engine.envs.number_guess import NumberGuess
from llm_rl_engine.envs.pendulum_gate import PendulumGate

ENV_REGISTRY: dict[str, type] = {
    "NumberGuess": NumberGuess,
    "StationaryBandit": StationaryBandit,
    "ReversalBandit": ReversalBandit,
    "PeriodicBandit": PeriodicBandit,
    "DriftingBandit": DriftingBandit,
    "PendulumGate": PendulumGate,
}


def make_env(name: str, **kwargs: Any) -> Env:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_REGISTRY)}")
    cls = ENV_REGISTRY[name]
    return cls(**_normalize_kwargs(name, kwargs))


def _normalize_kwargs(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    range_fields = {
        "ReversalBandit": ["reversal_time_range"],
        "PeriodicBandit": ["period_range", "amplitude_range", "base_range"],
        "PendulumGate": ["period_range"],
    }
    for field in range_fields.get(name, []):
        if field in out and isinstance(out[field], list):
            out[field] = tuple(out[field])
    return out