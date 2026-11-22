from __future__ import annotations

from typing import Any

from llm_rl_engine.envs.base import Env
from llm_rl_engine.envs.bandits import (
    DriftingBandit,
    PeriodicBandit,
    PhaseLockedBandit,
    ReversalBandit,
    StationaryBandit,
)
from llm_rl_engine.envs.number_guess import NumberGuess
from llm_rl_engine.envs.pendulum_gate import PendulumGate
from llm_rl_engine.envs.exploit_decay_bandit import ExploitDecayBandit
from llm_rl_engine.envs.staged_bandit import StagedBandit
from llm_rl_engine.envs.traffic_light_gate import TrafficLightGate

ENV_REGISTRY: dict[str, type] = {
    "NumberGuess": NumberGuess,
    "StationaryBandit": StationaryBandit,
    "ReversalBandit": ReversalBandit,
    "PeriodicBandit": PeriodicBandit,
    "DriftingBandit": DriftingBandit,
    "PhaseLockedBandit": PhaseLockedBandit,
    "StagedBandit": StagedBandit,
    "ExploitDecayBandit": ExploitDecayBandit,
    "PendulumGate": PendulumGate,
    "TrafficLightGate": TrafficLightGate,
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
        "PendulumGate": ["period_range", "phase_shift_time_range", "phase_shift_delta_range"],
        "TrafficLightGate": ["period_range", "phase_shift_time_range", "phase_shift_delta_range"],
        "PhaseLockedBandit": ["period_range"],
        "StagedBandit": ["period_range", "amplitude_range", "base_range"],
    }
    for field in range_fields.get(name, []):
        if field in out and isinstance(out[field], list):
            out[field] = tuple(out[field])
    return out