from __future__ import annotations

from typing import Any

from llm_rl_engine.envs.base import Env
from llm_rl_engine.envs.bandits import ReversalBandit, StationaryBandit
from llm_rl_engine.envs.number_guess import NumberGuess

ENV_REGISTRY: dict[str, type] = {
    "NumberGuess": NumberGuess,
    "StationaryBandit": StationaryBandit,
    "ReversalBandit": ReversalBandit,
}


def make_env(name: str, **kwargs: Any) -> Env:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_REGISTRY)}")
    cls = ENV_REGISTRY[name]
    return cls(**_normalize_kwargs(name, kwargs))


def _normalize_kwargs(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    if name == "ReversalBandit" and "reversal_time_range" in out:
        r = out["reversal_time_range"]
        if isinstance(r, list):
            out["reversal_time_range"] = tuple(r)
    return out