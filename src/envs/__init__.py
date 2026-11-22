"""ParadigmBench environment registry."""

from __future__ import annotations

from src.envs.alpha_desk import AlphaDeskEnv
from src.envs.alpha_desk_v2 import AlphaDeskV2Env
from src.envs.alpha_desk_v3 import AlphaDeskV3Env, AlphaDeskV3ForkEnv
from src.envs.base import ParadigmEnv, PublicTaskSpec, StepOutcome
from src.envs.coalition_ops import CoalitionOpsEnv
from src.envs.life_companion import LifeCompanionEnv


def make_env(spec_name: str, *, horizon: int = 30, seed: int = 42) -> ParadigmEnv:
    """Instantiate a ParadigmBench environment by `<domain>:<mode>` key."""
    mapping = {
        "alpha_desk:stationary": lambda: AlphaDeskEnv(mode="stationary", horizon=horizon, seed=seed),
        "alpha_desk:gradual_decay": lambda: AlphaDeskEnv(mode="gradual_decay", horizon=horizon, seed=seed),
        "alpha_desk:regime_cycle": lambda: AlphaDeskEnv(mode="regime_cycle", horizon=horizon, seed=seed),
        "alpha_desk_v2:stationary": lambda: AlphaDeskV2Env(mode="stationary", horizon=horizon, seed=seed),
        "alpha_desk_v2:abrupt_reversal": lambda: AlphaDeskV2Env(mode="abrupt_reversal", horizon=horizon, seed=seed),
        "alpha_desk_v2:gradual_decay": lambda: AlphaDeskV2Env(mode="gradual_decay", horizon=horizon, seed=seed),
        "alpha_desk_v2:opportunity_rise": lambda: AlphaDeskV2Env(mode="opportunity_rise", horizon=horizon, seed=seed),
        "alpha_desk_v2:off_scope_shock": lambda: AlphaDeskV2Env(mode="off_scope_shock", horizon=horizon, seed=seed),
        "alpha_desk_v3:stationary": lambda: AlphaDeskV3Env(mode="stationary", horizon=max(24, horizon), seed=seed),
        "alpha_desk_v3:multi_abrupt_recurrence": lambda: AlphaDeskV3Env(mode="multi_abrupt_recurrence", horizon=max(24, horizon), seed=seed),
        "alpha_desk_v3:mixed_decay_opportunity": lambda: AlphaDeskV3Env(mode="mixed_decay_opportunity", horizon=max(24, horizon), seed=seed),
        "life_companion:stationary": lambda: LifeCompanionEnv(mode="stationary", horizon=horizon, seed=seed),
        "life_companion:lifecycle": lambda: LifeCompanionEnv(mode="lifecycle_maturation", horizon=horizon, seed=seed),
        "coalition_ops:stationary": lambda: CoalitionOpsEnv(mode="stationary", horizon=horizon, seed=seed),
        "coalition_ops:reversal": lambda: CoalitionOpsEnv(mode="utility_reversal", horizon=horizon, seed=seed),
    }
    if spec_name not in mapping:
        raise ValueError(f"Unknown environment '{spec_name}'. Valid: {sorted(mapping)}")
    return mapping[spec_name]()


__all__ = [
    "AlphaDeskEnv",
    "AlphaDeskV2Env",
    "AlphaDeskV3Env",
    "AlphaDeskV3ForkEnv",
    "CoalitionOpsEnv",
    "LifeCompanionEnv",
    "ParadigmEnv",
    "PublicTaskSpec",
    "StepOutcome",
    "make_env",
]
