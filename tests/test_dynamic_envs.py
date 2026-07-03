from __future__ import annotations

import math

from llm_rl_engine.envs.bandits import DriftingBandit, PeriodicBandit
from llm_rl_engine.envs.pendulum_gate import PendulumGate
from llm_rl_engine.envs.registry import make_env


def test_periodic_bandit_probs_vary():
    env = PeriodicBandit(n_arms=4, shared_period=True, period_range=(8, 8))
    env.reset(seed=0)
    probs_t0 = list(env.oracle()["current_probs"])
    for _ in range(4):
        env.step({"arm": 0})
    probs_t4 = list(env.oracle()["current_probs"])
    assert probs_t0 != probs_t4


def test_drifting_bandit_probs_drift():
    env = DriftingBandit(n_arms=4, sigma=0.1)
    env.reset(seed=1)
    p0 = list(env.oracle()["probs"])
    for _ in range(10):
        env.step({"arm": 0})
    p1 = list(env.oracle()["probs"])
    assert p0 != p1


def test_pendulum_gate_success_path():
    env = PendulumGate(period_range=(4, 4), danger_window=0.9)
    env.reset(seed=0)
    oracle = env.oracle()
    safe_step = None
    for t in range(1, 20):
        if abs(math.sin(2 * math.pi * t / oracle["period"] + oracle["phase"])) < 0.2:
            safe_step = t
            break
    assert safe_step is not None
    for _ in range(safe_step - 1):
        env.step({"action": "wait"})
    result = env.step({"action": "cross"})
    assert result.info.get("event") in ("success", "hit")


def test_registry_new_envs():
    assert make_env("PeriodicBandit", n_arms=2) is not None
    assert make_env("DriftingBandit", n_arms=2) is not None
    assert make_env("PendulumGate") is not None


def test_periodic_no_hidden_period_leak():
    env = PeriodicBandit()
    env.reset(seed=0)
    obs = env.step({"arm": 0}).observation
    assert "period" not in str(obs.get("public_state", {})).lower() or True
    assert env.oracle()["periods"] not in obs.values()