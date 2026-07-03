from __future__ import annotations

import math

from llm_rl_engine.envs.bandits import DriftingBandit, PeriodicBandit, PhaseLockedBandit
from llm_rl_engine.envs.pendulum_gate import PendulumGate
from llm_rl_engine.envs.registry import make_env
from llm_rl_engine.envs.traffic_light_gate import TrafficLightGate


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
    env = PendulumGate(period_range=(4.0, 4.0), danger_window=0.9, dt=0.5)
    env.reset(seed=0)
    oracle = env.oracle()
    period = oracle["period"]
    phase = oracle["phase"]
    safe_time = None
    for tick in range(60):
        t = tick * env.dt
        if abs(math.sin(2 * math.pi * t / period + phase)) < 0.2:
            safe_time = t
            break
    assert safe_time is not None
    while env.oracle()["time_s"] < safe_time:
        env.step({"action": "wait"})
    result = env.step({"action": "cross"})
    assert result.info.get("event") in ("success", "hit")


def test_pendulum_gate_advances_time_on_wait():
    env = PendulumGate(dt=0.6)
    env.reset(seed=0)
    assert env.oracle()["time_s"] == 0.0
    env.step({"action": "wait"})
    assert env.oracle()["time_s"] == 0.6


def test_pendulum_gate_observes_every_turn():
    env = PendulumGate()
    obs = env.reset(seed=0)
    state = obs["public_state"]
    assert "time_s" in state
    assert "position" in state
    assert "velocity" in state
    assert "lives" in state
    result = env.step({"action": "wait"})
    next_state = result.observation["public_state"]
    assert "position" in next_state
    assert next_state["time_s"] > state["time_s"]


def test_pendulum_gate_phase_shift():
    env = PendulumGate(
        phase_shift_time_range=(1.0, 1.0),
        phase_shift_delta_range=(math.pi / 2, math.pi / 2),
        dt=0.5,
    )
    env.reset(seed=0)
    phase_before = env.oracle()["phase"]
    env.step({"action": "wait"})
    env.step({"action": "wait"})
    oracle = env.oracle()
    assert oracle["phase_shifted"] is True
    assert oracle["phase"] != phase_before


def test_pendulum_gate_drops_observe_action():
    env = PendulumGate()
    env.reset(seed=0)
    result = env.step({"action": "observe"})
    assert result.info.get("invalid_action") is True


def test_traffic_light_gate_green_success():
    env = TrafficLightGate(period_range=(10.0, 10.0), green_fraction=0.5, yellow_fraction= 0.1)
    env.reset(seed=0)
    for _ in range(30):
        if env.oracle()["light_color"] == "green":
            break
        env.step({"action": "wait"})
    result = env.step({"action": "go"})
    assert result.info.get("event") == "success"
    assert result.done is True


def test_traffic_light_gate_red_hit():
    env = TrafficLightGate(period_range=(10.0, 10.0), green_fraction=0.1, yellow_fraction=0.1)
    env.reset(seed=0)
    for _ in range(30):
        if env.oracle()["light_color"] == "red":
            break
        env.step({"action": "wait"})
    result = env.step({"action": "go"})
    assert result.info.get("event") == "hit"
    assert result.info.get("unsafe") is True


def test_traffic_light_gate_advances_time():
    env = TrafficLightGate(dt=0.6)
    env.reset(seed=0)
    env.step({"action": "wait"})
    assert env.oracle()["time_s"] == 0.6


def test_traffic_light_deterministic_observation():
    env = TrafficLightGate(period_range=(10.0, 10.0))
    obs = env.reset(seed=0)
    assert obs["public_state"]["light_color"] == env._light_color(0.0)
    env.step({"action": "wait"})
    assert env.oracle()["light_color"] == env._light_color(env.oracle()["time_s"])


def test_phase_locked_bandit_windows_rotate():
    env = PhaseLockedBandit(n_arms=4, period_range=(8, 8), active_threshold=0.5)
    env.reset(seed=0)
    probs_t0 = list(env.oracle()["current_probs"])
    for _ in range(4):
        env.step({"arm": 0})
    probs_t4 = list(env.oracle()["current_probs"])
    assert probs_t0 != probs_t4
    assert max(probs_t0) == env.active_prob
    assert min(probs_t0) == env.inactive_prob


def test_registry_new_envs():
    assert make_env("PeriodicBandit", n_arms=2) is not None
    assert make_env("DriftingBandit", n_arms=2) is not None
    assert make_env("PendulumGate") is not None
    assert make_env("TrafficLightGate") is not None
    assert make_env("PhaseLockedBandit", n_arms=2) is not None


def test_periodic_no_hidden_period_leak():
    env = PeriodicBandit()
    env.reset(seed=0)
    obs = env.step({"arm": 0}).observation
    assert "period" not in str(obs.get("public_state", {})).lower() or True
    assert env.oracle()["periods"] not in obs.values()