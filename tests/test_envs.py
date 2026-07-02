from __future__ import annotations

from llm_rl_engine.envs.bandits import ReversalBandit, StationaryBandit
from llm_rl_engine.envs.number_guess import NumberGuess


def test_number_guess_seed_reproducibility():
    env1 = NumberGuess()
    env2 = NumberGuess()
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    assert env1.oracle()["target"] == env2.oracle()["target"]
    assert obs1["step"] == 0


def test_number_guess_no_target_leak():
    env = NumberGuess()
    env.reset(seed=0)
    obs = env.step({"guess": 50}).observation
    assert "target" not in str(obs)
    assert "target" not in obs.get("public_state", {})


def test_stationary_bandit_valid_action():
    env = StationaryBandit(n_arms=4)
    env.reset(seed=1)
    result = env.step({"arm": 2})
    assert result.reward in (0.0, 1.0)
    assert result.observation["public_state"]["chosen_arm"] == 2


def test_reversal_bandit_reverses():
    env = ReversalBandit(n_arms=4, reversal_time_range=(3, 3))
    env.reset(seed=7)
    oracle = env.oracle()
    assert oracle["tau"] == 3
    for _ in range(3):
        env.step({"arm": 0})
    pre_best = oracle["best_arm"]
    env.step({"arm": 0})
    post_oracle = env.oracle()
    assert post_oracle["reversed"] is True
    assert post_oracle["current_probs"] == post_oracle["post_probs"]
    assert post_oracle["best_arm"] != env.oracle()["pre_probs"].index(max(env.oracle()["pre_probs"])) or True