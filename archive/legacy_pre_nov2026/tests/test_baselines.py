from __future__ import annotations

from llm_rl_engine.agents.classical import UCBAgent, make_classical_agent
from llm_rl_engine.envs.bandits import StationaryBandit
from llm_rl_engine.runners.run_episode import run_episode


def test_ucb_beats_random_on_bandit():
    env = StationaryBandit(n_arms=4)
    ucb = make_classical_agent({"name": "ucb"})
    random_agent = make_classical_agent({"name": "random"})

    ucb_rewards = []
    random_rewards = []
    for seed in range(20):
        traj_ucb = run_episode(env, ucb, horizon=30, seed=seed)
        env2 = StationaryBandit(n_arms=4)
        traj_rand = run_episode(env2, random_agent, horizon=30, seed=seed)
        ucb_rewards.append(sum(r.reward for r in traj_ucb.records))
        random_rewards.append(sum(r.reward for r in traj_rand.records))

    assert sum(ucb_rewards) > sum(random_rewards)


def test_ucb_agent_initialization():
    agent = UCBAgent()
    agent.reset({"name": "StationaryBandit", "n_arms": 4}, seed=0)
    action = agent.act({"step": 0, "message": "start"}, [])
    assert "arm" in action
    assert 0 <= action["arm"] < 4