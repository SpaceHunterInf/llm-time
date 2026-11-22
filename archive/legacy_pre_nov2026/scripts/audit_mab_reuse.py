"""Offline legacy MAB probes; no API calls, model loading, or result mutations.

Run: uv run --offline --no-sync python scripts/audit_mab_reuse.py
Reports observations rather than asserting that old bugs must remain.
"""
from __future__ import annotations

import json

from llm_rl_engine.agents.base import Agent
from llm_rl_engine.agents.classical import GreedyAgent, ThompsonSamplingAgent, UCBAgent
from llm_rl_engine.agents.llm import GenerationResult, LLMAgent
from llm_rl_engine.envs.staged_bandit import StagedBandit
from llm_rl_engine.runners.run_episode import run_episode


def baseline_probe():
    history = [
        {"action": {"arm": a}, "reward": r}
        for a, r in [(0, 1.0), (1, 0.0), (0, 0.0)]
    ]
    ucb, ts = UCBAgent(), ThompsonSamplingAgent()
    for agent in (ucb, ts):
        agent.reset({"n_arms": 2}, seed=0)
        for length in range(1, 4):
            agent.act({"step": length}, history[:length])
    greedy_history = [
        {"action": {"arm": a}, "reward": r}
        for a, r in [(0, 1.0), (1, 1.0), (1, 0.0), (0, 0.0), (0, 0.0)]
    ]
    incremental, fresh = GreedyAgent(), GreedyAgent()
    for agent in (incremental, fresh):
        agent.reset({"n_arms": 2}, seed=0)
    for length in range(1, 6):
        incremental_action = incremental.act({"step": length}, greedy_history[:length])
    return {
        "ucb_counts": ucb._counts,
        "expected_ucb_counts": [2, 1],
        "thompson_alpha": ts._alpha,
        "expected_thompson_alpha": [2.0, 1.0],
        "thompson_beta": ts._beta,
        "expected_thompson_beta": [2.0, 2.0],
        "greedy_incremental_action": incremental_action,
        "greedy_fresh_action_same_history": fresh.act({"step": 5}, greedy_history),
    }


class InspectingAgent(Agent):
    def reset(self, task_spec, seed=None):
        self.task_spec = task_spec
        self.history_keys = []
        self.public_keys = []

    def act(self, observation, history):
        if history:
            self.history_keys = sorted(history[-1])
            self.public_keys = sorted(history[-1].get("public_info", {}))
        return {"arm": 0}


def boundary_probe():
    agent = InspectingAgent()
    run_episode(StagedBandit(stage_length=3), agent, horizon=2, seed=0)
    return {
        "agent_reset_metadata_keys": sorted(agent.task_spec),
        "agent_history_keys": agent.history_keys,
        "agent_public_info_keys": agent.public_keys,
        "note": "Interface exposure does not imply current LLM prompts render these fields.",
    }


class RetryClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return GenerationResult(
            text="invalid" if self.calls == 1 else '{"action":{"arm":0}}',
            input_tokens=10, output_tokens=5, total_tokens=15,
        )


def retry_probe():
    client = RetryClient()
    agent = LLMAgent(model_client=client, max_retries=1)
    agent.reset({"n_arms": 2, "description": "Choose an arm."})
    _, info = agent.act_with_info({"step": 0}, [])
    return {
        "fake_provider_calls": client.calls,
        "recorded_total_tokens": info.token_usage.total_tokens,
        "expected_total_tokens": 15 * client.calls,
    }


def paired_noise_probe():
    mismatches = []
    for seed in range(20):
        changed = StagedBandit(stage_length=10, stages=["constant", "constant"])
        stable = StagedBandit(stage_length=10, stages=["constant", "constant"], no_change=True)
        changed.reset(seed)
        stable.reset(seed)
        assert changed.oracle()["current_probs"] == stable.oracle()["current_probs"]
        prefix = [changed.step({"arm": 0}).reward for _ in range(10)]
        control = [stable.step({"arm": 0}).reward for _ in range(10)]
        if prefix != control:
            mismatches.append(seed)
    return {
        "seeds_checked": 20,
        "different_pre_change_reward_prefixes": len(mismatches),
        "mismatch_seeds": mismatches,
        "note": "Identical initial probabilities/actions/seed; regime sampling shares the reward RNG.",
    }


def main():
    print(json.dumps({
        "baseline_updates": baseline_probe(),
        "information_boundary": boundary_probe(),
        "retry_accounting": retry_probe(),
        "paired_randomness": paired_noise_probe(),
    }, indent=2))


if __name__ == "__main__":
    main()
