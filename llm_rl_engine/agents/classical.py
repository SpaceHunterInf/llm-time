from __future__ import annotations

import math
import random
from typing import Any

from llm_rl_engine.agents.base import Agent
from llm_rl_engine.types import Action, Observation


class RandomAgent(Agent):
    def __init__(self, name: str = "random") -> None:
        self.name = name
        self._rng = random.Random()
        self._task_spec: dict[str, Any] = {}

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        if seed is not None:
            self._rng.seed(seed)

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        env_name = self._task_spec.get("name", "")
        if env_name == "NumberGuess":
            lo = self._task_spec.get("target_min", 0)
            hi = self._task_spec.get("target_max", 100)
            return {"guess": self._rng.randint(lo, hi)}
        n_arms = self._task_spec.get("n_arms", 4)
        return {"arm": self._rng.randint(0, n_arms - 1)}


class GreedyAgent(Agent):
    def __init__(self, name: str = "greedy") -> None:
        self.name = name
        self._task_spec: dict[str, Any] = {}
        self._arm_rewards: dict[int, list[float]] = {}
        self._arm_counts: dict[int, int] = {}

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        self._arm_rewards = {}
        self._arm_counts = {}

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        env_name = self._task_spec.get("name", "")
        if env_name == "NumberGuess":
            return self._number_guess_act(history)
        return self._bandit_act(history)

    def _number_guess_act(self, history: list[dict[str, Any]]) -> Action:
        lo = self._task_spec.get("target_min", 0)
        hi = self._task_spec.get("target_max", 100)
        if not history:
            return {"guess": (lo + hi) // 2}
        last = history[-1]
        action = last.get("action", {})
        info = last.get("private_info", {}) or last.get("public_info", {})
        guess = action.get("guess")
        feedback = info.get("feedback")
        if guess is None or feedback is None:
            return {"guess": (lo + hi) // 2}
        if feedback == "too_low":
            return {"guess": min(hi, guess + max(1, (hi - guess) // 2))}
        if feedback == "too_high":
            return {"guess": max(lo, guess - max(1, (guess - lo) // 2))}
        return {"guess": guess}

    def _bandit_act(self, history: list[dict[str, Any]]) -> Action:
        n_arms = self._task_spec.get("n_arms", 4)
        for rec in history:
            action = rec.get("action", {})
            arm = action.get("arm")
            reward = rec.get("reward", 0.0)
            if arm is not None:
                self._arm_rewards.setdefault(arm, []).append(reward)
                self._arm_counts[arm] = self._arm_counts.get(arm, 0) + 1

        if not self._arm_rewards:
            return {"arm": 0}

        best_arm = max(
            range(n_arms),
            key=lambda a: (
                sum(self._arm_rewards.get(a, [])) / max(len(self._arm_rewards.get(a, [])), 1),
                -a,
            ),
        )
        return {"arm": best_arm}


class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, epsilon: float = 0.1, name: str = "epsilon_greedy") -> None:
        super().__init__(name=name)
        self.epsilon = epsilon
        self._rng = random.Random()

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        super().reset(task_spec, seed)
        if seed is not None:
            self._rng.seed(seed)

    def _bandit_act(self, history: list[dict[str, Any]]) -> Action:
        n_arms = self._task_spec.get("n_arms", 4)
        if self._rng.random() < self.epsilon or not history:
            return {"arm": self._rng.randint(0, n_arms - 1)}
        return super()._bandit_act(history)


class UCBAgent(Agent):
    def __init__(self, c: float = 2.0, name: str = "ucb") -> None:
        self.name = name
        self.c = c
        self._task_spec: dict[str, Any] = {}
        self._counts: list[int] = []
        self._rewards: list[float] = []

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        n = task_spec.get("n_arms", 4)
        self._counts = [0] * n
        self._rewards = [0.0] * n

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        n_arms = self._task_spec.get("n_arms", 4)
        for rec in history:
            action = rec.get("action", {})
            arm = action.get("arm")
            if arm is not None and 0 <= arm < n_arms:
                self._counts[arm] += 1
                self._rewards[arm] += rec.get("reward", 0.0)

        for arm in range(n_arms):
            if self._counts[arm] == 0:
                return {"arm": arm}

        t = max(sum(self._counts), 1)
        scores = []
        for arm in range(n_arms):
            mean = self._rewards[arm] / self._counts[arm]
            bonus = self.c * math.sqrt(math.log(t) / self._counts[arm])
            scores.append((mean + bonus, arm))
        return {"arm": max(scores)[1]}


class ThompsonSamplingAgent(Agent):
    def __init__(self, name: str = "thompson_sampling") -> None:
        self.name = name
        self._task_spec: dict[str, Any] = {}
        self._rng = random.Random()
        self._alpha: list[float] = []
        self._beta: list[float] = []

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        if seed is not None:
            self._rng.seed(seed)
        n = task_spec.get("n_arms", 4)
        self._alpha = [1.0] * n
        self._beta = [1.0] * n

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        n_arms = self._task_spec.get("n_arms", 4)
        for rec in history:
            action = rec.get("action", {})
            arm = action.get("arm")
            if arm is not None and 0 <= arm < n_arms:
                if rec.get("reward", 0.0) >= 0.5:
                    self._alpha[arm] += 1
                else:
                    self._beta[arm] += 1

        samples = [
            self._rng.betavariate(self._alpha[a], self._beta[a]) for a in range(n_arms)
        ]
        return {"arm": max(range(n_arms), key=lambda a: samples[a])}


class SlidingWindowUCBAgent(Agent):
    def __init__(self, window: int = 10, c: float = 2.0, name: str = "sliding_window_ucb") -> None:
        self.name = name
        self.window = window
        self.c = c
        self._task_spec: dict[str, Any] = {}
        self._history: list[tuple[int, float]] = []

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        self._history = []

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        n_arms = self._task_spec.get("n_arms", 4)
        self._history = []
        for rec in history[-self.window :]:
            action = rec.get("action", {})
            arm = action.get("arm")
            if arm is not None:
                self._history.append((arm, rec.get("reward", 0.0)))

        if not self._history:
            return {"arm": 0}

        counts = [0] * n_arms
        rewards = [0.0] * n_arms
        for arm, reward in self._history:
            counts[arm] += 1
            rewards[arm] += reward

        for arm in range(n_arms):
            if counts[arm] == 0:
                return {"arm": arm}

        t = max(len(self._history), 1)
        scores = []
        for arm in range(n_arms):
            mean = rewards[arm] / counts[arm]
            bonus = self.c * math.sqrt(math.log(t) / counts[arm])
            scores.append((mean + bonus, arm))
        return {"arm": max(scores)[1]}


class OracleAgent(Agent):
    """Upper-bound agent with access to hidden parameters (analysis only)."""

    def __init__(self, name: str = "oracle") -> None:
        self.name = name
        self._task_spec: dict[str, Any] = {}
        self._oracle: dict[str, Any] = {}
        self._step: int = 0

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        self._oracle = {}
        self._step = 0

    def set_oracle(self, oracle: dict[str, Any]) -> None:
        self._oracle = oracle

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        self._step = observation.get("step", self._step)
        env_name = self._task_spec.get("name", "")
        if env_name == "NumberGuess":
            return {"guess": self._oracle.get("target", 0)}
        if env_name in ("StationaryBandit", "ReversalBandit"):
            probs = self._oracle.get("current_probs") or self._oracle.get("probs", [])
            if probs:
                return {"arm": max(range(len(probs)), key=lambda i: probs[i])}
            return {"arm": self._oracle.get("best_arm", 0)}
        return {"arm": 0}


CLASSICAL_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "epsilon_greedy": EpsilonGreedyAgent,
    "ucb": UCBAgent,
    "thompson_sampling": ThompsonSamplingAgent,
    "sliding_window_ucb": SlidingWindowUCBAgent,
    "oracle": OracleAgent,
}


def make_classical_agent(config: dict[str, Any]) -> Agent:
    name = config.get("name", "random")
    if name not in CLASSICAL_REGISTRY:
        raise ValueError(f"Unknown classical agent: {name}")
    cls = CLASSICAL_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if "epsilon" in kwargs and name == "epsilon_greedy":
        return EpsilonGreedyAgent(epsilon=kwargs["epsilon"], name=name)
    if "window" in kwargs and name == "sliding_window_ucb":
        return SlidingWindowUCBAgent(
            window=kwargs.get("window", 10),
            c=kwargs.get("c", 2.0),
            name=name,
        )
    if "c" in kwargs and name == "ucb":
        return UCBAgent(c=kwargs["c"], name=name)
    return cls(name=name)