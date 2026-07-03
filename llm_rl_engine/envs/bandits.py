from __future__ import annotations

import math
import random
from typing import Any

from llm_rl_engine.envs.base import ActionSpace, Env
from llm_rl_engine.types import Action, Observation, StepResult

BANDIT_ENV_NAMES = (
    "StationaryBandit",
    "ReversalBandit",
    "PeriodicBandit",
    "DriftingBandit",
)


class _BanditBase(Env):
    """Shared bandit mechanics."""

    def __init__(
        self,
        n_arms: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        invalid_penalty: float = -0.1,
    ) -> None:
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
        self.invalid_penalty = invalid_penalty
        self._rng = random.Random()
        self._probs: list[float] = []
        self._step: int = 0
        self._last_action: Action | None = None
        self._last_reward: float | None = None
        self._done: bool = False

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="arm",
            schema={"arm": f"integer in [0, {self.n_arms - 1}]"},
            description=(
                f"Return {{\"arm\": <integer>}} where 0 <= arm < {self.n_arms}."
            ),
        )

    def _validate_arm(self, action: Action) -> tuple[int | None, str | None]:
        valid, err = self.action_space.validate(action)
        if not valid:
            return None, err
        if "arm" not in action:
            return None, "Missing 'arm' field."
        arm = action["arm"]
        if not isinstance(arm, int) or not (0 <= arm < self.n_arms):
            return None, f"Arm must be an integer in [0, {self.n_arms - 1}]."
        return arm, None

    def _reward_prob(self, arm: int) -> float:
        return self._probs[arm]

    def _on_pre_step(self) -> None:
        """Hook for time-varying bandits to update probabilities."""
        return None

    def _pull_arm(self, arm: int) -> tuple[float, StepResult]:
        self._on_pre_step()
        self._step += 1
        self._last_action = {"arm": arm}
        reward = 1.0 if self._rng.random() < self._reward_prob(arm) else 0.0
        self._last_reward = reward
        obs = self._make_observation(
            f"You pulled arm {arm} and received reward {reward:.0f}.",
            public_state={"chosen_arm": arm, "reward": reward},
        )
        return reward, StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"chosen_arm": arm, "reward": reward, "step_probs": list(self._probs)},
        )

    def _invalid_step(self, error: str) -> StepResult:
        self._step += 1
        obs = self._make_observation(f"Invalid action: {error}")
        return StepResult(
            observation=obs,
            reward=self.invalid_penalty,
            done=False,
            info={"error": error, "invalid_action": True},
        )

    def _make_observation(
        self,
        message: str,
        public_state: dict[str, Any] | None = None,
    ) -> Observation:
        state = {"step": self._step}
        if public_state:
            state.update(public_state)
        return {
            "step": self._step,
            "message": message,
            "public_state": state,
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "done": self._done,
        }

    def _sample_probs(self) -> list[float]:
        return [self._rng.betavariate(self.alpha, self.beta) for _ in range(self.n_arms)]

    def _best_arm(self, probs: list[float] | None = None) -> int:
        p = probs if probs is not None else self._probs
        return max(range(len(p)), key=lambda i: p[i])


class StationaryBandit(_BanditBase):
    """Static task: fixed Bernoulli reward probabilities per arm."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "StationaryBandit",
            "description": (
                f"A multi-armed bandit with {self.n_arms} arms. Each arm has a "
                "fixed but unknown reward probability. Pull one arm per step to "
                "maximize cumulative reward."
            ),
            "action_space": self.action_space.description,
            "n_arms": self.n_arms,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self._probs = self._sample_probs()
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        return self._make_observation(
            f"Episode started. Choose an arm from 0 to {self.n_arms - 1}."
        )

    def step(self, action: Action) -> StepResult:
        arm, err = self._validate_arm(action)
        if arm is None:
            return self._invalid_step(err or "Invalid arm.")
        _, result = self._pull_arm(arm)
        return result

    def oracle(self) -> dict[str, Any]:
        return {
            "probs": list(self._probs),
            "best_arm": self._best_arm(),
        }


class ReversalBandit(_BanditBase):
    """Dynamic task: reward probabilities flip at a hidden reversal time."""

    def __init__(
        self,
        n_arms: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        reversal_schedule: str = "fixed_hidden_time",
        reversal_time_range: tuple[int, int] = (15, 25),
        invalid_penalty: float = -0.1,
    ) -> None:
        super().__init__(n_arms=n_arms, alpha=alpha, beta=beta, invalid_penalty=invalid_penalty)
        self.reversal_schedule = reversal_schedule
        self.reversal_time_range = reversal_time_range
        self._pre_probs: list[float] = []
        self._post_probs: list[float] = []
        self._tau: int = 0
        self._reversed: bool = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "ReversalBandit",
            "description": (
                f"A multi-armed bandit with {self.n_arms} arms. Reward "
                "probabilities may change at some point during the episode. "
                "Adapt your strategy as evidence accumulates."
            ),
            "action_space": self.action_space.description,
            "n_arms": self.n_arms,
            "reversal_schedule": self.reversal_schedule,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self._pre_probs = self._sample_probs()
        self._post_probs = self._make_post_probs(self._pre_probs)
        lo, hi = self.reversal_time_range
        self._tau = self._rng.randint(lo, hi)
        self._reversed = False
        self._probs = list(self._pre_probs)
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        return self._make_observation(
            f"Episode started. Choose an arm from 0 to {self.n_arms - 1}."
        )

    def step(self, action: Action) -> StepResult:
        arm, err = self._validate_arm(action)
        if arm is None:
            return self._invalid_step(err or "Invalid arm.")

        reward, result = self._pull_arm(arm)
        self._maybe_reverse()
        result.info["reversed"] = self._reversed
        return result

    def oracle(self) -> dict[str, Any]:
        return {
            "pre_probs": list(self._pre_probs),
            "post_probs": list(self._post_probs),
            "tau": self._tau,
            "reversed": self._reversed,
            "current_probs": list(self._probs),
            "best_arm": self._best_arm(),
        }

    def _make_post_probs(self, pre: list[float]) -> list[float]:
        post = list(pre)
        best = self._best_arm(pre)
        worst = min(range(len(pre)), key=lambda i: pre[i])
        post[best], post[worst] = post[worst], post[best]
        for i in range(len(post)):
            noise = self._rng.uniform(-0.05, 0.05)
            post[i] = min(0.95, max(0.05, post[i] + noise))
        return post

    def _maybe_reverse(self) -> None:
        if self._reversed:
            return
        if self._step >= self._tau:
            self._probs = list(self._post_probs)
            self._reversed = True


class PeriodicBandit(_BanditBase):
    """Dynamic task: arm reward probabilities vary periodically over time."""

    def __init__(
        self,
        n_arms: int = 4,
        shared_period: bool = True,
        period_range: tuple[int, int] = (8, 20),
        amplitude_range: tuple[float, float] = (0.15, 0.35),
        base_range: tuple[float, float] = (0.25, 0.65),
        invalid_penalty: float = -0.1,
    ) -> None:
        super().__init__(n_arms=n_arms, invalid_penalty=invalid_penalty)
        self.shared_period = shared_period
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.base_range = base_range
        self._bases: list[float] = []
        self._amps: list[float] = []
        self._phases: list[float] = []
        self._periods: list[int] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "PeriodicBandit",
            "description": (
                f"A multi-armed bandit with {self.n_arms} arms. Each arm's reward "
                "probability changes periodically over time. Infer the temporal "
                "pattern and exploit the best arm at each phase."
            ),
            "action_space": self.action_space.description,
            "n_arms": self.n_arms,
            "shared_period": self.shared_period,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        lo_p, hi_p = self.period_range
        lo_b, hi_b = self.base_range
        lo_a, hi_a = self.amplitude_range
        shared = self._rng.randint(lo_p, hi_p)
        self._periods = [shared if self.shared_period else self._rng.randint(lo_p, hi_p) for _ in range(self.n_arms)]
        self._bases = [self._rng.uniform(lo_b, hi_b) for _ in range(self.n_arms)]
        self._amps = [self._rng.uniform(lo_a, hi_a) for _ in range(self.n_arms)]
        self._phases = [self._rng.uniform(0, 2 * math.pi) for _ in range(self.n_arms)]
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        self._refresh_probs()
        return self._make_observation(
            f"Episode started. Choose an arm from 0 to {self.n_arms - 1}."
        )

    def step(self, action: Action) -> StepResult:
        arm, err = self._validate_arm(action)
        if arm is None:
            return self._invalid_step(err or "Invalid arm.")
        _, result = self._pull_arm(arm)
        return result

    def oracle(self) -> dict[str, Any]:
        return {
            "bases": list(self._bases),
            "amplitudes": list(self._amps),
            "phases": list(self._phases),
            "periods": list(self._periods),
            "current_probs": list(self._probs),
            "best_arm": self._best_arm(),
        }

    def _prob_formula(self, arm: int, t: int) -> float:
        period = max(self._periods[arm], 1)
        val = self._bases[arm] + self._amps[arm] * math.sin(
            2 * math.pi * t / period + self._phases[arm]
        )
        return min(0.99, max(0.01, val))

    def _refresh_probs(self) -> None:
        t = max(self._step, 0)
        self._probs = [self._prob_formula(i, t) for i in range(self.n_arms)]

    def _on_pre_step(self) -> None:
        self._refresh_probs()


class DriftingBandit(_BanditBase):
    """Dynamic task: reward probabilities drift via a random walk each step."""

    def __init__(
        self,
        n_arms: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        sigma: float = 0.03,
        invalid_penalty: float = -0.1,
    ) -> None:
        super().__init__(n_arms=n_arms, alpha=alpha, beta=beta, invalid_penalty=invalid_penalty)
        self.sigma = sigma
        self._prob_history: list[list[float]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "DriftingBandit",
            "description": (
                f"A multi-armed bandit with {self.n_arms} arms. Reward probabilities "
                "drift gradually over time. Track recent evidence and adapt continuously."
            ),
            "action_space": self.action_space.description,
            "n_arms": self.n_arms,
            "sigma": self.sigma,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self._probs = self._sample_probs()
        self._prob_history = [list(self._probs)]
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        return self._make_observation(
            f"Episode started. Choose an arm from 0 to {self.n_arms - 1}."
        )

    def step(self, action: Action) -> StepResult:
        arm, err = self._validate_arm(action)
        if arm is None:
            return self._invalid_step(err or "Invalid arm.")
        _, result = self._pull_arm(arm)
        return result

    def oracle(self) -> dict[str, Any]:
        return {
            "probs": list(self._probs),
            "prob_history": list(self._prob_history),
            "best_arm": self._best_arm(),
            "sigma": self.sigma,
        }

    def _on_pre_step(self) -> None:
        if self._step == 0:
            return
        for i in range(self.n_arms):
            noise = self._rng.gauss(0, self.sigma)
            self._probs[i] = min(0.95, max(0.05, self._probs[i] + noise))
        self._prob_history.append(list(self._probs))