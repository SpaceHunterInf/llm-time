from __future__ import annotations

import random
from typing import Any

from llm_rl_engine.envs.base import ActionSpace, Env
from llm_rl_engine.types import Action, Observation, StepResult


class NumberGuess(Env):
    """Static task: guess a hidden integer with comparison feedback."""

    def __init__(
        self,
        target_min: int = 0,
        target_max: int = 100,
        feedback: str = "exact_comparison",
        reward_mode: str = "sparse",
        invalid_penalty: float = -0.1,
    ) -> None:
        self.target_min = target_min
        self.target_max = target_max
        self.feedback = feedback
        self.reward_mode = reward_mode
        self.invalid_penalty = invalid_penalty
        self._rng = random.Random()
        self._target: int = 0
        self._step: int = 0
        self._last_action: Action | None = None
        self._last_reward: float | None = None
        self._done: bool = False

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="guess",
            schema={"guess": "integer"},
            description=(
                f"Return {{\"guess\": <integer>}} where "
                f"{self.target_min} <= guess <= {self.target_max}."
            ),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "NumberGuess",
            "description": (
                f"Guess a hidden integer between {self.target_min} and "
                f"{self.target_max}. After each guess you receive feedback "
                "about whether your guess is too low, too high, or correct."
            ),
            "action_space": self.action_space.description,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "feedback": self.feedback,
            "reward_mode": self.reward_mode,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self._target = self._rng.randint(self.target_min, self.target_max)
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        return self._make_observation("Episode started. Make your first guess.")

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_observation("Episode already finished."),
                reward=0.0,
                done=True,
                info={"error": "episode_done"},
            )

        valid, err = self.action_space.validate(action)
        if not valid or "guess" not in action:
            return self._invalid_step(err or "Missing 'guess' field.")

        guess = action["guess"]
        if not isinstance(guess, int) or not (self.target_min <= guess <= self.target_max):
            return self._invalid_step(
                f"Guess must be an integer in [{self.target_min}, {self.target_max}]."
            )

        self._step += 1
        self._last_action = action

        if guess == self._target:
            feedback = "correct"
            reward = 1.0
            self._done = True
        elif guess < self._target:
            feedback = "too_low"
            reward = self._compute_dense_reward(guess)
        else:
            feedback = "too_high"
            reward = self._compute_dense_reward(guess)

        self._last_reward = reward
        public_state = {"feedback": feedback}
        if self.feedback == "distance_bucket":
            public_state["distance_bucket"] = self._distance_bucket(guess)

        obs = self._make_observation(
            f"Your guess was {guess}. Feedback: {feedback}.",
            public_state=public_state,
        )
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"feedback": feedback, "guess": guess},
        )

    def oracle(self) -> dict[str, Any]:
        return {"target": self._target}

    def _compute_dense_reward(self, guess: int) -> float:
        if self.reward_mode == "dense":
            span = max(self.target_max - self.target_min, 1)
            return -abs(guess - self._target) / span
        return 0.0

    def _distance_bucket(self, guess: int) -> str:
        dist = abs(guess - self._target)
        span = max(self.target_max - self.target_min, 1)
        ratio = dist / span
        if ratio > 0.5:
            return "very_far"
        if ratio > 0.25:
            return "far"
        if ratio > 0.1:
            return "near"
        return "very_near"

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
        return {
            "step": self._step,
            "message": message,
            "public_state": public_state or {},
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "done": self._done,
        }