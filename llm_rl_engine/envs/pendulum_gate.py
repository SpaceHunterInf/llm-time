from __future__ import annotations

import math
import random
from typing import Any

from llm_rl_engine.envs.base import ActionSpace, Env
from llm_rl_engine.types import Action, Observation, StepResult

VALID_ACTIONS = ("wait", "observe", "cross")
POSITION_HINTS = ("far_left", "left", "center", "right", "far_right")


class PendulumGate(Env):
    """Dynamic timing task: cross a gate blocked by a swinging pendulum."""

    def __init__(
        self,
        period_range: tuple[int, int] = (8, 16),
        danger_window: float = 0.35,
        wait_cost: float = -0.05,
        observe_cost: float = -0.05,
        success_reward: float = 10.0,
        hit_penalty: float = -1.0,
        max_lives: int = 3,
        invalid_penalty: float = -0.1,
    ) -> None:
        self.period_range = period_range
        self.danger_window = danger_window
        self.wait_cost = wait_cost
        self.observe_cost = observe_cost
        self.success_reward = success_reward
        self.hit_penalty = hit_penalty
        self.max_lives = max_lives
        self.invalid_penalty = invalid_penalty
        self._rng = random.Random()
        self._period: int = 10
        self._phase: float = 0.0
        self._step: int = 0
        self._lives: int = max_lives
        self._done: bool = False
        self._last_action: Action | None = None
        self._last_reward: float | None = None
        self._cross_attempts: int = 0
        self._unsafe_crosses: int = 0

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="pendulum_action",
            schema={"action": "one of wait | observe | cross"},
            description=(
                'Return {"action": "wait"}, {"action": "observe"}, or {"action": "cross"}.'
            ),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "PendulumGate",
            "description": (
                "Cross a gate guarded by a hidden swinging pendulum. You may wait, "
                "observe (noisy position hint), or attempt to cross. Crossing during "
                "the danger window costs a life; success ends the episode with a large reward."
            ),
            "action_space": self.action_space.description,
            "max_lives": self.max_lives,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        lo, hi = self.period_range
        self._period = self._rng.randint(lo, hi)
        self._phase = self._rng.uniform(0, 2 * math.pi)
        self._step = 0
        self._lives = self.max_lives
        self._done = False
        self._last_action = None
        self._last_reward = None
        self._cross_attempts = 0
        self._unsafe_crosses = 0
        return self._make_observation(
            "Episode started. The pendulum period and phase are hidden. Choose wait, observe, or cross.",
            public_state={"event": "episode_start", "lives": self._lives},
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_observation("Episode finished."),
                reward=0.0,
                done=True,
                info={"error": "episode_done"},
            )

        parsed, err = self._parse_action(action)
        if parsed is None:
            return self._invalid_step(err or "Invalid action.")

        self._step += 1
        self._last_action = {"action": parsed}

        if parsed == "wait":
            return self._handle_wait()
        if parsed == "observe":
            return self._handle_observe()
        return self._handle_cross()

    def oracle(self) -> dict[str, Any]:
        return {
            "period": self._period,
            "phase": self._phase,
            "danger_window": self.danger_window,
            "position_now": self._pendulum_position(self._step),
            "in_danger_now": self._is_danger(self._step),
            "lives": self._lives,
            "cross_attempts": self._cross_attempts,
            "unsafe_crosses": self._unsafe_crosses,
        }

    def _parse_action(self, action: Action) -> tuple[str | None, str | None]:
        if not isinstance(action, dict) or "action" not in action:
            return None, "Missing 'action' field."
        val = action["action"]
        if not isinstance(val, str) or val not in VALID_ACTIONS:
            return None, f"Action must be one of {VALID_ACTIONS}."
        return val, None

    def _pendulum_position(self, t: int) -> float:
        return math.sin(2 * math.pi * t / max(self._period, 1) + self._phase)

    def _is_danger(self, t: int) -> bool:
        return abs(self._pendulum_position(t)) >= self.danger_window

    def _position_hint(self, t: int) -> str:
        pos = self._pendulum_position(t)
        if pos <= -0.6:
            return "far_left"
        if pos <= -0.2:
            return "left"
        if pos < 0.2:
            return "center"
        if pos < 0.6:
            return "right"
        return "far_right"

    def _handle_wait(self) -> StepResult:
        self._last_reward = self.wait_cost
        obs = self._make_observation(
            "You waited.",
            public_state={"event": "waited", "lives": self._lives},
        )
        return StepResult(
            observation=obs,
            reward=self.wait_cost,
            done=False,
            info={"event": "wait"},
        )

    def _handle_observe(self) -> StepResult:
        hint = self._position_hint(self._step)
        if self._rng.random() < 0.15:
            hint = self._rng.choice(POSITION_HINTS)
        self._last_reward = self.observe_cost
        obs = self._make_observation(
            f"You observed the pendulum. Position hint: {hint}.",
            public_state={
                "event": "observed",
                "pendulum_position_hint": hint,
                "lives": self._lives,
            },
        )
        return StepResult(
            observation=obs,
            reward=self.observe_cost,
            done=False,
            info={"event": "observe", "hint": hint},
        )

    def _handle_cross(self) -> StepResult:
        self._cross_attempts += 1
        in_danger = self._is_danger(self._step)
        if in_danger:
            self._unsafe_crosses += 1
            self._lives -= 1
            self._last_reward = self.hit_penalty
            if self._lives <= 0:
                self._done = True
                status = "dead"
            else:
                status = "alive"
            obs = self._make_observation(
                f"You were hit by the pendulum. Lives remaining: {self._lives}.",
                public_state={
                    "event": "hit",
                    "reward": self.hit_penalty,
                    "episode_status": status,
                    "lives": self._lives,
                },
            )
            return StepResult(
                observation=obs,
                reward=self.hit_penalty,
                done=self._done,
                info={"event": "hit", "unsafe": True, "lives": self._lives},
            )

        self._done = True
        self._last_reward = self.success_reward
        obs = self._make_observation(
            "You crossed safely!",
            public_state={
                "event": "success",
                "reward": self.success_reward,
                "episode_status": "done",
                "lives": self._lives,
            },
        )
        return StepResult(
            observation=obs,
            reward=self.success_reward,
            done=True,
            info={"event": "success", "unsafe": False},
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
        return {
            "step": self._step,
            "message": message,
            "public_state": public_state or {},
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "done": self._done,
        }