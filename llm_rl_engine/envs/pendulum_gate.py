from __future__ import annotations

import math
import random
from typing import Any

from llm_rl_engine.envs.base import ActionSpace, Env
from llm_rl_engine.types import Action, Observation, StepResult

VALID_ACTIONS = ("wait", "cross")


class PendulumGate(Env):
    """Dynamic timing task: cross a gate blocked by a swinging pendulum.

    Environment time advances by a fixed ``dt`` every agent turn (independent of
    action). Deliberation is frozen — LLM thinking consumes zero simulation time.
    """

    def __init__(
        self,
        period_range: tuple[float, float] = (8.0, 16.0),
        danger_window: float = 0.35,
        dt: float = 0.6,
        episode_max_time_s: float = 30.0,
        wait_cost: float = -0.02,
        success_reward: float = 10.0,
        hit_penalty: float = -1.0,
        max_lives: int = 3,
        invalid_penalty: float = -0.1,
        phase_shift_time_range: tuple[float, float] = (10.0, 20.0),
        phase_shift_delta_range: tuple[float, float] = (math.pi / 4, math.pi),
    ) -> None:
        self.period_range = period_range
        self.danger_window = danger_window
        self.dt = dt
        self.episode_max_time_s = episode_max_time_s
        self.wait_cost = wait_cost
        self.success_reward = success_reward
        self.hit_penalty = hit_penalty
        self.max_lives = max_lives
        self.invalid_penalty = invalid_penalty
        self.phase_shift_time_range = phase_shift_time_range
        self.phase_shift_delta_range = phase_shift_delta_range
        self._rng = random.Random()
        self._period: float = 10.0
        self._phase: float = 0.0
        self._time_s: float = 0.0
        self._step: int = 0
        self._lives: int = max_lives
        self._done: bool = False
        self._last_action: Action | None = None
        self._last_reward: float | None = None
        self._cross_attempts: int = 0
        self._unsafe_crosses: int = 0
        self._phase_shift_time: float = 15.0
        self._phase_shift_delta: float = 0.0
        self._phase_shifted: bool = False

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="pendulum_action",
            schema={"action": "one of wait | cross"},
            description='Return {"action": "wait"} or {"action": "cross"}.',
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "PendulumGate",
            "description": (
                "Cross a gate guarded by a swinging pendulum. Each turn you receive "
                "the pendulum position and velocity at the current simulation time. "
                "The period and phase are hidden; the rhythm may shift mid-episode. "
                "Cross when |position| is below the danger threshold (0.35); "
                "otherwise you lose a life. Waiting has a small cost. "
                "Simulation time advances by a fixed interval every turn regardless "
                "of your action; deliberation takes no simulation time."
            ),
            "action_space": self.action_space.description,
            "max_lives": self.max_lives,
            "dt": self.dt,
            "episode_max_time_s": self.episode_max_time_s,
            "danger_window": self.danger_window,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        lo, hi = self.period_range
        self._period = self._rng.uniform(lo, hi)
        self._phase = self._rng.uniform(0, 2 * math.pi)
        shift_lo, shift_hi = self.phase_shift_time_range
        self._phase_shift_time = self._rng.uniform(shift_lo, shift_hi)
        delta_lo, delta_hi = self.phase_shift_delta_range
        self._phase_shift_delta = self._rng.uniform(delta_lo, delta_hi)
        self._phase_shifted = False
        self._time_s = 0.0
        self._step = 0
        self._lives = self.max_lives
        self._done = False
        self._last_action = None
        self._last_reward = None
        self._cross_attempts = 0
        self._unsafe_crosses = 0
        return self._make_observation(
            "Episode started. Pendulum period and phase are hidden. "
            "You see position and velocity each turn. Choose wait or cross.",
            public_state=self._public_state(),
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

        if parsed == "cross":
            result = self._handle_cross()
            if result.done:
                return result
        else:
            result = self._handle_wait()

        self._advance_time()
        if self._done:
            return StepResult(
                observation=self._make_observation(
                    "Episode ended: time limit reached.",
                    public_state=self._public_state(event="timeout"),
                ),
                reward=0.0,
                done=True,
                info={"event": "timeout"},
            )

        obs = self._make_observation(
            result.observation.get("message", ""),
            public_state=self._public_state(),
        )
        return StepResult(
            observation=obs,
            reward=result.reward,
            done=False,
            info=result.info,
        )

    def oracle(self) -> dict[str, Any]:
        pos = self._pendulum_position(self._time_s)
        vel = self._pendulum_velocity(self._time_s)
        return {
            "period": self._period,
            "phase": self._phase,
            "time_s": self._time_s,
            "danger_window": self.danger_window,
            "position_now": pos,
            "velocity_now": vel,
            "in_danger_now": self._is_danger(self._time_s),
            "lives": self._lives,
            "cross_attempts": self._cross_attempts,
            "unsafe_crosses": self._unsafe_crosses,
            "phase_shift_time": self._phase_shift_time,
            "phase_shift_delta": self._phase_shift_delta,
            "phase_shifted": self._phase_shifted,
            "dt": self.dt,
            "episode_max_time_s": self.episode_max_time_s,
        }

    def _parse_action(self, action: Action) -> tuple[str | None, str | None]:
        if not isinstance(action, dict) or "action" not in action:
            return None, "Missing 'action' field."
        val = action["action"]
        if not isinstance(val, str) or val not in VALID_ACTIONS:
            return None, f"Action must be one of {VALID_ACTIONS}."
        return val, None

    def _pendulum_position(self, t: float) -> float:
        return math.sin(2 * math.pi * t / max(self._period, 1e-6) + self._phase)

    def _pendulum_velocity(self, t: float) -> float:
        return (
            (2 * math.pi / max(self._period, 1e-6))
            * math.cos(2 * math.pi * t / max(self._period, 1e-6) + self._phase)
        )

    def _is_danger(self, t: float) -> bool:
        return abs(self._pendulum_position(t)) >= self.danger_window

    def _public_state(self, event: str | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "time_s": round(self._time_s, 2),
            "position": round(self._pendulum_position(self._time_s), 2),
            "velocity": round(self._pendulum_velocity(self._time_s), 2),
            "lives": self._lives,
        }
        if event:
            state["event"] = event
        return state

    def _handle_wait(self) -> StepResult:
        self._last_reward = self.wait_cost
        obs = self._make_observation(
            "You waited.",
            public_state={**self._public_state(event="waited"), "lives": self._lives},
        )
        return StepResult(
            observation=obs,
            reward=self.wait_cost,
            done=False,
            info={"event": "wait"},
        )

    def _handle_cross(self) -> StepResult:
        self._cross_attempts += 1
        in_danger = self._is_danger(self._time_s)
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
                f"You were hit by the pendulum at t={self._time_s:.2f}s. "
                f"Lives remaining: {self._lives}.",
                public_state={
                    **self._public_state(event="hit"),
                    "reward": self.hit_penalty,
                    "episode_status": status,
                },
            )
            if self._done:
                return StepResult(
                    observation=obs,
                    reward=self.hit_penalty,
                    done=True,
                    info={"event": "hit", "unsafe": True, "lives": self._lives},
                )
            self._advance_time()
            if self._done:
                return StepResult(
                    observation=self._make_observation(
                        "Episode ended: time limit reached after hit.",
                        public_state=self._public_state(event="timeout"),
                    ),
                    reward=self.hit_penalty,
                    done=True,
                    info={"event": "hit", "unsafe": True, "lives": self._lives},
                )
            next_obs = self._make_observation(
                obs["message"],
                public_state=self._public_state(),
            )
            return StepResult(
                observation=next_obs,
                reward=self.hit_penalty,
                done=False,
                info={"event": "hit", "unsafe": True, "lives": self._lives},
            )

        self._done = True
        self._last_reward = self.success_reward
        obs = self._make_observation(
            f"You crossed safely at t={self._time_s:.2f}s!",
            public_state={
                **self._public_state(event="success"),
                "reward": self.success_reward,
                "episode_status": "done",
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
        self._advance_time()
        obs = self._make_observation(
            f"Invalid action: {error}",
            public_state=self._public_state(event="invalid"),
        )
        done = self._done
        return StepResult(
            observation=obs,
            reward=self.invalid_penalty,
            done=done,
            info={"error": error, "invalid_action": True},
        )

    def _advance_time(self) -> None:
        self._time_s += self.dt
        self._maybe_apply_phase_shift()
        if self._time_s >= self.episode_max_time_s:
            self._done = True

    def _maybe_apply_phase_shift(self) -> None:
        if not self._phase_shifted and self._time_s >= self._phase_shift_time:
            self._phase = (self._phase + self._phase_shift_delta) % (2 * math.pi)
            self._phase_shifted = True

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