from __future__ import annotations

import random
from typing import Any

from llm_rl_engine.envs.base import ActionSpace, Env
from llm_rl_engine.types import Action, Observation, StepResult

VALID_ACTIONS = ("wait", "go")
LIGHT_COLORS = ("green", "yellow", "red")


class TrafficLightGate(Env):
    """Cross an intersection once while a hidden cyclic traffic light runs.

    Simulation time advances by a fixed ``dt`` every agent turn. Observations
    are deterministic: the agent sees exact ``time_s`` and ``light_color``.
    """

    def __init__(
        self,
        period_range: tuple[float, float] = (8.0, 16.0),
        green_fraction: float = 0.30,
        yellow_fraction: float = 0.15,
        dt: float = 0.6,
        episode_max_time_s: float = 30.0,
        wait_cost: float = -0.02,
        success_reward: float = 10.0,
        hit_penalty: float = -1.0,
        max_lives: int = 3,
        invalid_penalty: float = -0.1,
        phase_shift_time_range: tuple[float, float] | None = (12.0, 22.0),
        phase_shift_delta_range: tuple[float, float] = (0.25, 0.45),
    ) -> None:
        if green_fraction + yellow_fraction >= 1.0:
            raise ValueError("green_fraction + yellow_fraction must be < 1.")
        self.period_range = period_range
        self.green_fraction = green_fraction
        self.yellow_fraction = yellow_fraction
        self.red_fraction = 1.0 - green_fraction - yellow_fraction
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
        self._phase_offset: float = 0.0
        self._time_s: float = 0.0
        self._step: int = 0
        self._lives: int = max_lives
        self._done: bool = False
        self._last_action: Action | None = None
        self._last_reward: float | None = None
        self._go_attempts: int = 0
        self._unsafe_goes: int = 0
        self._phase_shift_time: float | None = None
        self._phase_shift_delta: float = 0.0
        self._phase_shifted: bool = False

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="traffic_action",
            schema={"action": "one of wait | go"},
            description='Return {"action": "wait"} or {"action": "go"}.',
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "TrafficLightGate",
            "description": (
                "Cross an intersection guarded by a hidden cyclic traffic light. "
                "Each turn you see the current simulation time and exact light color "
                "(green, yellow, or red). Going on green succeeds and ends the episode; "
                "going on yellow or red costs a life. Waiting has a small cost. "
                "The light cycle period and phase are hidden and may shift mid-episode. "
                "Simulation time advances by a fixed interval every turn."
            ),
            "action_space": self.action_space.description,
            "max_lives": self.max_lives,
            "dt": self.dt,
            "episode_max_time_s": self.episode_max_time_s,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        lo, hi = self.period_range
        self._period = self._rng.uniform(lo, hi)
        self._phase_offset = self._rng.uniform(0.0, 1.0)
        self._phase_shift_time = None
        self._phase_shift_delta = 0.0
        self._phase_shifted = False
        if self.phase_shift_time_range is not None:
            shift_lo, shift_hi = self.phase_shift_time_range
            self._phase_shift_time = self._rng.uniform(shift_lo, shift_hi)
            delta_lo, delta_hi = self.phase_shift_delta_range
            self._phase_shift_delta = self._rng.uniform(delta_lo, delta_hi)
        self._time_s = 0.0
        self._step = 0
        self._lives = self.max_lives
        self._done = False
        self._last_action = None
        self._last_reward = None
        self._go_attempts = 0
        self._unsafe_goes = 0
        return self._make_observation(
            "Episode started. The traffic light cycle is hidden. Choose wait or go.",
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

        if parsed == "go":
            result = self._handle_go()
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
                reward=result.reward,
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
        color = self._light_color(self._time_s)
        return {
            "period": self._period,
            "phase_offset": self._phase_offset,
            "green_fraction": self.green_fraction,
            "yellow_fraction": self.yellow_fraction,
            "red_fraction": self.red_fraction,
            "time_s": self._time_s,
            "light_color": color,
            "safe_to_go": color == "green",
            "lives": self._lives,
            "go_attempts": self._go_attempts,
            "unsafe_goes": self._unsafe_goes,
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

    def _cycle_fraction(self, t: float) -> float:
        return ((t / max(self._period, 1e-6)) + self._phase_offset) % 1.0

    def _light_color(self, t: float) -> str:
        frac = self._cycle_fraction(t)
        if frac < self.green_fraction:
            return "green"
        if frac < self.green_fraction + self.yellow_fraction:
            return "yellow"
        return "red"

    def _public_state(self, event: str | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "time_s": round(self._time_s, 2),
            "light_color": self._light_color(self._time_s),
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

    def _handle_go(self) -> StepResult:
        self._go_attempts += 1
        color = self._light_color(self._time_s)
        if color == "green":
            self._done = True
            self._last_reward = self.success_reward
            obs = self._make_observation(
                f"You crossed on green at t={self._time_s:.2f}s!",
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
                info={"event": "success", "unsafe": False, "light_color": color},
            )

        self._unsafe_goes += 1
        self._lives -= 1
        self._last_reward = self.hit_penalty
        if self._lives <= 0:
            self._done = True
            status = "dead"
        else:
            status = "alive"
        obs = self._make_observation(
            f"You went on {color} at t={self._time_s:.2f}s. Lives remaining: {self._lives}.",
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
                info={"event": "hit", "unsafe": True, "light_color": color, "lives": self._lives},
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
                info={"event": "hit", "unsafe": True, "light_color": color, "lives": self._lives},
            )
        next_obs = self._make_observation(
            obs["message"],
            public_state=self._public_state(),
        )
        return StepResult(
            observation=next_obs,
            reward=self.hit_penalty,
            done=False,
            info={"event": "hit", "unsafe": True, "light_color": color, "lives": self._lives},
        )

    def _invalid_step(self, error: str) -> StepResult:
        self._step += 1
        self._advance_time()
        obs = self._make_observation(
            f"Invalid action: {error}",
            public_state=self._public_state(event="invalid"),
        )
        return StepResult(
            observation=obs,
            reward=self.invalid_penalty,
            done=self._done,
            info={"error": error, "invalid_action": True},
        )

    def _advance_time(self) -> None:
        self._time_s += self.dt
        self._maybe_apply_phase_shift()
        if self._time_s >= self.episode_max_time_s:
            self._done = True

    def _maybe_apply_phase_shift(self) -> None:
        if (
            self._phase_shift_time is not None
            and not self._phase_shifted
            and self._time_s >= self._phase_shift_time
        ):
            self._phase_offset = (self._phase_offset + self._phase_shift_delta) % 1.0
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