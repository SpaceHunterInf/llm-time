from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_rl_engine.types import Action, Observation, StepResult


@dataclass
class ActionSpace:
    """Describes valid actions for an environment."""

    name: str
    schema: dict[str, Any]
    description: str

    def validate(self, action: Action) -> tuple[bool, str | None]:
        if not isinstance(action, dict):
            return False, "Action must be a JSON object."
        return self._validate_impl(action)

    def _validate_impl(self, action: Action) -> tuple[bool, str | None]:
        return True, None


class Env(ABC):
    """Gym-like environment interface."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    @abstractmethod
    def oracle(self) -> dict[str, Any]:
        """Hidden parameters for analysis only — never shown to agents."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...

    def render_task_description(self) -> str:
        return self.metadata.get("description", "")