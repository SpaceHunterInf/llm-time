from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm_rl_engine.types import Action, Observation


@dataclass
class AgentStepInfo:
    """Optional per-step metadata from the agent (prompts, responses, etc.)."""

    prompt: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    parse_error: str | None = None


class Agent(ABC):
    @abstractmethod
    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        ...

    @abstractmethod
    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        ...

    def act_with_info(
        self, observation: Observation, history: list[dict[str, Any]]
    ) -> tuple[Action, AgentStepInfo]:
        action = self.act(observation, history)
        return action, AgentStepInfo()