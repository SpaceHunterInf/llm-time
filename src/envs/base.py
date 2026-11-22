"""Base multi-turn tool-calling environment protocol with strict public/private firewall."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PublicTaskSpec:
    """Information visible to the agent at the start of an episode (zero hidden state)."""

    env_name: str
    episode_idx: int
    timestamp: int
    domain_prompt: str
    task_prompt: str
    tool_schemas: list[dict[str, Any]]


@dataclass
class StepOutcome:
    """Result of executing a single tool call inside an episode."""

    observation: dict[str, Any]
    done: bool
    public_score: float | None = None  # Only revealed upon episode completion (0..100)
    # Evaluator-only fields — NEVER passed into agent prompts or harness memory!
    private_eval: dict[str, Any] = field(default_factory=dict)


class ParadigmEnv(ABC):
    """Abstract base class for sequential non-stationary reward-paradigm environments."""

    env_name: str = "base"
    max_turns_per_episode: int = 6

    @abstractmethod
    def num_episodes(self) -> int:
        """Total episodes in this continuous agent lifetime."""

    @abstractmethod
    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        """Initialize episode `episode_idx` and return strictly public task spec."""

    @abstractmethod
    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        """Execute a tool call in the active episode."""
