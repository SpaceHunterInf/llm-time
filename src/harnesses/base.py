"""Base interface for sequential agent memory / self-improvement harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.base import PublicTaskSpec


@dataclass(frozen=True)
class PublicEpisodeRecord:
    """Strictly public episode trace passed to a harness after episode completion."""

    env_name: str
    episode_idx: int
    timestamp: int
    task_prompt: str
    steps: list[dict[str, Any]]
    public_score: float


class EpistemicHarness(ABC):
    """Abstract harness governing cross-episode experience extraction and revision."""

    name: str = "base"

    @abstractmethod
    def build_context(self, spec: PublicTaskSpec) -> str:
        """Return persistent memory / policy guidance injected into the agent's system prompt."""

    @abstractmethod
    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        """Update internal state from public outcome; return diagnostic metadata (e.g. reflection cost)."""
