from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Action = dict[str, Any]
Observation = dict[str, Any]


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TrajectoryRecord:
    step: int
    observation: Observation
    action: Action
    reward: float
    done: bool
    public_info: dict[str, Any] = field(default_factory=dict)
    private_info: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    parse_error: str | None = None
    token_usage: TokenUsage | None = None


@dataclass
class Trajectory:
    records: list[TrajectoryRecord]
    oracle: dict[str, Any]
    metadata: dict[str, Any]
    agent_name: str = ""
    run_id: str = ""
    seed: int = 0
    episode_id: int = 0