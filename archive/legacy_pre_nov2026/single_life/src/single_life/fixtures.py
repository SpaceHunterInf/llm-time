"""Evaluator-owned Bernoulli fixture.

Schedule parameters, outcome innovations, and agent randomness are separate
streams. An innovation is keyed only by (innovation_seed, lifetime_id,
action_index, action_id). Selecting an action reveals that innovation alone.
Unreachable schedule metadata is stored for leakage tests and is not an input
to rewards or to the public view.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from single_life import PROTOCOL_VERSION
from single_life.contracts import ContractError, PrivateEvent, canonical_dumps, sha256_text


class EvaluatorFailure(RuntimeError):
    """The evaluator could not score the committed action. No other arm is queried."""


CONFIG_KEYS = frozenset(
    {
        "action_ids",
        "agent",
        "agent_seed",
        "artifact_period",
        "artifact_window",
        "compute_budget",
        "evaluator_private_note",
        "experiment_id",
        "fixture_id",
        "horizon",
        "innovation_seed",
        "interaction_budget",
        "lifetime_id",
        "protocol_version",
        "schedule_seed",
        "segments",
        "ucb_window",
    }
)
SEGMENT_KEYS = frozenset(
    {
        "end_index",
        "probabilities",
        "regime_id",
        "start_index",
        "unreachable_metadata",
    }
)


def innovation_unit(innovation_seed: str, lifetime_id: str, action_index: int, action_id: str) -> float:
    """Uniform draw in [0, 1), independent of schedule generation and agent RNG."""

    material = f"v1|{innovation_seed}|{lifetime_id}|{action_index}|{action_id}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") >> 11
    return value / float(1 << 53)


def innovation_ref(innovation_seed: str, lifetime_id: str, action_index: int, action_id: str) -> str:
    return f"innovation|{innovation_seed}|{lifetime_id}|{action_index}|{action_id}"


@dataclass(frozen=True)
class Segment:
    start_index: int
    end_index: int
    probabilities: dict[str, float]
    regime_id: str
    unreachable_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "probabilities": dict(self.probabilities),
            "regime_id": self.regime_id,
            "unreachable_metadata": self.unreachable_metadata,
        }


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    fixture_id: str
    lifetime_id: str
    action_ids: tuple[str, ...]
    horizon: int
    interaction_budget: int
    compute_budget: int
    schedule_seed: str
    innovation_seed: str
    agent_seed: str
    agent_name: str
    ucb_window: int
    artifact_period: int
    artifact_window: int
    segments: tuple[Segment, ...]
    evaluator_private_note: str
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "fixture_id": self.fixture_id,
            "lifetime_id": self.lifetime_id,
            "action_ids": list(self.action_ids),
            "horizon": self.horizon,
            "interaction_budget": self.interaction_budget,
            "compute_budget": self.compute_budget,
            "schedule_seed": self.schedule_seed,
            "innovation_seed": self.innovation_seed,
            "agent_seed": self.agent_seed,
            "agent": self.agent_name,
            "ucb_window": self.ucb_window,
            "artifact_period": self.artifact_period,
            "artifact_window": self.artifact_window,
            "segments": [segment.to_dict() for segment in self.segments],
            "evaluator_private_note": self.evaluator_private_note,
            "protocol_version": self.protocol_version,
        }

    def fingerprint(self) -> str:
        return sha256_text(canonical_dumps(self.to_dict()))


def load_spec(path: str | Path) -> RunSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec_from_dict(raw)


def spec_from_dict(data: Any) -> RunSpec:
    if not isinstance(data, dict):
        raise ContractError("fixture config must be an object")
    present = set(data)
    if present != set(CONFIG_KEYS):
        missing = sorted(set(CONFIG_KEYS) - present)
        extra = sorted(present - set(CONFIG_KEYS))
        raise ContractError(f"fixture config keys mismatch missing={missing} extra={extra}")
    if data["protocol_version"] != PROTOCOL_VERSION:
        raise ContractError(f"fixture protocol_version must be {PROTOCOL_VERSION}")
    action_ids = data["action_ids"]
    if (
        not isinstance(action_ids, list)
        or len(action_ids) < 2
        or any(not isinstance(item, str) or item == "" for item in action_ids)
        or len(set(action_ids)) != len(action_ids)
    ):
        raise ContractError("action_ids must be at least two unique strings")
    segments_raw = data["segments"]
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ContractError("segments must be a non-empty list")
    segments = tuple(_segment_from_dict(item, tuple(action_ids)) for item in segments_raw)
    horizon = _pos_int(data["horizon"], "horizon")
    _validate_coverage(segments, horizon, tuple(action_ids))
    agent_name = data["agent"]
    if agent_name not in {"random", "ucb", "artifact", "llm-recent", "llm-note"}:
        raise ContractError(
            "agent must be random, ucb, artifact, llm-recent, or llm-note"
        )
    note = data["evaluator_private_note"]
    if not isinstance(note, str):
        raise ContractError("evaluator_private_note must be a string")
    return RunSpec(
        experiment_id=_nonempty(data["experiment_id"], "experiment_id"),
        fixture_id=_nonempty(data["fixture_id"], "fixture_id"),
        lifetime_id=_nonempty(data["lifetime_id"], "lifetime_id"),
        action_ids=tuple(action_ids),
        horizon=horizon,
        interaction_budget=_pos_int(data["interaction_budget"], "interaction_budget"),
        compute_budget=_pos_int(data["compute_budget"], "compute_budget"),
        schedule_seed=_nonempty(data["schedule_seed"], "schedule_seed"),
        innovation_seed=_nonempty(data["innovation_seed"], "innovation_seed"),
        agent_seed=_nonempty(data["agent_seed"], "agent_seed"),
        agent_name=agent_name,
        ucb_window=_pos_int(data["ucb_window"], "ucb_window"),
        artifact_period=_pos_int(data["artifact_period"], "artifact_period"),
        artifact_window=_pos_int(data["artifact_window"], "artifact_window"),
        segments=segments,
        evaluator_private_note=note,
        protocol_version=PROTOCOL_VERSION,
    )


def _segment_from_dict(data: Any, action_ids: tuple[str, ...]) -> Segment:
    if not isinstance(data, dict):
        raise ContractError("segment must be an object")
    present = set(data)
    if present != set(SEGMENT_KEYS):
        missing = sorted(set(SEGMENT_KEYS) - present)
        extra = sorted(present - set(SEGMENT_KEYS))
        raise ContractError(f"segment keys mismatch missing={missing} extra={extra}")
    probabilities = data["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(action_ids):
        raise ContractError("segment probabilities must match action_ids exactly")
    cleaned: dict[str, float] = {}
    for action_id in action_ids:
        value = probabilities[action_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"probability for {action_id} must be a number")
        if isinstance(value, float) and not _finite(value):
            raise ContractError(f"probability for {action_id} must be finite")
        if value < 0 or value > 1:
            raise ContractError(f"probability for {action_id} must be in [0, 1]")
        cleaned[action_id] = float(value)
    metadata = data["unreachable_metadata"]
    if not isinstance(metadata, dict):
        raise ContractError("unreachable_metadata must be an object")
    canonical_dumps(metadata)
    return Segment(
        start_index=_pos_int(data["start_index"], "start_index"),
        end_index=_pos_int(data["end_index"], "end_index"),
        probabilities=cleaned,
        regime_id=_nonempty(data["regime_id"], "regime_id"),
        unreachable_metadata=dict(metadata),
    )


def _validate_coverage(segments: tuple[Segment, ...], horizon: int, action_ids: tuple[str, ...]) -> None:
    cursor = 1
    for segment in segments:
        if segment.start_index != cursor or segment.end_index < segment.start_index:
            raise ContractError("segments must be contiguous and ordered from action 1")
        if set(segment.probabilities) != set(action_ids):
            raise ContractError("action ids changed inside a lifetime")
        cursor = segment.end_index + 1
    if cursor != horizon + 1:
        raise ContractError("segments must cover the horizon exactly")


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _pos_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{where} must be an integer >= 1")
    return value


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


class BanditEvaluator:
    """Scores one chosen action. Counterfactual arms are not revealed."""

    def __init__(self, spec: RunSpec, fault_decisions: set[int] | None = None) -> None:
        self.spec = spec
        self.fault_decisions = set(fault_decisions or ())
        self.reveal_calls: list[tuple[int, str]] = []

    def segment_at(self, action_index: int) -> tuple[int, Segment]:
        for position, segment in enumerate(self.spec.segments):
            if segment.start_index <= action_index <= segment.end_index:
                return position, segment
        raise EvaluatorFailure(f"no segment covers action {action_index}")

    def reveal(self, action_index: int, action_id: str) -> tuple[int, PrivateEvent]:
        if action_index in self.fault_decisions:
            raise EvaluatorFailure(f"injected evaluator failure at {action_index}")
        if action_id not in self.spec.action_ids:
            raise EvaluatorFailure("refusing to score an action id outside the fixture")
        position, segment = self.segment_at(action_index)
        probability = segment.probabilities[action_id]
        unit = innovation_unit(
            self.spec.innovation_seed, self.spec.lifetime_id, action_index, action_id
        )
        reward = 1 if unit < probability else 0
        self.reveal_calls.append((action_index, action_id))
        private = PrivateEvent(
            public_event_id=f"{self.spec.lifetime_id}:event:{action_index}",
            evaluator_state={
                "chosen_probability": probability,
                "regime_id": segment.regime_id,
                "probabilities": dict(segment.probabilities),
                "segment_start": segment.start_index,
                "segment_end": segment.end_index,
            },
            schedule_position=position,
            innovation_ref=innovation_ref(
                self.spec.innovation_seed, self.spec.lifetime_id, action_index, action_id
            ),
            evaluator_diagnostics={
                "innovation_u": unit,
                "schedule_seed": self.spec.schedule_seed,
                "unreachable_metadata": segment.unreachable_metadata,
                "evaluator_private_note": self.spec.evaluator_private_note,
                "reward": reward,
            },
        )
        return reward, private


def replace_spec(spec: RunSpec, **changes: Any) -> RunSpec:
    updated = replace(spec, **changes)
    _validate_coverage(updated.segments, updated.horizon, updated.action_ids)
    return updated
