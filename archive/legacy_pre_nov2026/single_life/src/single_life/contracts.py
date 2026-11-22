"""Strict records for protocol single-life-v0.2.

Serialization is allowlisted. Extra fields, non-finite numbers, and booleans
standing in for numbers are rejected. These types are not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from single_life import CLOCK_CONVENTION, PROTOCOL_VERSION

PUBLIC_VIEW_KEYS = frozenset(
    {
        "accessible_artifacts",
        "decision_index",
        "lifetime_id",
        "observed_events",
        "permitted_action_ids",
        "protocol_version",
        "public_current_observation",
        "remaining_public_resources",
    }
)
OBSERVED_EVENT_KEYS = frozenset(
    {
        "action_index",
        "actual_interaction_cost",
        "chosen_action",
        "event_id",
        "observed_outcome",
        "public_observation",
    }
)
ARTIFACT_KEYS = frozenset(
    {
        "artifact_id",
        "content",
        "content_hash",
        "creation_cutoff",
        "creation_decision_index",
        "deployment_decision_index",
        "kind",
        "parent_version",
        "supporting_observed_event_ids",
        "version",
    }
)
RESOURCE_KEYS = frozenset({"compute_budget_remaining", "interactions_remaining"})
PRIVATE_EVENT_KEYS = frozenset(
    {
        "evaluator_diagnostics",
        "evaluator_state",
        "innovation_ref",
        "public_event_id",
        "schedule_position",
    }
)


class ContractError(ValueError):
    """A record broke the public schema or an internal consistency check."""


class EvidenceError(ContractError):
    """Evidence identity, cutoff, or creation/deployment order is invalid."""


class CheckpointError(ContractError):
    """A checkpoint is incomplete or does not match the lifetime being resumed."""


def canonical_dumps(value: Any) -> str:
    _reject_nonjson(value, "value")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def thaw_json(value: Any) -> Any:
    """Plain JSON structure with no shared mutable containers."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def freeze_json(value: Any) -> Any:
    """Recursive read-only JSON structure."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


def content_hash(content: Mapping[str, Any]) -> str:
    return sha256_text(canonical_dumps(thaw_json(content)))


def _reject_nonjson(value: Any, where: str) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite number at {where}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonjson(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"non-string key at {where}")
            _reject_nonjson(item, f"{where}.{key}")
        return
    raise ContractError(f"non-JSON value at {where}: {type(value).__name__}")


def _exact_mapping(data: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractError(f"{where} must be a JSON object")
    present = set(data)
    if present != set(keys):
        missing = sorted(set(keys) - present)
        extra = sorted(present - set(keys))
        raise ContractError(f"{where} keys mismatch missing={missing} extra={extra}")
    return data


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where} must be an integer")
    return value


def _require_positive_int(value: Any, where: str) -> int:
    number = _require_int(value, where)
    if number < 1:
        raise ContractError(f"{where} must be >= 1")
    return number


def _require_nonnegative_int(value: Any, where: str) -> int:
    number = _require_int(value, where)
    if number < 0:
        raise ContractError(f"{where} must be >= 0")
    return number


def _require_finite_number(value: Any, where: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{where} must be a finite number")
    return value


def _optional_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, where)


@dataclass(frozen=True)
class ObservedEvent:
    event_id: str
    action_index: int
    chosen_action: str
    observed_outcome: int | float
    public_observation: Any
    actual_interaction_cost: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "action_index": self.action_index,
            "chosen_action": self.chosen_action,
            "observed_outcome": self.observed_outcome,
            "public_observation": self.public_observation,
            "actual_interaction_cost": self.actual_interaction_cost,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ObservedEvent":
        body = _exact_mapping(data, OBSERVED_EVENT_KEYS, "ObservedEvent")
        observation = body["public_observation"]
        _reject_nonjson(observation, "ObservedEvent.public_observation")
        return cls(
            event_id=_require_str(body["event_id"], "ObservedEvent.event_id"),
            action_index=_require_positive_int(body["action_index"], "ObservedEvent.action_index"),
            chosen_action=_require_str(body["chosen_action"], "ObservedEvent.chosen_action"),
            observed_outcome=_require_finite_number(
                body["observed_outcome"], "ObservedEvent.observed_outcome"
            ),
            public_observation=observation,
            actual_interaction_cost=_require_positive_int(
                body["actual_interaction_cost"], "ObservedEvent.actual_interaction_cost"
            ),
        )


@dataclass(frozen=True)
class ArtifactVersion:
    artifact_id: str
    version: str
    kind: str
    parent_version: str | None
    creation_cutoff: int
    creation_decision_index: int
    supporting_observed_event_ids: tuple[str, ...]
    content: dict[str, Any]
    content_hash: str
    deployment_decision_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "kind": self.kind,
            "parent_version": self.parent_version,
            "creation_cutoff": self.creation_cutoff,
            "creation_decision_index": self.creation_decision_index,
            "supporting_observed_event_ids": list(self.supporting_observed_event_ids),
            "content": thaw_json(self.content),
            "content_hash": self.content_hash,
            "deployment_decision_index": self.deployment_decision_index,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ArtifactVersion":
        body = _exact_mapping(data, ARTIFACT_KEYS, "ArtifactVersion")
        content = body["content"]
        if not isinstance(content, Mapping) or isinstance(content, (str, bytes)):
            raise ContractError("ArtifactVersion.content must be an object")
        plain_content = thaw_json(content)
        if not isinstance(plain_content, dict):
            raise ContractError("ArtifactVersion.content must be an object")
        _reject_nonjson(plain_content, "ArtifactVersion.content")
        digest = _require_str(body["content_hash"], "ArtifactVersion.content_hash")
        if digest != content_hash(plain_content):
            raise ContractError("ArtifactVersion.content_hash does not match content")
        support = body["supporting_observed_event_ids"]
        if not isinstance(support, list) or any(not isinstance(item, str) or item == "" for item in support):
            raise ContractError("ArtifactVersion.supporting_observed_event_ids must be a list of strings")
        deployment = body["deployment_decision_index"]
        if deployment is not None:
            deployment = _require_positive_int(deployment, "ArtifactVersion.deployment_decision_index")
        cutoff = _require_nonnegative_int(body["creation_cutoff"], "ArtifactVersion.creation_cutoff")
        created_at = _require_positive_int(
            body["creation_decision_index"], "ArtifactVersion.creation_decision_index"
        )
        if created_at <= cutoff:
            raise EvidenceError("creation_decision_index must be strictly after creation_cutoff")
        if deployment is not None and deployment <= cutoff:
            raise EvidenceError(
                "deployment_decision_index must be strictly after creation_cutoff"
            )
        if deployment is not None and deployment < created_at:
            raise EvidenceError("deployment cannot precede creation")
        return cls(
            artifact_id=_require_str(body["artifact_id"], "ArtifactVersion.artifact_id"),
            version=_require_str(body["version"], "ArtifactVersion.version"),
            kind=_require_str(body["kind"], "ArtifactVersion.kind"),
            parent_version=_optional_str(body["parent_version"], "ArtifactVersion.parent_version"),
            creation_cutoff=cutoff,
            creation_decision_index=created_at,
            supporting_observed_event_ids=tuple(support),
            content=freeze_json(plain_content),
            content_hash=digest,
            deployment_decision_index=deployment,
        )


@dataclass(frozen=True)
class PublicView:
    protocol_version: str
    lifetime_id: str
    decision_index: int
    permitted_action_ids: tuple[str, ...]
    public_current_observation: Any
    observed_events: tuple[ObservedEvent, ...]
    accessible_artifacts: tuple[ArtifactVersion, ...]
    remaining_public_resources: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "lifetime_id": self.lifetime_id,
            "decision_index": self.decision_index,
            "permitted_action_ids": list(self.permitted_action_ids),
            "public_current_observation": self.public_current_observation,
            "observed_events": [event.to_dict() for event in self.observed_events],
            "accessible_artifacts": [artifact.to_dict() for artifact in self.accessible_artifacts],
            "remaining_public_resources": dict(self.remaining_public_resources),
        }

    def canonical_json(self) -> str:
        return canonical_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> "PublicView":
        body = _exact_mapping(data, PUBLIC_VIEW_KEYS, "PublicView")
        if body["protocol_version"] != PROTOCOL_VERSION:
            raise ContractError(
                f"PublicView.protocol_version must be {PROTOCOL_VERSION}"
            )
        observation = body["public_current_observation"]
        _reject_nonjson(observation, "PublicView.public_current_observation")
        actions = body["permitted_action_ids"]
        if (
            not isinstance(actions, list)
            or not actions
            or any(not isinstance(item, str) or item == "" for item in actions)
            or len(set(actions)) != len(actions)
        ):
            raise ContractError("PublicView.permitted_action_ids must be unique non-empty strings")
        events_raw = body["observed_events"]
        artifacts_raw = body["accessible_artifacts"]
        if not isinstance(events_raw, list) or not isinstance(artifacts_raw, list):
            raise ContractError("PublicView history fields must be lists")
        resources = _exact_mapping(
            body["remaining_public_resources"], RESOURCE_KEYS, "remaining_public_resources"
        )
        decision_index = _require_positive_int(body["decision_index"], "PublicView.decision_index")
        observed_events = tuple(ObservedEvent.from_dict(item) for item in events_raw)
        accessible_artifacts = tuple(ArtifactVersion.from_dict(item) for item in artifacts_raw)
        _validate_causal_view(decision_index, tuple(actions), observed_events, accessible_artifacts)
        return cls(
            protocol_version=PROTOCOL_VERSION,
            lifetime_id=_require_str(body["lifetime_id"], "PublicView.lifetime_id"),
            decision_index=decision_index,
            permitted_action_ids=tuple(actions),
            public_current_observation=observation,
            observed_events=observed_events,
            accessible_artifacts=accessible_artifacts,
            remaining_public_resources={
                "interactions_remaining": _require_nonnegative_int(
                    resources["interactions_remaining"], "interactions_remaining"
                ),
                "compute_budget_remaining": _require_nonnegative_int(
                    resources["compute_budget_remaining"], "compute_budget_remaining"
                ),
            },
        )


@dataclass(frozen=True)
class PrivateEvent:
    public_event_id: str
    evaluator_state: dict[str, Any]
    schedule_position: int
    innovation_ref: str
    evaluator_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_event_id": self.public_event_id,
            "evaluator_state": self.evaluator_state,
            "schedule_position": self.schedule_position,
            "innovation_ref": self.innovation_ref,
            "evaluator_diagnostics": self.evaluator_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PrivateEvent":
        body = _exact_mapping(data, PRIVATE_EVENT_KEYS, "PrivateEvent")
        state = body["evaluator_state"]
        diagnostics = body["evaluator_diagnostics"]
        if not isinstance(state, dict) or not isinstance(diagnostics, dict):
            raise ContractError("PrivateEvent evaluator objects must be objects")
        _reject_nonjson(state, "PrivateEvent.evaluator_state")
        _reject_nonjson(diagnostics, "PrivateEvent.evaluator_diagnostics")
        return cls(
            public_event_id=_require_str(body["public_event_id"], "PrivateEvent.public_event_id"),
            evaluator_state=dict(state),
            schedule_position=_require_nonnegative_int(
                body["schedule_position"], "PrivateEvent.schedule_position"
            ),
            innovation_ref=_require_str(body["innovation_ref"], "PrivateEvent.innovation_ref"),
            evaluator_diagnostics=dict(diagnostics),
        )


def _validate_causal_view(
    decision_index: int,
    permitted_action_ids: tuple[str, ...],
    observed_events: tuple[ObservedEvent, ...],
    accessible_artifacts: tuple[ArtifactVersion, ...],
) -> None:
    """Reject current/future outcomes, duplicate identities, and unavailable artifacts.

    Invalid-action gaps are legal: action indexes must be strictly increasing
    and each strictly less than the decision, but they need not be consecutive.
    """

    permitted = set(permitted_action_ids)
    by_id: dict[str, ObservedEvent] = {}
    previous_index = 0
    for event in observed_events:
        if event.action_index >= decision_index:
            raise ContractError(
                f"observed event {event.event_id} at {event.action_index} is not before decision {decision_index}"
            )
        if event.action_index <= previous_index:
            raise ContractError("observed events must have strictly increasing action indexes")
        if event.event_id in by_id:
            raise ContractError(f"duplicate observed event id {event.event_id}")
        if event.chosen_action not in permitted:
            raise ContractError(
                f"observed action {event.chosen_action} is not in the permitted set"
            )
        by_id[event.event_id] = event
        previous_index = event.action_index
    by_version: dict[str, ArtifactVersion] = {}
    for artifact in accessible_artifacts:
        if artifact.version in by_version:
            raise ContractError(f"ambiguous artifact version {artifact.version}")
        if artifact.deployment_decision_index is None:
            raise ContractError(f"artifact {artifact.version} is accessible but not deployed")
        if artifact.deployment_decision_index > decision_index:
            raise ContractError(f"artifact {artifact.version} deployment is in the future")
        if artifact.creation_decision_index > decision_index:
            raise ContractError(f"artifact {artifact.version} creation is in the future")
        if artifact.creation_cutoff >= decision_index:
            raise ContractError(f"artifact {artifact.version} cutoff is not before the decision")
        for event_id in artifact.supporting_observed_event_ids:
            event = by_id.get(event_id)
            if event is None:
                raise EvidenceError(f"artifact support {event_id} is not in the observed record")
            if event.action_index > artifact.creation_cutoff:
                raise EvidenceError(f"artifact support {event_id} is newer than cutoff")
        by_version[artifact.version] = artifact
    for artifact in accessible_artifacts:
        if artifact.parent_version is None:
            continue
        parent = by_version.get(artifact.parent_version)
        if parent is None:
            raise EvidenceError(f"artifact parent {artifact.parent_version} is not available")
        if parent.creation_decision_index >= artifact.creation_decision_index:
            raise EvidenceError("artifact parent is not strictly older")


def bind_public_view(view: PublicView) -> tuple[str, PublicView]:
    """Return the allowlisted JSON payload and the only object an agent may see."""

    payload = view.canonical_json()
    parsed = PublicView.from_dict(json.loads(payload))
    return payload, parsed


def assert_separated_streams(manifest: Mapping[str, Any], public_payload: str) -> None:
    """Research manifests carry source identity; public payloads must not."""

    for marker in ("git_commit", "source_sha256", "uv_lock_sha256", "worktree_dirty"):
        if marker in public_payload:
            raise ContractError(f"public payload contains manifest field {marker}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ContractError("manifest protocol_version mismatch")
    if manifest.get("clock_convention") != CLOCK_CONVENTION:
        raise ContractError("manifest clock_convention mismatch")
