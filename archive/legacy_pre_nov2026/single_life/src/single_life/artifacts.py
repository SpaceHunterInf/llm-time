"""Scripted artifact updater.

At decision t, before the action, if the latest observed action index c is a
positive multiple of ``period`` and no artifact already has that cutoff, write
empirical arm means from the last ``window`` global-time observations. The
parent is the previous version. Evidence ids must already exist and have
action_index <= c. Deployment is for decision t, which must be > c.

Content is computed only from public observed events. Regime labels, schedule
seeds, and unreachable metadata are not read.
"""

from __future__ import annotations

from typing import Iterable

from single_life.baselines import dedupe_events, window_events
from single_life.contracts import (
    ArtifactVersion,
    EvidenceError,
    ObservedEvent,
    PublicView,
    content_hash,
)

MEAN_CONTENT_KEYS = frozenset({"arm_means", "evidence_count", "window", "window_definition"})
UPDATER_NAME = "scripted-mean-updater-v1"
NULL_UPDATER_ID = "null-updater-v1"
SUPPORTED_UPDATER_IDS = frozenset({UPDATER_NAME, NULL_UPDATER_ID})


class ArtifactStore:
    """Host-bound artifact log.

    ``bind_decision`` installs the current decision and the observed-evidence
    catalog. Callers cannot supply either. Creation is stamped at the bound
    decision; deployment at any other decision is rejected. Evidence cutoff
    may be older than that decision.
    """

    def __init__(self) -> None:
        self.versions: list[ArtifactVersion] = []
        self.current_decision: int | None = None
        self.catalog: dict[str, ObservedEvent] = {}

    def bind_decision(self, decision_index: int, events: Iterable[ObservedEvent]) -> None:
        if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 1:
            raise EvidenceError("current decision must be an integer >= 1")
        catalog: dict[str, ObservedEvent] = {}
        for event in events:
            if event.action_index >= decision_index:
                raise EvidenceError(
                    f"catalog event {event.event_id} is not before decision {decision_index}"
                )
            previous = catalog.get(event.event_id)
            if previous is not None and previous.to_dict() != event.to_dict():
                raise EvidenceError(f"conflicting catalog event {event.event_id}")
            catalog[event.event_id] = event
        self.current_decision = decision_index
        self.catalog = catalog

    def by_version(self, version: str) -> ArtifactVersion | None:
        matches = [artifact for artifact in self.versions if artifact.version == version]
        if not matches:
            return None
        if len(matches) > 1:
            raise EvidenceError(f"duplicate artifact version {version}")
        return matches[0]

    def create(
        self,
        *,
        artifact_id: str,
        version: str,
        kind: str,
        parent_version: str | None,
        creation_cutoff: int,
        supporting_observed_event_ids: Iterable[str],
        content: dict,
    ) -> ArtifactVersion:
        if self.current_decision is None:
            raise EvidenceError("artifact store is not bound to a decision")
        if self.by_version(version) is not None:
            raise EvidenceError(f"artifact version {version} already exists")
        if creation_cutoff >= self.current_decision:
            raise EvidenceError("creation cutoff must precede the current decision")
        support = tuple(supporting_observed_event_ids)
        if len(set(support)) != len(support):
            raise EvidenceError("supporting evidence ids must be unique")
        for event_id in support:
            event = self.catalog.get(event_id)
            if event is None:
                raise EvidenceError(f"nonexistent evidence reference {event_id}")
            if event.action_index > creation_cutoff:
                raise EvidenceError(
                    f"evidence {event_id} at {event.action_index} is after cutoff {creation_cutoff}"
                )
        if parent_version is not None:
            parent = self.by_version(parent_version)
            if parent is None:
                raise EvidenceError(f"parent version {parent_version} does not exist")
            if parent.creation_cutoff >= creation_cutoff:
                raise EvidenceError("parent artifact is not strictly older than its child")
        artifact = ArtifactVersion.from_dict(
            {
                "artifact_id": artifact_id,
                "version": version,
                "kind": kind,
                "parent_version": parent_version,
                "creation_cutoff": creation_cutoff,
                "creation_decision_index": self.current_decision,
                "supporting_observed_event_ids": list(support),
                "content": dict(content),
                "content_hash": content_hash(content),
                "deployment_decision_index": None,
            }
        )
        self.versions.append(artifact)
        return artifact

    def deploy(self, version: str, decision_index: int | None = None) -> ArtifactVersion:
        if self.current_decision is None:
            raise EvidenceError("artifact store is not bound to a decision")
        if decision_index is None:
            decision_index = self.current_decision
        if decision_index != self.current_decision:
            raise EvidenceError(
                f"deployment at {decision_index} does not match host decision {self.current_decision}"
            )
        artifact = self.by_version(version)
        if artifact is None:
            raise EvidenceError(f"cannot deploy unknown version {version}")
        if artifact.deployment_decision_index is not None:
            raise EvidenceError(f"version {version} is already deployed")
        if decision_index <= artifact.creation_cutoff:
            raise EvidenceError(
                "deployment_decision_index must be strictly after creation_cutoff"
            )
        if artifact.creation_decision_index != self.current_decision:
            raise EvidenceError("only an artifact created at the current decision can be deployed")
        for existing in self.versions:
            if existing.deployment_decision_index == decision_index:
                raise EvidenceError(f"decision {decision_index} already has a deployment")
        deployed = ArtifactVersion.from_dict(
            {**artifact.to_dict(), "deployment_decision_index": decision_index}
        )
        self.versions = [
            deployed if item.version == version else item for item in self.versions
        ]
        return deployed

    def assert_update_committed(self, decision_index: int, prior: list[ArtifactVersion]) -> None:
        """Reject forged lifecycle times and edits to immutable prior versions."""

        prior_by_version = {item.version: item for item in prior}
        if len(prior_by_version) != len(prior):
            raise EvidenceError("prior artifact versions are ambiguous")
        seen: set[str] = set()
        for artifact in self.versions:
            if artifact.version in seen:
                raise EvidenceError(f"ambiguous artifact version {artifact.version}")
            seen.add(artifact.version)
            old = prior_by_version.get(artifact.version)
            if old is None:
                if artifact.creation_decision_index != decision_index:
                    raise EvidenceError("creation decision is not the host decision")
                if artifact.deployment_decision_index not in (None, decision_index):
                    raise EvidenceError("deployment decision is not the host decision")
                continue
            if old.deployment_decision_index is None and artifact.deployment_decision_index == decision_index:
                rewritten = dict(old.to_dict())
                rewritten["deployment_decision_index"] = decision_index
                if rewritten != artifact.to_dict():
                    raise EvidenceError("prior artifact version was modified")
                continue
            if old.to_dict() != artifact.to_dict():
                raise EvidenceError("prior artifact version was modified")
        if not set(prior_by_version).issubset(seen):
            raise EvidenceError("prior artifact version was removed")

    def install_historical(
        self, artifacts: Iterable[ArtifactVersion], events: Iterable[ObservedEvent], committed_index: int
    ) -> None:
        """Restore checkpoint artifacts without restamping their lifecycle times."""

        event_list = list(events)
        catalog = {event.event_id: event for event in event_list}
        if len(catalog) != len(event_list):
            raise EvidenceError("historical evidence ids are not unique")
        ordered = tuple(artifacts)
        versions = [artifact.version for artifact in ordered]
        if len(versions) != len(set(versions)):
            raise EvidenceError("historical artifact versions are ambiguous")
        by_version = {artifact.version: artifact for artifact in ordered}
        for artifact in ordered:
            if artifact.deployment_decision_index is None:
                raise EvidenceError(f"historical artifact {artifact.version} was not deployed")
            if artifact.creation_decision_index > committed_index:
                raise EvidenceError("historical creation is after the committed prefix")
            if artifact.deployment_decision_index > committed_index:
                raise EvidenceError("historical deployment is after the committed prefix")
            for event_id in artifact.supporting_observed_event_ids:
                event = catalog.get(event_id)
                if event is None:
                    raise EvidenceError(f"historical support {event_id} is not in the prefix")
                if event.action_index > artifact.creation_cutoff:
                    raise EvidenceError(f"historical support {event_id} is newer than cutoff")
            if artifact.parent_version is not None:
                parent = by_version.get(artifact.parent_version)
                if parent is None:
                    raise EvidenceError(f"historical parent {artifact.parent_version} is missing")
                if parent.creation_decision_index >= artifact.creation_decision_index:
                    raise EvidenceError("historical parent is not strictly older")
        self.versions = list(ordered)
        self.current_decision = None
        self.catalog = {}

    def snapshot(self) -> list[ArtifactVersion]:
        """Independent copies. Nested content is not aliased to the live store."""

        return [ArtifactVersion.from_dict(artifact.to_dict()) for artifact in self.versions]


class ScriptedMeanUpdater:
    updater_id = UPDATER_NAME

    def __init__(self, period: int, window: int) -> None:
        if period < 1 or window < 1:
            raise EvidenceError("artifact period and window must be >= 1")
        self.period = period
        self.window = window

    def apply(self, view: PublicView, store: ArtifactStore) -> ArtifactVersion | None:
        events = dedupe_events(view.observed_events)
        if not events:
            return None
        cutoff = max(event.action_index for event in events)
        if cutoff % self.period != 0:
            return None
        if any(artifact.creation_cutoff == cutoff for artifact in store.versions):
            return None
        used = window_events(events, view.decision_index, self.window)
        # window_events is relative to the decision. At decision t the window
        # is [t - W, t - 1]. When cutoff == t - 1 this matches the evidence
        # available at creation. Refuse to include anything newer than cutoff.
        used = [event for event in used if event.action_index <= cutoff]
        means: dict[str, float] = {}
        for action_id in sorted(view.permitted_action_ids):
            rewards = [
                float(event.observed_outcome)
                for event in used
                if event.chosen_action == action_id
            ]
            if rewards:
                means[action_id] = sum(rewards) / len(rewards)
        content = {
            "arm_means": means,
            "window": self.window,
            "evidence_count": len(used),
            "window_definition": "global-time",
        }
        if set(content) != MEAN_CONTENT_KEYS:
            raise EvidenceError("scripted artifact content keys drifted")
        parents = [
            artifact
            for artifact in store.versions
            if artifact.creation_cutoff < cutoff
        ]
        parent_version = None
        if parents:
            parent_version = max(parents, key=lambda artifact: artifact.creation_cutoff).version
        version = f"v{len(store.versions) + 1}"
        created = store.create(
            artifact_id=f"{view.lifetime_id}:means",
            version=version,
            kind="empirical-arm-means",
            parent_version=parent_version,
            creation_cutoff=cutoff,
            supporting_observed_event_ids=[event.event_id for event in used],
            content=content,
        )
        return store.deploy(created.version)


class NullUpdater:
    updater_id = NULL_UPDATER_ID

    def apply(self, view: PublicView, store: ArtifactStore) -> None:
        del view, store
        return None
