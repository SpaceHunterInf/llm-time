"""One interaction-indexed lifetime.

A completed step is either an observed reward or an invalid action that
consumed its tick. Checkpoints are written only on those boundaries. An
evaluator failure aborts without a reward and without scoring a second arm;
that attempt is not a committed step and is not retried inside the run.

Invalid actions consume one interaction and reveal nothing. The runner does
not resample or substitute another arm. Budget exhaustion stops before the
next decision and does not probe the environment.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from single_life import CLOCK_CONVENTION, PROTOCOL_VERSION
from single_life.artifacts import (
    SUPPORTED_UPDATER_IDS,
    ArtifactStore,
    NullUpdater,
    ScriptedMeanUpdater,
)
from single_life.baselines import ArtifactConfigAgent, RandomAgent, SlidingWindowUCB, make_policy
from single_life.contracts import (
    ArtifactVersion,
    CheckpointError,
    ContractError,
    EvidenceError,
    ObservedEvent,
    PrivateEvent,
    PublicView,
    bind_public_view,
    canonical_dumps,
    sha256_text,
)
from single_life.fixtures import BanditEvaluator, EvaluatorFailure, RunSpec

CHECKPOINT_FORMAT = "single-life-checkpoint-v3"
CHECKPOINT_V1_FORMAT = "single-life-checkpoint-v1"
CHECKPOINT_V2_FORMAT = "single-life-checkpoint-v2"
CHECKPOINT_ENVELOPE_KEYS = frozenset({"complete", "format", "sha256", "state"})
CHECKPOINT_STATE_KEYS = frozenset(
    {
        "agent_name",
        "agent_rng",
        "agent_state",
        "artifacts",
        "committed_index",
        "next_decision_index",
        "observed_events",
        "private_events",
        "policy_type",
        "policy_window",
        "protocol_version",
        "public_failures",
        "remaining_compute",
        "remaining_interactions",
        "seeds",
        "spec_sha256",
        "updater_id",
        "updater_period",
        "updater_type",
        "updater_window",
    }
)
POLICY_TYPES = {
    "random": RandomAgent,
    "ucb": SlidingWindowUCB,
    "artifact": ArtifactConfigAgent,
}
SUPPORTED_AGENT_NAMES = frozenset({"random", "ucb", "artifact"})


@dataclass
class RunResult:
    status: str
    committed_index: int
    observed_events: list[dict[str, Any]]
    private_events: list[dict[str, Any]]
    public_failures: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    actor_inputs: list[str]
    updater_inputs: list[str]
    remaining_interactions: int
    remaining_compute: int
    interactions_consumed: int
    compute_units_consumed: int
    error_private: str | None = None
    public_abort: dict[str, Any] | None = None
    spec_sha256: str = ""

    def semantic_stream(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "committed_index": self.committed_index,
            "observed_events": self.observed_events,
            "private_events": self.private_events,
            "public_failures": self.public_failures,
            "artifacts": self.artifacts,
            "actions": self.actions,
            "remaining_interactions": self.remaining_interactions,
            "remaining_compute": self.remaining_compute,
            "public_abort": self.public_abort,
        }


@dataclass
class _State:
    committed_index: int
    next_decision_index: int
    remaining_interactions: int
    remaining_compute: int
    interactions_consumed: int
    compute_units_consumed: int
    rng: random.Random
    store: ArtifactStore
    observed: list[ObservedEvent] = field(default_factory=list)
    private: list[PrivateEvent] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)


def run_lifetime(
    spec: RunSpec,
    *,
    policy: Any | None = None,
    updater: Any | None = None,
    evaluator: BanditEvaluator | None = None,
    resume_path: str | Path | None = None,
    stop_after: int | None = None,
    checkpoint_path: str | Path | None = None,
    forced_actions: Sequence[str] | None = None,
    fault_decisions: Sequence[int] = (),
    on_commit: Callable[[_State], None] | None = None,
) -> RunResult:
    if policy is None:
        policy = make_policy(spec.agent_name, spec.ucb_window)
    if updater is None:
        updater = ScriptedMeanUpdater(spec.artifact_period, spec.artifact_window)
    require_supported_components(spec, policy, updater, allow_test_doubles=True)
    if evaluator is None:
        evaluator = BanditEvaluator(spec, set(fault_decisions))
    if resume_path is None:
        state = _fresh_state(spec)
    else:
        state = _state_from_checkpoint(Path(resume_path), spec, policy, updater)
    result = RunResult(
        status="running",
        committed_index=state.committed_index,
        observed_events=[],
        private_events=[],
        public_failures=[],
        artifacts=[],
        actions=[],
        actor_inputs=[],
        updater_inputs=[],
        remaining_interactions=state.remaining_interactions,
        remaining_compute=state.remaining_compute,
        interactions_consumed=state.interactions_consumed,
        compute_units_consumed=state.compute_units_consumed,
        spec_sha256=spec.fingerprint(),
    )
    while True:
        if stop_after is not None and state.committed_index >= stop_after:
            result.status = "stopped_for_checkpoint"
            break
        decision_index = state.next_decision_index
        if decision_index > spec.horizon:
            result.status = "completed"
            break
        if state.remaining_interactions <= 0:
            result.status = "budget_exhausted"
            break
        if state.remaining_compute < 1:
            result.status = "compute_budget_exhausted"
            break
        state.remaining_compute -= 1
        state.compute_units_consumed += 1
        try:
            view = _public_view(spec, state, decision_index)
            updater_payload, updater_view = bind_public_view(view)
        except (ContractError, EvidenceError) as exc:
            _abort(result, "contract_failure", decision_index, exc, tick_consumed=False, reward_revealed=False)
            break
        result.updater_inputs.append(updater_payload)
        prior_versions = state.store.snapshot()
        try:
            state.store.bind_decision(decision_index, state.observed)
            updater.apply(updater_view, state.store)
            state.store.assert_update_committed(decision_index, prior_versions)
        except Exception as exc:
            state.store.versions = prior_versions
            _abort(result, "updater_failure", decision_index, exc, tick_consumed=False, reward_revealed=False)
            break
        try:
            view = _public_view(spec, state, decision_index)
            actor_payload, actor_view = bind_public_view(view)
        except (ContractError, EvidenceError) as exc:
            state.store.versions = prior_versions
            _abort(result, "contract_failure", decision_index, exc, tick_consumed=False, reward_revealed=False)
            break
        result.actor_inputs.append(actor_payload)
        forced = None
        if forced_actions is not None and decision_index <= len(forced_actions):
            forced = forced_actions[decision_index - 1]
        # A missing or explicit null entry uses the policy. One selection only.
        try:
            if isinstance(forced, str):
                action = forced
            else:
                action = policy.select(actor_view, state.rng)
        except Exception as exc:
            state.store.versions = prior_versions
            _abort(result, "actor_failure", decision_index, exc, tick_consumed=False, reward_revealed=False)
            break
        if not isinstance(action, str) or action not in spec.action_ids:
            # Tick is consumed. No reward is drawn and no second action is tried.
            failure = {
                "record_type": "public_failure",
                "decision_index": decision_index,
                "failure_kind": "invalid_action",
                "attempted_action": action if isinstance(action, str) else None,
                "tick_consumed": True,
                "reward_revealed": False,
            }
            state.failures.append(failure)
            state.actions.append(
                {
                    "decision_index": decision_index,
                    "action": failure["attempted_action"],
                    "kind": "invalid_action",
                }
            )
            state.remaining_interactions -= 1
            state.interactions_consumed += 1
            state.committed_index = decision_index
            state.next_decision_index = decision_index + 1
            if not _journal(result, on_commit, state, decision_index):
                break
            continue
        try:
            reward, private = evaluator.reveal(decision_index, action)
        except EvaluatorFailure as exc:
            state.store.versions = prior_versions
            _abort(
                result,
                "evaluator_failure",
                decision_index,
                exc,
                tick_consumed=False,
                reward_revealed=False,
            )
            break
        except Exception as exc:
            state.store.versions = prior_versions
            _abort(
                result,
                "evaluator_failure",
                decision_index,
                exc,
                tick_consumed=False,
                reward_revealed=False,
            )
            break
        event = ObservedEvent(
            event_id=f"{spec.lifetime_id}:event:{decision_index}",
            action_index=decision_index,
            chosen_action=action,
            observed_outcome=reward,
            public_observation=None,
            actual_interaction_cost=1,
        )
        state.observed.append(event)
        state.private.append(private)
        state.actions.append(
            {"decision_index": decision_index, "action": action, "kind": "observed"}
        )
        state.remaining_interactions -= 1
        state.interactions_consumed += 1
        state.committed_index = decision_index
        state.next_decision_index = decision_index + 1
        # The reward is already in ``state``. A journal or observe failure must
        # not drop it or score another arm.
        if not _journal(result, on_commit, state, decision_index):
            break
        try:
            if hasattr(policy, "observe"):
                policy.observe([event])
        except Exception as exc:
            _abort(
                result,
                "post_observation_failure",
                decision_index,
                exc,
                tick_consumed=True,
                reward_revealed=True,
            )
            break
    if checkpoint_path is not None and result.status == "stopped_for_checkpoint":
        save_checkpoint(Path(checkpoint_path), spec, policy, updater, state)
    _fill_result(result, state)
    return result


def _abort(
    result: RunResult,
    status: str,
    decision_index: int,
    exc: BaseException,
    *,
    tick_consumed: bool,
    reward_revealed: bool,
) -> None:
    result.status = status
    result.error_private = f"{type(exc).__name__}: {exc}"
    result.public_abort = {
        "record_type": "run_aborted",
        "decision_index": decision_index,
        "failure_kind": status,
        "tick_consumed": tick_consumed,
        "reward_revealed": reward_revealed,
    }


def _journal(
    result: RunResult,
    on_commit: Callable[[_State], None] | None,
    state: _State,
    decision_index: int,
) -> bool:
    if on_commit is None:
        return True
    try:
        on_commit(state)
    except Exception as exc:
        _abort(
            result,
            "journal_failure",
            decision_index,
            exc,
            tick_consumed=True,
            reward_revealed=any(event.action_index == decision_index for event in state.observed),
        )
        return False
    return True


def require_supported_components(
    spec: RunSpec, policy: Any, updater: Any, *, allow_test_doubles: bool
) -> None:
    """Exact classes only. A subclass that inherits a supported id is rejected.

    Unrelated test doubles may run in-process. They cannot be checkpointed.
    """

    policy_type = type(policy)
    exact_policies = tuple(POLICY_TYPES.values())
    if policy_type in exact_policies:
        if policy_type is SlidingWindowUCB and policy.window != spec.ucb_window:
            raise ContractError("UCB window does not match the spec")
    elif any(issubclass(policy_type, cls) for cls in exact_policies):
        raise ContractError(f"unsupported policy implementation {policy_type.__name__}")
    elif not allow_test_doubles:
        raise CheckpointError(f"unsupported policy implementation {policy_type.__name__}")
    updater_type = type(updater)
    if updater_type is ScriptedMeanUpdater:
        if (updater.period, updater.window) != (spec.artifact_period, spec.artifact_window):
            raise ContractError("ScriptedMeanUpdater period/window does not match the spec")
    elif updater_type is NullUpdater:
        return
    elif issubclass(updater_type, (ScriptedMeanUpdater, NullUpdater)):
        raise ContractError(f"unsupported updater implementation {updater_type.__name__}")
    elif not allow_test_doubles:
        raise CheckpointError(f"unsupported updater implementation {updater_type.__name__}")


def component_identity(policy: Any, updater: Any) -> dict[str, Any]:
    policy_type = type(policy)
    if policy_type not in POLICY_TYPES.values():
        raise CheckpointError(f"unsupported policy implementation {policy_type.__name__}")
    if type(updater) is ScriptedMeanUpdater:
        return {
            "policy_type": policy_type.__name__,
            "policy_window": policy.window if policy_type is SlidingWindowUCB else None,
            "updater_type": "ScriptedMeanUpdater",
            "updater_period": updater.period,
            "updater_window": updater.window,
        }
    if type(updater) is NullUpdater:
        return {
            "policy_type": policy_type.__name__,
            "policy_window": policy.window if policy_type is SlidingWindowUCB else None,
            "updater_type": "NullUpdater",
            "updater_period": None,
            "updater_window": None,
        }
    raise CheckpointError(f"unsupported updater implementation {type(updater).__name__}")


def _updater_id(updater: Any) -> str:
    updater_id = getattr(updater, "updater_id", None)
    if updater_id not in SUPPORTED_UPDATER_IDS:
        raise CheckpointError(
            "unsupported updater for checkpoint; supported updaters are "
            + ", ".join(sorted(SUPPORTED_UPDATER_IDS))
        )
    return updater_id


def save_checkpoint(path: Path, spec: RunSpec, policy: Any, updater: Any, state: _State) -> None:
    require_supported_components(spec, policy, updater, allow_test_doubles=False)
    identity = component_identity(policy, updater)
    body = {
        "committed_index": state.committed_index,
        "next_decision_index": state.next_decision_index,
        "remaining_interactions": state.remaining_interactions,
        "remaining_compute": state.remaining_compute,
        "agent_name": spec.agent_name,
        "agent_rng": _rng_to_json(state.rng),
        "agent_state": policy.state_dict(),
        "observed_events": [event.to_dict() for event in state.observed],
        "private_events": [event.to_dict() for event in state.private],
        "protocol_version": PROTOCOL_VERSION,
        "public_failures": list(state.failures),
        "artifacts": [artifact.to_dict() for artifact in state.store.versions],
        "spec_sha256": spec.fingerprint(),
        "seeds": {
            "agent_seed": spec.agent_seed,
            "innovation_seed": spec.innovation_seed,
            "schedule_seed": spec.schedule_seed,
            "lifetime_id": spec.lifetime_id,
        },
        "updater_id": _updater_id(updater),
        **identity,
    }
    if set(body) != CHECKPOINT_STATE_KEYS:
        raise CheckpointError("checkpoint state keys drifted")
    _validate_checkpoint_body(body, spec, policy, updater)
    encoded = canonical_dumps(body)
    envelope = {
        "format": CHECKPOINT_FORMAT,
        "complete": True,
        "sha256": sha256_text(encoded),
        "state": json.loads(encoded),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_dumps(envelope), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointError(f"incomplete checkpoint: {exc}") from exc
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CheckpointError("incomplete checkpoint: invalid JSON") from exc
    if not isinstance(envelope, dict):
        raise CheckpointError("incomplete checkpoint: envelope mismatch")
    if envelope.get("format") == CHECKPOINT_V1_FORMAT:
        raise CheckpointError(
            "incompatible checkpoint format single-life-checkpoint-v1; refusing to reinterpret v0.1"
        )
    if envelope.get("format") == CHECKPOINT_V2_FORMAT:
        raise CheckpointError(
            "incompatible checkpoint format single-life-checkpoint-v2; refusing to reinterpret it"
        )
    if set(envelope) != set(CHECKPOINT_ENVELOPE_KEYS) or envelope.get("complete") is not True:
        raise CheckpointError("incomplete checkpoint: envelope mismatch")
    if envelope.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"incompatible checkpoint format {envelope.get('format')}")
    state = envelope.get("state")
    if not isinstance(state, dict) or set(state) != set(CHECKPOINT_STATE_KEYS):
        raise CheckpointError("incomplete checkpoint: state mismatch")
    encoded = canonical_dumps(state)
    if envelope.get("sha256") != sha256_text(encoded):
        raise CheckpointError("incomplete checkpoint: hash mismatch")
    return state


def semantic_stream(result: RunResult) -> dict[str, Any]:
    return result.semantic_stream()


def _fresh_state(spec: RunSpec) -> _State:
    return _State(
        committed_index=0,
        next_decision_index=1,
        remaining_interactions=spec.interaction_budget,
        remaining_compute=spec.compute_budget,
        interactions_consumed=0,
        compute_units_consumed=0,
        rng=random.Random(spec.agent_seed),
        store=ArtifactStore(),
    )


def _state_from_checkpoint(path: Path, spec: RunSpec, policy: Any, updater: Any) -> _State:
    body = load_checkpoint(path)
    _validate_checkpoint_body(body, spec, policy, updater)
    if body["agent_name"] != spec.agent_name:
        raise CheckpointError("checkpoint agent_name does not match the run")
    observed = [ObservedEvent.from_dict(item) for item in body["observed_events"]]
    private = [PrivateEvent.from_dict(item) for item in body["private_events"]]
    store = ArtifactStore()
    try:
        artifacts = [ArtifactVersion.from_dict(item) for item in body["artifacts"]]
        store.install_historical(artifacts, observed, body["committed_index"])
    except (ContractError, EvidenceError) as exc:
        raise CheckpointError(f"checkpoint artifacts are inconsistent: {exc}") from exc
    try:
        policy.load_state_dict(body["agent_state"])
    except (ContractError, EvidenceError) as exc:
        raise CheckpointError(f"checkpoint policy state is inconsistent: {exc}") from exc
    consumed = body["committed_index"]
    return _State(
        committed_index=body["committed_index"],
        next_decision_index=body["next_decision_index"],
        remaining_interactions=body["remaining_interactions"],
        remaining_compute=body["remaining_compute"],
        interactions_consumed=consumed,
        compute_units_consumed=spec.compute_budget - body["remaining_compute"],
        rng=_rng_from_json(body["agent_rng"]),
        store=store,
        observed=observed,
        private=private,
        failures=list(body["public_failures"]),
        actions=_actions_from_history(observed, body["public_failures"]),
    )


def _validate_checkpoint_body(
    body: dict[str, Any], spec: RunSpec, policy: Any | None = None, updater: Any | None = None
) -> None:
    if body.get("protocol_version") != PROTOCOL_VERSION:
        raise CheckpointError(
            f"incompatible checkpoint protocol {body.get('protocol_version')}; refusing to reinterpret it"
        )
    if updater is not None and policy is not None:
        require_supported_components(spec, policy, updater, allow_test_doubles=False)
        identity = component_identity(policy, updater)
        for key, value in identity.items():
            if body.get(key) != value:
                raise CheckpointError(
                    f"checkpoint {key} does not match the resume component"
                )
        if body.get("updater_id") != _updater_id(updater):
            raise CheckpointError("checkpoint updater is not the supported updater for this resume")
    if body["spec_sha256"] != spec.fingerprint():
        raise CheckpointError("checkpoint does not match fixture fingerprint")
    seeds = body["seeds"]
    if not isinstance(seeds, dict):
        raise CheckpointError("incomplete checkpoint: seeds")
    expected_seeds = {
        "agent_seed": spec.agent_seed,
        "innovation_seed": spec.innovation_seed,
        "schedule_seed": spec.schedule_seed,
        "lifetime_id": spec.lifetime_id,
    }
    if seeds != expected_seeds:
        raise CheckpointError("checkpoint seeds do not match the run")
    committed = body["committed_index"]
    nxt = body["next_decision_index"]
    if not isinstance(committed, int) or isinstance(committed, bool) or committed < 0:
        raise CheckpointError("incomplete checkpoint: committed index")
    if nxt != committed + 1:
        raise CheckpointError("incomplete checkpoint: next decision is ambiguous")
    if committed == 0:
        raise CheckpointError("incomplete checkpoint: no completed step")
    if committed > spec.horizon or nxt > spec.horizon + 1:
        raise CheckpointError("checkpoint is outside the fixture horizon")
    events = body["observed_events"]
    failures = body["public_failures"]
    private_events = body["private_events"]
    if not isinstance(events, list) or not isinstance(failures, list) or not isinstance(private_events, list):
        raise CheckpointError("incomplete checkpoint: logs")
    if len(private_events) != len(events):
        raise CheckpointError("incomplete checkpoint: private log length")
    observed_indexes = []
    seen_event_ids: set[str] = set()
    previous_index = 0
    for raw, private in zip(events, private_events):
        if not isinstance(raw, dict) or not isinstance(private, dict):
            raise CheckpointError("incomplete checkpoint: event")
        try:
            event = ObservedEvent.from_dict(raw)
            private_event = PrivateEvent.from_dict(private)
        except ContractError as exc:
            raise CheckpointError(f"incomplete checkpoint: event record: {exc}") from exc
        if event.action_index <= previous_index:
            raise CheckpointError("observed events are not in chronological order")
        if event.event_id in seen_event_ids:
            raise CheckpointError(f"duplicate event id {event.event_id}")
        if private_event.public_event_id != event.event_id:
            raise CheckpointError("private event does not match the public event")
        previous_index = event.action_index
        seen_event_ids.add(event.event_id)
        observed_indexes.append(event.action_index)
    failure_indexes = []
    for raw in failures:
        if not isinstance(raw, dict) or raw.get("tick_consumed") is not True:
            raise CheckpointError("incomplete checkpoint: failure record")
        failure_indexes.append(raw.get("decision_index"))
    covered = observed_indexes + failure_indexes
    if sorted(covered) != list(range(1, committed + 1)) or len(covered) != committed:
        raise CheckpointError("incomplete checkpoint: committed steps do not match logs")
    remaining = body["remaining_interactions"]
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        raise CheckpointError("incomplete checkpoint: remaining interactions")
    if remaining != spec.interaction_budget - committed:
        raise CheckpointError("incomplete checkpoint: interaction accounting")
    compute_left = body["remaining_compute"]
    if not isinstance(compute_left, int) or isinstance(compute_left, bool) or compute_left < 0:
        raise CheckpointError("incomplete checkpoint: remaining compute")
    if compute_left != spec.compute_budget - committed:
        raise CheckpointError(
            "checkpoint compute accounting does not match the committed prefix"
        )
    if body.get("agent_name") not in SUPPORTED_AGENT_NAMES:
        raise CheckpointError(f"unsupported policy {body.get('agent_name')}")
    _validate_baseline_state(body["agent_name"], body["agent_state"], events, spec)
    try:
        parsed_events = [ObservedEvent.from_dict(item) for item in events]
        parsed_artifacts = [ArtifactVersion.from_dict(item) for item in body["artifacts"]]
        ArtifactStore().install_historical(parsed_artifacts, parsed_events, committed)
    except (ContractError, EvidenceError, CheckpointError) as exc:
        raise CheckpointError(f"checkpoint artifacts are inconsistent: {exc}") from exc


def _validate_baseline_state(
    agent_name: str, agent_state: Any, observed_events: list[dict[str, Any]], spec: RunSpec
) -> None:
    if agent_name in {"random", "artifact"}:
        if agent_state != {}:
            raise CheckpointError(f"{agent_name} checkpoint state is not empty")
        return
    if not isinstance(agent_state, dict) or set(agent_state) != {"events", "window"}:
        raise CheckpointError("UCB checkpoint state does not match observed evidence")
    if agent_state["window"] != spec.ucb_window:
        raise CheckpointError("UCB window does not match the fixture")
    if agent_state["events"] != observed_events:
        raise CheckpointError("UCB state does not match observed evidence")


def _actions_from_history(
    observed: list[ObservedEvent], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for event in observed:
        actions.append(
            {
                "decision_index": event.action_index,
                "action": event.chosen_action,
                "kind": "observed",
            }
        )
    for failure in failures:
        actions.append(
            {
                "decision_index": failure["decision_index"],
                "action": failure.get("attempted_action"),
                "kind": "invalid_action",
            }
        )
    actions.sort(key=lambda item: item["decision_index"])
    return actions


def _public_view(spec: RunSpec, state: _State, decision_index: int) -> PublicView:
    accessible: list[ArtifactVersion] = []
    for artifact in state.store.versions:
        deployment = artifact.deployment_decision_index
        if deployment is None or deployment > decision_index:
            continue
        if artifact.creation_cutoff >= decision_index:
            continue
        accessible.append(artifact)
    return PublicView(
        protocol_version=PROTOCOL_VERSION,
        lifetime_id=spec.lifetime_id,
        decision_index=decision_index,
        permitted_action_ids=spec.action_ids,
        public_current_observation=None,
        observed_events=tuple(state.observed),
        accessible_artifacts=tuple(accessible),
        remaining_public_resources={
            "interactions_remaining": state.remaining_interactions,
            "compute_budget_remaining": state.remaining_compute,
        },
    )


def _fill_result(result: RunResult, state: _State) -> None:
    result.committed_index = state.committed_index
    result.observed_events = [event.to_dict() for event in state.observed]
    result.private_events = [event.to_dict() for event in state.private]
    result.public_failures = list(state.failures)
    result.artifacts = [artifact.to_dict() for artifact in state.store.versions]
    result.actions = list(state.actions)
    result.remaining_interactions = state.remaining_interactions
    result.remaining_compute = state.remaining_compute
    result.interactions_consumed = state.interactions_consumed
    result.compute_units_consumed = state.compute_units_consumed


def _rng_to_json(rng: random.Random) -> dict[str, Any]:
    version, internal, gauss = rng.getstate()
    return {"version": version, "internal": list(internal), "gauss": gauss}


def _rng_from_json(data: Any) -> random.Random:
    if not isinstance(data, dict) or set(data) != {"gauss", "internal", "version"}:
        raise CheckpointError("incomplete checkpoint: rng")
    internal = data["internal"]
    if not isinstance(internal, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in internal):
        raise CheckpointError("incomplete checkpoint: rng state")
    rng = random.Random()
    rng.setstate((data["version"], tuple(internal), data["gauss"]))
    return rng


def clock_fields() -> dict[str, str]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "clock_convention": CLOCK_CONVENTION,
        "clock_limitation": (
            "One chosen action consumes one tick. Host computation between ticks "
            "is logged as compute units and does not advance simulated time."
        ),
    }
