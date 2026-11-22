"""Contextual lifetime runner for executable JSON procedures.

Checkpoint and resume are disabled. A failed call keeps the committed prefix
and does not take another environment action for that call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from single_life.contracts import sha256_text
from single_life.gemini_adapter import TransportError, Usage
from single_life.procedure_client import (
    INPUT_TOKEN_CAP,
    MAX_ATTEMPTS,
    OUTPUT_TOKEN_CAP,
    ProcedureGeneration,
)
from single_life.procedure_dsl import (
    ACTION_ORDER,
    HORIZON,
    LOCAL_ALGORITHM,
    RANDOM_ALGORITHM,
    RANDOM_SEED,
    WHOLE_ALGORITHM,
    build_prompt,
    calls_for,
    canonical_procedure,
    initial_counters,
    next_action,
    parse_procedure,
    procedure_hash,
    run_reference_actions,
    type_at,
)
from single_life.procedure_dsl import ProcedureParseError
from single_life.procedure_progress import ProcedureLedger
from single_life.procedure_world import formula_worlds, table_phase, utility_tenths

UtilityFn = Callable[[int, str, str], int]


class LifetimeAbort(Exception):
    def __init__(self, reason: str, *, stop_task: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stop_task = stop_task


@dataclass
class LifetimeResult:
    status: str
    reason: str
    stop_task: bool
    committed_index: int
    call_indices: list[int]
    events: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    lifetime_id: str
    world: str
    condition: str


@dataclass
class _Deployed:
    version: int
    parent_version: int | None
    creation_cutoff: int
    deployment_index: int
    procedure: dict[str, list[str]]
    content_hash: str
    rationale: str
    evidence_ids: list[int]
    semantic_change: bool
    counters: dict[str, int] = field(default_factory=initial_counters)

    def record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "creation_cutoff": self.creation_cutoff,
            "deployment_index": self.deployment_index,
            "content_hash": self.content_hash,
            "semantic_change": self.semantic_change,
            "procedure": canonical_procedure(self.procedure),
            "rationale": self.rationale,
            "evidence_ids": list(self.evidence_ids),
            "counters_at_deployment": dict(self.counters),
        }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _model_observation(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": event["action"],
        "index": event["index"],
        "type": event["type"],
        "utility_tenths": event["utility_tenths"],
    }


def _empty_usage() -> Usage:
    return Usage("unknown", None, None, None, None, None)


class _Journal:
    def __init__(self, run_dir: Path, *, lifetime_id: str, world: str, condition: str, horizon: int) -> None:
        self.run_dir = run_dir
        self.lifetime_id = lifetime_id
        self.world = world
        self.condition = condition
        self.horizon = horizon
        self.events: list[dict[str, Any]] = []
        self.private: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "responses").mkdir(parents=True, exist_ok=True)

    def flush(self) -> None:
        atomic_write_text(self.run_dir / "public_log.jsonl", _jsonl(self.events))
        atomic_write_text(self.run_dir / "private_log.jsonl", _jsonl(self.private))
        _write_json(self.run_dir / "artifacts.json", self.artifacts)

    def write_response(self, action_index: int, attempt: int, payload: dict[str, Any]) -> None:
        path = self.run_dir / "responses" / f"c{action_index:03d}-a{attempt}.json"
        _write_json(path, payload)

    def write_side(self, name: str, payload: dict[str, Any]) -> None:
        _write_json(self.run_dir / "responses" / name, payload)


def run_lifetime(
    *,
    world: str,
    condition: str,
    run_dir: Path,
    client: ScriptedProcedureClient | Any,
    ledger: ProcedureLedger | None = None,
    horizon: int | None = None,
    utility_of: UtilityFn | None = None,
    lifetime_id: str | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> LifetimeResult:
    horizon = HORIZON if horizon is None else horizon
    schedule = calls_for(condition)
    lifetime_id = lifetime_id or f"life-006-{world}-{condition}"
    utility_of = utility_of or (lambda index, typ, action: utility_tenths(world, index, typ, action))
    journal = _Journal(run_dir, lifetime_id=lifetime_id, world=world, condition=condition, horizon=horizon)
    _write_public_config(run_dir, world, condition, horizon, schedule, lifetime_id)
    _write_private_table(run_dir, world)
    current: _Deployed | None = None
    call_indices: list[int] = []
    observations: list[dict[str, Any]] = []
    status = "completed"
    reason = ""
    stop_task = False
    try:
        for index in range(1, horizon + 1):
            if index in schedule:
                if ledger is not None:
                    ledger.locate(world=world, condition=condition, tick=index)
                parsed = _request_procedure(
                    client=client,
                    ledger=ledger,
                    journal=journal,
                    action_index=index,
                    horizon=horizon,
                    observations=observations,
                    deployed=current,
                )
                call_indices.append(index)
                current = _deploy(current, parsed["procedure_body"], index)
                journal.artifacts.append(current.record())
                journal.flush()
                if ledger is not None:
                    ledger.complete_call()
            if current is None:
                raise LifetimeAbort("no procedure is deployed")
            typ = type_at(index)
            action, slot = next_action(current.procedure, current.counters, typ)
            current.counters[typ] += 1
            utility = int(utility_of(index, typ, action))
            event = {
                "action": action,
                "cycle_slot": slot,
                "index": index,
                "procedure_hash": current.content_hash,
                "procedure_version": current.version,
                "semantic_deployment": current.deployment_index == index and current.semantic_change,
                "type": typ,
                "utility_tenths": utility,
            }
            private = {
                **event,
                "table_phase": table_phase(world, index),
                "world": world,
            }
            journal.events.append(event)
            journal.private.append(private)
            observations.append(_model_observation(event))
            journal.flush()
            if ledger is not None:
                ledger.complete_tick()
    except LifetimeAbort as exc:
        status = "aborted"
        reason = exc.reason
        stop_task = exc.stop_task
    result = LifetimeResult(
        status=status,
        reason=reason,
        stop_task=stop_task,
        committed_index=len(journal.events),
        call_indices=call_indices,
        events=list(journal.events),
        artifacts=list(journal.artifacts),
        lifetime_id=lifetime_id,
        world=world,
        condition=condition,
    )
    _write_manifest(journal, result, schedule, manifest_extra or {}, kind="gemini-procedure")
    return result


def _deploy(current: _Deployed | None, body: dict[str, Any], action_index: int) -> _Deployed:
    procedure = canonical_procedure(body["procedure"])
    content_hash = procedure_hash(procedure)
    semantic = current is None or content_hash != current.content_hash
    if semantic or current is None:
        counters = initial_counters()
        parent = None if current is None else current.version
        version = 1 if current is None else current.version + 1
    else:
        counters = current.counters
        parent = current.version
        version = current.version + 1
    return _Deployed(
        version=version,
        parent_version=parent,
        creation_cutoff=action_index - 1,
        deployment_index=action_index,
        procedure=procedure,
        content_hash=content_hash,
        rationale=body["rationale"],
        evidence_ids=list(body["evidence_ids"]),
        semantic_change=semantic,
        counters=counters,
    )


def _request_procedure(
    *,
    client: Any,
    ledger: ProcedureLedger | None,
    journal: _Journal,
    action_index: int,
    horizon: int,
    observations: list[dict[str, Any]],
    deployed: _Deployed | None,
) -> dict[str, Any]:
    prompt = build_prompt(
        action_index=action_index,
        horizon=horizon,
        observations=observations,
        procedure=None if deployed is None else deployed.procedure,
        rationale=None if deployed is None else deployed.rationale,
    )
    counted: int | None = None
    count_auth = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if ledger is not None:
            ledger.begin_token_count()
        try:
            counted = int(client.count_tokens(prompt))
        except TransportError as exc:
            if ledger is not None:
                ledger.finish_token_count(ok=False)
            journal.write_side(
                f"c{action_index:03d}-count-a{attempt}.json",
                {"action_index": action_index, "attempt": attempt, "error": str(exc), "kind": "token_count"},
            )
            if exc.auth_or_quota:
                count_auth = True
            if attempt == MAX_ATTEMPTS:
                raise LifetimeAbort("token_count_failed", stop_task=count_auth) from None
            continue
        if ledger is not None:
            ledger.finish_token_count(ok=True)
        break
    if counted is None or counted > INPUT_TOKEN_CAP:
        journal.write_side(
            f"c{action_index:03d}-cap.json",
            {
                "action_index": action_index,
                "counted_input_tokens": counted,
                "input_token_cap": INPUT_TOKEN_CAP,
                "kind": "input_cap",
                "prompt": prompt,
            },
        )
        if ledger is not None:
            ledger.note_failure()
        raise LifetimeAbort("input_cap")
    generation_auth = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if ledger is not None:
            ledger.begin_attempt()
        try:
            generation = client.generate(prompt, token_count=counted)
        except TransportError as exc:
            _write_generation(
                journal,
                action_index,
                attempt,
                prompt,
                text="",
                generation=None,
                parser_status="transport_error",
                error=str(exc),
                counted=counted,
                world=journal.world,
                condition=journal.condition,
            )
            if ledger is not None:
                ledger.finish_attempt(usage=None, failed=True)
            if exc.auth_or_quota:
                generation_auth = True
            if attempt == MAX_ATTEMPTS:
                raise LifetimeAbort("transport_failed", stop_task=generation_auth) from None
            continue
        try:
            body = parse_procedure(generation.text, cutoff=action_index - 1)
        except ProcedureParseError as exc:
            _write_generation(
                journal,
                action_index,
                attempt,
                prompt,
                text=generation.text,
                generation=generation,
                parser_status="invalid",
                error=str(exc),
                counted=counted,
                world=journal.world,
                condition=journal.condition,
            )
            if ledger is not None:
                ledger.finish_attempt(usage=_priced(generation.usage), failed=True)
            if attempt == MAX_ATTEMPTS:
                raise LifetimeAbort("invalid_procedure") from None
            continue
        _write_generation(
            journal,
            action_index,
            attempt,
            prompt,
            text=generation.text,
            generation=generation,
            parser_status="ok",
            error="",
            counted=counted,
            world=journal.world,
            condition=journal.condition,
            body=body,
        )
        if ledger is not None:
            ledger.finish_attempt(usage=_priced(generation.usage), failed=False)
        return {"procedure_body": body}
    raise LifetimeAbort("request_failed")


def _priced(usage: Usage) -> Usage | None:
    if usage.status != "available" or usage.cost_usd is None:
        return None
    return usage


def _write_generation(
    journal: _Journal,
    action_index: int,
    attempt: int,
    prompt: str,
    *,
    text: str,
    generation: ProcedureGeneration | None,
    parser_status: str,
    error: str,
    counted: int,
    world: str,
    condition: str,
    body: dict[str, Any] | None = None,
) -> None:
    usage = _empty_usage() if generation is None else generation.usage
    payload = {
        "action_index": action_index,
        "attempt": attempt,
        "condition": condition,
        "counted_input_tokens": counted,
        "error": error,
        "finish_reason": None if generation is None else generation.finish_reason,
        "input_token_cap": INPUT_TOKEN_CAP,
        "max_output_tokens": OUTPUT_TOKEN_CAP,
        "model": "gemini-3.5-flash",
        "model_version": None if generation is None else generation.model_version,
        "parsed_procedure": None if body is None else body["procedure"],
        "parser_status": parser_status,
        "prompt": prompt,
        "provider_metadata": None if generation is None else generation.provider_metadata,
        "response_text": text,
        "temperature": 0,
        "thinking_level": "low",
        "usage": usage.to_dict(),
        "world": world,
    }
    journal.write_response(action_index, attempt, payload)


def _write_public_config(
    run_dir: Path,
    world: str,
    condition: str,
    horizon: int,
    schedule: tuple[int, ...],
    lifetime_id: str,
) -> None:
    _write_json(
        run_dir / "config.json",
        {
            "call_schedule": list(schedule),
            "condition": condition,
            "horizon": horizon,
            "lifetime_id": lifetime_id,
            "task": "006",
            "utility_unit": "tenths of a utility point",
            "world_id": world,
        },
    )


def _write_private_table(run_dir: Path, world: str) -> None:
    spec = formula_worlds()[world]
    _write_json(
        run_dir / "private_table.json",
        {
            "change_at_action": 43 if spec["switches"] else None,
            "post": spec["post"],
            "pre": spec["pre"],
            "switches": spec["switches"],
            "world_id": world,
        },
    )


def _write_manifest(
    journal: _Journal,
    result: LifetimeResult,
    schedule: tuple[int, ...],
    extra: dict[str, Any],
    *,
    kind: str,
    reference_algorithm: str | None = None,
) -> None:
    public_text = _jsonl(journal.events)
    private_text = _jsonl(journal.private)
    artifacts_text = json.dumps(journal.artifacts, indent=2, sort_keys=True) + "\n"
    responses = sorted((journal.run_dir / "responses").glob("*.json")) if kind == "gemini-procedure" else []
    known = Decimal("0")
    unknown = 0
    for path in responses:
        if "-a" not in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        usage = data.get("usage") or {}
        if usage.get("status") == "available" and usage.get("cost_usd") is not None:
            known += Decimal(str(usage["cost_usd"]))
        elif data.get("parser_status") == "transport_error" or usage.get("status") == "unknown":
            unknown += 1
    manifest = {
        "artifacts": len(journal.artifacts),
        "baseline_definition": None,
        "call_schedule": list(schedule),
        "committed_index": result.committed_index,
        "condition": result.condition,
        "hashes": {
            "artifacts_sha256": sha256_text(artifacts_text),
            "private_log_sha256": sha256_text(private_text),
            "public_log_sha256": sha256_text(public_text),
        },
        "kind": kind,
        "lifetime_id": result.lifetime_id,
        "llm_checkpoint": "disabled" if kind == "gemini-procedure" else "not_used",
        "protocol": "single-life-v0.3-specified",
        "reason": result.reason,
        "reference_algorithm": reference_algorithm,
        "status": result.status,
        "task": "006",
        "usage_estimate_usd": format(known, "f") if kind == "gemini-procedure" else "0",
        "unknown_usage_attempts": unknown,
        "utility_unit": "tenths of a utility point",
        "world": result.world,
    }
    for key, value in extra.items():
        if key not in manifest:
            manifest[key] = value
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_write_text(journal.run_dir / "manifest.json", text)


def run_reference(
    *,
    world: str,
    algorithm: str,
    run_dir: Path,
    lifetime_id: str | None = None,
    manifest_extra: dict[str, Any] | None = None,
    horizon: int = HORIZON,
    utility_of: UtilityFn | None = None,
) -> LifetimeResult:
    lifetime_id = lifetime_id or f"life-006-{world}-{algorithm}"
    utility_of = utility_of or (lambda index, typ, action: utility_tenths(world, index, typ, action))
    raw = run_reference_actions(utility_of, algorithm=algorithm, horizon=horizon)
    journal = _Journal(run_dir, lifetime_id=lifetime_id, world=world, condition=algorithm, horizon=horizon)
    _write_public_config(run_dir, world, algorithm, horizon, (), lifetime_id)
    _write_private_table(run_dir, world)
    for item in raw:
        event = {
            "action": item["action"],
            "cycle_slot": None,
            "index": item["index"],
            "procedure_hash": None,
            "procedure_version": None,
            "semantic_deployment": False,
            "type": item["type"],
            "utility_tenths": item["utility_tenths"],
        }
        journal.events.append(event)
        journal.private.append({**event, "table_phase": table_phase(world, item["index"]), "world": world})
    journal.flush()
    result = LifetimeResult(
        status="completed",
        reason="",
        stop_task=False,
        committed_index=len(journal.events),
        call_indices=[],
        events=list(journal.events),
        artifacts=[],
        lifetime_id=lifetime_id,
        world=world,
        condition=algorithm,
    )
    extra = {
        "action_order": list(ACTION_ORDER),
        "api_status": "offline_reference",
        "random_seed": RANDOM_SEED if algorithm == RANDOM_ALGORITHM else None,
        **(manifest_extra or {}),
    }
    _write_manifest(journal, result, (), extra, kind="offline-reference", reference_algorithm=algorithm)
    return result
