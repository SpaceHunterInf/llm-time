"""Public procedure DSL, prompt, and offline reference learners.

This module has no reward table and no private sentinel. The host interpreter,
not the model, chooses actions between revision calls.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from single_life.contracts import ContractError, sha256_text

TYPES = ("T1", "T2", "T3")
ACTION_ORDER = ("A|single", "A|pair", "B|single", "B|pair")
HORIZON = 96
REVISION_CALLS = (1, 13, 25, 37, 49, 61, 73, 85)
FROZEN_CALLS = (1, 13, 25)
SMOKE_CALLS = (1, 2)
SMOKE_HORIZON = 2
RATIONALE_LIMIT = 1000
RANDOM_SEED = 601
RANDOM_ALGORITHM = "uniform-random-v1"
LOCAL_ALGORITHM = "local-relearning-v1"
WHOLE_ALGORITHM = "whole-relearning-v1"

INSTRUCTION = (
    "You maintain an executable procedure for three request types, T1, T2, and T3. "
    "Before some actions the host asks you for a procedure. Between those calls, the host "
    "executes the current procedure. You do not choose those intervening actions yourself. "
    "The procedure is a JSON object with exactly the keys T1, T2, and T3. Each value is a "
    "list of 1 to 4 distinct action ids. The permitted action ids are A|single, A|pair, "
    "B|single, and B|pair. A list is read in the order you write it. On each occurrence of "
    "that type since this procedure was deployed, the host uses the next entry and then wraps "
    "to the start. A one-element list repeats that action. A longer list rotates, which you "
    "can use to explore. If you return action lists identical to the current procedure, the "
    "host keeps the occurrence counters. If any list differs, the host sets all three counters "
    "back to zero. Only the action lists control later actions. The rationale is your note; "
    "rewriting it does not by itself move the counters. "
    "utility_tenths is an integer number of tenths of a utility point: 8 means 0.8 and -4 means -0.4. "
    "You see the current request type, the current procedure if one has been deployed, and every "
    "completed record so far. Each record has an index, a type, the action that was executed, and "
    "its utility_tenths. Unchosen actions stay hidden. Later request types are not included. "
    'Reply with one JSON object and no other text: {"procedure": {"T1": ["<action id>"], '
    '"T2": ["<action id>"], "T3": ["<action id>"]}, "rationale": "<at most 1000 characters>", '
    '"evidence_ids": [<integer index of a completed record>]}. '
    "evidence_ids may cite only completed records in the payload. Use an empty list when you cite none."
)

PARSER_POLICY = {
    "schema": {
        "procedure": "object with exactly T1, T2, T3; each a list of 1 to 4 distinct permitted action ids",
        "rationale": "string of at most 1000 characters",
        "evidence_ids": "list of integer indexes of completed observations, none past the cutoff",
    },
    "semantic_identity": "the three action lists, not the rationale",
    "identical_lists": "keep per-type occurrence counters",
    "different_lists": "reset all three counters to zero at deployment",
    "interpreter": "on each occurrence of a type, choose list[counter % length], then increment that counter",
    "invalid_procedure": "retry the same prompt once with no environment action; a second invalid response aborts the lifetime and keeps the committed prefix",
    "transport_error": "retry once; a repeated auth, permission, or quota error stops the batch; any other repeated transport error aborts that lifetime",
    "input_cap": "provider token count must be at most 8192 before generateContent; over-cap calls do not generate",
    "checkpoint": "disabled",
    "evidence_rule": "cited ids must already be completed; the model is not required to cite every completed id",
}


class ProcedureParseError(Exception):
    pass


def type_at(action_index: int) -> str:
    if action_index < 1:
        raise ContractError("action_index starts at 1")
    return TYPES[(action_index - 1) % 3]


def calls_for(condition: str) -> tuple[int, ...]:
    if condition == "revision":
        return REVISION_CALLS
    if condition == "frozen":
        return FROZEN_CALLS
    if condition == "smoke":
        return SMOKE_CALLS
    raise ContractError(f"unknown condition {condition}")


def canonical_procedure(procedure: dict[str, list[str]]) -> dict[str, list[str]]:
    return {typ: list(procedure[typ]) for typ in TYPES}


def procedure_text(procedure: dict[str, list[str]]) -> str:
    return json.dumps(canonical_procedure(procedure), sort_keys=True, separators=(",", ":"))


def procedure_hash(procedure: dict[str, list[str]]) -> str:
    return sha256_text(procedure_text(procedure))


def build_prompt(
    *,
    action_index: int,
    horizon: int,
    observations: list[dict[str, Any]],
    procedure: dict[str, list[str]] | None,
    rationale: str | None,
) -> str:
    payload = {
        "actions_completed": len(observations),
        "actions_remaining": horizon - action_index + 1,
        "current_procedure": None if procedure is None else canonical_procedure(procedure),
        "current_rationale": rationale,
        "current_type": type_at(action_index),
        "evidence_cutoff": action_index - 1,
        "observations": observations,
        "permitted_action_ids": list(ACTION_ORDER),
        "request_types": list(TYPES),
        "utility_unit": "tenths of a utility point",
    }
    text = INSTRUCTION + "\n\npublic_payload:\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    for marker in ("SENTINEL_PRIVATE", "table_phase", "change_at", "oracle_best", "PROGRESS"):
        if marker in text:
            raise ContractError("public prompt contains a private marker")
    return text


def parse_procedure(text: str, *, cutoff: int) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProcedureParseError("response is not JSON") from exc
    if not isinstance(data, dict) or set(data) != {"procedure", "rationale", "evidence_ids"}:
        raise ProcedureParseError("response keys are not procedure, rationale, evidence_ids")
    procedure = data["procedure"]
    rationale = data["rationale"]
    evidence = data["evidence_ids"]
    if not isinstance(procedure, dict) or set(procedure) != set(TYPES):
        raise ProcedureParseError("procedure must have exactly T1, T2, and T3")
    normalized: dict[str, list[str]] = {}
    for typ in TYPES:
        items = procedure[typ]
        if not isinstance(items, list) or not 1 <= len(items) <= 4:
            raise ProcedureParseError(f"{typ} list must contain 1 to 4 actions")
        if any(type(item) is not str or item not in ACTION_ORDER for item in items):
            raise ProcedureParseError(f"{typ} has an action outside the permitted ids")
        if len(set(items)) != len(items):
            raise ProcedureParseError(f"{typ} repeats an action")
        normalized[typ] = list(items)
    if not isinstance(rationale, str) or len(rationale) > RATIONALE_LIMIT:
        raise ProcedureParseError("rationale must be a string of at most 1000 characters")
    if not isinstance(evidence, list):
        raise ProcedureParseError("evidence_ids must be a list")
    checked: list[int] = []
    for item in evidence:
        if type(item) is not int or not 1 <= item <= cutoff:
            raise ProcedureParseError("evidence id is outside the completed prefix")
        checked.append(item)
    return {
        "procedure": normalized,
        "rationale": rationale,
        "evidence_ids": checked,
        "content_hash": procedure_hash(normalized),
    }


def initial_counters() -> dict[str, int]:
    return {typ: 0 for typ in TYPES}


def next_action(procedure: dict[str, list[str]], counters: dict[str, int], typ: str) -> tuple[str, int]:
    rules = procedure[typ]
    slot = counters[typ] % len(rules)
    return rules[slot], slot


class CacheLearner:
    """Per-type sample-then-exploit. A mismatch clears caches and keeps the new cell."""

    def __init__(self, *, whole: bool) -> None:
        self.whole = whole
        self.cache: dict[str, dict[str, int]] = {typ: {} for typ in TYPES}

    def choose(self, typ: str) -> str:
        missing = [action for action in ACTION_ORDER if action not in self.cache[typ]]
        if missing:
            return missing[0]
        best = ACTION_ORDER[0]
        best_value = self.cache[typ][best]
        for action in ACTION_ORDER[1:]:
            value = self.cache[typ][action]
            if value > best_value:
                best = action
                best_value = value
        return best

    def observe(self, typ: str, action: str, utility: int) -> None:
        previous = self.cache[typ].get(action)
        if previous is not None and previous != utility:
            if self.whole:
                self.cache = {name: {} for name in TYPES}
            else:
                self.cache[typ] = {}
        self.cache[typ][action] = utility


def run_reference_actions(
    utility_of: Callable[[int, str, str], int],
    *,
    algorithm: str,
    horizon: int = HORIZON,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    learner: CacheLearner | None
    rng = random.Random(seed)
    if algorithm == LOCAL_ALGORITHM:
        learner = CacheLearner(whole=False)
    elif algorithm == WHOLE_ALGORITHM:
        learner = CacheLearner(whole=True)
    elif algorithm == RANDOM_ALGORITHM:
        learner = None
    else:
        raise ContractError(f"unknown reference {algorithm}")
    events: list[dict[str, Any]] = []
    for index in range(1, horizon + 1):
        typ = type_at(index)
        if learner is None:
            action = ACTION_ORDER[rng.randrange(len(ACTION_ORDER))]
        else:
            action = learner.choose(typ)
        utility = int(utility_of(index, typ, action))
        if learner is not None:
            learner.observe(typ, action, utility)
        events.append(
            {
                "index": index,
                "type": typ,
                "action": action,
                "utility_tenths": utility,
            }
        )
    return events
