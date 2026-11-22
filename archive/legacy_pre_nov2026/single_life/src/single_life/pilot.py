"""Bounded Gemini note-retention diagnostic on the Bernoulli fixture.

Offline UCB and random runs are references. The live portion is one smoke
lifetime plus four 48-step lifetimes. Checkpoint/resume of the Gemini policy
is disabled. This module does not decide whether a reward gap is harmful
memory, novelty, or RSI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from single_life.artifacts import NullUpdater
from single_life.baselines import UCB_DEFINITION, make_policy
from single_life.cli import (
    AGENT_IDS,
    _LogView,
    _public_lines,
    atomic_write_bytes,
    atomic_write_text,
)
from single_life.contracts import (
    assert_separated_streams,
    canonical_dumps,
    sha256_bytes,
    sha256_text,
)
from single_life.fixtures import RunSpec, load_spec, spec_from_dict
from single_life.gemini_adapter import (
    INSTRUCTION,
    INPUT_TOKEN_CAP,
    INPUT_USD_PER_MILLION,
    MAX_ATTEMPTS,
    MAX_REQUESTS,
    MODEL_ID,
    NOTE_CHAR_LIMIT,
    OBSERVATION_WINDOW,
    OUTPUT_TOKEN_CAP,
    OUTPUT_USD_PER_MILLION,
    PARSER_POLICY,
    PILOT_DECISIONS,
    PLANNED_DECISIONS,
    SMOKE_DECISIONS,
    TEMPERATURE,
    THINKING_LEVEL,
    GeminiClient,
    GeminiNotePolicy,
    NoteCarryUpdater,
    NoteHandoff,
    TextClient,
    TransportError,
    attempt_cap_usd,
    completion_cap_usd,
    load_api_key,
    pricing_record,
    redact,
    token_cost,
)
from single_life.identity import dependency_identity, source_identity
from single_life.progress import ProgressLedger
from single_life.runner import RunResult, clock_fields, run_lifetime

WORLD_FILES = {"w1": "005_world_w1.json", "w2": "005_world_w2.json"}
CONDITIONS = (
    ("recent", "llm-recent", False, "recent-only"),
    ("persistent", "llm-note", True, "persistent-note"),
)
REFERENCES = (("ucb", "ucb-reference"), ("random", "random-reference"))
SUCCESS_PARSER = frozenset({"ok", "truncated"})
TIMING_NOTE = (
    "A note proposed at decision t is written before that decision's reward and "
    "deployed at decision t+1. Its creation cutoff is t-1. The prompt at decision "
    "17 therefore includes observations through action 16 and the note proposed at "
    "16. For the second world, the first reward scored under the middle probability "
    "table is the outcome of decision 17, and it is visible starting at decision 18. "
    "Those boundaries are evaluator-side and are not part of the prompt."
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_dir(root: Path) -> Path:
    return root / "experiments" / "005-gemini-lifetime"


def smoke_run_dir(root: Path) -> Path:
    return root / "runs" / "005-gemini-smoke"


def pilot_run_dir(root: Path, world: str, condition: str) -> Path:
    return root / "runs" / "005-gemini-pilot" / world / condition


def reference_run_dir(root: Path, world: str, agent: str) -> Path:
    return root / "runs" / "005-gemini-references" / world / agent


def ensure_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing nonempty directory {path}")


def assert_outputs_fresh(root: Path) -> None:
    for path in (
        experiment_dir(root),
        smoke_run_dir(root),
        root / "runs" / "005-gemini-pilot",
        root / "runs" / "005-gemini-references",
    ):
        ensure_fresh(path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_world_dict(root: Path, world: str) -> dict[str, Any]:
    path = root / "configs" / WORLD_FILES[world]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not an object")
    return data


def with_agent(data: dict[str, Any], agent: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(data))
    copied["agent"] = agent
    return copied


def smoke_dict(root: Path) -> dict[str, Any]:
    data = with_agent(load_world_dict(root, "w1"), "llm-note")
    data["fixture_id"] = "005-smoke"
    data["lifetime_id"] = "life-005-smoke"
    data["innovation_seed"] = "innovation-005-smoke-v1"
    data["schedule_seed"] = "schedule-005-smoke-v1"
    data["agent_seed"] = "agent-005-smoke-v1"
    data["horizon"] = SMOKE_DECISIONS
    data["interaction_budget"] = SMOKE_DECISIONS
    data["artifact_period"] = SMOKE_DECISIONS
    segment = dict(data["segments"][0])
    segment["end_index"] = SMOKE_DECISIONS
    data["segments"] = [segment]
    return data


def current_settings() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "thinking_level": THINKING_LEVEL,
        "input_token_cap": INPUT_TOKEN_CAP,
        "max_output_tokens": OUTPUT_TOKEN_CAP,
        "instruction_sha256": sha256_text(INSTRUCTION),
        "note_char_limit": NOTE_CHAR_LIMIT,
        "observation_window": OBSERVATION_WINDOW,
        "max_attempts_per_decision": MAX_ATTEMPTS,
        "ucb_window": 12,
    }


def _response_accounting(responses_dir: Path | None) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    if responses_dir is not None and responses_dir.is_dir():
        for path in sorted(responses_dir.glob("d*-a*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                attempts.append(payload)
    known = Decimal("0")
    unknown = 0
    succeeded = 0
    failed = 0
    input_tokens = 0
    output_tokens = 0
    thoughts = 0
    elapsed = 0.0
    for item in attempts:
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        if item.get("parser_status") in SUCCESS_PARSER:
            succeeded += 1
        else:
            failed += 1
        elapsed += float(item.get("elapsed_seconds") or 0.0)
        if usage.get("status") == "available" and usage.get("cost_usd") is not None:
            known += Decimal(str(usage["cost_usd"]))
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            thoughts += int(usage.get("thoughts_tokens") or 0)
        else:
            unknown += 1
    reserved = attempt_cap_usd() * unknown
    return {
        "attempts": len(attempts),
        "parser_successes": succeeded,
        "parser_or_transport_failures": failed,
        "known_usage_attempts": len(attempts) - unknown,
        "unknown_usage_attempts": unknown,
        "known_spend_usd": format(known, "f"),
        "unknown_reserved_usd": format(reserved, "f"),
        "usage_estimate_usd": format(known + reserved, "f"),
        "known_input_tokens": input_tokens,
        "known_output_tokens_including_thinking": output_tokens,
        "known_thoughts_tokens": thoughts,
        "provider_elapsed_seconds": round(elapsed, 3),
        "invoice": False,
    }


class LifetimeJournal:
    """Atomic run logs. Decision commits do not resume a crashed lifetime."""

    def __init__(
        self,
        *,
        run_dir: Path,
        spec: RunSpec,
        run_id: str,
        condition: str,
        memory_condition: str,
        updater_id: str,
        note_carry: bool,
        llm: bool,
        code: dict[str, Any],
        dependencies: dict[str, Any],
        on_committed: Callable[[], None] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.spec = spec
        self.run_id = run_id
        self.condition = condition
        self.memory_condition = memory_condition
        self.updater_id = updater_id
        self.note_carry = note_carry
        self.llm = llm
        self.code = code
        self.dependencies = dependencies
        self.on_committed = on_committed
        self.config_text = canonical_dumps(spec.to_dict())
        self.config_bytes = self.config_text.encode("utf-8")
        self.manifest: dict[str, Any] = {}
        self.manifest_text = ""

    def commit(self, state: Any) -> None:
        if self.on_committed is not None:
            self.on_committed()
        self._persist(
            status="running",
            artifacts=[artifact.to_dict() for artifact in state.store.versions],
            observed_events=[event.to_dict() for event in state.observed],
            private_events=[event.to_dict() for event in state.private],
            public_failures=list(state.failures),
            public_abort=None,
            error_private=None,
            interactions_consumed=state.interactions_consumed,
            compute_units_consumed=state.compute_units_consumed,
            spec_sha256=self.spec.fingerprint(),
        )

    def write(self, status: str, result: RunResult | None) -> None:
        if result is None:
            self._persist(
                status=status,
                artifacts=[],
                observed_events=[],
                private_events=[],
                public_failures=[],
                public_abort=None,
                error_private=None,
                interactions_consumed=0,
                compute_units_consumed=0,
                spec_sha256=self.spec.fingerprint(),
            )
            return
        self._persist(
            status=status,
            artifacts=list(result.artifacts),
            observed_events=list(result.observed_events),
            private_events=list(result.private_events),
            public_failures=list(result.public_failures),
            public_abort=result.public_abort,
            error_private=result.error_private,
            interactions_consumed=result.interactions_consumed,
            compute_units_consumed=result.compute_units_consumed,
            spec_sha256=result.spec_sha256,
        )

    def _persist(
        self,
        *,
        status: str,
        artifacts: list[dict[str, Any]],
        observed_events: list[dict[str, Any]],
        private_events: list[dict[str, Any]],
        public_failures: list[dict[str, Any]],
        public_abort: dict[str, Any] | None,
        error_private: str | None,
        interactions_consumed: int,
        compute_units_consumed: int,
        spec_sha256: str,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.run_dir / "config.json", self.config_bytes)
        view = _LogView(artifacts, observed_events, public_failures)
        public_lines = _public_lines(view)
        private_lines = [canonical_dumps(event) for event in private_events]
        if public_abort is not None:
            public_lines.append(canonical_dumps(public_abort))
            private_lines.append(
                canonical_dumps(
                    {
                        "record_type": public_abort["failure_kind"],
                        "decision_index": public_abort["decision_index"],
                        "message": error_private,
                    }
                )
            )
        public_text = "".join(line + "\n" for line in public_lines)
        private_text = "".join(line + "\n" for line in private_lines)
        artifacts_text = canonical_dumps(artifacts)
        atomic_write_text(self.run_dir / "public_log.jsonl", public_text)
        atomic_write_text(self.run_dir / "private_log.jsonl", private_text)
        atomic_write_text(self.run_dir / "artifacts.json", artifacts_text)
        reward = sum(int(event["observed_outcome"]) for event in observed_events)
        if self.llm:
            usage = _response_accounting(self.run_dir / "responses")
            usage_status = "usage_estimate"
        else:
            usage = {
                "attempts": 0,
                "parser_successes": 0,
                "parser_or_transport_failures": 0,
                "known_usage_attempts": 0,
                "unknown_usage_attempts": 0,
                "known_spend_usd": "0",
                "unknown_reserved_usd": "0",
                "usage_estimate_usd": "0",
                "known_input_tokens": 0,
                "known_output_tokens_including_thinking": 0,
                "known_thoughts_tokens": 0,
                "provider_elapsed_seconds": 0,
                "invoice": False,
            }
            usage_status = "offline_reference"
        manifest = {
            **clock_fields(),
            "run_id": self.run_id,
            "experiment_id": self.spec.experiment_id,
            "status": status,
            "condition": self.condition,
            "memory_condition": self.memory_condition,
            "note_carry": self.note_carry,
            "agent_name": self.spec.agent_name,
            "agent_id": AGENT_IDS[self.spec.agent_name],
            "updater_id": self.updater_id,
            "llm_checkpoint": "disabled" if self.llm else "not_used",
            "seeds_pilot_memory": False,
            "config_id": self.spec.fixture_id,
            "config_sha256": sha256_bytes(self.config_bytes),
            "spec_sha256": spec_sha256,
            "code": self.code,
            "dependency_identity": self.dependencies,
            "seeds": {
                "schedule_seed": self.spec.schedule_seed,
                "innovation_seed": self.spec.innovation_seed,
                "agent_seed": self.spec.agent_seed,
                "lifetime_id": self.spec.lifetime_id,
            },
            "resource_caps": {
                "interaction_budget": self.spec.interaction_budget,
                "compute_budget": self.spec.compute_budget,
                "horizon": self.spec.horizon,
            },
            "actual_usage": {
                "interactions_consumed": interactions_consumed,
                "compute_units_consumed": compute_units_consumed,
                "observed_events": len(observed_events),
                "invalid_actions": len(public_failures),
                "artifacts": len(artifacts),
                "realized_reward": reward,
                "api_usage_status": usage_status,
                **usage,
            },
            "paths": {
                "public_log": "public_log.jsonl",
                "private_log": "private_log.jsonl",
                "artifacts": "artifacts.json",
                "config": "config.json",
                "responses": "responses" if self.llm else None,
            },
            "hashes": {
                "public_log_sha256": sha256_text(public_text),
                "private_log_sha256": sha256_text(private_text),
                "artifacts_sha256": sha256_text(artifacts_text),
            },
            "completion_reason": status,
            "failure_status": None if status in {"running", "completed"} else status,
            "performance_claim": None,
            "written_at_utc": utc_now(),
            "baseline_definition": UCB_DEFINITION,
            "preservation_limit": (
                "Files are replaced atomically. A crash between files is an inconsistent "
                "directory and is not resumed. LLM checkpoints are disabled."
            ),
        }
        if self.llm:
            manifest["model"] = {
                "id": MODEL_ID,
                "temperature": TEMPERATURE,
                "thinking_level": THINKING_LEVEL,
                "max_output_tokens": OUTPUT_TOKEN_CAP,
                "input_token_cap": INPUT_TOKEN_CAP,
            }
            manifest["parser_policy"] = PARSER_POLICY
        manifest_text = canonical_dumps(manifest)
        assert_separated_streams(manifest, public_text)
        atomic_write_text(self.run_dir / "manifest.json", manifest_text)
        self.manifest = manifest
        self.manifest_text = manifest_text


def run_note_lifetime(
    *,
    root: Path,
    spec: RunSpec,
    run_dir: Path,
    run_id: str,
    carry: bool,
    world: str,
    condition: str,
    memory_condition: str,
    client: TextClient,
    ledger: ProgressLedger,
    phase: str,
    evaluator: Any | None = None,
    stop_after: int | None = None,
) -> tuple[RunResult, GeminiNotePolicy]:
    ensure_fresh(run_dir)
    responses = run_dir / "responses"
    handoff = NoteHandoff(carry=carry)
    updater = NoteCarryUpdater(handoff)
    policy = GeminiNotePolicy(
        carry_note=carry,
        client=client,
        handoff=handoff,
        responses_dir=responses,
        ledger=ledger,
        world=world,
        condition=condition,
    )
    journal = LifetimeJournal(
        run_dir=run_dir,
        spec=spec,
        run_id=run_id,
        condition=condition,
        memory_condition=memory_condition,
        updater_id=updater.updater_id,
        note_carry=carry,
        llm=True,
        code=source_identity(root),
        dependencies=dependency_identity(root),
        on_committed=ledger.complete_decision,
    )
    ledger.set_phase(phase)
    journal.write("running", None)
    result = run_lifetime(
        spec,
        policy=policy,
        updater=updater,
        evaluator=evaluator,
        stop_after=stop_after,
        on_commit=journal.commit,
    )
    journal.write(result.status, result)
    return result, policy


def run_reference_lifetime(
    *,
    root: Path,
    spec: RunSpec,
    run_dir: Path,
    run_id: str,
    condition: str,
) -> RunResult:
    ensure_fresh(run_dir)
    policy = make_policy(spec.agent_name, spec.ucb_window)
    updater = NullUpdater()
    journal = LifetimeJournal(
        run_dir=run_dir,
        spec=spec,
        run_id=run_id,
        condition=condition,
        memory_condition="none",
        updater_id=updater.updater_id,
        note_carry=False,
        llm=False,
        code=source_identity(root),
        dependencies=dependency_identity(root),
    )
    journal.write("running", None)
    result = run_lifetime(spec, policy=policy, updater=updater, on_commit=journal.commit)
    journal.write(result.status, result)
    return result


def render_preflight(record: dict[str, Any]) -> str:
    lines = [
        "PREFLIGHT gemini note-retention diagnostic",
        (
            f"model={record['model']} temperature={record['temperature']} "
            f"thinking_level={record['thinking_level']}"
        ),
        (
            f"input_token_cap={record['input_token_cap']} "
            f"max_output_tokens={record['max_output_tokens']} "
            "output_cap_includes_thinking=true"
        ),
        (
            f"planned_decisions={record['planned_decisions']} "
            f"smoke={record['planned_smoke']} pilot={record['planned_pilot']} "
            f"max_attempts_including_retries={record['max_attempts']}"
        ),
        f"price_source={record['price_source']}",
        (
            f"price_page_updated={record['price_page_updated']} "
            f"price_checked_on={record['price_checked_on']}"
        ),
        (
            f"input_usd_per_million={record['input_usd_per_million']} "
            "output_usd_per_million_including_thinking="
            f"{record['output_usd_per_million_including_thinking']}"
        ),
        (
            f"per_attempt_cap_usd={record['per_attempt_cap_usd']} = "
            f"{INPUT_TOKEN_CAP} * {INPUT_USD_PER_MILLION}/1e6 + "
            f"{OUTPUT_TOKEN_CAP} * {OUTPUT_USD_PER_MILLION}/1e6"
        ),
        (
            f"conservative_completion_cap_usd={record['completion_cap_usd']} = "
            f"per_attempt_cap * {MAX_REQUESTS}"
        ),
        (
            f"illustrative_short_run_usd={record['illustrative_short_run_usd']} "
            "assumes 196 single attempts at 800 input and 300 output tokens; "
            "this is not a measurement"
        ),
        (
            f"configured_spend_cap_usd={record['completion_cap_usd']} "
            "user_ceiling_usd=20"
        ),
        (
            "projection=after 4 completed decisions, remaining cost uses the observed "
            "mean spend per completed decision; before that, remaining uses two full-cap "
            "attempts per unfinished decision; the conservative cap is not replaced"
        ),
        "unknown_usage=reserved at the per-attempt cap, never priced as zero",
        "billing_note=usage estimate from published token rates, not a provider invoice",
        str(record["other_published_rows"]),
    ]
    return "\n".join(lines) + "\n"


def build_registration(root: Path) -> dict[str, Any]:
    prices = pricing_record()
    illustrative = token_cost(800, 300) * PLANNED_DECISIONS
    worlds = {}
    for name, filename in WORLD_FILES.items():
        path = root / "configs" / filename
        raw = path.read_bytes()
        worlds[name] = {
            "path": f"configs/{filename}",
            "sha256": sha256_bytes(raw),
            "spec": spec_from_dict(json.loads(raw.decode("utf-8"))).to_dict(),
        }
    smoke = spec_from_dict(smoke_dict(root)).to_dict()
    pilots = {}
    for world in WORLD_FILES:
        base = load_world_dict(root, world)
        for condition, agent, carry, memory in CONDITIONS:
            spec = spec_from_dict(with_agent(base, agent))
            pilots[f"{world}-{condition}"] = {
                "agent": agent,
                "note_carry": carry,
                "memory_condition": memory,
                "spec_sha256": spec.fingerprint(),
                "segment_sha256": sha256_text(canonical_dumps(base["segments"])),
                "lifetime_id": spec.lifetime_id,
                "innovation_seed": spec.innovation_seed,
            }
    record = {
        "experiment_id": "005-gemini-lifetime",
        "registered_at_utc": utc_now(),
        "settings": current_settings(),
        "instruction": INSTRUCTION,
        "parser_policy": PARSER_POLICY,
        "pricing": prices,
        "illustrative_short_run_usd": format(illustrative, "f"),
        "planned_smoke": SMOKE_DECISIONS,
        "planned_pilot": PILOT_DECISIONS,
        "planned_decisions": PLANNED_DECISIONS,
        "max_attempts": MAX_REQUESTS,
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "thinking_level": THINKING_LEVEL,
        "input_token_cap": INPUT_TOKEN_CAP,
        "max_output_tokens": OUTPUT_TOKEN_CAP,
        "per_attempt_cap_usd": prices["per_attempt_cap_usd"],
        "completion_cap_usd": prices["completion_cap_usd"],
        "price_source": prices["price_source"],
        "price_page_updated": prices["price_page_updated"],
        "price_checked_on": prices["price_checked_on"],
        "input_usd_per_million": prices["input_usd_per_million"],
        "output_usd_per_million_including_thinking": prices[
            "output_usd_per_million_including_thinking"
        ],
        "other_published_rows": prices["other_published_rows"],
        "failure_rules": PARSER_POLICY,
        "checkpoint": "disabled",
        "reference_settings": {
            "policies": ["ucb", "random"],
            "ucb_window": 12,
            "ucb_definition": UCB_DEFINITION,
            "updater": "null-updater-v1",
            "cost_usd": "0",
            "label": "offline reference, not an equally costly LLM method",
            "shared_innovation": "same lifetime_id and innovation_seed as the paired LLM runs in that world",
        },
        "worlds": worlds,
        "smoke_spec": smoke,
        "pilot_index": pilots,
        "timing_note": TIMING_NOTE,
        "source": source_identity(root),
        "dependency_identity": dependency_identity(root),
        "interpretation_limit": (
            "Integration baseline and a narrow note-retention comparison. "
            "Not a novelty claim, a harmful-memory result, or RSI."
        ),
    }
    return record


def confirm_model(api_key: str) -> dict[str, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}"
    last: TransportError | None = None
    payload: dict[str, Any] | None = None
    for attempt in (1, 2):
        request = urllib.request.Request(
            url,
            headers={"x-goog-api-key": api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            payload = parsed if isinstance(parsed, dict) else None
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp is not None else ""
            last = TransportError(
                redact(f"HTTP {exc.code}: {detail}", api_key),
                status=exc.code,
                auth_or_quota=exc.code in {401, 403, 429},
            )
            if exc.code < 500 or attempt == 2:
                raise last from None
        except urllib.error.URLError as exc:
            last = TransportError(redact(f"URL error: {exc.reason}", api_key))
            if attempt == 2:
                raise last from None
    else:
        raise last or TransportError("model check failed")
    if not isinstance(payload, dict):
        raise TransportError("model endpoint did not return an object")
    name = str(payload.get("name", ""))
    if MODEL_ID not in name:
        raise TransportError(f"model endpoint returned {name!r}")
    display = payload.get("displayName")
    return {
        "name": name,
        "display_name": "" if display is None else str(display),
    }


def smoke_problem(run_dir: Path, result: RunResult) -> str | None:
    if result.status != "completed":
        return f"smoke status {result.status}: {result.error_private}"
    if result.public_failures:
        return "smoke produced an invalid action"
    if result.committed_index != SMOKE_DECISIONS:
        return f"smoke committed {result.committed_index}"
    paths = sorted((run_dir / "responses").glob("d*-a*.json"))
    if len(paths) < SMOKE_DECISIONS:
        return "smoke wrote fewer responses than decisions"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("parser_status") not in SUCCESS_PARSER:
            return f"{path.name} parser_status={data.get('parser_status')}"
        if data.get("finish_reason") != "STOP":
            return f"{path.name} finish_reason={data.get('finish_reason')}"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _responses(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    directory = run_dir / "responses"
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("d*-a*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict):
            item["path"] = str(path)
            rows.append(item)
    return rows


def _public_payload(prompt: str) -> dict[str, Any]:
    marker = "public_payload:\n"
    index = prompt.rfind(marker)
    if index < 0:
        return {}
    payload = json.loads(prompt[index + len(marker) :])
    return payload if isinstance(payload, dict) else {}


def _accepted(responses: list[dict[str, Any]], decision: int) -> dict[str, Any] | None:
    matches = [
        item
        for item in responses
        if item.get("decision_index") == decision and item.get("parser_status") in SUCCESS_PARSER
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: int(item.get("attempt") or 0))


def _quote(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated in summary)"


def describe_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = load_spec(run_dir / "config.json")
    public = _read_jsonl(run_dir / "public_log.jsonl")
    events = [row for row in public if row.get("record_type") == "observed_event"]
    failures = [row for row in public if row.get("record_type") == "public_failure"]
    responses = _responses(run_dir)
    by_index = {int(event["action_index"]): event for event in events}
    deployed: dict[int, str] = {}
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("kind") != "strategy-note":
                continue
            deployment = artifact.get("deployment_decision_index")
            content = artifact.get("content") if isinstance(artifact.get("content"), dict) else {}
            note = content.get("strategy_note")
            if isinstance(deployment, int) and isinstance(note, str):
                deployed[deployment] = note
    rows = []
    previous_note = None
    for decision in range(1, spec.horizon + 1):
        event = by_index.get(decision)
        failure = next((item for item in failures if item.get("decision_index") == decision), None)
        accepted = _accepted(responses, decision)
        proposed = None if accepted is None else accepted.get("parsed_note")
        payload = {} if accepted is None else _public_payload(str(accepted.get("prompt") or ""))
        retained = payload.get("retained_strategy_note")
        note_now = deployed.get(decision)
        changed = note_now is not None and note_now != previous_note
        if note_now is not None:
            previous_note = note_now
        action = None
        reward = None
        kind = "missing"
        if event is not None:
            action = event.get("chosen_action")
            reward = event.get("observed_outcome")
            kind = "observed"
        elif failure is not None:
            action = failure.get("attempted_action")
            kind = "invalid_action"
        rows.append(
            {
                "decision": decision,
                "action": action,
                "reward": reward,
                "kind": kind,
                "retained_strategy_note": _quote(retained if isinstance(retained, str) else None, 180),
                "proposed_note": _quote(proposed if isinstance(proposed, str) else None, 180),
                "note_changed": changed,
            }
        )
    windows = []
    ranges = ((1, spec.horizon),)
    if spec.fixture_id == "005-w2":
        ranges = ((1, 16), (17, 34), (35, 48))
    for start, end in ranges:
        chosen = [event for event in events if start <= int(event["action_index"]) <= end]
        counts: dict[str, int] = {}
        for event in chosen:
            action = str(event["chosen_action"])
            counts[action] = counts.get(action, 0) + 1
        modal = None
        if counts:
            modal = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        oracle = None
        for segment in spec.segments:
            if segment.start_index == start:
                oracle = segment.unreachable_metadata.get("oracle_best_arm")
        windows.append(
            {
                "actions": f"{start}_{end}",
                "observed": len(chosen),
                "reward_sum": sum(int(event["observed_outcome"]) for event in chosen),
                "action_counts": counts,
                "modal_action": modal,
                "evaluator_oracle_arm": oracle,
                "evaluator_only": True,
            }
        )
    excerpts = []
    for decision in (1, 16, 17):
        accepted = _accepted(responses, decision)
        if accepted is None:
            continue
        payload = _public_payload(str(accepted.get("prompt") or ""))
        observations = payload.get("observations")
        last_index = None
        if isinstance(observations, list) and observations:
            last = observations[-1]
            if isinstance(last, dict):
                last_index = last.get("action_index")
        excerpts.append(
            {
                "decision": decision,
                "attempt": accepted.get("attempt"),
                "action": accepted.get("parsed_action"),
                "proposed_note": _quote(
                    accepted.get("parsed_note") if isinstance(accepted.get("parsed_note"), str) else None,
                    300,
                ),
                "retained_strategy_note": payload.get("retained_strategy_note")
                if not isinstance(payload.get("retained_strategy_note"), str)
                else _quote(payload.get("retained_strategy_note"), 300),
                "observation_count": len(observations) if isinstance(observations, list) else 0,
                "latest_observation_index": last_index,
                "response_path": accepted.get("path"),
                "quoted_prompt_has_evaluator_annotation": False,
            }
        )
    usage = manifest.get("actual_usage", {})
    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "condition": manifest.get("condition"),
        "memory_condition": manifest.get("memory_condition"),
        "agent_name": manifest.get("agent_name"),
        "status": manifest.get("status"),
        "lifetime_id": spec.lifetime_id,
        "innovation_seed": spec.innovation_seed,
        "realized_reward": usage.get("realized_reward"),
        "invalid_actions": usage.get("invalid_actions"),
        "api_usage_status": usage.get("api_usage_status"),
        "usage_estimate_usd": usage.get("usage_estimate_usd"),
        "known_spend_usd": usage.get("known_spend_usd"),
        "unknown_reserved_usd": usage.get("unknown_reserved_usd"),
        "provider_elapsed_seconds": usage.get("provider_elapsed_seconds"),
        "windows": windows,
        "timeline": rows,
        "excerpts": excerpts,
        "note_changes": sum(1 for row in rows if row["note_changed"]),
    }


def write_svg(path: Path, runs: list[dict[str, Any]]) -> None:
    band_h = 36
    width = 1100
    height = 28 + band_h * max(1, len(runs))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    colors = {"arm_a": "#1b4f72", "arm_b": "#b9770e", "arm_c": "#1e8449"}
    for index, run in enumerate(runs):
        top = 24 + index * band_h
        label = f"{run.get('condition')} {run.get('agent_name')} reward={run.get('realized_reward')}"
        parts.append(f'<text x="8" y="{top - 6}" font-size="11" font-family="sans-serif">{label}</text>')
        rows = run.get("timeline") or []
        if not rows:
            continue
        cell = min(18, (width - 20) / len(rows))
        for row in rows:
            x = 8 + (int(row["decision"]) - 1) * cell
            color = colors.get(str(row.get("action")), "#922b21")
            reward = row.get("reward")
            h = 10 if reward == 1 else 6
            parts.append(
                f'<rect x="{x:.1f}" y="{top + (14 - h)}" width="{max(cell - 1, 1):.1f}" '
                f'height="{h}" fill="{color}"/>'
            )
            if row.get("note_changed"):
                parts.append(f'<circle cx="{x + 2:.1f}" cy="{top + 18}" r="1.5" fill="#111111"/>')
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def write_summary(root: Path, *, status: str, blocker: str | None, ledger: ProgressLedger) -> None:
    experiment = experiment_dir(root)
    experiment.mkdir(parents=True, exist_ok=True)
    described = []
    smoke = smoke_run_dir(root)
    groups = [("smoke", smoke)] if (smoke / "manifest.json").is_file() else []
    for world in WORLD_FILES:
        for agent, _label in REFERENCES:
            path = reference_run_dir(root, world, agent)
            if (path / "manifest.json").is_file():
                groups.append(("reference", path))
        for condition, _agent, _carry, _memory in CONDITIONS:
            path = pilot_run_dir(root, world, condition)
            if (path / "manifest.json").is_file():
                groups.append(("pilot", path))
    for kind, path in groups:
        item = describe_run(path)
        item["group"] = kind
        described.append(item)
        atomic_write_text(
            path / "timeline.json",
            json.dumps(item["timeline"], indent=2, sort_keys=True) + "\n",
        )
    write_svg(experiment / "timeline.svg", described)
    snapshot = ledger.snapshot()
    succeeded = 0
    failed = 0
    for item in described:
        if item["group"] == "reference":
            continue
        responses = _responses(Path(item["run_dir"]))
        for response in responses:
            if response.get("parser_status") in SUCCESS_PARSER:
                succeeded += 1
            else:
                failed += 1
    summary = {
        "experiment_id": "005-gemini-lifetime",
        "status": status,
        "blocker": blocker,
        "performance_claim": None,
        "interpretation_limit": (
            "One paired lifetime per world is not a population effect. "
            "A reward difference is not, by itself, harmful memory. "
            "This is an integration baseline and a narrow note-retention comparison."
        ),
        "timing_note": TIMING_NOTE,
        "progress": snapshot,
        "live_attempts": {
            "parser_successes": succeeded,
            "parser_or_transport_failures": failed,
            "ledger_attempts": snapshot["attempts"],
            "ledger_retries": snapshot["retries"],
            "ledger_failures": snapshot["failures"],
        },
        "runs": described,
        "written_at_utc": utc_now(),
    }
    atomic_write_text(
        experiment / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# 005 Gemini lifetime",
        "",
        f"Status: {status}.",
        "",
        TIMING_NOTE,
        "",
        "Reward sums below are descriptive. Reference rows cost 0 and are not LLM methods.",
        "",
        "| group | run | status | reward | usage estimate USD | invalid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in described:
        lines.append(
            f"| {item['group']} | {item['run_id']} | {item['status']} | "
            f"{item['realized_reward']} | {item['usage_estimate_usd']} | {item['invalid_actions']} |"
        )
    lines.extend(["", "Evaluator-only window counts are in summary.json under each run's windows.", ""])
    if blocker:
        lines.extend(["Blocker:", "", blocker, ""])
    atomic_write_text(experiment / "summary.md", "\n".join(lines))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute(root: Path) -> int:
    assert_outputs_fresh(root)
    experiment = experiment_dir(root)
    experiment.mkdir(parents=True, exist_ok=True)
    ledger = ProgressLedger(
        experiment / "progress.json",
        planned_smoke=SMOKE_DECISIONS,
        planned_pilot=PILOT_DECISIONS,
    )
    secret = ""
    closed = False

    def close(status: str) -> None:
        nonlocal closed
        if not closed:
            ledger.finish(status)
            closed = True

    try:
        registration = build_registration(root)
        _write_json(experiment / "registration.json", registration)
        for world in WORLD_FILES:
            base = load_world_dict(root, world)
            segment_hash = sha256_text(canonical_dumps(base["segments"]))
            for agent, label in REFERENCES:
                spec = spec_from_dict(with_agent(base, agent))
                if sha256_text(canonical_dumps(base["segments"])) != segment_hash:
                    raise RuntimeError("reference segments drifted")
                print(
                    f"REFERENCE offline world={world} policy={agent} cost_usd=0",
                    flush=True,
                )
                result = run_reference_lifetime(
                    root=root,
                    spec=spec,
                    run_dir=reference_run_dir(root, world, agent),
                    run_id=f"{world}-{agent}",
                    condition=label,
                )
                if result.status != "completed":
                    close("failed")
                    write_summary(root, status="failed", blocker=f"reference {world} {agent} {result.status}", ledger=ledger)
                    return 1
        preflight = render_preflight(registration)
        atomic_write_text(experiment / "preflight.txt", preflight)
        print(preflight, flush=True)
        secret = load_api_key()
        if not secret:
            close("blocked")
            write_summary(root, status="blocked", blocker="GEMINI_API_KEY is not set", ledger=ledger)
            print("BLOCKED GEMINI_API_KEY is not set", file=sys.stderr)
            return 2
        try:
            identity = confirm_model(secret)
        except TransportError as exc:
            message = redact(str(exc), secret)
            close("blocked")
            _write_json(
                experiment / "model_check.json",
                {"requested_model": MODEL_ID, "ok": False, "error": message, "counted_as_decision": False},
            )
            write_summary(root, status="blocked", blocker=message, ledger=ledger)
            print(f"BLOCKED model check: {message}", file=sys.stderr)
            return 2
        _write_json(
            experiment / "model_check.json",
            {
                "requested_model": MODEL_ID,
                "ok": True,
                "endpoint_name": identity["name"],
                "display_name": identity["display_name"],
                "counted_as_decision": False,
            },
        )
        print(
            f"MODEL ok name={identity['name']} display={identity['display_name']} counted_as_decision=false",
            flush=True,
        )
        client = GeminiClient(secret)
        ledger.start_heartbeat()
        smoke_spec = spec_from_dict(smoke_dict(root))
        if canonical_dumps(smoke_spec.to_dict()) != canonical_dumps(registration["smoke_spec"]):
            close("blocked")
            write_summary(root, status="blocked", blocker="smoke spec drifted before the call", ledger=ledger)
            return 2
        smoke_result, smoke_policy = run_note_lifetime(
            root=root,
            spec=smoke_spec,
            run_dir=smoke_run_dir(root),
            run_id="smoke",
            carry=True,
            world="smoke",
            condition="smoke",
            memory_condition="persistent-note",
            client=client,
            ledger=ledger,
            phase="smoke",
        )
        problem = smoke_problem(smoke_run_dir(root), smoke_result)
        if problem or smoke_policy.stop_task:
            reason = problem or "smoke auth or quota failure"
            close("blocked" if smoke_policy.stop_task else "failed")
            write_summary(root, status="blocked" if smoke_policy.stop_task else "failed", blocker=reason, ledger=ledger)
            print(f"BLOCKED smoke: {reason}", file=sys.stderr)
            return 2
        frozen = current_settings()
        _write_json(experiment / "frozen.json", {"settings": frozen, "frozen_at_utc": utc_now()})
        for world in WORLD_FILES:
            base = load_world_dict(root, world)
            file_hash = sha256_bytes((root / "configs" / WORLD_FILES[world]).read_bytes())
            if file_hash != registration["worlds"][world]["sha256"]:
                close("blocked")
                write_summary(root, status="blocked", blocker=f"{world} config changed after registration", ledger=ledger)
                return 2
            if current_settings() != frozen:
                close("blocked")
                write_summary(root, status="blocked", blocker="settings changed after the smoke", ledger=ledger)
                return 2
            for condition, agent, carry, memory in CONDITIONS:
                spec = spec_from_dict(with_agent(base, agent))
                result, policy = run_note_lifetime(
                    root=root,
                    spec=spec,
                    run_dir=pilot_run_dir(root, world, condition),
                    run_id=f"{world}-{condition}",
                    carry=carry,
                    world=world,
                    condition=condition,
                    memory_condition=memory,
                    client=client,
                    ledger=ledger,
                    phase="pilot",
                )
                if policy.stop_task or result.status != "completed":
                    reason = result.error_private or result.status
                    close("blocked" if policy.stop_task else "failed")
                    write_summary(
                        root,
                        status="blocked" if policy.stop_task else "failed",
                        blocker=redact(str(reason), secret),
                        ledger=ledger,
                    )
                    print(f"BLOCKED lifetime {world} {condition}: {redact(str(reason), secret)}", file=sys.stderr)
                    return 2 if policy.stop_task else 1
        close("complete")
        write_summary(root, status="complete", blocker=None, ledger=ledger)
        print("PILOT complete", flush=True)
        return 0
    except Exception as exc:
        message = redact(f"{type(exc).__name__}: {exc}", secret)
        try:
            close("failed")
        except Exception:
            pass
        try:
            write_summary(root, status="failed", blocker=message, ledger=ledger)
        except Exception:
            pass
        print(f"BLOCKED {message}", file=sys.stderr)
        return 1
    finally:
        ledger.stop_heartbeat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="single_life.pilot")
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args(argv)
    try:
        return execute(args.root)
    except FileExistsError as exc:
        print(f"BLOCKED {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
