"""Gemini text adapter for the bandit note-retention diagnostic.

Request shape follows the public generateContent REST reference
(https://ai.google.dev/api/generate-content). Accounting for missing usage is
implemented here: unknown usage keeps its reserve and is never priced as zero.
No parent-project imports.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from single_life.artifacts import ArtifactStore
from single_life.contracts import ContractError, PublicView

MODEL_ID = "gemini-3.5-flash"
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash"
PRICE_PAGE_UPDATED = "2026-09-24"
PRICE_CHECKED_ON = "2026-09-29"
INPUT_USD_PER_MILLION = Decimal("1.50")
OUTPUT_USD_PER_MILLION = Decimal("9.00")
INPUT_TOKEN_CAP = 4096
OUTPUT_TOKEN_CAP = 2048
THINKING_LEVEL = "low"
TEMPERATURE = 0
MAX_ATTEMPTS = 2
NOTE_CHAR_LIMIT = 1000
OBSERVATION_WINDOW = 12
SMOKE_DECISIONS = 4
PILOT_DECISIONS = 192
PLANNED_DECISIONS = SMOKE_DECISIONS + PILOT_DECISIONS
MAX_REQUESTS = PLANNED_DECISIONS * MAX_ATTEMPTS

INSTRUCTION = (
    "You choose the next action in a three-action task. "
    "Each completed observation has an action and a reward of 0 or 1. "
    "Unchosen actions stay hidden. "
    "You may also see a strategy note retained from your own earlier reply. "
    "If retained_strategy_note is null, no note has been retained. "
    "Use only the public payload. "
    'Reply with one JSON object and no other text: {"action": "<permitted action id>", '
    '"strategy_note": "<short note, at most 1000 characters>"}. '
    "The note is your own working memory for a later decision. "
    "Do not invent probabilities, regime labels, or hidden schedules."
)

PARSER_POLICY = {
    "schema": {"action": "permitted action id", "strategy_note": "string"},
    "note_char_limit": NOTE_CHAR_LIMIT,
    "excess_note": "truncate to 1000 characters, log truncated=true, keep the action",
    "invalid_json_or_action": "retry the same prompt once; then consume the tick as an invalid action with no reward and no second environment action",
    "transport_error": "retry once; a repeated auth, permission, or quota error stops the task; any other repeated transport error aborts that lifetime without another environment action",
    "max_attempts_per_decision": MAX_ATTEMPTS,
    "checkpoint": "disabled; this policy is not in the checkpoint allowlist",
}


class ParseError(Exception):
    pass


class TransportError(Exception):
    def __init__(self, message: str, *, status: int | None = None, auth_or_quota: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.auth_or_quota = auth_or_quota


class UsageUnknown(Exception):
    pass


@dataclass(frozen=True)
class Usage:
    status: str
    input_tokens: int | None
    output_tokens: int | None
    thoughts_tokens: int | None
    total_tokens: int | None
    cost_usd: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": None if self.cost_usd is None else format(self.cost_usd, "f"),
        }


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    finish_reason: str | None
    model_version: str | None
    elapsed_seconds: float


class TextClient(Protocol):
    def generate(self, prompt: str) -> Generation: ...


def attempt_cap_usd() -> Decimal:
    """Worst-case price of one attempt at the input cap and the output+thinking cap."""

    return token_cost(INPUT_TOKEN_CAP, OUTPUT_TOKEN_CAP)


def completion_cap_usd() -> Decimal:
    return attempt_cap_usd() * MAX_REQUESTS


def token_cost(input_tokens: int, billed_output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * INPUT_USD_PER_MILLION
        + Decimal(billed_output_tokens) * OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)


def pricing_record() -> dict[str, Any]:
    per_attempt = attempt_cap_usd()
    return {
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "thinking_level": THINKING_LEVEL,
        "input_token_cap": INPUT_TOKEN_CAP,
        "max_output_tokens": OUTPUT_TOKEN_CAP,
        "output_cap_includes_thinking": True,
        "thinking_budget_integer": None,
        "thinking_budget_note": (
            "gemini-3.5-flash uses thinkingLevel, not a separate integer thinking budget. "
            "The generateContent REST examples spell the level in lowercase, so this run sends \"low\". "
            "maxOutputTokens is the hard cap on visible output plus thinking tokens. "
            f"{OUTPUT_TOKEN_CAP} reserves 1024 visible tokens plus 1024 thinking tokens "
            "because those caps are not separately enforceable."
        ),
        "other_published_rows": (
            "The same Gemini 3.5 Flash section also lists paid rows at $0.75/$4.50 and $2.70/$16.20. "
            "This run does not request batch, flex, or priority. It prices synchronous generateContent "
            "at the standard $1.50 input and $9.00 output row."
        ),
        "input_usd_per_million": format(INPUT_USD_PER_MILLION, "f"),
        "output_usd_per_million_including_thinking": format(OUTPUT_USD_PER_MILLION, "f"),
        "price_row": "standard paid tier, synchronous generateContent, no batch, cache, or grounding",
        "price_source": PRICE_SOURCE,
        "price_page_updated": PRICE_PAGE_UPDATED,
        "price_checked_on": PRICE_CHECKED_ON,
        "per_attempt_cap_usd": format(per_attempt, "f"),
        "planned_decisions": PLANNED_DECISIONS,
        "max_attempts": MAX_REQUESTS,
        "completion_cap_usd": format(completion_cap_usd(), "f"),
        "estimate_note": (
            "The completion cap prices every planned decision at two full-cap attempts. "
            "It is a usage estimate from the published token rates, not a provider invoice. "
            "Unknown usage keeps this per-attempt reserve and is not entered as zero."
        ),
    }


def load_api_key(env_path: Path | None = None) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = env_path or Path("/home/xz479/llm-time/.env")
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def redact(text: str, secret: str) -> str:
    if secret and secret in text:
        text = text.replace(secret, "[redacted]")
    return text


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def retained_note(view: PublicView) -> str | None:
    notes = [artifact for artifact in view.accessible_artifacts if artifact.kind == "strategy-note"]
    if not notes:
        return None
    latest = max(
        notes,
        key=lambda artifact: (
            artifact.deployment_decision_index or 0,
            artifact.creation_cutoff,
            artifact.version,
        ),
    )
    text = latest.content.get("strategy_note")
    if not isinstance(text, str):
        return None
    return text


def build_prompt(view: PublicView, *, carry_note: bool) -> str:
    window = list(view.observed_events)[-OBSERVATION_WINDOW:]
    payload = {
        "decision_index": view.decision_index,
        "interactions_remaining": view.remaining_public_resources["interactions_remaining"],
        "observation_window": OBSERVATION_WINDOW,
        "observations": [
            {
                "action": event.chosen_action,
                "action_index": event.action_index,
                "reward": event.observed_outcome,
            }
            for event in window
        ],
        "permitted_action_ids": list(view.permitted_action_ids),
        "retained_strategy_note": retained_note(view) if carry_note else None,
    }
    return INSTRUCTION + "\n\npublic_payload:\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_model_json(text: str, permitted: tuple[str, ...]) -> tuple[str, str, bool]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError("response is not JSON") from exc
    if not isinstance(data, dict):
        raise ParseError("response is not an object")
    action = data.get("action")
    note = data.get("strategy_note")
    if not isinstance(action, str) or action not in permitted:
        raise ParseError("action is not a permitted id")
    if not isinstance(note, str):
        raise ParseError("strategy_note is missing")
    truncated = len(note) > NOTE_CHAR_LIMIT
    if truncated:
        note = note[:NOTE_CHAR_LIMIT]
    return action, note, truncated


def usage_from_metadata(metadata: dict[str, Any] | None) -> Usage:
    if not isinstance(metadata, dict):
        return Usage("unknown", None, None, None, None, None)
    if "promptTokenCount" not in metadata or "candidatesTokenCount" not in metadata:
        return Usage("unknown", None, None, None, None, None)
    if "thoughtsTokenCount" not in metadata:
        return Usage("unknown", None, None, None, None, None)
    try:
        prompt = int(metadata["promptTokenCount"])
        candidates = int(metadata["candidatesTokenCount"])
        thoughts = int(metadata["thoughtsTokenCount"])
        total = int(metadata.get("totalTokenCount", prompt + candidates + thoughts))
    except (TypeError, ValueError):
        return Usage("unknown", None, None, None, None, None)
    billed_output = candidates + thoughts
    return Usage(
        "available",
        prompt,
        billed_output,
        thoughts,
        total,
        token_cost(prompt, billed_output),
    )


class GeminiClient:
    def __init__(self, api_key: str, *, opener: Any | None = None, timeout: float = 120.0) -> None:
        if not api_key:
            raise TransportError("GEMINI_API_KEY is not set", auth_or_quota=True)
        self._api_key = api_key
        self._opener = opener or urllib.request.urlopen
        self._timeout = timeout

    def generate(self, prompt: str) -> Generation:
        if estimate_tokens(prompt) > INPUT_TOKEN_CAP:
            raise ContractError("prompt exceeds the input token cap")
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": OUTPUT_TOKEN_CAP,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
            },
        }
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent",
            data=encoded,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        started = time.monotonic()
        try:
            with self._opener(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else ""
            message = redact(f"HTTP {exc.code}: {detail[:300]}", self._api_key)
            raise TransportError(
                message,
                status=exc.code,
                auth_or_quota=exc.code in {401, 403, 429},
            ) from None
        except urllib.error.URLError as exc:
            raise TransportError(redact(f"URL error: {exc.reason}", self._api_key)) from None
        elapsed = time.monotonic() - started
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportError("provider response is not JSON") from exc
        if not isinstance(payload, dict):
            raise TransportError("provider response is not an object")
        candidates = payload.get("candidates") or []
        text = ""
        finish = None
        if candidates and isinstance(candidates[0], dict):
            finish_raw = candidates[0].get("finishReason")
            finish = None if finish_raw is None else str(finish_raw)
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            chunks = []
            for part in parts:
                if not isinstance(part, dict) or part.get("thought") is True:
                    continue
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            text = "".join(chunks)
        version = payload.get("modelVersion")
        return Generation(
            text=text,
            usage=usage_from_metadata(payload.get("usageMetadata")),
            finish_reason=finish,
            model_version=None if version is None else str(version),
            elapsed_seconds=elapsed,
        )


class ScriptedClient:
    """In-process stand-in. Responses are queued and never call a network."""

    def __init__(self, replies: list[Generation | Exception | str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        if not self._replies:
            raise TransportError("scripted client has no reply")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return Generation(
                text=item,
                usage=Usage("available", 100, 20, 0, 120, token_cost(100, 20)),
                finish_reason="STOP",
                model_version="scripted",
                elapsed_seconds=0.01,
            )
        return item


def scripted_reply(action: str, note: str, *, usage: str = "available") -> str | Generation:
    text = json.dumps({"action": action, "strategy_note": note})
    if usage == "available":
        return text
    return Generation(
        text=text,
        usage=Usage("unknown", None, None, None, None, None),
        finish_reason="STOP",
        model_version="scripted",
        elapsed_seconds=0.01,
    )


@dataclass
class NoteHandoff:
    carry: bool
    pending: dict[str, Any] | None = None


class NoteCarryUpdater:
    """Deploys the note written at the previous decision. Checkpointing is unsupported."""

    updater_id = "strategy-note-carry-v1"

    def __init__(self, handoff: NoteHandoff) -> None:
        self.handoff = handoff

    def apply(self, view: PublicView, store: ArtifactStore) -> None:
        pending = self.handoff.pending
        if not self.handoff.carry or pending is None:
            return None
        if pending["source_decision"] >= view.decision_index:
            return None
        cutoff = pending["creation_cutoff"]
        support = [
            event_id
            for event_id in pending["support"]
            if event_id in store.catalog and store.catalog[event_id].action_index <= cutoff
        ]
        parents = [artifact for artifact in store.versions if artifact.kind == "strategy-note"]
        parent_version = None
        if parents:
            parent_version = max(parents, key=lambda artifact: artifact.creation_cutoff).version
        version = f"note-{pending['source_decision']}"
        content = {
            "strategy_note": pending["strategy_note"],
            "truncated": pending["truncated"],
            "written_at_decision": pending["source_decision"],
        }
        created = store.create(
            artifact_id=f"{view.lifetime_id}:strategy-note",
            version=version,
            kind="strategy-note",
            parent_version=parent_version,
            creation_cutoff=cutoff,
            supporting_observed_event_ids=support,
            content=content,
        )
        store.deploy(created.version)
        self.handoff.pending = None
        return None


class GeminiNotePolicy:
    """One provider call sequence per decision. The note is staged for the next decision."""

    name = "gemini-note-v1"

    def __init__(
        self,
        *,
        carry_note: bool,
        client: TextClient,
        handoff: NoteHandoff,
        responses_dir: Path,
        ledger: Any,
        world: str,
        condition: str,
    ) -> None:
        self.carry_note = carry_note
        self.client = client
        self.handoff = handoff
        self.responses_dir = responses_dir
        self.ledger = ledger
        self.world = world
        self.condition = condition
        self.prompts: list[str] = []
        self.stop_task = False
        self.saw_auth_or_quota = False

    def select(self, view: PublicView, rng: Any) -> str:
        del rng
        prompt = build_prompt(view, carry_note=self.carry_note)
        self.prompts.append(prompt)
        if estimate_tokens(prompt) > INPUT_TOKEN_CAP:
            raise ContractError("prompt exceeds the input token cap")
        self.ledger.locate(
            world=self.world,
            condition=self.condition,
            step=view.decision_index,
        )
        permitted = tuple(view.permitted_action_ids)
        last_error = "no attempt"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.ledger.begin_attempt()
            try:
                generation = self.client.generate(prompt)
            except TransportError as exc:
                self._write_attempt(
                    view.decision_index,
                    attempt,
                    prompt,
                    text="",
                    usage=Usage("unknown", None, None, None, None, None),
                    finish_reason=None,
                    model_version=None,
                    elapsed=0.0,
                    parser_status="transport_error",
                    error=str(exc),
                )
                self.ledger.finish_attempt(usage=None, failed=True)
                last_error = str(exc)
                if exc.auth_or_quota:
                    self.saw_auth_or_quota = True
                if attempt == MAX_ATTEMPTS:
                    self.stop_task = self.saw_auth_or_quota
                    self.ledger.complete_decision()
                    raise
                continue
            try:
                action, note, truncated = parse_model_json(generation.text, permitted)
            except ParseError as exc:
                self._write_attempt(
                    view.decision_index,
                    attempt,
                    prompt,
                    text=generation.text,
                    usage=generation.usage,
                    finish_reason=generation.finish_reason,
                    model_version=generation.model_version,
                    elapsed=generation.elapsed_seconds,
                    parser_status="invalid",
                    error=str(exc),
                )
                self.ledger.finish_attempt(
                    usage=generation.usage if generation.usage.status == "available" else None,
                    failed=True,
                )
                last_error = str(exc)
                if attempt == MAX_ATTEMPTS:
                    return "invalid"
                continue
            self._write_attempt(
                view.decision_index,
                attempt,
                prompt,
                text=generation.text,
                usage=generation.usage,
                finish_reason=generation.finish_reason,
                model_version=generation.model_version,
                elapsed=generation.elapsed_seconds,
                parser_status="truncated" if truncated else "ok",
                error="",
                action=action,
                note=note,
            )
            self.ledger.finish_attempt(
                usage=generation.usage if generation.usage.status == "available" else None,
                failed=False,
            )
            if self.carry_note:
                self.handoff.pending = {
                    "source_decision": view.decision_index,
                    "strategy_note": note,
                    "truncated": truncated,
                    "creation_cutoff": view.decision_index - 1,
                    "support": [event.event_id for event in view.observed_events],
                }
            return action
        raise ContractError(last_error)

    def _write_attempt(
        self,
        decision_index: int,
        attempt: int,
        prompt: str,
        *,
        text: str,
        usage: Usage,
        finish_reason: str | None,
        model_version: str | None,
        elapsed: float,
        parser_status: str,
        error: str,
        action: str | None = None,
        note: str | None = None,
    ) -> None:
        payload = {
            "decision_index": decision_index,
            "attempt": attempt,
            "model": MODEL_ID,
            "model_version": model_version,
            "temperature": TEMPERATURE,
            "thinking_level": THINKING_LEVEL,
            "max_output_tokens": OUTPUT_TOKEN_CAP,
            "prompt": prompt,
            "response_text": text,
            "parsed_action": action,
            "parsed_note": note,
            "parser_status": parser_status,
            "error": error,
            "finish_reason": finish_reason,
            "usage": usage.to_dict(),
            "elapsed_seconds": elapsed,
            "world": self.world,
            "condition": self.condition,
        }
        path = self.responses_dir / f"d{decision_index:03d}-a{attempt}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
