"""Gemini transport for task 006 procedure calls.

Generation is refused unless a provider token count for that prompt is at most
the input cap. characters/4 is not an acceptance check. Raw nonsecret usage
metadata is kept so later audits can reconstruct the token normalization.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from single_life.contracts import ContractError
from single_life.gemini_adapter import (
    INPUT_USD_PER_MILLION,
    MODEL_ID,
    OUTPUT_USD_PER_MILLION,
    TEMPERATURE,
    THINKING_LEVEL,
    TransportError,
    Usage,
    redact,
    usage_from_metadata,
)

INPUT_TOKEN_CAP = 8192
OUTPUT_TOKEN_CAP = 2048
MAX_ATTEMPTS = 2
SMOKE_CALLS = 2
PILOT_CALLS = 44
SCHEDULED_CALLS = SMOKE_CALLS + PILOT_CALLS
MAX_GENERATION_ATTEMPTS = SCHEDULED_CALLS * MAX_ATTEMPTS
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash"
PRICE_CHECKED_ON = "2026-09-29"


def token_cost(input_tokens: int, billed_output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * INPUT_USD_PER_MILLION
        + Decimal(billed_output_tokens) * OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)


def attempt_cap_usd() -> Decimal:
    return token_cost(INPUT_TOKEN_CAP, OUTPUT_TOKEN_CAP)


def completion_cap_usd() -> Decimal:
    return attempt_cap_usd() * MAX_GENERATION_ATTEMPTS


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
        "input_cap_enforcement": "provider countTokens totalTokens must be <= 8192 before generateContent; characters/4 is not used to accept a prompt",
        "token_count_requests": "counted separately from generation attempts; no countTokens price is listed on the Gemini 3.5 Flash standard row, so they are not added to the generation estimate",
        "other_published_rows": (
            "The same section also lists $0.75/$4.50 and $2.70/$16.20. "
            "This run prices synchronous generateContent at the standard $1.50 input and $9.00 output row."
        ),
        "input_usd_per_million": format(INPUT_USD_PER_MILLION, "f"),
        "output_usd_per_million_including_thinking": format(OUTPUT_USD_PER_MILLION, "f"),
        "price_row": "standard paid tier, synchronous generateContent",
        "price_source": PRICE_SOURCE,
        "price_checked_on": PRICE_CHECKED_ON,
        "per_attempt_cap_usd": format(per_attempt, "f"),
        "scheduled_calls": SCHEDULED_CALLS,
        "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
        "completion_cap_usd": format(completion_cap_usd(), "f"),
        "estimate_note": "Usage estimate from the published rates, not a provider invoice. Unknown generation usage keeps the per-attempt reserve.",
    }


def provider_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usageMetadata")
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    finish = None
    thought_flags: list[bool] = []
    if candidates and isinstance(candidates[0], dict):
        finish = candidates[0].get("finishReason")
        content = candidates[0].get("content") if isinstance(candidates[0].get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        for part in parts:
            thought_flags.append(bool(isinstance(part, dict) and part.get("thought") is True))
    return {
        "modelVersion": payload.get("modelVersion"),
        "responseId": payload.get("responseId"),
        "finishReason": finish,
        "usageMetadata": usage if isinstance(usage, dict) else None,
        "candidate_count": len(candidates),
        "part_thought_flags": thought_flags,
    }


@dataclass(frozen=True)
class ProcedureGeneration:
    text: str
    usage: Usage
    finish_reason: str | None
    model_version: str | None
    elapsed_seconds: float
    provider_metadata: dict[str, Any]
    counted_input_tokens: int


class ProcedureGeminiClient:
    def __init__(self, api_key: str, *, opener: Any | None = None, timeout: float = 120.0) -> None:
        if not api_key:
            raise TransportError("GEMINI_API_KEY is not set", auth_or_quota=True)
        self._api_key = api_key
        self._opener = opener or urllib.request.urlopen
        self._timeout = timeout

    def count_tokens(self, prompt: str) -> int:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        payload = self._post(f"models/{MODEL_ID}:countTokens", body, timeout=60.0)
        total = payload.get("totalTokens", payload.get("total_tokens"))
        if type(total) is not int or total < 1:
            raise TransportError("countTokens did not return totalTokens")
        return total

    def generate(self, prompt: str, *, token_count: int) -> ProcedureGeneration:
        if type(token_count) is not int or token_count > INPUT_TOKEN_CAP:
            raise ContractError("refusing generation above the counted input cap")
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": OUTPUT_TOKEN_CAP,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
            },
        }
        started = time.monotonic()
        payload = self._post(f"models/{MODEL_ID}:generateContent", body, timeout=self._timeout)
        elapsed = time.monotonic() - started
        meta = provider_metadata(payload)
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
        return ProcedureGeneration(
            text=text,
            usage=usage_from_metadata(payload.get("usageMetadata")),
            finish_reason=finish,
            model_version=None if version is None else str(version),
            elapsed_seconds=elapsed,
            provider_metadata=meta,
            counted_input_tokens=token_count,
        )

    def _post(self, method: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/{method}",
            data=encoded,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with self._opener(request, timeout=timeout) as response:
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
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportError("provider response is not JSON") from exc
        if not isinstance(payload, dict):
            raise TransportError("provider response is not an object")
        return payload


class ScriptedProcedureClient:
    """Queued stand-in. It never opens a socket."""

    def __init__(self, replies: list[ProcedureGeneration | Exception | str] | None = None) -> None:
        self._replies = list(replies or [])
        self.prompts: list[str] = []
        self.generate_calls = 0
        self.count_calls = 0
        self.token_count = 128
        self.count_errors: list[Exception] = []

    def count_tokens(self, prompt: str) -> int:
        del prompt
        self.count_calls += 1
        if self.count_errors:
            raise self.count_errors.pop(0)
        return self.token_count

    def generate(self, prompt: str, *, token_count: int) -> ProcedureGeneration:
        if type(token_count) is not int or token_count > INPUT_TOKEN_CAP:
            raise ContractError("refusing generation above the counted input cap")
        self.generate_calls += 1
        self.prompts.append(prompt)
        if not self._replies:
            raise TransportError("scripted client has no reply")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return ProcedureGeneration(
                text=item,
                usage=Usage("available", 100, 20, 0, 120, token_cost(100, 20)),
                finish_reason="STOP",
                model_version="scripted",
                elapsed_seconds=0.01,
                provider_metadata={
                    "modelVersion": "scripted",
                    "responseId": None,
                    "finishReason": "STOP",
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 20,
                        "thoughtsTokenCount": 0,
                        "totalTokenCount": 120,
                    },
                    "candidate_count": 1,
                    "part_thought_flags": [False],
                },
                counted_input_tokens=token_count,
            )
        return item


def confirm_model(api_key: str) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}"
    last: TransportError | None = None
    payload: dict[str, Any] | None = None
    for attempt in (1, 2):
        request = urllib.request.Request(url, headers={"x-goog-api-key": api_key}, method="GET")
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
    if not isinstance(payload, dict) or MODEL_ID not in str(payload.get("name", "")):
        raise TransportError("model endpoint did not return gemini-3.5-flash")
    display = payload.get("displayName")
    return {
        "name": str(payload.get("name", "")),
        "display_name": "" if display is None else str(display),
        "counted_as_generation": False,
    }


