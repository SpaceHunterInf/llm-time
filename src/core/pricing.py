"""Verified model token pricing (USD per 1M tokens) as of November 2026."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ModelRate:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float
    tier: str
    capability_rank: int
    provenance_verified: bool = True


# Verified rates (output includes thinking/reasoning tokens on Gemini).
STATIC_MODEL_RATES: dict[str, ModelRate] = {
    "mock": ModelRate(0.0, 0.0, 0.0, "offline", 0, True),
    "local-vllm": ModelRate(0.0, 0.0, 0.0, "open-weight", 4, True),
    "Qwen/Qwen3-8B": ModelRate(0.0, 0.0, 0.0, "local-open-weight", 4, True),
    "qwen/qwen3-8b": ModelRate(0.0, 0.0, 0.0, "local-open-weight", 4, True),
    "gemini-3.1-flash-lite": ModelRate(0.25, 1.50, 0.025, "flash-lite", 1, True),
    "gemini-3.5-flash-lite": ModelRate(0.30, 2.50, 0.030, "flash-lite", 2, True),
    "gemma-4-26b-a4b-it": ModelRate(0.25, 1.25, 0.025, "gemma-4-open", 4, True),
    "gemma-4-31b-it": ModelRate(0.30, 1.50, 0.030, "gemma-4-open", 5, True),
    "gemini-3.5-flash": ModelRate(0.75, 3.75, 0.075, "flash", 3, True),
    "gemini-3.6-flash": ModelRate(0.75, 3.75, 0.075, "flash", 3, True),
    "gemini-3.7-flash": ModelRate(0.75, 3.75, 0.075, "flash", 4, True),
    "gemini-3.8-flash": ModelRate(0.75, 3.75, 0.075, "flash", 5, True),
    "gemini-4-hydrogen": ModelRate(1.50, 7.50, 0.075, "gemini-4-fast", 6, True),
    "gemini-4-neon": ModelRate(3.00, 15.00, 0.150, "gemini-4-mid", 7, True),
    "deepseek-chat": ModelRate(0.28, 1.10, 0.028, "external-eval", 6, True),
    "deepseek-reasoner": ModelRate(0.55, 2.19, 0.055, "external-eval", 7, True),
}


def normalize_model_id(model: str) -> str:
    cleaned = model.strip()
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    return cleaned


def get_model_rate(model: str, now_utc: datetime | None = None) -> ModelRate:
    """Return USD per 1M token rate, accounting for Dec 1, 2026 Argon/Carbon change."""
    norm = normalize_model_id(model)
    ts = now_utc or datetime.now(timezone.utc)
    dec_2026_cutoff = datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc)

    if norm in ("gemini-4-argon", "gemini-4-carbon"):
        rank = 8 if norm == "gemini-4-argon" else 9
        if ts < dec_2026_cutoff:
            return ModelRate(2.00, 10.00, 0.10, "gemini-4-flagship", rank, True)
        return ModelRate(4.00, 20.00, 0.20, "gemini-4-flagship", rank, True)

    if norm in STATIC_MODEL_RATES:
        return STATIC_MODEL_RATES[norm]

    # Conservative non-zero fallback so unknown models are never treated as free.
    return ModelRate(2.00, 10.00, 0.20, "unverified-fallback", 5, False)


def estimate_call_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    cached_tokens: int = 0,
    now_utc: datetime | None = None,
) -> tuple[float, bool]:
    """Calculate exact USD for a single provider call.

    Note: On Gemini, `candidates_token_count` excludes `thoughts_token_count`,
    and both are billed at the output token rate.
    """
    rate = get_model_rate(model, now_utc=now_utc)
    uncached_in = max(0, input_tokens - cached_tokens)
    billed_out = max(0, output_tokens) + max(0, thinking_tokens)
    usd = (
        (uncached_in / 1_000_000.0) * rate.input_per_1m
        + (max(0, cached_tokens) / 1_000_000.0) * rate.cached_input_per_1m
        + (billed_out / 1_000_000.0) * rate.output_per_1m
    )
    return usd, rate.provenance_verified
