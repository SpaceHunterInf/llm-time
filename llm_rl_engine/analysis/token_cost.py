from __future__ import annotations

from typing import Any

# USD per 1M tokens (estimates; override via config pricing section)
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gemini-2.5-flash": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gemini-2.0-flash": {"input_per_1m": 0.10, "output_per_1m": 0.40},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any] | None = None,
) -> dict[str, float]:
    table = pricing or DEFAULT_MODEL_PRICING
    rates = table.get(model) or table.get("gemini-3.5-flash", {"input_per_1m": 0.15, "output_per_1m": 0.60})
    input_cost = (input_tokens / 1_000_000) * rates["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * rates["output_per_1m"]
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def aggregate_token_usage(
    steps: list[dict[str, Any]],
    model: str = "gemini-3.5-flash",
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    n_llm_steps = 0
    for step in steps:
        usage = step.get("token_usage") or {}
        if not usage:
            continue
        n_llm_steps += 1
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))

    costs = estimate_cost_usd(model, input_tokens, output_tokens, pricing)
    return {
        "model": model,
        "n_llm_steps": n_llm_steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens_per_step": input_tokens / max(n_llm_steps, 1),
        "avg_output_tokens_per_step": output_tokens / max(n_llm_steps, 1),
        **costs,
        "pricing_note": "Estimated from configured per-1M-token rates; verify against Google billing.",
    }


def aggregate_run_token_report(
    trajectories_path: str,
    model: str = "gemini-3.5-flash",
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import json
    from pathlib import Path

    by_agent_env: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_llm_steps: list[dict[str, Any]] = []

    with Path(trajectories_path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("prompt"):
                continue
            key = (row["agent_name"], row["env_name"])
            by_agent_env.setdefault(key, []).append(row)
            all_llm_steps.append(row)

    per_combo = {
        f"{agent}__{env}": aggregate_token_usage(steps, model=model, pricing=pricing)
        for (agent, env), steps in by_agent_env.items()
    }
    total = aggregate_token_usage(all_llm_steps, model=model, pricing=pricing)
    return {"total": total, "by_agent_env": per_combo}