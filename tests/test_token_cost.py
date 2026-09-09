from __future__ import annotations

from llm_rl_engine.analysis.token_cost import aggregate_token_usage, estimate_cost_usd


def test_estimate_cost_usd():
    costs = estimate_cost_usd("gemini-3.5-flash", 1_000_000, 500_000)
    assert costs["input_cost_usd"] == 1.50
    assert costs["output_cost_usd"] == 4.50
    assert costs["total_cost_usd"] == 6.00


def test_aggregate_token_usage():
    steps = [
        {"token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
        {"token_usage": {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240}},
    ]
    agg = aggregate_token_usage(steps, model="gemini-3.5-flash")
    assert agg["input_tokens"] == 300
    assert agg["output_tokens"] == 60
    assert agg["total_cost_usd"] > 0
