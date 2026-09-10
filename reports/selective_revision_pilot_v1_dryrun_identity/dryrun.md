# Selective revision pilot: dry-run

No LLM calls, no `.env` read, no network. This is the fixed-history diagnostic
package, not a model evaluation and not a performance benchmark.

## Grid

- Public cases: 24 (3 structures × 4 A-success counts × 2 diagnostic costs).
- Treatments: raw, counts, counts-plus-reference predictions (3).
- Planned decision calls: **72**.
- Retry reserve (one retry per call): **144** attempts.
- Requested model ID: `gemini-3.5-flash`.

A diagnose choice commits to using the revealed latent rate optimally. This probe
tests valuation of information, not interpretation of diagnostic results.

## Cost bound

Each attempt is reserved at `max_input_tokens` input plus the **provider**
output limit (`configured_max_output_tokens + thinking_budget`) at the output
rate. 2048 is not an independently enforced visible-text cap. Interrupted
calls keep their reservation. Dry-run token figures are an offline heuristic.

- Input cap: 2000 tokens
- Configured max_output_tokens: 2048
- Provider output limit (sum): 2048
- Rates: $1.5/M input, $9.0/M output
- Reserved per attempt: $0.021432
- Reserved for full grid plus retries: **$3.086208**
- Largest prompt token estimate: 1461
- Prompts exceeding the input cap: 0

## Integrity

- Config hash: `5dc02da811a38c21425a19adc9aff0efba350b6e333fb0628a3df7d4060844db`
- Effective hash: `5dc02da811a38c21425a19adc9aff0efba350b6e333fb0628a3df7d4060844db`
- Git HEAD: `9aff4fad14e9f29f5f998a2b4ed154a2386161c3`
- Dirty files hashed: 83 (credentials omitted)
- Public prompts are under `prompts/`. Evaluator targets are in `evaluator.json` and are not sent to the provider.
- Resume key: `case_id|treatment|effective_hash|prompt_hash`.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --config llm_rl_engine/configs/pilot_selective_revision_v1.yaml
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q tests/test_selective_revision_pilot.py
```

Artifacts: `reports/selective_revision_pilot_v1_dryrun_identity`

## Limitations

Cost bound is reserved_per_attempt * jobs * (1 + retries) using max_input_tokens at the input rate and (max_output_tokens + thinking_budget) at the output rate. thinking_budget=-1 is rejected. Dry-run made no model calls.

Labels and rates are held fixed. This grid is not structural-transfer evidence.
Live execution requires a follow-up with explicit model, max_calls, token caps,
and max_usd. No claim of model performance is made here.
