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

Each attempt is reserved at `max_input_tokens` input plus `max_output_tokens`
output using the verified per-million rates. Thinking is requested disabled
(`thinking_budget=0`) so billed output is intended to
lie inside `max_output_tokens`. Interrupted calls keep their reservation.

- Input cap: 2000 tokens
- Output cap: 2048 tokens
- Rates: $1.5/M input, $9.0/M output
- Reserved per attempt: $0.021432
- Reserved for full grid plus retries: **$3.086208**
- Largest prompt token estimate: 1461
- Prompts exceeding the input cap: 0

## Integrity

- Config hash: `ecb3f5add70689916200b07873a79efb0c52a97fee52fc03224393990f5592af`
- Git HEAD: `c913e21daef0ec77ba2e7dec49c4e25616214eeb`
- Dirty files hashed: 95 (credentials omitted)
- Public prompts are under `prompts/`. Evaluator targets are in `evaluator.json` and are not sent to the provider.
- Resume key: `case_id|treatment|model|prompt_hash|config_hash`.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --config llm_rl_engine/configs/pilot_selective_revision_v1.yaml
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q tests/test_selective_revision_pilot.py
```

Artifacts: `/home/xz479/llm-time/reports/selective_revision_pilot_v1_dryrun`

## Limitations

Cost bound is reserved_per_attempt * jobs * (1 + retries) using max_input_tokens at the input rate and max_output_tokens at the output rate, with thinking_budget=0 so thinking is not billed extra. If a live response reports thoughts_token_count>0, stop; the dollar ceiling would no longer be valid. Dry-run made no model calls.

Labels and rates are held fixed. This grid is not structural-transfer evidence.
Live execution requires a follow-up with explicit model, max_calls, token caps,
and max_usd. No claim of model performance is made here.
