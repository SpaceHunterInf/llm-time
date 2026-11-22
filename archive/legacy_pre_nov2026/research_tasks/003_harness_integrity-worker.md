# Task 003 return

Author: Grok (worker). CPU/fake-client only. Existing live run files under `runs/` were not rewritten.

## Changed files

- `llm_rl_engine/analysis/selective_revision_harness.py` — effective identity, resume rejection, request hashes, unknown usage, saved-response recovery, provider metadata
- `llm_rl_engine/analysis/selective_revision_pilot.py` — `posterior_mse`, expected Brier, C abs error, equal-utility ties, forecast-consistency metric
- `llm_rl_engine/agents/llm.py` — optional `model_version`, `finish_reason`, `usage_status` on `GenerationResult` (defaults; historical agents unchanged)
- `scripts/selective_revision_pilot.py` — identity check **before** constructing a client; do not clobber the v1 dry-run report
- `tests/test_selective_revision_pilot.py` — mismatch/resume/unknown-usage/recovery/legacy-manifest regressions
- Fresh dry-run: `reports/selective_revision_pilot_v1_dryrun_identity/` (v1 dir left for provenance)

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **111 passed**.

Local v1 grids still score 0 vs 7 decision errors (`test_v1_saved_responses_still_score_0_vs_7`).

## Effective-key rules

Scientific identity hash includes: model, temperature, thinking_budget, configured and provider output limits, max_input_tokens, response MIME/schema, retry policy, grid fingerprint, implementation file hashes.

**Not** in the scientific hash: `max_calls`, `max_usd` (may increase; spend ledger persists).

Job key: `case_id|treatment|effective_hash|prompt_hash`.

Resume into a directory whose `scientific_identity.json` / `effective_hash` differs **raises `IdentityError` before any generate**. Legacy manifests without `effective_hash` refuse silent resume.

CLI model/thinking/temperature overrides change the hash. Reproduced: thinking=0 then thinking=8192 in the same dir is rejected.

Provider output limit is `configured_max_output_tokens + thinking_budget`, not a separate 2048 visible cap.

Token planning uses `client.count_tokens` when present (cached by request hash), including a dummy repair-prompt allowance. The old `estimate_tokens` function is labelled an offline heuristic.

Missing usage (`usage_status=unknown`) keeps the full reservation as spend. A saved `responses/*.json` with `scored=false` is scored on resume without a new generate.

## Limitations

- Gemini `count_tokens` is only used in live `BoundedGeminiClient`; fake tests use the heuristic via `ReferenceAnswerClient.count_tokens`.
- Finish reason is recorded when the SDK exposes it; `unavailable` is stored explicitly and does not by itself invalidate JSON.
- Thinking text is never requested or logged.

## Ready invocation for task 004 live (later; not run here)

After the commander reviews 004 dry-run fixtures:

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_sequential.py --live \
  --model gemini-3.5-flash --temperature 0 --thinking-budget 8192 \
  --max-input-tokens 4000 --max-output-tokens 2048 \
  --max-calls 192 --max-usd 20 \
  --output-dir runs/selective_revision_sequential_v1
```

Caps must be re-checked from the 004 dry-run token counts before that command is used.
