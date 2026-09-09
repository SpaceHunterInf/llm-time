# Task 002 implementation return

Author: Grok (worker). Dry-run only. No Gemini, DeepSeek, training, or `.env` reads.

## Done

Fixed-history pilot harness, tests, versioned config, and dry-run artifacts. Historical benchmark code was not changed.

## Files

| Path | Role |
|---|---|
| `llm_rl_engine/analysis/selective_revision_pilot.py` | Public cases, shared prompts, strict JSON validation, expected-utility scoring |
| `llm_rl_engine/analysis/selective_revision_harness.py` | Budget preflight, resume, attempt ledger, dry-run/live runner |
| `llm_rl_engine/configs/pilot_selective_revision_v1.yaml` | Versioned grid, pricing, token caps; empty `live:` so live needs explicit flags |
| `scripts/selective_revision_pilot.py` | CLI; dry-run default |
| `tests/test_selective_revision_pilot.py` | Integrity and budget/resume checks |
| `reports/selective_revision_pilot_v1_dryrun/` | Prompts, evaluator (separate), hashes, manifest |
| `reports/selective_revision_pilot_dryrun.md` | Human dry-run report |
| `reports/selective_revision_pilot_dryrun.json` | Machine dry-run report |

## Commands actually run

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q tests/test_selective_revision_pilot.py
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --config llm_rl_engine/configs/pilot_selective_revision_v1.yaml
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --live
```

Results: **17/17** focused tests passed; **101/101** full suite passed. Dry-run wrote 72 prompts. Bare `--live` exited 2 *before* any client or `.env` read, missing `max_calls`, `max_usd`, `max_input_tokens`, `max_output_tokens`.

## Planned calls and cost bound

- 24 public cases × 3 treatments = **72** planned decision calls.
- Retry reserve: one retry each → **144** attempts.
- Caps: 2000 input tokens, 2048 output tokens, `thinking_budget=0`.
- Verified rates (9 September): $1.50 / $9.00 per million.
- Reserved per attempt: $0.021432.
- Reserved for the full grid plus retries: **$3.086208**.

Mechanism: every attempt is reserved at the token caps before the call; actual usage replaces the reservation after a completed response; an interrupted call keeps the reservation as spend. Resume key is `case_id|treatment|model|prompt_hash|config_hash`. Spend is never reset. Thinking must stay disabled; if a live response reports `thoughts_token_count>0`, the run must stop because the dollar ceiling would no longer be valid.

Largest dry-run prompt estimate: 1461 tokens (conservative 2-chars-per-token). None exceed the input cap.

## Limitations (visible)

- Diagnose commits to using the revealed rate optimally; this probe does not test interpretation of diagnostic outcomes.
- Labels and rates are fixed. Not structural-transfer evidence.
- No model outputs. No performance claim.
- Live Gemini uses `generate_content` with `max_output_tokens` and `thinking_budget=0`, not the historical `interactions.create` path. Historical agents are unchanged.
- Output-token bounding depends on the provider honouring those fields.
- Do not launch the 72-call batch from this return. The lead supplies model/budget, starting with `--smoke` (≤3 calls).

## Not done, as specified

No Gemini calls. No DeepSeek. No training. No commits. No subagents.
