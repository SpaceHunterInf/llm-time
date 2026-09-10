# Task 004 return

Author: Grok (worker). Phase A/B implementation and dry-run only. No Gemini, no training.

## Files

- `llm_rl_engine/analysis/selective_revision_sequential.py` — 8-state filter, frozen tapes, prompts, scoring
- `scripts/selective_revision_sequential.py` — dry-run CLI (`--live` exits 2)
- `tests/test_selective_revision_sequential.py`
- `reports/selective_revision_sequential_v1_dryrun/` — tapes, 96 prompts, hashes, figure, manifest
- `reports/selective_revision_sequential_dryrun.md`

## Checks run

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **126 passed**.

Includes: h=0 identity; h=1 independent redraw; one-step match to task 001; two-step shared/independent brute force; label permutation; chunked=replay; shared correlation; no hidden/future fields in public prefix/prompts; fake sequential resume without duplicate calls; fixtures where safe / B / diagnose are each optimal (`decision_values` unit cases plus 6/12/14 counts over 32 tape×probe cells).

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_sequential.py \
  --output-dir reports/selective_revision_sequential_v1_dryrun
```

## Grid

8 tapes × 4 checkpoints × 3 conditions = **96** jobs. Retry reserve 192.

Heuristic max prompt ≈ 1363 tokens. Illustrative reserve at 4000 input and 10240 provider output (2048+8192): **$18.85**. Recheck with `count_tokens` before live; 2000 input may already fit these templates.

Figure: `reports/selective_revision_sequential_v1_dryrun/figures/shared_s0_linked.png`

## Limitations

- Later **self-state** prompts in the dry-run use a size-matched reference joint as a placeholder. Live must insert the condition's actual previous parsed joint and its hash.
- Probes do not change the tape or reveal diagnostics.
- `--live` is disabled until the commander reviews fixtures and supplies the invocation from the task-003 worker note.
- Did not reduce tape count; illustrative reserve is under $20 at those caps.

## Not done

No paid sequential batch. No training.
