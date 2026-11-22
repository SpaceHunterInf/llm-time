# Task 005 return

Author: Grok. v1 runs were not rewritten. No training.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **133 passed** before live (includes v1 audit 15/13, redraw-law prompt check, keyed-sum rejection, changed-tape identity, schema retry, saved-response recovery).

## Protocol

- Shared generative paragraph with 0.5 redraw in **every** structure.
- Keyed joint `000..111` = observed, linked, unrelated. Marginals derived, not requested.
- Sequential identity hashes tape contents, not only IDs. Resume rejects tape-ID collision with flipped outcomes.
- Smoke jobs are the first three of the v2 96: reconstruct n=12 shared/independent/stable.

## Live

- Dir: `runs/selective_revision_sequential_v2`
- Effective hash: `a85781f4…35375d`
- Smoke 3/3 valid, $0.216, then resume 93. Total 96 jobs, 104 attempts, **$8.511**, 95 valid, 1 parse fail, 31 valid decision errors.
- Report: `reports/selective_revision_sequential_v2_live.md`

## Limitations

v1→v2 mixes prompt completeness and schema; not an ablation. One parse failure. n=12 self/reference are repeat inputs. Self-state gap is real on these tapes but outlier-dominated (`shared_s2`). Stopped here.
