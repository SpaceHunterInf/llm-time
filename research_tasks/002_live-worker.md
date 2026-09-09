# Task 002 live return

Author: Grok (worker). User authorized ordinary runs up to about US$20. Smoke, thinking grid, and no-thinking ablation are done. No training.

## Spend

| Run | Spent | Cap |
|---|---:|---:|
| Smoke, thinking on | $0.088 | $2 |
| 72-call grid, thinking on | $1.683 | $20 |
| 72-call grid, thinking off | $0.155 | $20 |
| **Total** | **$1.93** | |

No DeepSeek. No `.env` in reports.

## Outcome

Thinking grid: 72/72 valid, 0 decision errors. No-thinking: 72/72 valid, 7 decision errors, all on **shared** structure, mostly missed `diagnose_B`. Reference treatment can have exact probabilities and still miss the diagnostic. Report: `reports/selective_revision_pilot_live.md`.

I am not launching training or another paid batch.

## Operator visibility (after the live runs)

`run_jobs(..., progress=True)` from the CLI. Each `generate()` return is flushed to `output_dir/responses/{case}_{treatment}_{attempt}.json` and `last_response.json` before parse/score. `progress.json` and `summary.json` rewrite after every completed job. tqdm on stderr.

## Tests still passing

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`: 102 passed after the progress/save change.
