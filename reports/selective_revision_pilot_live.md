# Selective revision Gemini pilot (live)

9 September 2026. Frozen `gemini-3.5-flash`, temperature 0. Enumerated 24-case grid × 3 treatments = 72 calls per run. This is a diagnostic screen, not a powered evaluation and not a training result.

## What was run

| Run | Thinking | Calls | Valid JSON | Decision errors | Spent |
|---|---|---:|---:|---:|---:|
| Smoke (`case_01` × 3 treatments) | budget 8192 | 3 | 3 | 0 | $0.088 |
| Full grid | budget 8192 | 72 | 72 | **0** | $1.683 |
| Full grid ablation | **off** | 72 | 72 | **7** | $0.155 |

Caps: 2000 input tokens, 2048 visible output, thinking 8192 when on. Worst-case reserve for the thinking grid was about $13.70; actual spend was $1.68. No retries, no auth/quota stop, no truncation. Artifacts: `runs/selective_revision_pilot_v1_{smoke,grid,grid_nothink}/` (gitignored). Live traces stay on the machine; this report is the shareable summary.

Later harness change (after these runs): each provider response is written to `responses/` *before* scoring, `summary.json`/`progress.json` update after every job, and stderr shows a tqdm bar plus one line per job. A crash mid-grid should no longer lose the last model output.

Reproduce:

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --live \
  --model gemini-3.5-flash --max-calls 144 --max-usd 20 \
  --max-input-tokens 2000 --max-output-tokens 2048 --thinking-budget 8192 \
  --output-dir runs/selective_revision_pilot_v1_grid
```

## Result

With thinking, the frozen model matches the observation-limited reference on this supplied-structure one-event task: every action is optimal, C never drifts, Brier on P(high) is ~0 (worst cell 3e-5). Input was 601–730 tokens; thinking used up to 5116 (under 8192).

Without thinking, JSON still parses 72/72, independent and stable stay perfect, and **all 7 errors are on shared A/B resets**. C still does not drift. Mean ITT decision loss over the grid is 0.0067 (below the 0.02 screening average). Loss on the missed cells is not small: 0.15 and 0.10 on the five-failure shared cases.

The missed cells are exactly where the optimum is `diagnose_B` (cheap diagnostic, B is in the uncertain band after A failures/mixed evidence):

| Case | A successes / 5 | Cost | Treatment | Chose | Optimal | Loss | Brier |
|---|---:|---:|---|---|---|---:|---:|
| case_01 | 0 | 0.04 | raw, counts | choose_B | diagnose_B | 0.149 | 0.03–0.04 |
| case_01 | 0 | 0.04 | reference | choose_safe | diagnose_B | 0.047 | ~0 |
| case_02 | 0 | 0.20 | counts | choose_B | choose_safe | 0.102 | 0.034 |
| case_03 | 2 | 0.04 | raw, counts, reference | choose_B | diagnose_B | 0.011 | 0–0.14 |

On `case_01` and `case_03` **reference**, supplied posterior probabilities are exact (Brier ~0) and the action is still wrong. That is a valuation-of-information miss, not an inference miss.

## What this does and does not justify

- It does **not** justify training on this numeric task for Gemini 3.5 Flash. A frozen thinking pass removes the entire decision loss.
- It does **not** justify posterior SFT. Correct probabilities were already enough in the reference treatment, and the remaining errors were in using them to buy the diagnostic.
- It does show a correctable, structure-specific gap without thinking: shared-reset VOI, not C-overgeneralization, not independent-reset leakage, not formatting.
- The one-event supplied-structure grid is a ceiling for this model with thinking. Broader claims need hidden structure, longer histories, or a language workflow.

## Next (lead decides)

Cheapest remaining probes, if any: hide the dependency graph; counterbalance labels; or keep thinking off and teach diagnostic valuation. Do not start weight training because this grid is unsolved. Do not auto-launch another paid batch from this file.
