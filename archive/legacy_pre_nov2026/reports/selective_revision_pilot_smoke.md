# Selective revision pilot run: `selective_revision_pilot_v1_smoke`

Descriptive grid summary, not a population estimate. No confidence intervals.

## Totals

- Attempted jobs: 3
- Valid JSON: 3
- API failures: 0
- Parse failures: 0
- Decision errors among valid: 0
- ITT mean decision loss: 0.0
- Valid mean Brier (P high): 4.981099200538363e-14
- Valid mean C drift from prior predictive: 0.0
- Overpriced diagnostics (valid): 0
- Ledger spent_usd: 0.0880725
- Ledger attempts: 3

## By treatment

| Treatment | n | valid | decision errors | mean ITT loss |
|---|---:|---:|---:|---:|
| raw | 1 | 1 | 0 | 0.0 |
| counts | 1 | 1 | 0 | 0.0 |
| reference | 1 | 1 | 0 | 0.0 |

## Token usage (completed attempts)

- n attempts logged complete: 3
- input tokens min/max: 645 / 730
- billed output min/max: 1552 / 3962
- thoughts min/max: 1408 / 3844

## Per job

| case | treatment | valid | action | optimal | loss | failure |
|---|---|---|---|---|---:|---|
| case_01 | raw | True | diagnose_B | diagnose_B | 0.0 | None |
| case_01 | counts | True | diagnose_B | diagnose_B | 0.0 | None |
| case_01 | reference | True | diagnose_B | diagnose_B | 0.0 | None |
