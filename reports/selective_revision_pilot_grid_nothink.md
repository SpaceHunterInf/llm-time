# Selective revision pilot run: `selective_revision_pilot_v1_grid_nothink`

Descriptive grid summary, not a population estimate. No confidence intervals.

## Totals

- Attempted jobs: 72
- Valid JSON: 72
- API failures: 0
- Parse failures: 0
- Decision errors among valid: 7
- ITT mean decision loss: 0.006667226085076042
- Valid mean Brier (P high): 0.023540258241838904
- Valid mean C drift from prior predictive: 0.0
- Overpriced diagnostics (valid): 0
- Ledger spent_usd: 0.15540600000000002
- Ledger attempts: 72

## By treatment

| Treatment | n | valid | decision errors | mean ITT loss |
|---|---:|---:|---:|---:|
| raw | 24 | 24 | 2 | 0.006667226085076042 |
| counts | 24 | 24 | 3 | 0.010899351188976164 |
| reference | 24 | 24 | 2 | 0.0024351009811759187 |

## Token usage (completed attempts)

- n attempts logged complete: 72
- input tokens min/max: 601 / 730
- billed output min/max: 100 / 160
- thoughts min/max: 0 / 0

## Per job

| case | treatment | valid | action | optimal | loss | failure |
|---|---|---|---|---|---:|---|
| case_01 | raw | True | choose_B | diagnose_B | 0.14867825187020223 | None |
| case_01 | counts | True | choose_B | diagnose_B | 0.14867825187020223 | None |
| case_01 | reference | True | choose_safe | diagnose_B | 0.04710724937659927 | None |
| case_02 | raw | True | choose_safe | choose_safe | 0.0 | None |
| case_02 | counts | True | choose_B | choose_safe | 0.10157100249360296 | None |
| case_02 | reference | True | choose_safe | choose_safe | 0.0 | None |
| case_03 | raw | True | choose_B | diagnose_B | 0.011335174171622775 | None |
| case_03 | counts | True | choose_B | diagnose_B | 0.011335174171622775 | None |
| case_03 | reference | True | choose_B | diagnose_B | 0.011335174171622775 | None |
| case_04 | raw | True | choose_B | choose_B | 0.0 | None |
| case_04 | counts | True | choose_B | choose_B | 0.0 | None |
| case_04 | reference | True | choose_B | choose_B | 0.0 | None |
| case_05 | raw | True | choose_B | choose_B | 0.0 | None |
| case_05 | counts | True | choose_B | choose_B | 0.0 | None |
| case_05 | reference | True | choose_B | choose_B | 0.0 | None |
| case_06 | raw | True | choose_B | choose_B | 0.0 | None |
| case_06 | counts | True | choose_B | choose_B | 0.0 | None |
| case_06 | reference | True | choose_B | choose_B | 0.0 | None |
| case_07 | raw | True | choose_B | choose_B | 0.0 | None |
| case_07 | counts | True | choose_B | choose_B | 0.0 | None |
| case_07 | reference | True | choose_B | choose_B | 0.0 | None |
| case_08 | raw | True | choose_B | choose_B | 0.0 | None |
| case_08 | counts | True | choose_B | choose_B | 0.0 | None |
| case_08 | reference | True | choose_B | choose_B | 0.0 | None |
| case_09 | raw | True | choose_B | choose_B | 0.0 | None |
| case_09 | counts | True | choose_B | choose_B | 0.0 | None |
| case_09 | reference | True | choose_B | choose_B | 0.0 | None |
| case_10 | raw | True | choose_B | choose_B | 0.0 | None |
| case_10 | counts | True | choose_B | choose_B | 0.0 | None |
| case_10 | reference | True | choose_B | choose_B | 0.0 | None |
| case_11 | raw | True | choose_B | choose_B | 0.0 | None |
| case_11 | counts | True | choose_B | choose_B | 0.0 | None |
| case_11 | reference | True | choose_B | choose_B | 0.0 | None |
| case_12 | raw | True | choose_B | choose_B | 0.0 | None |
| case_12 | counts | True | choose_B | choose_B | 0.0 | None |
| case_12 | reference | True | choose_B | choose_B | 0.0 | None |
| case_13 | raw | True | choose_B | choose_B | 0.0 | None |
| case_13 | counts | True | choose_B | choose_B | 0.0 | None |
| case_13 | reference | True | choose_B | choose_B | 0.0 | None |
| case_14 | raw | True | choose_B | choose_B | 0.0 | None |
| case_14 | counts | True | choose_B | choose_B | 0.0 | None |
| case_14 | reference | True | choose_B | choose_B | 0.0 | None |
| case_15 | raw | True | choose_B | choose_B | 0.0 | None |
| case_15 | counts | True | choose_B | choose_B | 0.0 | None |
| case_15 | reference | True | choose_B | choose_B | 0.0 | None |
| case_16 | raw | True | choose_B | choose_B | 0.0 | None |
| case_16 | counts | True | choose_B | choose_B | 0.0 | None |
| case_16 | reference | True | choose_B | choose_B | 0.0 | None |
| case_17 | raw | True | choose_B | choose_B | 0.0 | None |
| case_17 | counts | True | choose_B | choose_B | 0.0 | None |
| case_17 | reference | True | choose_B | choose_B | 0.0 | None |
| case_18 | raw | True | choose_B | choose_B | 0.0 | None |
| case_18 | counts | True | choose_B | choose_B | 0.0 | None |
| case_18 | reference | True | choose_B | choose_B | 0.0 | None |
| case_19 | raw | True | choose_B | choose_B | 0.0 | None |
| case_19 | counts | True | choose_B | choose_B | 0.0 | None |
| case_19 | reference | True | choose_B | choose_B | 0.0 | None |
| case_20 | raw | True | choose_B | choose_B | 0.0 | None |
| case_20 | counts | True | choose_B | choose_B | 0.0 | None |
| case_20 | reference | True | choose_B | choose_B | 0.0 | None |
| case_21 | raw | True | choose_B | choose_B | 0.0 | None |
| case_21 | counts | True | choose_B | choose_B | 0.0 | None |
| case_21 | reference | True | choose_B | choose_B | 0.0 | None |
| case_22 | raw | True | choose_B | choose_B | 0.0 | None |
| case_22 | counts | True | choose_B | choose_B | 0.0 | None |
| case_22 | reference | True | choose_B | choose_B | 0.0 | None |
| case_23 | raw | True | choose_B | choose_B | 0.0 | None |
| case_23 | counts | True | choose_B | choose_B | 0.0 | None |
| case_23 | reference | True | choose_B | choose_B | 0.0 | None |
| case_24 | raw | True | choose_B | choose_B | 0.0 | None |
| case_24 | counts | True | choose_B | choose_B | 0.0 | None |
| case_24 | reference | True | choose_B | choose_B | 0.0 | None |
