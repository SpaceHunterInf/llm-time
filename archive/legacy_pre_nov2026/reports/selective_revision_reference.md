# Selective revision: exact reference

CPU-only mathematical check. No LLM calls, training, or sampled performance evaluation.

## Assumptions

A/B/C independently start high with probability .99; rates are .2 or .8. The prior summarizes earlier evidence; no raw prehistory is asserted. One possible reset occurs before the observation batch with probability .1. Shared A/B resets share an event but redraw their values independently. Independent resets have matched marginal hazards. C cannot reset. Stable explicitly means a known zero hazard; it is not a secretly labelled no-event realization under a nonzero hazard.

A diagnostic reveals B's rate, not its next reward. After diagnosis choose the higher expected-utility option and charge the diagnostic cost once. The alternative pays .65. Ties within 1e-12 prefer safe, then B, then diagnosis.

## Worked example: five failures on A

| Structure | Forecast A | Forecast B | Forecast C | B utility | Safe utility | Diagnostic utility | Best action |
|---|---:|---:|---:|---:|---:|---:|---|
| shared | 0.209202 | 0.548429 | 0.794000 | 0.548429 | 0.650000 | 0.697107 | diagnose_B |
| independent | 0.209202 | 0.764600 | 0.794000 | 0.764600 | 0.650000 | 0.751150 | choose_B |

The reference predicts the same A outcome in both cases, preserves C, and recommends different B decisions. This establishes a discriminating task, not that an LLM fails it or that training helps.

## Diagnostic grid

54 enumerated cases: six A success counts, three structures, three diagnostic costs. These are not independent episodes. Full inputs and targets are stored separately in the adjacent JSON; reference targets must not leak into ordinary prompts.

| Structure | A successes / 5 | Cost | Forecast B | Best action |
|---|---:|---:|---:|---|
| shared | 0 | 0.00 | 0.548429 | diagnose_B |
| shared | 0 | 0.04 | 0.548429 | diagnose_B |
| shared | 0 | 0.20 | 0.548429 | choose_safe |
| shared | 1 | 0.00 | 0.591435 | diagnose_B |
| shared | 1 | 0.04 | 0.591435 | diagnose_B |
| shared | 1 | 0.20 | 0.591435 | choose_safe |
| shared | 2 | 0.00 | 0.731553 | diagnose_B |
| shared | 2 | 0.04 | 0.731553 | diagnose_B |
| shared | 2 | 0.20 | 0.731553 | choose_B |
| shared | 3 | 0.00 | 0.774774 | diagnose_B |
| shared | 3 | 0.04 | 0.774774 | choose_B |
| shared | 3 | 0.20 | 0.774774 | choose_B |
| shared | 4 | 0.00 | 0.778150 | diagnose_B |
| shared | 4 | 0.04 | 0.778150 | choose_B |
| shared | 4 | 0.20 | 0.778150 | choose_B |
| shared | 5 | 0.00 | 0.778364 | diagnose_B |
| shared | 5 | 0.04 | 0.778364 | choose_B |
| shared | 5 | 0.20 | 0.778364 | choose_B |
| independent | 0 | 0.00 | 0.764600 | diagnose_B |
| independent | 0 | 0.04 | 0.764600 | choose_B |
| independent | 0 | 0.20 | 0.764600 | choose_B |
| independent | 1 | 0.00 | 0.764600 | diagnose_B |
| independent | 1 | 0.04 | 0.764600 | choose_B |
| independent | 1 | 0.20 | 0.764600 | choose_B |
| independent | 2 | 0.00 | 0.764600 | diagnose_B |
| independent | 2 | 0.04 | 0.764600 | choose_B |
| independent | 2 | 0.20 | 0.764600 | choose_B |
| independent | 3 | 0.00 | 0.764600 | diagnose_B |
| independent | 3 | 0.04 | 0.764600 | choose_B |
| independent | 3 | 0.20 | 0.764600 | choose_B |
| independent | 4 | 0.00 | 0.764600 | diagnose_B |
| independent | 4 | 0.04 | 0.764600 | choose_B |
| independent | 4 | 0.20 | 0.764600 | choose_B |
| independent | 5 | 0.00 | 0.764600 | diagnose_B |
| independent | 5 | 0.04 | 0.764600 | choose_B |
| independent | 5 | 0.20 | 0.764600 | choose_B |
| stable | 0 | 0.00 | 0.794000 | diagnose_B |
| stable | 0 | 0.04 | 0.794000 | choose_B |
| stable | 0 | 0.20 | 0.794000 | choose_B |
| stable | 1 | 0.00 | 0.794000 | diagnose_B |
| stable | 1 | 0.04 | 0.794000 | choose_B |
| stable | 1 | 0.20 | 0.794000 | choose_B |
| stable | 2 | 0.00 | 0.794000 | diagnose_B |
| stable | 2 | 0.04 | 0.794000 | choose_B |
| stable | 2 | 0.20 | 0.794000 | choose_B |
| stable | 3 | 0.00 | 0.794000 | diagnose_B |
| stable | 3 | 0.04 | 0.794000 | choose_B |
| stable | 3 | 0.20 | 0.794000 | choose_B |
| stable | 4 | 0.00 | 0.794000 | diagnose_B |
| stable | 4 | 0.04 | 0.794000 | choose_B |
| stable | 4 | 0.20 | 0.794000 | choose_B |
| stable | 5 | 0.00 | 0.794000 | diagnose_B |
| stable | 5 | 0.04 | 0.794000 | choose_B |
| stable | 5 | 0.20 | 0.794000 | choose_B |

## Reproduction and limits

```bash
uv run python scripts/selective_revision_sanity.py
uv run pytest -q tests/test_selective_revision.py
```

The observer enumerates event/state pairs and normalizes log likelihoods. Impossible observations raise an error. It accepts observations of all three components, but supports one pre-batch event only and fixed A/B/C roles. Label permutation, learned dependency structure, repeated changes, and actual post-diagnostic model behavior remain follow-up work. Forecasts condition on the stated model, never the simulator's actual hidden event.
