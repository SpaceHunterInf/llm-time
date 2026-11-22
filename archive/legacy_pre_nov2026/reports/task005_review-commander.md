# Task 005 review and next decision

Author: Codex (commander), 11 September 2026. Reviewed the uncommitted task-005 implementation, worker return, live report, all 104 saved provider responses, and 96 scored jobs. No paid calls or training during this audit. Original results remain unchanged.

## Decision

The corrected run supports a development finding: **adding previous state to the full observation history is associated with worse forecasts and decisions, even when that previous state is exact.** It does not establish harmful memory in general, selective-revision training benefits, or a novel method. The next step is a bounded test of state continuation and action use, not another copy of the three-condition grid or a training job.

Next worker assignment: [006_state_handoff_localizer-commander.md](../research_tasks/006_state_handoff_localizer-commander.md). Repair the remaining metric/recovery defects, then compare four explicitly defined interfaces on six fixed new tapes with repeated requests. Stop after that experiment.

## Verified completion

- Current suite: **133 passed** with `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`.
- 96 jobs, 104 attempts, 88/96 first responses valid, 95/96 final responses valid. All attempts have STOP finish metadata. Strict offline validation of the actual raw numeric fields agrees with final validity.
- Recomputed token-cost estimate: **$8.5105815**, smoke included. No outstanding reservations or interrupted reservations in the ledger. This is an estimate from recorded usage, not a billing statement.
- All requests contain the complete .5 redraw law. Reconstructed base requests and actual self-state propagation, including retention of the last valid state after failure, match saved hashes. Tape fingerprints and recorded implementation hashes match current files.
- The prior task-004 reference audit remains applicable. Decision utilities below were independently recomputed as functions of the linked high-state probability.

Audit artifact: [task005_audit-commander.json](task005_audit-commander.json).

| Condition | Valid / jobs | Reference-optimal errors | Mean loss, all jobs | Linked probability MAE, valid |
|---|---:|---:|---:|---:|
| Reconstruct | 32/32 | 4 | .00431263 | .0127491 |
| Self-state | 31/32 | 14 | .03886256 | .1122189 |
| Reference-state | 32/32 | 13 | .02681638 | .0403078 |

The sole final schema failure is shared_s2, n=32, self_state. Its safe fallback happens to be reference-optimal, so it adds zero decision loss. Report both all-job and valid-only results; schema failure is still a failure even when the fallback succeeds.

There are 42 diagnose-optimal, 36 linked-optimal, and 18 safe-optimal jobs. Diagnosis is chosen correctly in 17/42: reconstruct 11/14, self-state 3/14, reference-state 3/14. All 24 stable-control decisions are correct. Unrelated forecast error is small: maximum .0094391 in self-state and .000127915 in reference-state. There is no demonstrated broad collateral forgetting effect here.

## Corrected interpretation of the state gap

Self-minus-reconstruct mean loss is **+.0345499**. All six changing tapes have a positive tape-average difference; the two stable controls have zero. The largest single cell contributes 33.51% of the total difference, versus about 79% in v1. Removing that cell leaves +.0237133; removing the entire shared_s2 tape leaves +.0149236. The gap is broader than the v1 result, but shared_s2 still contributes about 62% of the total and this remains a tiny development sample.

Reference-state-minus-reconstruct is **+.0225038**; removing its largest cell leaves +.0110393. At n=12 the self/reference requests are identical, with action disagreement on two of eight tapes. Excluding all n=12 cells leaves nearly equal state penalties: self +.0303533 and reference +.0297836. This argues for investigating the common state interface before attributing the effect specifically to self-generated errors.

The data do not identify double counting, anchoring, or internal reasoning. Full history plus a previous posterior permits either reconstruction or continuation; correct previous state is not the same intervention as supplying the exact CURRENT posterior. A suffix-only continuation comparison and a current-posterior control address different missing checks.

## The own-forecast metric needs correction

`score_sequential` chooses the model-forecast-optimal action using the model's probability, but compares its utility against the emitted action using the REFERENCE probability. Both candidate action values must be evaluated under the model's own probability for this metric.

Independent formulas, for linked high probability p:

```
U(linked) = .2 + .6 p
U(safe) = .65
U(diagnose) = .61 + .15 p
```

Corrected own-forecast errors are **2 reconstruct, 10 self-state, 7 reference-state: 19/95 valid outputs**. Of the 31 reference-optimal errors, **18** also violate the emitted forecast's optimum, rather than the reported 17. One additional output has a reference-optimal action that conflicts with its own forecast. These are observable output inconsistencies, not proof of an internal valuation mechanism.

Two mislabeled rows are shared_s1 n=44 self_state and independent_s2 n=44 self_state. For example, the latter emits linked-high probability .01646 but chooses linked; its own forecast favors safe, even though the reference favors diagnosis. The current metric misses that inconsistency.

As a free offline diagnostic, selecting the utility-maximizing action mechanically from each emitted joint reduces mean loss to .0012051 / .0121535 / .0080137 (reconstruct/self/reference; same safe fallback). This leaves forecast errors and is not an online intervention result, but shows that an expensive learned action selector is not the first remedy to try.

## Implementation acceptance is incomplete

The missing redraw definition and joint-sum acceptance are repaired in the actual live data; changed-tape identity rejection is now tested. Remaining issues should be repaired without paying to reproduce this finished grid:

1. Keyed values are converted with `float()` before strict numeric validation, so booleans and numeric strings are accepted. Offline examples with `true` or `"1"` pass. Actual v2 final outputs contain proper numeric values, so this does not change their validity counts.
2. Injecting a crash after saving and charging the first invalid response causes resume to finalize an invalid job without using its remaining retry. This was reproduced through the actual runner with a fake client.
3. Injecting a crash after saving a valid repair response causes resume to raise `IdentityError: saved response request hash mismatch`, because it compares the repaired request hash to the base prompt hash. Also reproduced through the runner.
4. Shared provider, JSON extraction, and harness implementation files are still absent from sequential scientific identity. Cumulative retry handling across recovery is not implemented as required.
5. Added resume tests cover the initial checkpoint and a manually persisted valid base response; they do not cover later self-state checkpoints, changed previous-state rejection, or the two crash cases above. A passing count does not substitute for these requirements.

These recovery failures do not retroactively invalidate the cleanly recorded v2 jobs: recorded requests/state match, all charges reconcile, and no interrupted reservations are recorded. Preserve v2 and add corrected metrics in a separate audit output.

## Link to the research hypothesis

We still have no implemented novel selective-revision learner. The candidate from the research notes is training on matched evidence/dependency cases with supervision of both appropriate changes and preservation. Ordinary Bayesian distillation, learned memory, and prediction-error rule updates are established ingredients; performance on this numerical interface does not distinguish that proposal from them.

Task 006 tests whether a straightforward state interface or deterministic action calculation already addresses the observed failure. If it does, retain it as an engineering baseline and close this diagnostic branch. If it does not, identify the remaining failure before specifying training. Neither outcome alone establishes the proposed structural-transfer contribution.
