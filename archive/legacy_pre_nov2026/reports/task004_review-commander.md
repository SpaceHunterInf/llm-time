# Review of tasks 003 and 004

Author: Codex (commander). 11 September 2026. Reviewed commit `1e76874`, Grok's task-003, task-004, and task-004-live returns, saved requests/responses, and current tests. This review made no model calls.

## Decision

Task 003 repaired important problems in the original pilot harness. Task 004 implemented the intended sequential reference and completed a live grid, but the new experiment has a missing task definition and substantial output-validation problems. **Do not accept its headline as clean evidence of independent-change failure or harmful memory. Correct the protocol and rerun the same development tapes once. No training or broader environment is justified yet.**

Next assignment: [005_sequential_protocol_correction-commander.md](../research_tasks/005_sequential_protocol_correction-commander.md).

## What actually completed

| Artifact/run | Verified status |
|---|---|
| Current test suite | **127 passed** (`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`) |
| Sequential reference | Independently enumerated reset masks and redraws; all three marginals match at all 32 tape/checkpoints to 1e-12 |
| Sequential smoke | 3 completed jobs, 5 attempts, $0.455568 in ledger estimates |
| Sequential grid | 96 completed jobs, 101 attempts, $8.819271 in ledger estimates |
| Outstanding reservations | Zero in both ledgers |
| Provider finish metadata | All 101 grid attempts report STOP; largest billed output 10,217, below configured 10,240 |

Total sequential ledger estimate: **$9.274839**. Costs are logged token estimates, not billing-statement verification. Old worker dry-run notes predate the live run and should not be read as current status.

Audit data: [task004_audit-commander.json](task004_audit-commander.json). Reconstructed base prompts, checked stored request hashes and repair-prefix continuity, verified actual self-state propagation, and recomputed final decision utilities. The reference is not the problem identified in this review.

Under the current evaluator and permissive parser, the worker's aggregate totals reproduce:

| Condition | Jobs | Decision errors | Mean loss |
|---|---:|---:|---:|
| Reconstruct | 32 | 12 | .0171064 |
| Self-state | 32 | 14 | .0321224 |
| Reference-state | 32 | 11 | .0135817 |

These numbers describe the original protocol, including the defects below. They are not results under a corrected protocol.

## 1. Independent-reset prompts omit the reset distribution

Actual saved requests say:

> A and B have independent Bernoulli(0.03) resets with the same redraw rule.

There is no preceding redraw rule in that request. The initial high-state probability .99 is stated, but the reset-high probability .5 is not. The shared prompt does explicitly define .5. Both the base request and retries inherit this omission. All 38 independent-condition attempts have the missing definition.

Therefore the two structures were not presented with equally complete generative models. The reported 29/36 independent-condition errors cannot cleanly be attributed to the model's reasoning about the intended uniform-redraw model.

As a diagnostic calculation, 34/36 independent-condition reported linked probabilities are closer to a filter that resets high with probability .99 than to the intended .5-reset filter. This is compatible with the ambiguity mattering; it does not reveal the model's internal reasoning or prove what a corrected prompt will do.

## 2. “96 valid” includes silently repaired distributions

`parse_sequential_response` accepts any eight nonnegative finite masses with a positive total and divides by that total. It does not enforce the requested sum of 1 or report the correction.

Among the 96 final responses, **15** have an absolute raw sum error above 1e-4. Examples:

- `independent_s2`, n=32, reconstruct: raw sum **93.7915293**.
- `independent_s1`, n=20, reference-state: raw sum **0.000036**.
- `shared_s1`, n=44, reconstruct: raw sum **0.0008**.

Thus the run has 96 parser-accepted responses, not 96 schema-compliant distributions. It would be wrong to retroactively pretend those 15 were fresh correctly retried outputs. Preserve the old scored results and add raw-validity diagnostics.

In **13** final responses, marginals implied by the normalized joint disagree with separately reported marginals by more than .001; the maximum disagreement is about .99992. The prose also says both “000..111” and least-significant-bit ordering, while the schema lists a nonlexicographic sequence. Although the explicit list specifies the intended mapping, this is avoidable representation ambiguity.

Because joint-derived unrelated error can approach 1 while another output field gives a different belief, we cannot confidently call this collateral forgetting. The next schema should remove positional ambiguity and redundant estimates, then derive marginals mechanically from the sole accepted joint.

## 3. Most action misses are consistent with wrong forecasts

Of the 37 evaluator errors, only **7** are suboptimal under the model's separately reported linked probability; **8** are suboptimal under the normalized joint's linked probability. The rest are consistent with its own forecasts. Calling the dominant failure diagnostic valuation is therefore premature. Prior-posterior errors, the undefined redraw law, and representation errors must be separated first.

The worker's exact action-error breakdown is also inaccurate. Actual counts are:

- 33 choose linked when diagnosis is optimal.
- 3 choose safe when diagnosis is optimal.
- 1 chooses linked when safe is optimal.

The last is `shared_s2`, n=32, self-state, loss .3779017. It contributes **78.65%** of the total self-minus-reconstruct excess loss. The overall paired difference is .015016, but only three shared tapes contribute nonzero differences; all independent/stationary tapes contribute zero. This is development evidence requiring replication, not a demonstrated general memory penalty.

At n=12, self-state and reference-state base prompts are identical: both receive the initial prior. Treat those two outputs as repeat requests with the same inputs, not as different memory mechanisms. Temperature zero does not establish repeatability of later comparisons.

## 4. Sequential runner reintroduces an identity hole

The original pilot now hashes effective settings. The separate sequential runner, however, hashes tape IDs rather than tape contents and skips completed jobs on either full key OR logical `(tape_id,n,condition)` equality.

Offline fake-client reproduction: run three jobs on a tape; flip its first observation while keeping its tape ID; resume the directory. The runner silently accepts it and makes zero new calls. It should reject the altered scientific input. Existing checked requests match their recorded tapes, so this does not by itself invalidate the completed data; it does invalidate the assumption that new sequential resumes enforce full identity.

Additional recovery concerns in this new path: the first self-state key is constructed inconsistently with an explicit initial-state hash versus `none`; saved-response lookup is by logical slot; the recovery scoring path does not check finish reason like the fresh-response path. Target tests must exercise the actual live runner, including later self-state checkpoints, not just a hand-built imitation loop.

## Next research decision after correction

Use a versioned complete prompt and keyed joint-state schema. Keep all eight tapes, conditions, and thinking configuration. Inspect dry-run requests and smoke results, then run the corrected 96-job development grid within a preflighted complete-experiment budget. Interpret v1→v2 changes as a protocol correction, not a proven intervention effect: several defects are repaired together.

If the corrected thinking-enabled model solves it, accept that and move to a separately identifiable harder task. If there is still a memory-specific gap, repeat on new tape seeds/role rotations and perform a targeted previous-state replacement before training. If forecasts are correct but decisions fail, supply exact current forecasts/action values in a small matched localizer before choosing a training objective. If schema failures remain dominant, redesign the representation instead of treating them as adaptation evidence.
