# Task 004 — Environment feasibility before a method pilot

Owner: Grok. Commander: Codex. Status: ready. Date: 29 September 2026.

## Read first

Read the research question, causal protocol, operating agreement, and [accepted kernel review](../reviews/003-review-commander.md). Tasks 001–003 are accepted for their offline fixture scope. This task investigates the scientific setting; do not reopen the kernel or add an LLM integration.

## Question

Can we specify a small continuing task in which experience produces useful procedures, a hidden partial change makes some of them harmful, and investigating/revising the affected procedure has a measurable benefit after its cost?

The strongest alternative is that a competent recency-weighted estimator or a flat online learner explains the entire benefit. The task must allow that alternative to win. Do not engineer a weak baseline to guarantee our hypothesis.

## Deliverable and budget

Produce a short environment-feasibility dossier under `experiments/004-environment-feasibility/`, with a small standalone CPU reference calculation if needed to check numerical examples. No full service simulator, provider adapter, benchmark suite, training, or sweep. No external services or paid calls. Use standard-library Python and the independent uv project.

This is a design/feasibility task, not evidence that an LLM learns or adapts. Hand-specified strategies and evaluator knowledge are permitted for calculating illustrative feasibility witnesses, but must be labeled as such and cannot be passed off as learner observations.

## Proposed starting setting

Consider a continuing job-service task with a persistent queue, job classes, deadlines, routing/batching decisions, and service/setup costs. Hidden changes can alter setup overhead, size-dependent service delay, or the value of a validation step. A procedure may combine batching, routing, and validation, so changing one relationship can leave other acquired knowledge useful.

This is a candidate, not a requirement to implement every mechanism. Compare it on paper with the existing MAB fixture and at most one simpler tool-workflow candidate, then recommend the smallest setting that exposes the proposed scientific distinction.

## Required dossier

1. **Concrete interaction contract.** State observable information, actions, hidden state, feedback timing, and one-life persistence. Give an explicit clock and reward formula with units. If actions take time, specify how arrivals/deadlines continue during execution and investigation. A current-regime simulator oracle must not be available to the learner.
2. **One complete worked trajectory.** Show acquisition, a hidden partial change, a costly diagnostic, and subsequent decisions using only feedback already received. Distinguish researcher annotations from runtime observations. Demonstrate that the agent could choose the wrong explanation; do not hand it the true mechanism/dependency graph.
3. **Feasibility witnesses.** Use exact arithmetic or a tiny deterministic reference program to check examples where: an old procedure earns value before change; it becomes costly after change; a selective repair preserves something useful; whole-system reset sacrifices something; and investigation has both benefit and cost. Also show a stable control where unnecessary investigation hurts. These are illustrative cases, not a tuned population result.
4. **Baseline challenge.** Specify a strong simple online reference using only permitted feedback, plus recent-history/reset and accumulating-procedure comparators. State where factorized estimation or a flat contextual bandit would suffice. If all proposed advantages collapse to ordinary recency estimation, recommend simplifying or rejecting the setting rather than increasing complexity just to defeat that baseline.
5. **Fairness and generalization plan.** Explain how the same information, action opportunities, and resource ceilings reach every method. Separate known-family shifts, recurrence, and withheld mechanisms/compositions. Do not claim universal open-world adaptation from a finite simulator.
6. **Recommendation and next implementation slice.** Recommend one setting, with the main confounds, unresolved identifiability issue, and the smallest experiment that could reject its research value. List exactly which existing kernel components remain valid and which would need changed semantics (especially clocks, endogenous state, checkpoints, and metrics).

A simple finite schedule is acceptable for a worked example when hidden from the learner. No claim that such a schedule proves generalization or that hand-designed witnesses establish method superiority.

## Acceptance criteria

The dossier must make the feedback and cost accounting unambiguous, include at least one checkable numerical example, name a competent simple baseline, and identify a plausible negative outcome. An “environment is unsuitable” recommendation is acceptable.

Do not optimize environment parameters using our eventual method's score; that method is not implemented. Do not invent scientific results from illustrative calculations. Use already documented primary sources only for context; any new related-work claim needs verification and an actual citation.

## Return and stop

Write `tasks/004-environment-feasibility-worker.md` with the dossier/reference paths, exact commands if code was run, findings, limitations, and recommended next slice. Keep the reference calculation isolated from production source and preserve all old results. Stop for commander review before building the full environment or making API calls.
