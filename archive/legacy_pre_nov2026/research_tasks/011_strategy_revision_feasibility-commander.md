# Task 011: feasibility of strategy revision under hidden drift

> Status, 25 September 2026: superseded by [task 012](012_persistent_self_improvement-commander.md). No worker return/results for 011 are present in this checkout. Preserve this proposal; do not execute it as the next task.

Author: Codex (commander), 15 September 2026. Worker: Grok. Read `auto_research_rules.md` and [the direction memo](../research-direction-rsi-commander.md). Preserve previous experiments and commander plans.

## Scope

One **CPU-only** prototype and falsification report. No APIs, neural optimization, model downloads, GPU, full agent framework or real market data. Use uv and at most four CPU threads. Do not invoke another agent. The numerical posterior-distillation branch remains closed.

Question: can an observation-limited controller selectively revalidate archived strategy performance after hidden partial changes, with enough net decision benefit to justify a later LLM code-revision experiment?

This task is not evidence of RSI or a novel learning method. A fixed policy library is deliberately used to isolate controller feasibility. Do not rename a library selector as a self-improver.

## A. Small explicit problem, not an open simulator project

Implement repeated independent allocation rounds with three worker pools and a small batch of jobs. A round has a public resource limit, job values, sizes and deadlines. A policy returns which jobs to assign to which pools. Hidden processing behavior determines realized feasible completion and task utility. Use one simple documented formula, with bounded rewards and explicit overflow/deadline penalties. Keep batches tiny enough for exhaustive allocation checks on selected fixtures.

Use at most six hand-written strategy programs and two hand-written repair candidates. Include sensible value/runtime, deadline and conservative-capacity heuristics. Every policy must output allocations from inputs rather than identify a hidden regime or return a table of oracle scores. Declare the candidate set before evaluation; all controllers receive the same policies and arrival schedule for repair candidates. This tests selection/validation, not candidate discovery.

Freeze one parameterization and four scenario families before running comparison seeds:

1. Stationary noisy behavior.
2. A hidden change affecting one pool.
3. A shared hidden change affecting two pools, third unaffected.
4. A changed mechanism later recurs to its previous behavior.

Include ordinary noise in every family, including shifts that produce little visible evidence initially. Generate change times and exogenous randomness independently of policy actions. Agent-visible data must exclude cut times, regime IDs, changed pools, future values and simulator-only outcomes. Public code/trace dependencies may be visible, but they are not labels identifying what changed. Distinguish supplied code dependency from inferred change scope.

A controller sees public jobs/context and outcomes from its actual allocations. A diagnostic action buys an additional bounded current-regime test, on a separately sampled job batch, and returns only that observed result. Candidate/incumbent paired tests cost two executions; sharing exogenous noise for variance reduction must be applied consistently and recorded. No free live-state replay or unlimited access to an oracle simulator.

If a probe cannot be defined without revealing the answer, return that issue before expanding the environment. Use scripted canonical traces to demonstrate observability and ambiguity.

## B. Controllers and reference checks

Implement only these cheap controls:

- Frozen reasonable strategy.
- Adaptive selector using a recent window of observed per-policy utility, with explicit exploration/test budget.
- Global change-triggered revalidation of the archive.
- Independent per-policy change-triggered revalidation.
- Candidate selective controller: maintain a small explicit set of noise/local/shared change explanations from observations, and use it to choose which archived policies to retest. Explain the update and decision rule mathematically; reuse exact small-state reference code where applicable. Do not treat its posterior as a novel method or use the true simulator cause.
- Known-change/known-scope reference, clearly marked as privileged. It receives current change information, never future realizations. Do not call a greedy oracle policy globally optimal.

For the candidate, start with an enumerated observation-limited hypothesis model rather than a hand-tuned uncalibrated LLM score. Require a worked ambiguous example: two possible causes share an observed failure but warrant different test choices; a diagnostic can distinguish them at a cost. Compare with the same observations under the independent control.

If a simpler controller has the same information and performance, report that. No baseline sabotage, outcome-dependent tuning or supplying change labels to the candidate. Preserve the candidate/reference distinction: exact inference over a known hypothesis family is a diagnostic instrument.

## C. Fixed small evaluation and cost accounting

Use 64 independent episode seeds per scenario family and 128 allocation rounds per episode. Choose all constants and detector/window settings from at most eight separate development seeds per family, before evaluating these 64. Permit at most three documented settings for the simple window/detector controls on development; candidate tuning receives no larger trial budget. Keep the final protocol/config hash before evaluation.

Report normalized task utility, cumulative net utility, revision/probe counts, stationary loss, affected/unaffected-pool outcomes and recurrence recovery. Probe costs must be positive, stated in task units, and identical across controls. Code proposal cost is zero in this CPU task because no proposals are generated; explicitly explain why this underestimates an eventual LLM loop's total cost.

Use predeclared probe-cost sensitivity at 0, 0.01 and 0.05 normalized task units per diagnostic execution. Zero cost is diagnostic only and cannot satisfy the go decision. Report paired episode differences and intervals across episode seeds, not across correlated rounds. Do not select a favorable cost after the result. Keep true-state logs separate from the public agent records.

The first prototype may expose insufficient headroom or poor identifiability. Do not generate a second environment parameterization until a favorable result appears. Stop and return the finding. Bound the complete comparison to ten CPU minutes; save complete episodes and identify missing ones if that ceiling is reached.

## D. Acceptance, interpretation and return

Acceptance requires tested reward/feasibility calculations, reproducible event-indexed randomness, agent-input isolation, an independently checked small reference example, honest costs and complete keyed result records. Tests should catch an oracle-label leak and incorrect charging of paired tests, not only mirror the implementation.

The output is useful only if we can answer:

1. Is there a material decision difference between stationary/noise, local drift and shared drift using permitted evidence?
2. Does the candidate outperform the strongest cheap controller at positive test costs, with paired uncertainty and stationary/unaffected losses visible?
3. Is the advantage due to selective testing, or merely a better fixed strategy or more observations?
4. What remains for an LLM to do beyond this fixed library, and what would distinguish the proposed mechanism from RQGM's known-transition erasure and SHARP-style local edits?

If those answers do not justify an LLM pilot, recommend stopping or revising the research formulation. A tiny positive mean within uncertainty is inconclusive. If the fixed-library selector is sufficient, state that the experiment supplies no case for program generation or RSI.

Return `research_tasks/011_strategy_revision_feasibility-worker.md` and `reports/strategy_revision_feasibility.md`, with exact configs, formulas, worked traces, tests, commands, runtime, raw episode metrics, comparisons and limitations. Do not edit commander plans. No API experiment, training or subsequent task is authorized by this brief.
