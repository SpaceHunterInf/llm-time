# Direction: transferable self-improvement over long trajectories

Author: Codex (commander), 25 September 2026. Read with `auto_research_rules.md`.

## Decision

Make persistent agent self-improvement the central object of study. Environmental drift becomes a later stress test. This supersedes the immediate fixed-library controller experiment in task 011, whose worker return/results are absent from this checkout. Tasks 001–010 remain historical evidence; their results do not establish self-improvement. Keep the numeric-training queue closed.

Working question: **When does an agent's acquired improvement procedure help it learn better solutions on fresh tasks, after accounting for the resources spent acquiring and using that procedure?**

This is a research hypothesis, not an established novel method. A plausible outcome is that improving the solver pays off but rewriting the improver does not. That would tell us to pursue persistent self-improvement without claiming a recursive advantage. A small negative toy experiment alone would not constitute a paper.

## What improves

Freeze the language-model weights. Persist three separately versioned artifacts across task blocks:

- **Solver:** executable code that solves instances of a task family.
- **Improver:** executable logic that selects past evidence and constructs the next code-edit request. It can propose changes to the solver and to its own improvement procedure.
- **Experience:** bounded records of attempts, failures, evaluations and costs. This is distinct from either program.

The immutable host owns model access, budget enforcement, evaluation, artifact validation and deployment. An improver cannot change the score, read future evaluation instances, grant itself calls or waive acceptance checks. Its outputs are requests to the host. Accepted improver changes actually construct subsequent requests, including later self-edit requests; saving an unused prompt is not recursion.

The initial method is a two-level propose–test–retain loop. Solver edits compete with the incumbent on fresh development instances. Improver edits are tested by letting old and candidate improvers each produce a solver edit from the same starting solver and permitted history, then evaluating their descendants. Those extra model calls are part of the improvement cost. This is an implementation baseline, not a claim to have invented downstream evaluation of improvers.

Our candidate methodological question is whether *selective, evidence-driven investment in improvement* can outperform a fixed schedule at limited budgets: which failure deserves a repair, which evidence still applies, and when is revising the improver worth giving up another solver proposal? Do not invent a new algorithm name before a reproducible failure and useful intervention exist.

## First environment

Use small, fully specified CPU resource-allocation problems: jobs have public values, resource requirements and deadlines; the solver constructs feasible assignments to public worker capacities. Start with deterministic execution and a stationary distribution of varied instances. Use a documented objective and an independent exact solver on tiny cases. Larger cases must have useful headroom above several competent handwritten heuristics; this must be measured, not assumed.

The model writes reusable algorithms, rather than selecting an arm from a fixed policy library. The solver handles variable instance sizes and combinations of constraints. Full public problem rules are available to every method. Hold out instance seeds and later structural combinations; do not create a convenient distribution where only our intervention is favored. If competent simple heuristics nearly solve it, return that finding rather than tuning the generator until a preferred method wins.

A trajectory is a sequence of task blocks sharing evolving artifacts, not one enormous context window. Begin with a short integration smoke. A later pilot should have at least 16 actual edit opportunities per lineage; hundreds of CPU instances with only one edit are not a long self-improvement trajectory. Longer runs and a second task family follow only if the pilot yields a reason to continue.

Stationarity first separates improvement from tracking a moving target. Later introduce local/shared changes and recurrence to test whether acquired procedures remain useful and whether selective revision protects unaffected competence. This is where the previous research can contribute; do not put hidden-change inference back on the critical path now.

## Comparisons and decisive tests

The first substantive comparison is fixed-improver solver evolution versus evolution of both solver and improver, with equal total model-token/dollar ceilings and CPU evaluation limits. Count self-edit proposals, failed edits, descendant rollouts, selection calls, retries and discarded candidates. Show performance against both elapsed tasks and accumulated resources. Do not make the recursive arm look cheaper by excluding its meta-level work.

Before a broad claim, add a bounded textual-memory control, ordinary search given the same total resources, and fixed external search over the same editable improver space. The last comparison distinguishes useful improver optimization from an advantage specifically due to the evolving improver directing its descendants. Give controls the same allowed evidence, context limits and model. Specify how each control uses its budget before evaluating it; memory-only should have an active way to affect decisions, not be a powerless strawman.

At selected checkpoints, independently transplant solver and improver into fresh runs:

| Starting solver | Starting improver | What this isolates |
|---|---|---|
| Initial | Initial | Reference learning curve |
| Evolved | Initial | Reusable solver improvement |
| Initial | Evolved | Transfer of the improvement procedure |
| Evolved | Evolved | Interaction of the two artifacts |

Reset task-specific history and solution archives in all four cells; use the same fresh task distribution and remaining budget. Also run a matched-history transfer check to detect an improver that requires a history format, rather than silently declaring it useless after removing its required inputs. No future test feedback returns to artifact selection.

An evolved improver earns the stronger claim only if it improves *subsequent improvement* from a common starting solver on fresh tasks. Higher end-of-run reward or larger memory alone does not establish that. This factorial test is our proposed diagnostic; its novelty is not established by this literature check.

Report cumulative deployed utility, fresh-task performance, learning curves, resource use, invalid edits, regressions and break-even horizon against ordinary search. Keep dollars and task utility separate unless a conversion is externally justified. Repeated rounds within one lineage are correlated: independent lineages, not individual tasks, are the main replication unit. Three seeds can screen feasibility, not support a strong statistical claim.

## Literature checkpoint, 25 September 2026

This is a targeted primary-source review, not an exhaustive novelty certification.

| Source | Relevant overlap | Consequence |
|---|---|---|
| [STOP](https://arxiv.org/abs/2310.02304) | A fixed language model supports an improving program that can improve itself. | A self-editing improver is already established. |
| [PILOT in the Loop](https://arxiv.org/html/2608.26530v1), 27 August | Live steering plus persistent skill/memory evolution for long-horizon agents. | Long trajectories and persistent experience are not a novelty claim. |
| [ModularRSI](https://arxiv.org/html/2609.14857v1), 14 September | Contrastive trajectories guide modular harness edits with validation gates. | Localized edits and modular attribution already have direct precedents. |
| [Dream-RSI](https://arxiv.org/html/2609.14858v1), 14 September | Improves programmable exploration using replay of accumulated discovery trees. | Learning how to search, and historical replay to lower its cost, are already occupied. Replay only covers recorded branches; it is not a free oracle for arbitrary new code. |
| [RRSI](https://arxiv.org/html/2609.24972v1), 21 September | Regularizes harness proposals/selection to address overfitting, complexity and transfer. | Conservative acceptance, pruning, cost awareness and OOD evaluation alone do not distinguish us. |

The [previous direction memo](research-direction-rsi-commander.md) also records DGM, AlphaEvolve, RQGM and SHARP. Its nonstationarity and selective-erasure cautions still apply. We have not reproduced these systems and must label simplified controls as adaptations rather than reproductions.

The possible contribution is a demonstrated mechanism or careful finding about **transferable, cost-effective improvement of the improver over sustained use**. That remains a hypothesis in a crowded area. Before scaling beyond a small pilot, audit the closest papers' exact transfer/budget experiments and show what question our experiment resolves that theirs does not. If no distinction survives, reproduce/adapt a strong baseline rather than claim a new RSI framework.

## Immediate work and stop conditions

[Task 012](research_tasks/012_persistent_self_improvement-commander.md) builds the smallest actual persistent code-edit loop, verifies it offline, and permits one bounded Gemini integration smoke under the existing ordinary-small-experiment authorization. No training, GPU experiment, new framework or broad benchmark campaign.

After the worker return, inspect actual code diffs, descendant requests and raw evaluations. If persistent useful solver edits are feasible, freeze a matched-budget pilot protocol. If self-edits cannot change improvement behavior, the implementation is incomplete. If they change behavior but fail to beat fixed improvement after costs, retain the simpler system and investigate that failure before another paid batch. Do not repeatedly increase budgets or change environments to obtain a positive curve.
