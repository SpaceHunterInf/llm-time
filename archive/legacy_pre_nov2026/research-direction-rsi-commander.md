# Direction decision: strategy improvement under hidden environmental change

> Status, 25 September 2026: preserved as a historical direction memo. The active direction is [persistent self-improvement over long trajectories](research-long-trajectory-self-improvement-commander.md); hidden drift is a later stress test. Follow task 012 rather than task 011.

Author: Codex (commander), 15 September 2026. Responds to the user's Gittins/alpha/RSI proposal. This is a research design and focused primary-source check, not a claim of novelty or an experimental result. The numerical posterior-distillation branch remains closed.

## Recommendation

Investigate **how an agent should allocate limited effort between exploiting its current strategy, checking whether its evidence has gone stale, and generating or validating a better strategy under hidden environmental change**.

Interpret “alpha” as an operational edge over a declared baseline, not necessarily financial-market alpha. Start with a cheap synthetic resource-allocation environment, where executable policies can earn more completed-job value for the same resources. Use an LLM occasionally to propose or repair policy code; execute that code on CPU between revisions. No weight training is required for this formulation.

The candidate contribution is **selective revalidation of strategy evidence when the source and scope of drift are unknown**, with a downstream gain after all improvement costs. Code generation, an archive, a changing simulator, or selective forgetting alone would not supply novelty. The closest new reference is RQGM, discussed below.

## What Gittins gives us

The classical discounted bandit theorem handles independent projects whose states freeze while inactive; choosing the largest per-project Gittins index is optimal under that model. It accounts for the future value of learning about a project, rather than selecting solely by current estimated reward. [Richard Weber's notes, §§6.5–7.3](https://www.statslab.cam.ac.uk/~rrw1/oc/oc2013.pdf)

Our original reset process evolves with calendar time even for unobserved components, and shared changes couple them. In the proposed setting, policies also compete for shared resources and new code can create new candidates or change future search. These break the classical decomposition. Restless-bandit/Whittle methods are closer for suitable special cases, but indexability is an additional condition and a Whittle policy has no general optimality guarantee. [Primary indexability study](https://arxiv.org/abs/2008.06111), [Weber and Weiss](https://www.statslab.cam.ac.uk/~rrw1/publications/Weber%20-%20Weiss%201990%20On%20an%20index%20policy%20for%20restless%20bandits.pdf)

Use value of information and opportunity cost to guide the design. A Gittins baseline belongs only in an explicitly reduced independent/rested/discounted setting; do not advertise a theorem for the full proposed problem. The theorem does not specify how to invent a strategy program or improve a search algorithm.

## Three levels of improvement

| Agent changes | What a successful result establishes |
|---|---|
| Estimates or selects among a fixed strategy library | Adaptation / online algorithm selection |
| Writes and revises executable strategy programs | Programmatic strategy improvement |
| Revises its own proposal, diagnostic or validation procedure, and the revision helps on later unseen problems | Improvement of the improvement procedure; a bounded system-level RSI experiment |

Initially target the second level. A later recursive claim requires an additional controlled experiment: transfer only the evolved improver to fresh environments, reset its task-specific policy library/evidence, give it the same backbone and budget as the original improver, and measure subsequent improvement efficiency. Compare against transferring only a better task policy. Rising accumulated reward within one changing trajectory cannot by itself establish recursive improvement. Frozen LLM weights must be reported explicitly.

## Primary-source overlap check

These sources constrain the claim; this is not an exhaustive literature review.

| Work | Relevant overlap and consequence |
|---|---|
| [STOP](https://arxiv.org/abs/2310.02304) | Uses an LM-based program improver to improve itself. A recursive scaffold with fixed LM weights is already a precedent. |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Evolves agent code and retains an archive of agent variants. Self-editing plus an archive is insufficient as our contribution. |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | Program proposals, automated evaluation and evolutionary search are established. We can adopt this execution pattern without claiming to invent it. |
| [Bilevel Autoresearch](https://arxiv.org/abs/2603.23420) | An outer loop modifies the inner search machinery. Merely adding a meta-optimization loop would overlap. |
| [Red Queen Gödel Machine, RQGM](https://arxiv.org/pdf/2606.26294) | Explicitly covers non-stationary utilities and selective erasure of evaluation records. Its inspected formulation uses controlled criterion transitions and fixed criteria within epochs. Our candidate difference is inferring hidden drift and deciding what to revalidate without receiving the transition or affected slots. This is a proposed distinction, not an established unoccupied niche. |
| [SHARP](https://arxiv.org/html/2605.06822v1) | Attributes trading errors, makes local rule edits and validates them in a walk-forward protocol. Its inspected test phase freezes the selected rubric. “Revise trading rules from P&L” alone is already covered; our proposed evaluation is continuous online improvement with intervention costs. |
| [DACBench](https://www.automl.org/automated-algorithm-design/dac/dacbench-benchmarking-dynamic-algorithm-configuration/) | Dynamic algorithm configuration is an established problem. Compare against algorithm selection/configuration, rather than crediting every policy switch to LLM reasoning. |

RQGM and SHARP are closer to this user proposal than the earlier numeric-learning literature alone. The broad title “RSI in a non-stationary environment” is already occupied. We must avoid saying these works lack a feature merely because it was absent from an abstract; the RQGM epoch/erasure formulation and SHARP inference/walk-forward sections were inspected specifically.

## Concrete initial environment

Use a synthetic service that repeatedly allocates short jobs to three worker pools. A strategy program receives permitted job attributes, observed operating history and its own state, and returns a feasible allocation under a resource limit. Reward is completed-job value minus declared operating penalties. Different task mixes reward different scheduling heuristics, so an edge can arise from a better program.

Introduce externally generated, unannounced changes in processing behavior. Two pools may share a runtime-estimation assumption or processing backend; a third may be unaffected. Some apparent failures are transient noise. Previously profitable heuristics can become poor; a later recurrence can make an archived strategy useful again. Initially keep these changes exogenous and independent of which candidate is selected. Defer adversaries that adapt to the agent, market crowding, price formation and real financial data.

Example: a size-dependent slowdown makes the previous value-per-estimated-runtime policy overcommit two pools. An LLM may repair the runtime estimator or scheduling rule. Evidence about one affected pool may justify retesting policies using the shared estimator, while a policy for the unaffected pool retains its evidence. Conversely, a one-off failure should not trigger a complete rewrite. These are intended cases to validate, not guaranteed performance advantages.

The environment's scoring, hidden state and future random draws are outside the mutable program. Candidate programs may change the policy and a bounded internal estimator; they do not change what counts as success. Pure CPU execution with an immutable evaluator makes both failure and gains measurable. Exact language-free solvers and fixed-library selectors remain legitimate competitors.

## Candidate method: infer which evidence needs revalidation

Represent each archived policy by its executable code and empirical evidence records: the public context, observed execution path, prediction/outcome, evaluation time and cost. Dependencies derived from visible code/trace are permitted; the simulator's changed component or change time is not.

At a decision point:

1. Compare new outcomes with the incumbent's predictions and retain competing explanations: noise, a local estimator failure, a shared failure, or broader workload change.
2. Estimate which past policy evaluations could be stale under each explanation. Preserve the alternatives instead of treating a single alarm as a known reset boundary.
3. Choose between continued exploitation and a paid diagnostic/revalidation that could change the deployment decision. Prioritize expected decision benefit after test cost; this is a proposed approximation, not a Gittins-optimal index.
4. Feed the resulting localized evidence to a fixed LLM proposer to repair or generate a policy. Evaluate candidate versus incumbent on matched fresh allowed tests, with a reserved confirmation budget, before deployment.
5. Keep versioned policy/evidence records so recurrence permits revalidation of old policies rather than compulsory rediscovery.

The first implementation should use a small explicit hypothesis set and an observation-limited reference for checking the controller. This reference is an engineering baseline, not a neural-method contribution. The research hypothesis is that *inferred evidence validity plus targeted retesting* saves enough budget under partial drift to improve net online performance against strong global and independent revalidation controls. We still need a concrete estimator/controller to demonstrate that claim; this memo does not declare the algorithm solved.

## Measurement and falsifiers

The primary score is cumulative deployment utility after operating, probing and revision costs. Count failed proposals and validation too. If adaptation downtime already reduces deployment reward, do not subtract it twice. Also report raw task utility, test count, proposal calls/tokens, CPU time and deployment interruptions separately. Any conversion from API/CPU costs to task utility must be declared beforehand and accompanied by a cost frontier.

Measure recovery after hidden shifts, regret in stationary intervals, impact on unaffected pools, and reuse under recurrence. Evaluate on fixed, withheld drift schedules and new workload compositions. A baseline-relative “alpha” is conditional on the declared benchmark and information budget; it is not a claim of financial alpha or unbounded compounding.

Necessary controls: a frozen strategy; a strong adaptive selector over a fixed policy library; global recent-window revalidation; independent per-policy change detection; the same proposal stream with no stale-evidence handling; and a known-change/known-scope reference. Give compared controllers the same candidates and test opportunities first, then compare end-to-end proposal loops. Do not give one agent counterfactual rewards for unexecuted actions while others receive only chosen-action outcomes.

Abandon or narrow the method claim if ordinary adaptive selection/recent-window validation matches it, its test cost removes its gain, it damages stationary/unaffected performance, or it works only when supplied the hidden change labels. If success is explained entirely by selecting an existing heuristic, call it algorithm selection. Do not turn that result into an RSI claim.

## Immediate next step

[Task 011](research_tasks/011_strategy_revision_feasibility-commander.md) is a single CPU-only feasibility and falsification check using a few hand-written strategy programs and explicit information/cost accounting. No LLM calls, no weight training, no full agent framework. Its role is to establish that hidden partial drift creates a decision-relevant revalidation problem with headroom over competent cheap controls. It cannot establish LLM or RSI novelty.

Only after that gate should we price one small fixed-backbone code-proposal pilot. Do not revive the old posterior-distillation experiments or automatically proceed to recursive modification of the improver. The direction is concrete and testable, but its publication case remains uncertain.
