# Learning What to Revise: a concrete research plan

Prepared 9 September 2026. This is a proposed research programme, not an experimental result. No model experiments were run for this plan. The original repository was left unchanged.

## Recommendation

Make **selective revision under hidden environmental change** the main project. Train a model to use contradictory evidence to update the appropriate predictions, retain still-valid knowledge, and transfer the correction to tasks that share the changed mechanism.

The central question is:

> Can targeted training teach an LLM a transferable procedure for deciding what to revise, improving decisions under change without unnecessarily disrupting stable knowledge?

This develops your current observation into a method paper. Your heuristics appear to trade faster adaptation for poorer stationary performance. Some trade-off is unavoidable: a noisy stable world and a recently changed world can produce identical observations. The objective should be to improve the achievable trade-off, especially by limiting the *scope* of revision, rather than promise free adaptation everywhere.

My recommended allocation is one main method, one small diagnostic environment, and one language-rich transfer environment. Additional models are replications after a useful effect appears. CEO-Bench is a possible later evaluation, not the first development priority.

## What the literature changes

The two papers you provided are useful starting points, but the contribution needs to go beyond their diagnostic framing.

| Paper | What to take from it | Consequence for this project |
|---|---|---|
| [Reflection-Bench](https://arxiv.org/html/2410.16270v3) | Reversal and other cognitive tasks expose adaptation failures. | Use controlled tasks, but distinguish predictive accuracy from reward maximization. Its choice-frequency-based reversal score is not a general decision-quality metric. |
| [BayesBench](https://arxiv.org/html/2606.30850v1) | Separates aspects of inference and prediction against tractable Bayesian references. | Measure what the model predicts separately from how it acts. Extend to changing latent variables and action-dependent evidence. |
| [MemoPilot](https://arxiv.org/html/2606.08656) | Trains a memory updater for a frozen player and evaluates opponent switches. | “Train an agent to update memory” is already close prior work. Compare targeted revision training against generic reward-trained memory. |
| [WorldEvolver](https://arxiv.org/html/2606.30639v1) | Converts prediction mismatches into reusable textual rules. | A prediction–reflection–memory scaffold alone is insufficient. Test attribution under noise and selective transfer. |
| [DynaMITE-RL](https://arxiv.org/html/2402.15957) | Learns latent inference with changes inside episodes. | Situate the work against non-stationary meta-RL, not only LLM benchmarks. |
| [Continual In-Context RL in Non-Stationary Environments](https://openreview.net/pdf?id=h8OJb8YGNa) | Directly studies ongoing interaction in changing worlds. | Non-stationary training itself cannot carry the novelty claim. Its indexed primary text was available; full implementation and publication status still need verification. |

The candidate contribution is therefore **evidence that a particular training intervention improves selective, structurally transferable revision**, beyond direct policy training, generic memory learning, and statistical forgetting. This is a defensible hypothesis to test, not an established novelty claim.

## Main idea: train the scope of revision

### A minimal task that distinguishes the explanations

Start with three latent components, A, B, and C. Each governs an observable outcome distribution. Build paired environments:

- **Shared change:** A and B can be reset by a common hidden event; C is separate.
- **Independent change:** A and B have independent reset events with the same marginal rates.
- **Stable control:** no reset occurs during the evaluation interval, but observations remain noisy.

Give the agent substantial initial evidence. Then present the same surprising sequence on A in the shared and independent environments, with no fresh evidence on B. Ask for predictions and decisions involving all three components.

A good learner should revise A in both cases. Evidence on A should affect uncertainty about B more in the shared case. It should preserve C. Mere global recency cannot make all of these distinctions. An independent per-component filter cannot propagate evidence across a shared reset unless it models that dependency.

First supply the dependency structure explicitly. This tests whether the model *uses* a known relationship. Then test a separate condition in which the relationship must be learned from a calibration history. Do not describe success with a supplied graph as discovering causal structure.

Include an explicit costly diagnostic action and a subsequent decision about B. Otherwise the study risks measuring numerical forecasts with little consequence for behaviour. The diagnostic should sometimes be worth buying and sometimes not.

### A checked example of why this task is discriminating

A small exact calculation is saved in [selective_revision_sanity.py](./selective_revision_sanity.py), with [outputs](./selective_revision_sanity.json). It is a mathematical check, not model evidence.

Assume each component's reward rate is either 0.2 or 0.8, initially with probability 0.99 on the high rate. Before a batch of observations, there is one possible reset with probability 0.1; a reset redraws the rate uniformly. A/B resets are shared or independent, and C is unaffected by this candidate event. After five failures on A:

| Observation-limited reference | Predicted reward A | Predicted reward B | Predicted reward C | Buy diagnostic on B? |
|---|---:|---:|---:|---|
| Shared A/B reset | 0.209 | 0.548 | 0.794 | Yes |
| Independent A/B resets | 0.209 | 0.765 | 0.794 | No |

Here a perfect diagnostic costs 0.04 before one choice between B and a known alternative worth 0.65. The appropriate action differs despite identical local evidence and the same posterior prediction for A. A reset can redraw the same value; detecting a reset is not equivalent to identifying a changed reward rate.

### The training intervention

Use one small trainable open model initially. Keep deployed weights frozen throughout each test episode. Do supervised training first; add interactive RL only if it addresses an observed limitation.

Generate histories from a small family of factored environments. Balance useful examples of local change, shared change, misleading noise, and stable evidence. Vary which components are related and which actions reveal them. Include ambiguous histories where uncertainty is the correct answer.

Train the model to produce:

1. A compact set of predictions about components or action outcomes, including uncertainty where identifiable.
2. An updated state after new evidence, preserving unsupported-to-change entries.
3. The next task or diagnostic action.

Actually feed the stored state into the next interaction. Use the same state format and context allowance in the relevant controls, so a schema advantage does not masquerade as a learning advantage.

The proposed objective combines action supervision with predictive supervision. Where the model family is tractable, distill the filtering posterior conditioned on permitted observations. Also assess forecasts with proper scores on realized outcomes. Teacher choices can use exact short-horizon planning in the small task; label approximations honestly for longer horizons.

Crucially, do not train the model to announce the simulator's true change time or component when observations do not identify it. Hidden simulator state can generate data, but it is not evidence available to the agent. Identical observable histories must receive identical posterior targets under the same stated prior.

The hypothesis is that targeted predictive training teaches a more reusable update procedure than action imitation alone. Prediction loss, explicit memory, and fine-tuning are established ingredients; the structural generalization result must justify this particular use of them.

## Four experiments, in order

### 1. Locate the correctable failure

Replay matched histories to the untreated model. Compare raw observations, verified summaries, and externally supplied reference predictions. Keep decision questions and evidence fixed. Separately compare retaining the model's previous state with reconstructing predictions from observations alone.

Measure whether the problem is evidence interpretation, inappropriate propagation to other components, failure to use a correct prediction, or poor choice of a diagnostic action. Supply diagnostic results for free in one condition to separate choosing an experiment from interpreting its outcome.

**Continue if:** there is a meaningful decision gap and a targeted information/state intervention closes part of it. A task that only reveals formatting failures or is already solved by the model needs redesign before training.

### 2. Test whether targeted training adds value

Start with an interface-matched untreated model, action-only SFT, and prediction-plus-action SFT trained on the same histories. Add a generic learned-memory comparator and a learned scalar forgetting rule for the main claim. Include tuned recent-context, discounted-statistics, and change-triggered-reset references; use a factor-aware statistical observer in the tractable setting.

If proceeding to RL, compare reward-only action training and a MemoPilot-style reward-trained updater. Do not credit predictive supervision for gains that a larger interaction budget explains.

**Continue if:** the candidate improves return or its stationary/adaptation trade-off over the strongest matched learned baseline. Better forecast scores without better decisions are a diagnostic finding, not the proposed performance result.

### 3. Test structural transfer and the role of stored state

Hold out dependency groupings and task compositions, rather than only seeds or names. Counterbalance labels and change directions. Test the first decision on an unobserved linked task after evidence on A; pair it with a task whose similar-looking component is independent. Include longer stable histories and different change rates as stress tests.

For the condition that learns dependencies from history, verify that an observation-limited reference can distinguish the candidate structures from the available calibration data. Otherwise apparent failure might be non-identifiability.

Replace or mask a single stored prediction with a corrected or stale version. Match length and include irrelevant replacements. Test whether these interventions selectively change the corresponding downstream actions.

**Continue if:** the correction transfers appropriately to unseen linked compositions while unrelated decisions remain reliable. State interventions support claims about the external memory interface; they do not reveal an internal neural belief mechanism.

### 4. Demonstrate the same skill in a language-rich workflow

Build a small, explicitly modified [ToolSandbox](https://github.com/apple/ToolSandbox) task using local reminders, message queries, contacts, and time conversion. Use scripted requests and deterministic database assertions.

Proposed scenario: a backend silently changes a time/unit convention while retaining its interface. A reminder failure supplies evidence about that conversion. The next task is a new date-filtered message query using the same backend. The model should transfer the correction, while preserving contact identities and untimed operations. In a paired control, message queries use an independent backend, so transferring the correction is wrong.

Provide a costly read-only calibration operation that makes the alternatives distinguishable. Include ordinary transient errors; a failure must not automatically certify a migration. Score successful state changes, correct query results, unintended edits, and all diagnostic/retry costs. Keep hidden migration state and evaluator internals inaccessible.

This is a proposed extension, not an existing ToolSandbox benchmark. It tests composed tool behaviour and shared semantics rather than dressing up arm selection with API names. A [recent set-shifting tool benchmark](https://arxiv.org/html/2607.13396v1) already covers hidden changes among redundant tool choices.

**Main-paper success condition:** improvement on first-attempt linked-task transfer and total utility, with preserved unrelated competence. If gains stop at the numeric environment, limit the claim accordingly.

## Measurement and controls to commit to

The primary outcome is cumulative utility, including diagnosis and failed-action costs. In exogenous bandits, also report regret against a clearly specified comparator. In action-dependent workflows, prefer actual return and a justified reference policy; an instantaneous reward oracle need not be an optimal long-horizon policy.

Report stationary utility versus post-change utility across validation-tuned settings. This exposes the trade-off your existing heuristic has, without hiding it in an arbitrary mixture average. Also measure affected-task recovery, unaffected-task retention, linked-task first-attempt transfer, forecast quality, and the fraction of changes never recovered from. Include lower-tail failures, not just mean return; [When Greedy Wins](https://arxiv.org/html/2509.24923) makes this particularly relevant for trained bandit LLMs.

Use independent episodes and training seeds as statistical units; cluster multiple change events within episodes. Pair exogenous observation streams where possible, indexed by event/request rather than relying on one sequential random-number generator. Match observable evidence, training examples, environment interactions, and inference allowances; report tokens and wall-clock cost separately.

Choose final sample sizes from pilot variance and a practically meaningful effect threshold. Freeze that threshold and analysis before final test evaluation. A small pilot might use 20–30 independent episodes per key condition; that is a planning choice, not a claim of adequate statistical power. Replicate training with several seeds and then a second model family after the method earns that investment.

Avoid treating emitted confidence or a written statement of doubt as belief revision. Outcome unpredictability is also not automatically epistemic uncertainty. In the first task keep the emission model known; separate unknown noise and unknown change rate in a later extension.

## Decision gates and fallback

| Finding | Decision |
|---|---|
| Action-only training matches predictive training | Use the simpler method; do not claim the auxiliary objective is necessary. |
| Generic memory learning matches the candidate | Drop special-purpose machinery or identify a narrower, reproducible benefit. |
| Component-wise filtering explains all gains with supplied structure | Require learned-structure or language-composition transfer for the central claim. |
| Forecasts improve but utility does not | Investigate the decision/diagnosis bottleneck before expanding the benchmark. |
| Stable performance collapses | Revisit training mixtures and uncertainty handling; do not average the collapse away. |
| No transfer beyond trained dependency patterns | Treat it as task-specific learning, not a general revision procedure. |

The smaller fallback is **learning to distinguish observation noise from environmental change**. Independently vary latent transition rate and observation reliability in a binary hidden-state model. Compare action-only training against prediction-plus-action training on held-out combinations, with exact filtering references and a stationary/adaptation frontier. Cognitive work on [noise and volatility](https://www.nature.com/articles/s41467-024-53459-z) supplies useful hypotheses; [recent binary-model work](https://pubmed.ncbi.nlm.nih.gov/42530995/) confirms that these sources of uncertainty can be separated without requiring continuous outcomes.

This fallback is cheaper and directly addresses your heuristic's cost, but its contribution is more crowded. I would choose it only if the main structural task proves too large or the pilot localizes the failure specifically to noise/change discrimination.

## Eight-week execution plan, when experiments begin

| Week | Work | Concrete deliverable / gate |
|---|---|---|
| 1 | Audit retained code; implement the three-component task and exact references; freeze core splits. | A trace demonstrating identifiable shared/independent cases and correctly charged diagnostics. |
| 2 | Run diagnostic interventions and tune statistical/prompt baselines on validation. | A go/no-go report locating a correctable decision failure. |
| 3–4 | Generate common training histories; train action-only and prediction-plus-action models; add generic-memory control. | A stationary/adaptation comparison with matched information and budgets. |
| 5 | Run structural holdouts and state interventions; replicate promising conditions. | Evidence for selective transfer or a decision to simplify/pivot. |
| 6 | Implement and validate the small ToolSandbox migration extension. | Deterministic paired workflow tests with no privileged feedback. |
| 7 | Evaluate workflow transfer and a second model family if justified. | Main result tables, cost accounting, and failure analysis. |
| 8 | Complete essential ablations and write. | A paper organized around the supported learning claim. |

This assumes access to a small open model and fine-tuning resources; throughput and cost remain to be measured. Avoid committing to a large training run before the diagnostic gate. RL and CEO-Bench are optional extensions, not prerequisites for this schedule.

As of 9 September, [ICLR 2027](https://www.iclr.cc/Conferences/2027/CallForPapers) lists an 18 September abstract deadline and 25 September paper deadline. That is too close for this new programme. The [ARR schedule](https://aclrollingreview.org/dates) lists January 2027 as ACL 2027's final submission cycle, with the exact date not yet provided in the inspected table. That is a more plausible planning horizon. An ACL paper should make language-mediated dependency inference and composed tool transfer central; an ICLR-oriented version should emphasize the learning objective and cross-task generalization.

## What to retain from the current repository

Reuse the rollout/logging infrastructure, exogenous bandits, randomized change schedules, and stationary controls after checking their current implementations. Treat existing reported results as hypotheses until the underlying runs are verified.

Resolve the raw-history versus summary-interface confound before attributing effects to heuristic wording. Explicitly persist prior predictive state if studying memory revision. Make recovery metrics include prior commitment and censored non-recoveries; returning to an old hypothesis can be correct when regimes recur.

Defer the broad environment zoo and extra model sweep. The new components worth building are the factored task, observation-limited reference observer, matched training data, and the small workflow transfer test.

## Reading order and supporting notes

Read MemoPilot and WorldEvolver first to constrain the method claim; then BayesBench for measurement, DynaMITE-RL and continual ICRL for the training setting. [Heilbron and Meyniel](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006972) are particularly useful for shared versus independent change structure. Return to Reflection-Bench for framing once the main experiment is specified.

The supporting audits contain more detailed source notes and limitations:

- [Cognitive mechanisms and measurement](./research-cognitive-mechanisms.md)
- [Training methods and closest learning baselines](./research-training-methods.md)
- [Novelty threats and transfer-environment design](./research-novelty-transfer.md)

The research bet is specific: **training on appropriately scoped predictive updates will improve decisions on new combinations of changing and stable mechanisms.** The experiments above are designed to tell us whether that bet is right, and whether a simpler training method already suffices.
