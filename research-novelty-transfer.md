# Novelty and transfer audit

Research date: 9 September 2026. This is a focused primary-source audit, not an exhaustive systematic review. Recommendations below are research proposals, not results. No model experiments, installations, or paid runs were performed.

## Recommendation

Pursue **training selective revision of predictive knowledge under noisy, partial environmental change**, with frozen deployed weights. The strongest claim would be that evidence about one changed mechanism leads to appropriate changes on unseen tasks sharing that mechanism, while preserving decisions that depend on unaffected mechanisms. Demonstrate this beyond raw outcome tracking, and establish the contribution through controlled interventions.

The broad ingredients—non-stationary in-context RL, learned memory, prediction-error-driven rule updates, confidence gating, and changing-tool benchmarks—all have close precedents. Their combination alone is not a defensible novelty claim. The discriminating result must be improved decision performance on held-out dependencies and dynamics, with causal evidence of selective use of learned state.

## Closest competing work

### Continual ICRL: direct threat to broad setting and training claims

[Towards Unpredictable Worlds: Continual In-Context Reinforcement Learning in Non-Stationary Environments](https://openreview.net/pdf?id=h8OJb8YGNa) studies uninterrupted interaction with changing environment dynamics and unrestarted context. Its indexed primary PDF describes symbolic and physics domains, architecture comparisons, and stationary versus non-stationary training, change frequency, context length, and interaction scale. Direct PDF/forum retrieval encountered OpenReview's browser challenge. The abstract and formulation were available through primary-source search indexing; full implementation and current acceptance status remain unverified. Do not call it an accepted ICLR paper.

**Implication:** Neither “within-episode non-stationary ICRL” nor “train on changing rather than static environments” should be the main novelty. Full-paper comparison is a required check before finalizing the method.

### WorldEvolver: closest method-level precedent

[Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/html/2606.30639v1) (June 2026 preprint) uses frozen world-model and agent weights. It retrieves realized transitions and turns prediction–observation mismatches into textual rules using factorized object/relation/action tuples. Rule evidence scores are updated; positive-score rules enter context. Foresight is filtered by token-probability confidence. Evaluation covers ALFWorld and ScienceWorld; memory persists across trials and tasks. The inspected experiments do not document injected, stochastic within-episode rule changes. Improvements over the no-world-model baseline vary across backbone/task combinations; best-of-five success is a reported measure.

**Implication:** A prediction–comparison–memory-update scaffold needs this as a baseline. The proposed advance is learning how to infer *which mechanism changed* under noise, then demonstrating selective generalization, rather than merely appending corrective rules.

### Set-shifting tool benchmark: direct threat to a simple tool extension

[Set-shifting Behavioral Test for Harnessed Agents](https://arxiv.org/html/2607.13396v1) (July 2026 preprint) hides reliability changes in redundant tool groups within persistent sessions. Branched schedules include returns to old groups, new groups, and no-shift controls. It tests two open-weight models, policy prompts, and tool-set framing. Tool calls receive binary success/failure; named domains include scheduling, DevOps, and multi-cloud storage. [Code](https://github.com/zwycl/wcst-tool-bench) is public.

**Implication:** Relabeling bandit arms as APIs and teaching “switch on failure” would overlap strongly. The proposed task must require modifying a shared transformation or dependency, rather than only choosing a currently reliable substitute.

### ContDa: tool adaptation already has a stability/adaptation method

[Beyond Static Toolsets: Self-Evolving LLM Tool Agents via Continual Documentation Adaptation](https://aclanthology.org/2026.findings-acl.1082/) is in Findings of ACL 2026. ContDa probes new tools using related tools and adjusts documentation to express overlap, preferences, and fallbacks. The full [paper](https://aclanthology.org/2026.findings-acl.1082.pdf), §§3–5 and Appendix A.9, evaluates expanding, shrinking, and mixed toolsets. Agents receive updated tool lists. Documentation maintenance can run asynchronously as tools appear/disappear; the procedure explicitly identifies newly introduced tools. [Code](https://github.com/Bingo-W/ContDa).

**Implication:** This is related but operationally different from detecting an unannounced change behind an invariant interface during interaction. Include a documentation-update baseline if the held-out task uses tool semantics. Charge probing costs rather than hiding them in offline maintenance.

### Learned memory: relevant baseline families, not a new contribution by themselves

| Work | Verified scope | Implication for the proposed project |
|---|---|---|
| [MEM1](https://arxiv.org/html/2506.15841v1), 2025 | RL trains compact state that combines memory consolidation and reasoning, with QA and web-shopping evaluations. | “Train a compact recurrent textual state with RL” is established. Compare to generic memory training with the same budget. |
| [Mem-α](https://arxiv.org/html/2509.25911v1), 2025 | RL trains memory construction; §3.4 explicitly excludes MemoryAgentBench conflict resolution, focusing on retrieval, test-time learning, and long-range understanding. Rewards also address call format, compression, and content. | Revising obsolete knowledge under ambiguous observations is not established by its evaluated dimensions. Use it as a learned-memory comparator, not evidence that revision is solved. |
| [Oblivion](https://arxiv.org/html/2604.00131v1), 2026 | Hierarchical external memory with uncertainty-gated retrieval and decay/reinforcement; evaluated on LongMemEval and GoodAILTM. | Accessibility-based forgetting and preserving high-level knowledge already have a scaffold. Its conversational evaluation differs from sequential reward-driven control. |
| [Neural Garbage Collection](https://arxiv.org/html/2604.18002v1), 2026 | Learns token generation and KV-cache eviction jointly from task reward, evaluated on Countdown and mathematical reasoning. | “Learning to forget” is too broad a title claim. Distinguish semantic validity under change from memory capacity optimization. |
| [Agentic Context Engineering](https://arxiv.org/abs/2510.04618), 2025/2026 | Evolving playbooks use generation, reflection, and incremental curation. [Author announcement](https://ace-agent.github.io/blogs/2026-01-30-iclr-acceptance/) confirms ICLR 2026 acceptance. | Strong training-free baseline for accumulating and revising reusable strategies. Preserve its incremental-update advantage in a fair comparison. |

### Survey verification and a reference that requires careful interpretation

[In-Context Reinforcement Learning under Non-Stationarity: A Survey](https://arxiv.org/html/2607.11906v1) is verified on arXiv, submitted 1 July 2026 by A Run and Ziluo Ding. It explicitly covers stale/recurring context, validity-aware retrieval, forgetting, and lifecycle evaluation. It traces the CICRL paper above and [PSBL](https://proceedings.mlr.press/v235/xu24o.html). It is useful as a literature map, but broad problem framing taken from it is not a novel contribution.

[PSBL: Meta-Reinforcement Learning Robust to Distributional Shift Via Performing Lifelong In-Context Learning](https://proceedings.mlr.press/v235/xu24o.html), ICML 2024, trains a transformer to approximate a predictive posterior over optimal policies and samples actions at deployment with fixed parameters. Its abstract emphasizes train/test task-distribution shift. That is not, by itself, evidence of adaptation to hidden changes during one uninterrupted episode. Avoid conflating those evaluations.

## Stress-testing selective revision as the contribution

The main danger is that a method appears selective because the environment labels each factor for it. A model can then run independent change detectors or route a correction to the explicitly named slot. That may be a useful system, but it does not establish learned structural reasoning.

The stronger protocol has three complementary cases:

1. **Affected and unaffected tasks:** establish several rules, change one, then interleave tasks requiring that rule and tasks depending on other rules. Reward-only performance should expose both recovery and collateral loss.
2. **Unobserved but linked tasks:** after contradictory evidence on one task, immediately test an unseen composition that shares the changed mechanism. Improved first-attempt performance is the critical result; it cannot be explained by relearning each action independently.
3. **Independent control:** present the same local contradiction in an environment where the other task depends on a separate mechanism. Generalizing the correction should now be wrong. Match observed histories as closely as the causal structures permit.

Counterbalance factor identities, change directions, tool names, surface language, and dependency assignments. Include recurrence. Hold out dependency graphs or compositions—not just random seeds and renamed actions. Separate fully supplied dependency structure from structure learned through experience; report each honestly.

Required method comparisons:

- Full-context and bounded recent-context policies with equalized observable data.
- Global reset/discounting and factorwise statistical updates where the factor structure is provided.
- A generic learned memory policy trained on the same trajectories and compute.
- A WorldEvolver-style mismatch-to-rule-memory update.
- Direct return training without prediction or selective-revision supervision.
- Oracle affected-factor/context availability for headroom; keep this privileged baseline separate.

Useful interventions replace, mask, or swap only the proposed predictive-state component while preserving context length and unrelated content. Test both removal of a changed-factor update and insertion of the correct update. If the action changes appropriately only through that component, this supports a causal role for the interface state. It still does not prove an internal neural mechanism. Avoid claiming mediation from rationales alone.

Precommit to a falsifier: if independent factorwise statistical policies match performance once factor identities are supplied, the advance must be demonstrated in learned dependency attribution or language-grounded transfer. If a generic memory learner matches the proposed method, remove special-purpose machinery rather than forcing the hypothesis.

## Recommended held-out task: a ToolSandbox device workflow with a shared backend migration

Use a small **modified ToolSandbox scenario** as the primary language-rich transfer task, with scripted user requests and deterministic state evaluation. This is a proposed extension, not an existing standard benchmark result.

### Verified reusable substrate

[ToolSandbox](https://github.com/apple/ToolSandbox) has composable local tools and per-turn state snapshots for settings, contacts, messages, and reminders. Its milestone evaluator supports exact database comparisons and unchanged-state constraints. Some existing scenarios use a user LLM or external RapidAPI; select the entirely local subset and fixed request scripts. Do not reproduce free-text response grading where exact state suffices.

The actual source contains [time conversion utilities](https://github.com/apple/ToolSandbox/blob/main/tool_sandbox/tools/utilities.py), [reminder creation/modification/search](https://github.com/apple/ToolSandbox/blob/main/tool_sandbox/tools/reminder.py), and [contact-dependent messaging and date-filtered message search](https://github.com/apple/ToolSandbox/blob/main/tool_sandbox/tools/messaging.py). These supply real multi-step data dependencies instead of a renamed arm choice.

### Proposed episode

The agent handles repeated reminder and message-retrieval requests: resolve a natural-language time, convert it, query or write records, and return the requested result. It learns how one time-conversion backend interprets local time. Mid-episode that backend silently changes its timezone or unit convention while keeping a compatible output shape. Contact identity, message contents, and unrelated service requirements remain valid. A read-only calibration endpoint or independent clock provides identifiable but costly evidence. All calls consume the same explicit interaction budget.

After failure on reminder creation, test the first date-filtered message retrieval or another unseen composition sharing the backend. A correctly revised backend assumption should transfer; changing unrelated contact rules should hurt. Include an independent-backend control where correction should *not* propagate. Compare retention on contact-only and untimed tasks. Deterministic tests inspect timestamps, record identities, and unintended database changes.

This needs a carefully written migration wrapper, calibration fixture, and longitudinal episode scheduler. It is substantially smaller than a business simulator, but still new engineering. Freeze request templates, hidden migration schedules, and test dependency graphs before model training. A clean prompt should not announce the hidden switch or mention the correct revised convention.

Why this is more than a relabeled bandit: the required response is a structured sequence with composed transformations and side effects. Evidence from one composition must change another through a shared semantic dependency, while preserving other mappings. No individual “correct arm” captures the policy across requests.

### Practical evaluator safeguards

- Keep backend migration state and evaluator code outside agent-visible files/APIs.
- Do not return privileged `changed_factor`, failure mode, target timestamp, or unit-test text as feedback.
- Diagnostic feedback must distinguish hypotheses without directly announcing the answer; include ordinary transient errors so every mismatch is not a change certificate.
- Count tool calls, retries, calibration, and state inspection in the cost. Cap repeated calls and charge unintended edits.
- Use pre-generated exogenous randomness indexed by episode/request/event, not one sequential RNG whose consumption changes with actions.
- Use identical scripted requests and starting states across compared branches; inspect one full trace to validate observability.
- Place generated paraphrases and compositions in disjoint train/development/test partitions. Preserve an untouched evaluation slice after design tuning.

## CEO-Bench and alternatives

### CEO-Bench: compelling later validation, weak first causal test

[CEO-Bench](https://arxiv.org/html/2606.18543v2) simulates 500 business days, with noisy delayed effects, customer dynamics, and competitor shocks. The paper's harness clears action history weekly and retains an editable memory file. Its reported model runs use seed 42, with repeated runs and best-run measures prominent. These are not a substitute for many independently seeded worlds when estimating treatment effects.

[Repository documentation](https://github.com/zlab-princeton/ceobench-src) exposes tunable configuration and separates the evaluated agent from social-post and enterprise-customer LLMs. Full runs thus involve more than agent inference. It also documents analysis access to the complete ledger. The source should not be placed in the evaluated agent's readable workspace.

The [actual configuration](https://github.com/zlab-princeton/ceobench-src/blob/main/src/saas_bench/config.py) contains separate competitor controls, a grace period, group/individual preference drift, macro-related parameters, and LLM simulator settings. Disabling competitor events alone does not establish stationarity. Competitor reactions also depend on accumulated player development, so action differences can alter subsequent environmental pressure.

**Recommendation:** defer until the method works in controlled tasks. Then use frozen simulator settings and hidden multi-seed forks with one controlled mechanism change. Freeze or cache language-generation components where possible. Report adaptation effects separately from better programming, budgeting, memory compression, or global strategy. Treat its built-in weekly memory reset as an experimental factor.

### ToolMaze: useful benchmark, does not meet the no-judge primary-test preference

[ToolMaze](https://arxiv.org/html/2606.05806v1) uses tool dependency DAGs and explicit/implicit × transient/permanent failures. It enumerates permitted recovery routes and injects deterministic faults. This is closer to structured recovery than a simple tool-bandit. However, its [released implementation](https://github.com/Zhudongsheng75/ToolMaze) configures an LLM judge, and fault activation can depend on the first tool touched within an alternative group. It is not a longitudinal study of established knowledge becoming stale.

**Recommendation:** optional external recovery evaluation with the published protocol; do not claim that it independently establishes selective adaptation. For deterministic scoring, a selected subset would require explicit state assertions and must be labeled a modified evaluation.

### AppWorld: stronger broad external transfer if resources permit

[AppWorld](https://github.com/stonybrooknlp/appworld) provides stateful apps, programmable APIs, and database-state unit tests that check completion and collateral changes. It supports API/environment server interfaces and controlled task state. This makes it a credible larger evaluation substrate without requiring an LLM judge.

**Recommendation:** an untouched held-out AppWorld slice can test general tool competence retention, while a separately labeled backend-drift extension tests adaptation. It is a larger integration than the local ToolSandbox subset. Do not conflate existing task success with non-stationarity; the change protocol would be new.

## Decision gates

1. Verify the inaccessible CICRL full paper before a novelty claim is finalized.
2. Demonstrate affected/unaffected selectivity in a compact factored task with a credible oracle and statistical reference.
3. Show the gain survives equal context and compute; compare direct training and generic memory training.
4. Require first-attempt improvement on held-out linked compositions plus preservation on independent compositions.
5. Only then invest in the ToolSandbox migration task; CEO-Bench follows if the result warrants a broader transfer study.

The credible publication contribution is an experimentally supported learning principle and an effective method. A new environment should serve that claim, not become the main product by default.
