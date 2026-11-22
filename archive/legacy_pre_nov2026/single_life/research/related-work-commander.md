# Related work and workflow sources

Checked 28 September 2026 using primary papers/project sources. This is a focused orientation and novelty checklist, not an exhaustive systematic review.

## Research-agent workflow

| Source | Verified pattern | Our choice |
|---|---|---|
| [Karpathy autoresearch: program.md](https://github.com/karpathy/autoresearch/blob/master/program.md) | A narrow editable training file, fixed preparation/evaluation, bounded runs, and a results ledger including discarded and crashed trials | Borrow the small task surface, fixed comparator, and complete ledger. Do not use its Git-reset loop on this dirty shared checkout or as the online agent's world-reset mechanism. |
| [AI Scientist-v2, official repository](https://github.com/SakanaAI/AI-Scientist-v2) and [paper](https://arxiv.org/abs/2504.08066) | Structured research ideas and experiment-manager-guided tree search, followed by analysis/write-up stages | Separate scientific review from implementation. One bounded task and an explicit worker return are sufficient now; a tree-search orchestrator and automatic paper writer are unnecessary. |

These are examples, not evidence that one universal organization is best. Our workflow is a design judgment: question + validity contract + registration + implementation return + review + next decision. Researchers may rerun experiments; the evaluated learner cannot rerun its past life.

## Closest scientific precedents

| Work | Established overlap | Consequence |
|---|---|---|
| [You Only Live Once: Single-Life Reinforcement Learning (2022)](https://arxiv.org/abs/2210.08863) | Adaptation during one uninterrupted deployment, using prior experience to help with novel situations | “Only one life” is not a novelty claim. Examine repeated hidden changes and acquired procedures, stating differences in prior training and task structure. |
| [Reinforcement Learning with Self-Modifying Policies (1997)](https://people.idsia.ch/~juergen/ssabook/ssabook.html) | Lifelong policies modifying their own learning components; success-story backtracking of internal modifications based on observed reward history | Self-modification and internal rollback have longstanding precedent. Investigate credit assignment when external conditions change; do not claim the general concept. |
| [GLOVE (2026)](https://arxiv.org/abs/2601.19249) | Active probing checks retrieved memories against fresh observations under drift | Direct overlap with applicability checking. Inspect its implementation/evaluation and use a faithful comparator or clearly label an adaptation. Checking stale memory alone is insufficient novelty. |
| [PACE-Bench (2026)](https://arxiv.org/abs/2608.14441) | Code-based adaptation across changed physical conditions using source-to-target pairs and diagnostic feedback | Changing gravity and inherited design failure are already benchmarked. Sustained chronological adaptation with costly validation is a possible distinction; verify exact reset/feedback differences before claiming it. |
| [Dream-RSI (2026)](https://arxiv.org/abs/2609.14858) | Refines exploration policies using replay simulations from historical discovery trees, then redeploys online | Past experience can support improvement without being future leakage. Replaying recorded outcomes does not establish current validity after exogenous change. Distinguish recorded, hypothetical, and newly observed outcomes. |

Current recheck covers the cited abstracts, the self-modifying-policy chapter abstract, and workflow source instructions; earlier discussion included fuller Dream-RSI/PACE/GLOVE reading. These entries do not establish implementation equivalence or exhaustive novelty.

## Follow-up before a paper claim

Review nonstationary bandits/continual RL, change detection, selective forgetting and recurrent contexts, dual control/active identification, online model selection, and performative prediction if actions induce change. Existing notes also mention BAM, EvoMem/EvoArena, RQGM, ModularRSI, and forgetting in self-evolving agents. These are leads requiring a direct source check, not certified baselines in this scaffold.

Build a full-protocol comparison table: learner resets, pretraining distribution, changes, true simulator/counterfactual access, mutable artifact, mutable improver, probing cost, and chronological selection. The strongest novelty objection is that our method could reduce to memory verification plus an ordinary nonstationary estimator. The pilot must be capable of supporting that objection.
