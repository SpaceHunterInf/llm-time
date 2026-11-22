# Literature Map & Novelty Positioning (Updated Nov 2026)

This file tracks our canonical reading list and defines **what is already occupied** vs. **our paper's exact contribution**.

---

## 0. Our Exact Wedge (In One Paragraph)
Recent benchmarks (`AdaExploreBench`, `ServeLearnBench`, `ReScopeBench`, `Are Self-Evolving Agents Continual Learners?`) establish that frozen LLMs and non-parametric RSI harnesses (`GEPA`, `Meta-Harness`, `ReasoningBank`) struggle under silent drift, while `The RSI Tax` (`2611.13524`) shows continuous verification wastes 75–94% of tokens and `EvoBelief` (`2610.12789`) wraps frozen LLMs in falsifiable world-model rules for environment dynamics. **What remains unaddressed:**
1. **The Reward-Paradigm RSI Ratchet:** When *world mechanics stay valid* (no tool crashes) and only the **reward/utility function** shifts (abrupt reversal, continuous alpha decay, lifecycle maturation), self-improvement methods actively *worsen* adaptation via score lock-in and self-compliance ("doubt without durable retraction / hypothesis resurrection").
2. **Low-Tax Surprise Gating + Scoped Shelving (Act 2):** Combining likelihood-ratio/CUSUM prediction-error triggers (zero verification tokens during stationary periods, minimal false alarms on noise) with post-changepoint evidence resets and regime shelving.
3. **Weight-Level Epistemic Plasticity (Act 3):** Meta-training open weights (`8B` / `27B` LoRA via GRPO/RFT) on non-stationary single-life reward-shift trajectories so *conservative doubt, active discriminative probing, and committed retraction* become native in-context behaviors without paying an external multi-call harness tax.

---

## 1. Closest Competitors & Non-Stationary Benchmarks
- ★ **AdaExploreBench** (silent rule drift, plain LLMs; 10.8% detection on direction-reversal vs. 33.5% on factor-replacement): https://arxiv.org/abs/2610.25950
- ★ **ReScopeBench** (silent partial changes across stateful streams, tracks saved structural/rule experience; code in `competitors/ReScopeBench`): https://arxiv.org/abs/2611.05450
- ★ **Safety Must Survive Self-Improvement** (historical scores keep invalid/unsafe programs alive; closest precedent to the score ratchet): https://arxiv.org/abs/2610.01073
- ★ **Are Self-Evolving Agents Continual Learners?** (evaluates `GEPA`, `Meta-Harness`, `ReasoningBank` under drifting streams; exposes plasticity–stability & retrieval collision): https://arxiv.org/abs/2610.25858
- **ServeLearnBench** (hidden business/utility policies evolve across windows, scalar score feedback; code in `competitors/ServeLearnBench`): https://arxiv.org/abs/2610.07792
- **Red Queen Gödel Machine (RQGM)** *(Added)* (RSI under evolving utilities across explicit epochs; contrasts with our *unannounced* midstream reward drift): https://arxiv.org/abs/2606.26294
- **EvoArena**: https://arxiv.org/abs/2606.13681
- **MirrorCraft**: https://arxiv.org/abs/2607.29218
- **EvoPathBench**: https://arxiv.org/abs/2609.24663
- **FinEvolveBench**: https://arxiv.org/abs/2606.06960
- **KellyBench**: https://arxiv.org/abs/2604.27865

## 2. Mechanism, Change Detection & Belief Updating Theory
- ★ **Many Steps into One** (proves symmetric/unclipped success counters are terrible change detectors; proposes likelihood-ratio weighting): https://arxiv.org/abs/2610.17544
- ★ **Comparative Reversal Learning in LLMs**: https://arxiv.org/abs/2604.04182
- **PEARL** (prediction-error-adaptive selective forgetting along surprised feature directions vs. uniform forgetting): https://arxiv.org/abs/2611.05842
- **Unable to Forget** (proactive interference in LLMs): https://arxiv.org/abs/2506.08184
- **When Context Changes** (stale binding / attention drift): https://arxiv.org/abs/2609.38866
- **LLMs Remember First, Forget Last**: https://arxiv.org/abs/2603.00270
- **LLMs as Discounted Bayesian Filters** (counter-evidence weighting): https://arxiv.org/abs/2512.18489
- **From Prompts to Posteriors** (Bayes in context; code access helps): https://arxiv.org/abs/2610.29331
- **In-Context Learning Agents Are Asymmetric Belief Updaters**: https://arxiv.org/abs/2402.03969
- **What Should World Models Forget?**: https://arxiv.org/abs/2610.03713

## 3. Stale Memory & Stale Self-Written Rules
- ★ **Closing the Feedback Loop** (insight governance: separating Rules, Evidence, and Skills layers for non-monotonic lifecycles): https://arxiv.org/abs/2606.17591
- **Memory Reward Inflation in Self-Improving LLM Agents** *(Added)* ("The Echo Gap" — self-graded memories inflate scores on confident mistakes): https://arxiv.org/abs/2608.00017
- **Do Self-Evolving Agents Forget? (CPE)** *(Added)* (capability degradation across workflow, skill, weight, and memory channels): https://arxiv.org/abs/2605.09315
- **STALE**: https://arxiv.org/abs/2605.06527
- **Supersede**: https://arxiv.org/abs/2606.27472
- **The Compliance Trap**: https://arxiv.org/abs/2607.10608
- **When Stale Constraints Go Unchecked**: https://arxiv.org/abs/2608.25553
- **Lessons Outlive Their Failures** (self-added agent rules persist after underlying failures vanish): https://arxiv.org/abs/2610.32578
- **When Instruction Files Go Stale**: https://arxiv.org/abs/2611.09802
- **When Continual Learning Moves to Memory**: https://arxiv.org/abs/2604.27003

## 4. Self-Improvement (RSI) & World-Model Methods
- ★ **The RSI Tax** (verification consumes 74.8–93.6% of RSI tokens; bounded Simple Reflection with Notes `SRN` matches heavy RSI at ~70% lower cost): https://arxiv.org/abs/2611.13524
- **EvoBelief** (falsifiable world-model hypotheses for frozen LLMs on ALFWorld/MiniGrid/DiscoveryWorld — key Act 2 comparator): https://arxiv.org/abs/2610.12789
- **MemoPilot** *(Added)* (trains a memory-updater via multi-turn GRPO across opponent switches — key Act 3 comparator): https://arxiv.org/abs/2606.08656
- **WorldEvolver** *(Added)* (converts prediction mismatches into textual world-model rules with frozen weights): https://arxiv.org/abs/2606.30639
- **GEPA**: https://arxiv.org/abs/2507.19457
- **ACE (Agentic Context Engineering)**: https://arxiv.org/abs/2510.04618
- **ReasoningBank**: https://arxiv.org/abs/2509.25140
- **Darwin Gödel Machine**: https://arxiv.org/abs/2505.22954
- **Meta-Harness**: https://arxiv.org/abs/2603.28052
- **AlphaEvolve**: https://arxiv.org/abs/2506.13131
- **Schema**: https://arxiv.org/abs/2609.39140
- **Memento 3**: https://arxiv.org/abs/2610.11794
- **RRSI**: https://arxiv.org/abs/2609.24972
- **Continual Harness**: https://arxiv.org/abs/2605.09998
- **Phantom Guardrails**: https://arxiv.org/abs/2607.13083
- **Can LLMs Improve Their Own Harness? (ARC-AGI-3)**: https://arxiv.org/abs/2610.30499
- **Reflexion**: https://arxiv.org/abs/2303.11366
- **ExpeL**: https://arxiv.org/abs/2308.10144
- **Voyager**: https://arxiv.org/abs/2305.16291
- **ADAS**: https://arxiv.org/abs/2408.08435
- **STOP**: https://arxiv.org/abs/2310.02304

## 5. Continual RL, Single-Life RL & Non-Stationarity Foundations
- ★ **You Only Live Once: Single-Life RL**: https://arxiv.org/abs/2210.08863
- ★ **In-Context RL under Non-Stationarity: A Survey**: https://arxiv.org/abs/2607.11906
- **DynaMITE-RL** *(Added)* (latent inference & session-boundary detection within episodes, NeurIPS 2024): https://arxiv.org/abs/2402.15957
- **Rethinking the Foundations for Continual RL**: https://arxiv.org/abs/2504.08161
- **A Definition of Continual RL**: https://arxiv.org/abs/2307.11046
- **Towards Continual RL: A Review and Perspectives**: https://arxiv.org/abs/2012.13490
- **Maintaining Plasticity in Deep Continual Learning**: https://arxiv.org/abs/2306.13812
- **Can Scale Save Us From Plasticity Loss in LLMs?**: https://arxiv.org/abs/2606.24752
- **Bayesian Online Changepoint Detection (BOCPD)**: https://arxiv.org/abs/0710.3742
- **UCB for Non-Stationary Bandits (Discounted / SW-UCB)**: https://arxiv.org/abs/0805.3415
- **RL²**: https://arxiv.org/abs/1611.02779
- **Learning to Reinforcement Learn**: https://arxiv.org/abs/1611.05763
- **ORBIT** (cross-episode meta-RL for LLMs): https://arxiv.org/abs/2602.04089

## 6. Exploration, Belief Revision & Rule Discovery
- **Can LLMs Explore In-Context?**: https://arxiv.org/abs/2403.15371
- **EVOLvE**: https://arxiv.org/abs/2410.06238
- **FALSIFYBENCH**: https://arxiv.org/abs/2606.04751
- **AI Scientists Produce Results Without Reasoning Scientifically**: https://arxiv.org/abs/2604.18805
- **LongChang** (altered-physics worlds): https://arxiv.org/abs/2610.37649
- **Reasoning or Reciting?** (counterfactual tasks): https://arxiv.org/abs/2307.02477
- **ARC-AGI-3**: https://arxiv.org/abs/2603.24621

## 7. Long-Horizon Host Environments
- **Vending-Bench**: https://arxiv.org/abs/2502.15840
- **EcoGym**: https://arxiv.org/abs/2602.09514
- **AgentOdyssey**: https://arxiv.org/abs/2606.24893
- **LatentGym**: https://arxiv.org/abs/2606.15306
