# Training methods and novelty audit

Prepared 9 September 2026. This is a focused primary-literature investigation, not an exhaustive systematic review. The studies below were inspected through their full-text method/result sections unless explicitly marked otherwise. Proposals are separated from reported findings. Page numbers refer to printed/PDF pages where available.

## Recommendation

The most defensible direction is **learning selective revision of an explicit predictive state under hidden, component-wise change**. The contribution must be stronger than training a bandit policy, adding reflection, training memory updates, or rewarding forecasts: all four already have close precedents. A useful claim would be that one specific training intervention improves the stable/adaptive performance frontier and transfers across unseen change mechanisms, with interventions demonstrating that it preserves still-valid knowledge while replacing invalid knowledge.

The strongest new overlap is MemoPilot. It already trains a memory updater for a frozen player, evaluates midstream switches, and explicitly identifies stale memory under frequent changes. Any proposed memory method should be compared against a MemoPilot-style reward-only updater, rather than only hand-written prompting baselines.

## Primary evidence

### EVOLvE — Nie et al., ICML 2025

[Paper](https://arxiv.org/pdf/2410.06238). Read §4.1 for information interfaces; §4.2, pp. 7–8, for trajectory generation and fine-tuning; §4.3 and Figs. 3–4 for representation and transfer ablations. UCB-generated histories supervise the next action, while an alternative supplies algorithmic statistics at inference. Their MAB datasets use 50 trajectories per stated configuration; contextual-bandit datasets use 200. Fine-tuning and prompting can improve exploration, and presentation effects differ between settings. The reported contextual-bandit generalization is held-out users within the training distribution. The experiments do not establish adaptation to hidden within-stream regime changes. **Implication:** non-stationary teacher distillation is a sensible baseline, but synthetic bandit trajectory SFT and interface ablations are established methods, not sufficient novelty.

### Algorithm Distillation — Laskin et al., ICLR 2023

[Paper](https://arxiv.org/pdf/2210.14215). Read §3, pp. 3–4, for the history-conditioned action objective; §5 and Figs. 6–7 for sample efficiency and context length. A causal transformer is trained on RL learning histories from multiple tasks. At evaluation, fixed weights can implement improvement through accumulating context. Multi-episode context is necessary in their context-size experiment; some distilled agents learn more sample-efficiently than the source algorithm while the source can retain higher asymptotic performance. **Implication:** “learning an update algorithm in frozen weights” is an established formulation. The paper does not provide the selective non-stationary revision intervention proposed here.

### Decision-Pretrained Transformer — Lee et al., NeurIPS 2023

[Paper](https://arxiv.org/pdf/2306.14892). Read §3/Algorithm 1, pp. 3–4, and §6.1–6.2, pp. 9–10. DPT predicts an optimal action from a query state and an interaction dataset. Under explicit coverage/realizability assumptions, a history-dependent variant corresponds to posterior sampling. Its formal “compliance” condition requires context-collection actions to depend only on observed history; expert-biased context data can violate this condition. Optimal-action training labels can nevertheless use the simulator's task specification. **Implication:** distinguish privileged training targets from privileged evidence in the input. Posterior-sampling correspondence does not automatically establish Bayes-optimal change detection or optimal information gathering under a new non-stationary model.

### DynaMITE-RL — Liang et al., NeurIPS 2024

[Paper](https://arxiv.org/html/2402.15957). Read §§3–4 for dynamic latent contextual MDPs and the training objective; §5, Fig. 5, and Appendix A.4 for experiments. The latent context can change inside an episode. The method combines session consistency, conditioning on the preceding latent belief, and session-restricted reconstruction. It learns session boundaries and compares with a boundary oracle. Experiments include changing goals, control targets, forces, and human preference locations; online and offline training are evaluated. **Implication:** learned latent inference under unannounced within-episode change, predictive reconstruction, and consistency regularization already exist outside LLMs. It is an essential conceptual baseline, especially if the new method claims principled belief maintenance.

### PSBL — Xu et al., ICML 2024

[Proceedings](https://proceedings.mlr.press/v235/xu24o.html), [PDF](https://openreview.net/pdf?id=laIOUtstMs). **Access limitation:** the authoritative abstract and indexed PDF first page were verified, but full-text retrieval was blocked by OpenReview browser verification and the alternative PDF endpoint failed. The verified claim is that a transformer learns a predictive posterior over optimal policies and samples actions with frozen parameters, improving performance under test-task distribution shift. **Do not infer from the title that it demonstrates hidden within-episode regime changes.** That specific protocol still requires full-text verification before using PSBL as direct evidence of non-stationary adaptation.

### When Greedy Wins — Chen et al., September 2025 preprint

[Paper](https://arxiv.org/html/2509.24923). Read §§3.1–3.3 for PPO/SFT objectives and reward alternatives; §5.2 and Appendix B.2 for behavioural failure analysis. The study trains bandit LLMs using observed reward, regret-based strategic rewards, or algorithm-imitation signals. It reports generalization to longer horizons and other bandit families. Improved average performance can accompany greedier policies and elevated suffix failure; model size changes which training methods work. SFT can fail on negative Gaussian rewards after Bernoulli training. **Implication:** compare reward-only RL, SFT, and teacher-guided methods with failure tails, not merely average regret. Improved reward cannot alone demonstrate better exploration or a general revision algorithm.

### LaMer — Jiang et al., ICLR 2026

[Paper](https://arxiv.org/html/2512.16848), [project](https://lamer-site.github.io/). Read §4 for cross-episode return and trained self-reflection; §5.1 and §6.2–6.3 for controls and cost. LaMer trains both action generation and reflection across repeated attempts, including ALFWorld and WebShop. Standard RL baselines receive equal numbers of trajectories per update, while LaMer's sequential rollouts cost approximately twice as much wall-clock time in their implementation. Reflection-only memory outperforms combined history/reflection in the reported ablation. **Implication:** trainable reflection, multi-episode credit assignment, and language-rich transfer are already established. Distinguish repeated attempts on a task from hidden changes to its rules within the retained context.

### MAGE — Yang et al., March 2026 preprint / ICLR 2026 Lifelong Agent workshop paper

[Paper](https://arxiv.org/html/2603.03680), [workshop PDF](https://openreview.net/pdf?id=39tQ6si2EK). Read §3.1 and §3.2.3 carefully: each meta-episode uses a stationary task or a single fixed opponent despite broad introductory language about non-stationarity. Reflections support adaptation across attempts; a differential episode reward targets terminal performance. §4.1 matches trajectory counts; §4.3 evaluates unseen opponents. **Implication:** population training and learned strategic adaptation are close precedents, but unseen opponent identity is not equivalent to an unannounced midstream opponent switch. There is also a metric distinction between final-episode return and cumulative deployment return, relevant when exploration has real costs.

### MemoPilot — Cai et al., June 2026 preprint

[Paper](https://arxiv.org/html/2606.08656). Read §§3.1–3.2 for a trained memory updater feeding a frozen player; §§5.4–5.6 and Tables 5–7 for switch tests and the maintenance/refinement trade-off. The method uses multi-turn GRPO, with a next-game reward assigned to each memory update; its cumulative-return alternative is less stable in the reported ablation. Separate memory models are trained for rock–paper–scissors and Limit Hold'em. Cross-opponent tests retain memory through a switch; frequent switches degrade performance. The authors explicitly discuss conservative memory becoming stale and opponents exploiting stored beliefs. **Implication:** the generic “train a memory updater to revise hypotheses from feedback” contribution is already taken. A targeted improvement to this failure, with controlled mechanisms and held-out dynamics, remains potentially useful.

### VAGEN — Wang et al., NeurIPS 2025

[Paper](https://arxiv.org/html/2510.16907), [project](https://vagen-ai.github.io/). Read §§2–3 for explicit state estimation/transition predictions and representation choices; §4.1–4.4 for rewards and ablations. Visual language agents receive dense feedback on their predicted/estimated states and a turn-aware credit assignment scheme. The implementation uses an LLM judge against ground-truth state information, with reward-hacking analysis in Appendix D/E. **Implication:** “forecasting supervision improves agents” and “explicit belief representation plus RL” are already established. A new method needs uncertainty-sensitive, component-wise change handling and downstream evidence, rather than repackaging state-prediction rewards.

## Proposed study: learn what changed, preserve what did not

This section is a research proposal, not a claim established by the cited papers.

### Main question and discriminating task

Can a model learn a reusable procedure for updating *parts* of its predictive state from feedback, improving cumulative reward under both stable and changing conditions?

Use a factored latent environment with 3–5 independently meaningful components. An example is operational routing: customer preference, supplier reliability, and tool behaviour jointly determine outcomes. Queries/actions expose different components and incur explicit costs. Inject hidden changes into one component, sometimes several; preserve the others. Include stable periods, observation noise, recurrence, and gradual change. Ensure that different latent causes initially induce similar failures but can be disambiguated by diagnostic actions.

Begin with a compact numeric/symbolic version where posterior references are tractable. Add a language-rich realization where descriptions convey relationships needed for transfer, not merely renamed arm labels. If a component is never queried after a change, the agent may legitimately remain uncertain; do not count a lack of omniscient knowledge as a reasoning failure.

### Minimal feasible training sequence

1. **Diagnostic pilot:** hold interaction histories fixed and compare raw evidence, verified sufficient summaries, model-generated predictive state, and reference predictive state. Use an unchanged action head/player when testing memory. This separates evidence interpretation from acting on beliefs. Proceed only if a substantial correctable gap exists.

2. **Common data:** generate matched mixed stationary/changing trajectories with random exploration and competent observation-limited teachers. Split by latent structure/change mechanism, not only seeds. Keep the same underlying examples, state visibility, output budget, and training-token budget across objectives. Separately mark any simulator-only targets; never put true cut times or hidden states into evaluation input.

3. **Strong cheap baseline:** supervised fine-tuning on teacher actions from histories. Also train the same representation with predictive supervision alone. This checks whether a custom update objective is needed before interactive RL investment.

4. **Candidate intervention:** maintain compact component-indexed predictive state with uncertainty. Train its update using action-conditioned outcome prediction, and add matched unchanged-component examples so successful updating requires retention as well as replacement. Score forecasts with a proper scoring rule on held-out outcomes; score decisions with cumulative return. Do not reward declarations of change or convincing rationales. A learned scalar forgetting gate is a necessary low-complexity comparator.

   If distilling an explicit posterior over changed components, the target must be a filtering posterior conditioned only on the observed history under the stated training prior. Do not supervise the model to assert the simulator's true changed component when the observations do not identify it. In particular, two examples with identical observable histories must have the same predictive/posterior target even when their hidden simulator states differ. No future outcomes may enter the filtering target. This avoids teaching clairvoyance and makes ambiguous noise-versus-change cases legitimate uncertainty tests.

5. **Optional RL:** start from the successful SFT model and optimize return across the mixed distribution, retaining predictive auxiliary loss only if it improves held-out results. Include a MemoPilot-style reward-only updater and direct action-only RL. Use shared simulator randomness where valid to reduce reward variance across candidate rollouts. Report wall-clock/inference tokens as well as matched environment steps.

For a pilot, a 3–8B open model with adapter fine-tuning and roughly 5,000–20,000 generated trajectories is a reasonable *planning range*, not an evidence-backed compute estimate. Estimate actual cost from a small throughput run before choosing the full budget. Replicate the main effect in a second model family only after the method shows a clear gain.

### Critical controls and failure criteria

- **Global recency versus selective update:** fixed/sliding windows, tuned exponential discounting, global change-triggered reset, component-wise statistical filtering, and a learned scalar recency gate. If these match the proposed method, do not claim a novel reasoning capability.
- **Information and capacity:** equal evidence, output tokens, calls, training data, optimization steps, and auxiliary supervision. Compare direct policy, structured predictive state, and generic reflection.
- **Stationarity:** plot stationary cost versus post-change regret for tuned hyperparameter sweeps. A change-only dataset rewards pathological perpetual doubt. Do not report only an arbitrary 50/50 average.
- **Generalization:** held-out change frequencies, substantially longer stable histories, unseen component combinations, recurrence, and a held-out change mechanism. A gain confined to trained cuts is insufficient.
- **Behavioural robustness:** include unrecovered changes, late collapses, cumulative regret, and lower-tail performance. Cluster uncertainty by independently generated episode and training seed.
- **Causal interface tests:** replace one predictive-state component with a correct/stale value and measure effects on forecasts and targeted actions. Ablate stored uncertainty and history provenance independently. This supports a causal claim about the implemented memory interface, not a claim to read the model's private beliefs.
- **Transfer threshold:** the language-rich task must require the same inferred update rule and preserve unrelated knowledge after local change. Generic success on a large benchmark without controlled changes is weak evidence for the proposed mechanism.

### What would be incremental

- More models failing reversal tasks, absent a new mechanism or intervention.
- Fine-tuning an LLM on UCB, Thompson sampling, or change-detection actions and reporting higher average bandit reward.
- Adding an explicit hypothesis field, reflection, predictive loss, or learned memory updater without comparison to the close methods above.
- Renaming bandit arms with a business narrative or adding a large simulator without identifying which adaptation skill transferred.

The plausible contribution is an experimentally demonstrated improvement in **selective evidence revision under latent change**, backed by a matched objective/control study and structural transfer. This literature search does not establish that such a method is novel; it narrows a defensible hypothesis and identifies the closest obstacles.

## Additional critique of posterior-predictive distillation

Posterior distillation in a known small latent family could simply compress an existing statistical filter into an expensive language model. A component-wise filter can already preserve unaffected beliefs, and a reward-trained policy may implicitly do the same. The publishable question is therefore whether explicit predictive supervision teaches a *transferable* selective updating procedure more effectively than direct policy or memory-reward training at equal data/compute. A structural holdout is more discriminating than new priors: for example, training on independent changes and testing on a new dependency graph conveyed through language, where one observation should revise several connected beliefs while leaving others intact. Conversely, train shared changes and test independent ones to detect indiscriminate propagation.

Active diagnosis strengthens the claim: the agent must select a costly observation that distinguishes two plausible causes, then revise only beliefs affected by the evidence. Compare the same agent with free diagnostic evidence to determine whether the bottleneck lies in choosing the experiment or interpreting its outcome. This is an extension proposed here, not a demonstrated finding from the reviewed studies. If prediction improves but diagnosis/return does not, retain that as a diagnostic result and reject the proposed method claim.
