# Task 007: CPU feasibility of a paired revision objective

Author: Codex (commander), 11 September 2026. Worker: Grok. Read `auto_research_rules.md`, [the task-006 review](../reports/task006_review-commander.md), and the training/novelty sections of the supplied `research-*.md` notes. Preserve earlier experiments and commander plans.

**No paid API calls, model downloads, GPU work, optimization/training runs, or new live benchmark.** Use uv and existing dependencies. Deliver a small mathematical/data prototype for the method hypothesis. This assignment deliberately closes the cycle of numeric prompt sweeps.

## 1. The hypothesis, explicitly provisional

Can training on matched evidence with different dependency structures improve the scope of predictive updates on unseen structures, compared with ordinary Bayesian posterior distillation using the SAME individual examples?

An updater receives a timestamped previous joint belief, subsequent observations, and the stated reset-group structure. It returns the updated joint. A fixed utility calculation turns predictions into decisions. The experiment should not credit a revision learner for merely learning action arithmetic or JSON formatting.

The proposed additional training term supervises the DIFFERENCE in updates between matched examples. It adds no privileged observations or new posterior labels. Its value, if any, is an inductive bias/weighting of prediction errors. Do not call it a new invention based on this brief. The structure is provided; this is use of dependencies, not discovery of a hidden graph.

Close comparators: [Bayesian Teaching](https://arxiv.org/html/2503.17523v3), [MemoPilot](https://arxiv.org/html/2606.08656), and WorldEvolver in `research-novelty-transfer.md`. Bayesian Teaching's main target is Bayesian assistant responses, not our full-joint loss; distinguish the actual published method from our stronger ordinary-distillation control. A full novelty clearance remains outstanding.

## 2. Candidate objective and mandatory algebra audit

For a matched pair a,b, let q_j be the exact current joint posterior conditioned only on the permitted evidence in example j. Let m(q_j) be its component-high marginals. Let r_j be the exact marginal prediction obtained from the supplied previous state after all elapsed transitions but WITHOUT assimilating the new observations. This separates evidence-driven revision from ordinary drift under hazard. Define:

```
d_j     = m(q_j)     - r_j
dhat_j  = m(qhat_j)  - r_j
L_base  = 0.5 * [KL(q_a || qhat_a) + KL(q_b || qhat_b)]
L_pair  = mean_k [ (dhat_a[k] - dhat_b[k]) - (d_a[k] - d_b[k]) ]^2
L_total = L_base + lambda * L_pair
```

qhat is a normalized probability vector, produced from logits for this prototype. Correct joint targets already supervise appropriate changes and retention; do not add a rule that forces every unobserved marginal to remain constant. Unobserved components can change through elapsed hazard or dependency evidence.

First derive, then numerically verify:

- r cancels in the loss residual. With marginal errors e_j=m(qhat_j)−m(q_j), the paired term is mean((e_a−e_b)^2).
- It equals a sum of individual squared errors minus a cross-example error term. State exactly what changes compared with independent prediction training, and what does not. It does NOT supply additional target information.
- It cannot detect a shared marginal bias by itself; L_base is essential. Exhibit that failure case.
- With independent zero-mean error vectors, its expectation can reduce to an extra marginal-error penalty. This makes a marginal-MSE/reweighting control essential; do not market a generic auxiliary loss as structural reasoning.

Use a finite-difference/analytic gradient check on a few small logit arrays, not a training run. If the proposed loss is equivalent to a simpler control under the actual pairing construction, explicitly recommend dropping it. Do not redesign it until some positive result appears.

## 3. Minimal exact data fixture

Use FOUR binary components (16 joint states), solely to support a held-out three-component reset group while retaining an unrelated fourth component. Implement a small reusable exact reset-group transition/filter module; do not build a simulator framework or tool workflow.

- Initial independent high probability .99; outcome rates .2/.8; reset-high probability .5.
- A partition defines disjoint reset groups. At each observation step every group has an independent Bernoulli(.03) reset event; when a group resets, its members redraw independently. Include explicit h=0 controls.
- This differs from the old special case where the unrelated component never resets. Serialize the actual group hazards and use them consistently. Under h>0 an unrelated group's forecast may drift even when it receives no new evidence.
- Keep the complete joint in reference computations; do not factorize a correlated group into its marginals.
- Give the solver the partition through a canonical representation and a deterministic plain-language rendering. Rotate names. Render no privileged reset event, latent bit, or which-entry-to-change label.

Generate at most **128 matched pairs**, split 96 training-fixture / 16 development-fixture / 16 structural-test-fixture. These are small feasibility fixtures, not a sufficient training dataset or sealed final benchmark.

Within each pair keep the initial prior, observation identities/outcomes, checkpoint and query fixed; change the reset partition. Compute the previous and current posterior separately under each stated model. The two previous states can legitimately differ; document that the pair isolates a different MODEL with the same evidence, not an identical sufficient state. Every displayed observation must have positive probability under both models. Do not assert the same hidden change occurred in the two worlds.

Training/development topology types: four singleton groups and one pair plus two singletons. Test topology types: two pairs, and one triple plus one singleton. Freeze these types before generating data. Vary label assignments within types; a renamed training partition is not a structural holdout. Keep every underlying evidence-sequence family and all of its pair members in one split. Include observed, unobserved same-group, and other-group forecast queries; use all components for prediction scoring.

Use short fixed/generated observation sequences, at most 12 observations, with a nonempty prefix and suffix. Include stable controls, noisy ambiguous evidence, and positive/negative outcomes without picking examples based on an LLM's errors. If you balance or enumerate evidence rather than sample it from the generative model, mark that explicitly and report likelihoods; do not present fixture frequencies as natural-world event frequencies.

Record full reference q_previous, q_current, transition-only r, update d, pair identity, model parameters, structure type, observations, split and hashes. Keep solver inputs and evaluator-only targets separate. Do not include q_current in the solver input.

## 4. Comparators and checks

All candidate/control objective calculations use the identical individual examples, prediction representation, teacher targets, and normalization:

1. Ordinary posterior KL alone (lambda=0).
2. KL plus an individual marginal-MSE penalty, to control for extra prediction-error weighting.
3. KL plus the proposed matched-pair term.
4. KL plus a shuffled-pair term, pairing examples differently but recomputing the correct target difference for each new pair. Do not create false labels. This controls for generic cross-example regularization.

For now, calculate these objectives on fixed synthetic prediction errors: exact teacher, global stale-state copying, independent-component updating, targeted scope error, common bias, and random normalized predictions. Such probes validate the objective and expose degeneracies; they do not demonstrate learned improvement. Never describe loss rankings as training results.

Meaningful CPU acceptance checks:

- Transition rows and all posteriors normalize; h=0 and h=1 limiting cases work.
- Independently enumerate reset events/redraws for at least one correlated four-component fixture.
- Prefix-plus-suffix filtering equals full-history replay, and label permutation preserves corresponding numeric results.
- A group receiving no observations in the suffix has zero evidence-driven update relative to its transition-only r when independent of observed groups; do not confuse that with zero temporal drift.
- Include at least one same-evidence/different-partition pair with a nonzero difference on an unobserved related component. If the chosen data fail to produce it, diagnose the fixture rather than claiming it measures revision scope.
- Algebra/gradient checks above, pair/split integrity, no current-posterior leakage, and topology holdout checks pass.

Do not add model weights or a neural architecture in 007. The eventual common prediction head should enforce normalized outputs so schema learning is not the treatment; how to attach it to a chosen model belongs to a later measured training plan. A future action-only Bayesian-teacher baseline and a generic learned-memory comparator remain necessary for broader claims; the CPU loss controls do not replace them.

## 5. Deliverables and stop

Return `research_tasks/007_paired_revision_feasibility-worker.md`, a reproducible CPU script, small fixtures/manifest, and `reports/paired_revision_feasibility.md` containing:

- The algebra, objective/gradient checks, exact-reference tests, fixture coverage/likelihoods, actual runtime and memory use, and commands.
- One concrete matched example showing observed/related/unrelated updates under both structures, including transition-only drift.
- An objective comparison on the synthetic error probes, with explicit common-bias and reweighting limitations.
- An evidence table distinguishing implemented feasibility from untested learning, transfer, and novelty claims.
- A short recommendation: retain this exact objective for a measured learning comparison, simplify to ordinary distillation, or reject it. Base this on algebra and fixture properties, not an asserted novel name or a fabricated performance gain.

Stop after the CPU report. No API batch, training, model download or language-workflow build follows automatically. Commander must first decide whether the proposed contribution is distinct enough to warrant a concrete model/compute plan and a full close-method comparison.

The old paid harness has remaining identity/recovery acceptance gaps listed in the task-006 review. They do not block this CPU-only module. Do not spend this assignment rebuilding the paid runner; address them before that runner is next used.
