# Paired revision objective: CPU feasibility (task 007)

11 September 2026. CPU only. No API calls, no model download, no training, no GPU. Fixtures are **hand-enumerated**, not draws from the generative model; reported likelihoods are those of the listed sequences, not natural-world frequencies.

Artifacts: `reports/paired_revision_feasibility/{manifest.json,solver_inputs.json,evaluator.json}`. Script: `scripts/paired_revision_feasibility.py`. Module: `llm_rl_engine/analysis/paired_revision.py`.

Runtime **4.23 s**, peak RSS **24.6 MB**. Tests: **155 passed**.

## Algebra

For a matched pair \(a,b\), \(q_j\) is the exact current joint, \(m(q_j)\) its component-high marginals, \(r_j\) the transition-only marginals from the supplied previous joint over the suffix length (hazard drift, no new observations). \(d_j=m(q_j)-r_j\), \(\hat d_j=m(\hat q_j)-r_j\).

\[
L_\mathrm{base}=\tfrac12\bigl(\mathrm{KL}(q_a\|\hat q_a)+\mathrm{KL}(q_b\|\hat q_b)\bigr)
\]
\[
L_\mathrm{pair}=\mathrm{mean}_k\bigl((\hat d_a[k]-\hat d_b[k])-(d_a[k]-d_b[k])\bigr)^2
\]
\[
L_\mathrm{total}=L_\mathrm{base}+\lambda L_\mathrm{pair}
\]

**\(r\) cancels.** Let \(e_j=m(\hat q_j)-m(q_j)\). Then

\[
(\hat d_a-\hat d_b)-(d_a-d_b)=e_a-e_b
\]

so \(L_\mathrm{pair}=\mathrm{mean}((e_a-e_b)^2)\). Numeric: `|with_r − without_r| = 0`.

**Expansion, no new labels.**

\[
\mathrm{mean}((e_a-e_b)^2)=2\,L_\mathrm{mse}-2\,\mathrm{mean}(e_a\odot e_b)
\]

where \(L_\mathrm{mse}=\tfrac12(\mathrm{mean}(e_a^2)+\mathrm{mean}(e_b^2))\). The pair term is individual squared marginal error minus a cross-example product. Teacher joints \(q_a,q_b\) are unchanged. Numeric expansion residual \(1.4\times10^{-17}\).

What changes versus independent prediction training: the \(-2\,\mathrm{mean}(e_a e_b)\) covariance. Shared errors are **downweighted**; opposite-signed errors are **upweighted**. What does not change: the information in the labels.

**Common bias is invisible to \(L_\mathrm{pair}\).** If \(e_a=e_b=c\), then \(L_\mathrm{pair}=0\) while \(L_\mathrm{base}\) need not be. Numeric common-bias construction: \(L_\mathrm{pair}=2.3\times10^{-33}\). \(L_\mathrm{base}\) is essential.

**Independent zero-mean errors.** If \(e_a,e_b\) are independent and zero-mean, \(\mathbb{E}[L_\mathrm{pair}]=2\,\mathbb{E}[L_\mathrm{mse}]\): an extra marginal-error penalty. Hence the KL + individual-MSE control.

**Gradient.** \(\hat q=\mathrm{softmax}(z)\). Finite-difference vs analytic on random 16-logit arrays: max abs error \(6.2\times10^{-11}\) and \(1.2\times10^{-10}\).

## Exact-reference checks

- Transition rows and all posteriors normalize. \(h=0\) is identity; \(h=1\) is uniform over 16 states (every group resets, every bit redraws independently at 1/2).
- Brute-force enumeration of group resets and member redraws matches the matrix builder to \(1\times10^{-12}\) on the correlated partition \(\{\{0,1\},\{2\},\{3\}\}\).
- Prefix-then-suffix filtering equals full-history replay to \(1\times10^{-12}\).
- Label permutation preserves corresponding joints.
- A group with no suffix observations, independent of observed groups, has **zero evidence-driven** \(d[k]\) relative to transition-only \(r\). Temporal drift of that group is in \(r\), not in \(d\).
- Unrelated groups **can** reset. This is not the old three-component special case where the unrelated bit never reset.

## Fixture

Four binary components (16 joints). Disjoint reset groups; each group has its own Bernoulli(\(h\)) event; members redraw independently at 1/2. Initial high probability 0.99; rates 0.2/0.8.

**128 matched pairs** (96 train / 16 dev / 16 structural-test). Same evidence, different partition. Previous joints are allowed to differ (96/128 do). Every observation has positive probability under both models (likelihoods in \([-5.92,-1.35]\)).

| Split | n | Topology types | Evidence families |
|---|---:|---|---|
| train | 96 | four singletons vs one pair + two singletons | T00–T15 |
| dev | 16 | same types, disjoint families | D00–D03 |
| test | 16 | two pairs vs triple+singleton (held out) | X00–X03 |

A renamed training partition is not in the test split. Every evidence family stays in one split. Sequences have a nonempty prefix and suffix, length ≤ 12. Solver inputs: previous joint, suffix, stated partition, names, hazards. Evaluator-only: \(q_\mathrm{current}\), \(r\), \(d\), likelihoods. `solver_inputs.json` contains no `q_current`.

**Unobserved related component.** 60/128 pairs have a nonzero \(d\)-gap on a suffix-unobserved component that is grouped with an observed component on exactly one side. Max gap 0.167. If this failed, the fixture would not measure revision scope.

## One matched example (T04)

Prefix and suffix: component A = 1, four times each. \(h=0.03\). Names A,B,C,D.

- **a** four singletons. **b** \(\{A,B\}\) share a reset; C and D singleton.

| | A (observed) | B (unobs. related in b) | C | D |
|---|---:|---:|---:|---:|
| \(m(q_\mathrm{prev})\) a | 0.9949 | 0.9338 | 0.9338 | 0.9338 |
| \(m(q_\mathrm{prev})\) b | 0.9949 | 0.9583 | 0.9338 | 0.9338 |
| \(r\) a (drift only) | 0.9382 | 0.8840 | 0.8840 | 0.8840 |
| \(r\) b | 0.9382 | 0.9057 | 0.8840 | 0.8840 |
| \(m(q_\mathrm{cur})\) a | 0.9949 | 0.8840 | 0.8840 | 0.8840 |
| \(m(q_\mathrm{cur})\) b | 0.9949 | 0.9309 | 0.8840 | 0.8840 |
| \(d\) a | 0.0568 | **0** | 0 | 0 |
| \(d\) b | 0.0568 | **0.0251** | 0 | 0 |

Previous states differ. C and D: evidence-driven update is 0; they still **drift** (\(r\) 0.934 → 0.884). B moves with A only under the pair partition. Likelihoods −1.9103 (a) and −1.9103 (b).

## Synthetic prediction-error probes

Fixed \(\hat q\), not optimization. \(\lambda=\lambda_\mathrm{mse}=1\). Train n=96. **Not training results.**

| Probe | \(L_\mathrm{base}\) | \(L_\mathrm{mse}\) | \(L_\mathrm{pair}\) | \(L_\mathrm{KL+MSE}\) | \(L_\mathrm{KL+pair}\) | shuffled pair term |
|---|---:|---:|---:|---:|---:|---:|
| exact teacher | 0 | 0 | 0 | 0 | 0 | 0 |
| common bias (+0.05 IPF) | 0.323 | 0.00151 | **6.5e-6** | 0.325 | 0.323 | 0.00071 |
| independent-component updater | 0.0179 | 0.000320 | 0.000639 | 0.0182 | 0.0186 | 0.000639 |
| targeted scope swap (\(\hat q_a=q_b\)) | 0.0324 | 0.000639 | 0.00256 | 0.0330 | 0.0349 | 0.00155 |
| stale previous joint | 0.139 | 0.00875 | 0.000303 | 0.148 | 0.139 | 0.0175 |
| random softmax | 2.388 | 0.208 | 0.0411 | 2.596 | 2.429 | 0.0885 |

Algebraic identities on these probes, not empirical wins:

- **Scope swap:** \(\hat q_a=q_b\), \(\hat q_b=q_a\) implies \(e_b=-e_a\), hence \(L_\mathrm{pair}=4\,L_\mathrm{mse}\) exactly. The pair term is a constant reweight of the same marginal errors MSE already measures.
- **Independent updater on this train pairing:** side a is four singletons, so \(e_a=0\) and \(L_\mathrm{pair}=2\,L_\mathrm{mse}\). The shuffled-pair control is degenerate here (same number), because every a-side error is already zero.
- **Common bias:** \(L_\mathrm{pair}\approx0\) while KL is large. The pair term does not see a shared marginal shift.
- **Stale copy:** pair term is small relative to MSE (errors are correlated across the two models). KL+MSE flags it; KL+pair barely adds.

Test-split probes (held-out two-pair vs triple) show the same pattern: scope-swap \(L_\mathrm{pair}=4\,L_\mathrm{mse}\) again; common-bias \(L_\mathrm{pair}=2.3\times10^{-5}\) vs \(L_\mathrm{base}=0.262\).

## Recommendation

**Simplify to ordinary posterior KL distillation.** If an extra error-weighting term is wanted, use **individual marginal MSE**, not this matched-pair difference.

Reasons (algebra and this pairing, not a named method and not a training score):

1. \(L_\mathrm{pair}\) supplies no extra teacher information.
2. Under the actual matched construction, the structural probes of interest make \(L_\mathrm{pair}\) a **constant multiple** of \(L_\mathrm{mse}\) (4× for scope swap, 2× for one-sided independent). MSE already ranks those errors.
3. The only non-MSE piece is \(-2\mathrm{cov}(e_a,e_b)\), which **hides shared bias** — including the global stale/common-shift failures.
4. The train pairing is asymmetric (always singletons vs one pair), which collapses the shuffled-pair control on the independent updater.
5. On every probe, \(L_\mathrm{base}\) dominates \(\lambda L_\mathrm{pair}\) at \(\lambda=1\).

Do not redesign a different pair term in this assignment. Do not start training. A later learning comparison, if any, should treat ordinary KL (and KL+MSE) as the controls that this objective failed to beat on paper.

## Evidence table

| Implemented here | Not shown |
|---|---|
| Exact 4-component reset-group filter; h=0/h=1; brute-force agreement; prefix/suffix replay; permutation invariance | Any trained updater, adapter, or LLM |
| 128 enumerated pairs with topology-type holdout and an unobserved-related \(d\)-gap | A sufficient training set or sealed benchmark |
| Algebra of \(L_\mathrm{pair}\): r-cancel, expansion, common-bias failure, gradient check | Transfer to held-out language workflows |
| Synthetic \(\hat q\) probe rankings | Improvement from optimizing \(L_\mathrm{total}\) |
| Distinction from Bayesian Teaching (assistant-response SFT) and MemoPilot (reward-trained memory) as **named comparators, not implemented** | A novelty clearance or a full close-method comparison |

Future method, if any, must beat **ordinary joint KL** on structural holdouts, with a fixed utility head so action arithmetic is not the treatment. Generic learned memory and an action-only Bayesian-teacher baseline remain required later; these CPU controls do not replace them.
