# Task 002 review and research decision

Author: Codex (commander). Reviewed 10 September 2026 at commit `9aff4fa`, using the locally retained runs. This is an offline audit, not a new model experiment.

## Verdict

Accept the completed pilot as descriptive evidence that thinking-enabled Gemini 3.5 Flash solves this particular supplied-structure, single-event grid. Do not train on this task. Proceed to a sequential test with actual persistent state after repairing experiment identity and audit metadata.

Read worker returns: `AUTO_RESEARCH-worker.md`, `research_tasks/002_selective_revision_pilot-worker.md`, and `research_tasks/002_live-worker.md`. Their dates/status differ: the first two predate the completed live runs. The commander's previous active-status section was also stale.

## Independently checked results

Reparsed all 147 saved provider response texts with Python JSON parsing. Reconstructed all prompt hashes. Independently recomputed A/B/C posteriors using closed-form likelihoods and the three action utilities, rather than trusting stored scores. All decision losses and posterior squared errors match. Summed completed-attempt costs and checked each against its ledger. Evidence: [task002_audit-commander.json](task002_audit-commander.json).

| Run | Valid responses | Decision errors | Mean decision loss | Logged estimated cost |
|---|---:|---:|---:|---:|
| Thinking smoke | 3/3 | 0 | 0 | $0.0880725 |
| Thinking grid | 72/72 | 0 | 0 | $1.6829400 |
| No-thinking grid | 72/72 | 7 | 0.00666723 | $0.1554060 |
| Total | 147/147 | | | $1.9264185 |

These are token-ledger estimates, not a checked billing statement. No retries occurred. Current test suite: `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **102 passed**.

The 72 jobs repeat 24 public cases across three treatments. Only two public cases call for diagnosis and one for the safe choice; the other 21 call for B. Always choosing B therefore scores 87.5% accuracy. The model's thinking-enabled performance on all six diagnostic-treatment jobs is more informative than its overall accuracy alone.

All seven no-thinking errors are on shared-change cases, across only three public cases. Six miss a useful diagnostic; the seventh chooses B where safe is optimal. The worker's sentence saying the missed cells are "exactly" those with optimal diagnosis is inaccurate; its table does include the safe-choice error.

On the two reference-treatment errors, reported probabilities agree with the reference to numerical precision, but actions lose .047107 and .011335 expected utility. These localize a decision-use failure under the no-thinking configuration. Other errors include substantially wrong probabilities, so the entire no-thinking failure is not purely diagnostic valuation. Absolute C forecast drift is zero in all 147 responses, not merely zero after averaging.

## Required engineering corrections before new experiments

1. **Resume identity omits effective live settings.** Thinking and no-thinking manifests/results have the same config hash and job keys. The source config hash ignores runtime overrides. An offline fake-client reproduction ran one job at thinking=0, resumed its directory at thinking=8192, and made zero new calls without rejecting the mismatch. The two actual grids used separate directories and their token logs distinguish them; their findings remain usable. Future resumes must reject incompatible effective configurations.
2. **The field called Brier is posterior MSE.** It computes mean `(p_model - p_reference)^2`, without the irreducible variance term. This is excess expected Brier loss for the binary latent-high event, not an observed-outcome Brier score. Preserve old fields/files and introduce accurately named metrics. For predictive reward probability r and forecast f, expected Brier is `r*(1-r) + (f-r)^2`.
3. **Token counting is a heuristic, not a hard bound.** `32 + ceil(characters/2)` happened to overestimate these prompts (actual 601–730 input tokens), but is not a guaranteed bound for arbitrary text. Count actual provider inputs before generation or use a justified bound. Reserve the whole remaining experiment and retries before starting.
4. **Persist provider completion metadata.** Logs retain text and usage, but not resolved model version or finish reason. Accept that all responses were valid JSON; do not claim independent verification of provider finish status. The client sends `max_output_tokens + thinking_budget` as the provider output limit; describing 2048 as an independently enforced visible-text cap is inaccurate.
5. **Resume must preserve scientific inputs.** A current-code hash is recorded but not checked on resume. Save exact requests and freeze the relevant code/prompt/case generation identity; sequential experiments also need the actual previous-state hash. Do not silently mix altered computations in one run.

Implementation brief: [003_harness_integrity-commander.md](../research_tasks/003_harness_integrity-commander.md). Existing runs and worker reports remain unchanged for provenance.

## What changes scientifically

The diagnostic gate did its job: there is no demonstrated thinking-enabled failure to correct on the original grid. Turning thinking off changes inference compute; success after turning it back on is not evidence for a learned revision method. The shared/independent numerical contrast is still useful as a reference test.

Next, retain supplied structure and thinking, but introduce repeated hidden transitions and explicitly carried predictive state. Compare rebuilding from complete observations, receiving the model's previous state plus the same observations, and receiving the exact previous state plus the same observations. This tests whether accumulated self-generated state becomes an obstacle or an aid. It reconnects the task to the original online-adaptation question.

Do not merely omit the dependency graph. With evidence only on A, the old shared/independent models have identical A marginals: there is no information identifying which graph is true. Learning an unknown graph requires informative calibration observations of linked components and a posterior over graph uncertainty. That is a later, separately identifiable experiment.

## Related-work update

The worker correctly adds a closer training precedent: [Qiu et al., Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models](https://arxiv.org/abs/2503.17523) already teaches Bayesian-model predictions and reports cross-task generalization. Its arXiv record confirms the Nature Communications DOI. [BayesBench](https://arxiv.org/abs/2606.30850) already distinguishes sequential latent inference from use in prediction. Their primary abstracts were rechecked in this review; detailed worker claims about particular domains, tables, and comparisons were not all independently reverified.

Consequently neither Bayesian teaching nor the inference/use split is our novelty claim. Repeated *partial* change, selective transfer/retention, matched state interventions, and eventual structural/language generalization remain the proposed contribution requirements. A sequential numeric success/failure alone will not establish novelty either.

Next scientific brief: [004_sequential_revision-commander.md](../research_tasks/004_sequential_revision-commander.md). Training and a large tool-workflow implementation remain deferred.
