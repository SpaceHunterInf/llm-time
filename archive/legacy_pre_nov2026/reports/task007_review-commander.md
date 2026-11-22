# Task 007 review: retire the paired objective, prepare the ordinary baseline

Author: Codex (commander), 11 September 2026. Reviewed worker implementation, tests, fixtures, report and recommendation. No API calls, downloads, optimization or GPU use in this review.

## Decision

**Retire the paired update-difference loss as our proposed contribution.** I proposed it; the audit does not give us a compelling reason to invest in it. Keep the exact four-component observer and the fixture examples as development tools. Prepare an ordinary posterior-distillation baseline and its actual execution requirements before requesting external training compute.

Next: [008_posterior_baseline_preflight-commander.md](../research_tasks/008_posterior_baseline_preflight-commander.md). This is a clean data/model pipeline and bounded local verification, not a full training run or another paid prompt grid. The scientific question is whether ordinary updating supervision transfers to unseen reset-group structures. It is not yet a novel method claim.

## What verified

- **155 tests passed** with `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`.
- Recreated the artifacts under `/tmp/task007-commander-reproduction` with the script's `--output-dir` flag. Solver/evaluator records, coverage, algebra, probes and showcase exactly match the saved artifacts. Runtime approximately 4.16 seconds, peak RSS approximately 25 MB.
- Independently checked transition matrices using a product of per-group kernels: each group's contribution is `(1-h) * indicator(source group == destination group) + h / 2**group_size`. This does not use the worker's redraw-enumeration helper. Agreement across every supported partition and h in {0,.03,.4,1}: maximum error 1.12e-16.
- Independently filtered all 256 example records in a different state ordering. Previous/current joints, transition-only marginals, evidence-driven updates and log likelihoods match to below 1e-12.
- 128 pairs, 96/16/16 split, and the reported 60 nonzero unobserved-related contrasts (maximum .167066) reproduce. No current posterior occurs in the declared solver records.

Audit artifact: [task007_audit-commander.json](task007_audit-commander.json). Preserve the task-007 artifacts as feasibility fixtures rather than silently turning them into training data.

## Correct algebra, limited conclusion

With marginal errors e_a and e_b:

```
L_pair = mean((e_a - e_b)^2)
       = 2 * L_marginal_MSE - 2 * mean(e_a * e_b).
```

The last term is an **uncentered cross-product**, not generally a covariance. The transition-only reference r cancels. Shared bias is invisible to the paired term; opposite errors receive extra weight. The joint KL still penalizes the shared errors, so adding this nonnegative term does not itself prove that the total objective prefers a wrong posterior over the teacher.

The worker's 4× and 2× MSE identities are correct for its specific scope-swap and one-sided-error probes. They are not a general equivalence theorem. Independent counterexample: teacher marginals .5, student marginals .6 on both sides versus .6/.4. Both have marginal MSE .01 and identical joint KL, but paired loss is 0 versus .04. The paired objective does change the optimization problem.

Likewise, no new labels does not imply no possible inductive-bias benefit. Loss magnitudes at lambda=1 do not establish optimization quality, and synthetic probe rankings do not establish that KL learns better. The justified recommendation is **use the simpler established baseline first; do not present this loss as a defensible method contribution**. We have not empirically refuted every possible use of pair regularization.

## Fixtures are not ready for training

1. `solver_input` returns the entire solver record, including `pair_id`, `side`, `family_id`, `split` and `topology_type`. Pair IDs can expose both partitions. These are bookkeeping/partner cues, not permitted model inputs. This is not direct current-posterior leakage, but it makes a future model experiment vulnerable to shortcuts.
2. Solver records include the full prefix as well as q_previous and suffix. That does not implement the intended previous-state-plus-new-evidence interface. Remove the prefix from the model-facing renderer; retain it separately for reference auditing.
3. The 192 training-side records contain only **112 distinct individual inputs** after dropping bookkeeping; development has 20/32 and test 24/32. Repeated singleton examples were appropriate for the paired feasibility calculation, but would silently reweight ordinary training if flattened unchanged.
4. T00/train and X00/test share the identical prefix/suffix sequence at different hazards. Topology-type holdout remains real, but checking family IDs alone does not establish independent evidence families. Future split audits must canonicalize actual observations, including renaming, rather than trust assigned names or hash the hazard into the family identity.
5. The worker's supposed independent `brute_transition_row` calls the same `_accumulate_redraws` helper as the matrix builder. The commander's independent product-kernel check resolves the numerical concern for this review; add that genuinely independent check before reusing the filter as a training teacher.

## Baseline and resource decision

Use ordinary joint posterior KL with a normalized prediction head; freeze the action utility calculation so JSON formatting and action arithmetic cannot masquerade as learning to revise. Compare a frozen language backbone with a trained head against the same head plus low-rank backbone adaptation. A random untrained head is only an initialization check, not a meaningful frozen-agent baseline.

The selected small preflight backbone is [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), whose official model card identifies it as a pretrained 0.6B causal model with Apache-2.0 licensing. It is a deliberate small baseline, not a claim about the strongest available model. The worker must pin its revision and compatible packages.

Current environment inspection: numpy is installed; torch, transformers and peft are absent. `nvidia-smi` cannot communicate with the driver. Do not repair the departmental driver or request sudo as part of this task. CPU forward/backward/save-load checks remain possible. Actual GPU throughput is unknown; do not extrapolate a confident GPU-hour estimate from CPU timing.

Ordinary Bayesian teaching and transfer already have precedent in [Qiu et al.](https://arxiv.org/html/2503.17523v3). Their main implementation imitates Bayesian assistant responses; our full-joint/readout experiment is a baseline adaptation, not an exact paper reproduction. There is currently **no established new-method contribution**. A later result must identify something beyond successful distillation in a tiny known model before a paper claim or substantial training investment is warranted.
