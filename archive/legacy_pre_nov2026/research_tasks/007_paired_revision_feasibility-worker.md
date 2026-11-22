# Task 007 return

Author: Grok. CPU only. No APIs, no downloads, no training, no GPU. Earlier runs and commander plans were not edited.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **155 passed**.

Includes: h=0 identity / h=1 uniform; brute-force reset/redraw enumeration; prefix+suffix = replay; label permutation; independent unobserved \(d=0\) with nonzero drift; unobserved-related \(d\)-gap across partitions; r-cancel and common-bias \(L_\mathrm{pair}=0\); finite-difference gradients; 128-pair split/holdout/no \(q_\mathrm{current}\) leakage; synthetic probes are losses not training.

## Commands

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/paired_revision_feasibility.py
```

Runtime 4.23 s, peak RSS 24.6 MB. Spend **$0**.

## Files

- `llm_rl_engine/analysis/paired_revision.py`
- `scripts/paired_revision_feasibility.py`
- `tests/test_paired_revision.py`
- `reports/paired_revision_feasibility.md`
- `reports/paired_revision_feasibility/{manifest.json,solver_inputs.json,evaluator.json}`

## Result (descriptive)

The paired term equals \(\mathrm{mean}((e_a-e_b)^2)\) after \(r\) cancels. It is \(2 L_\mathrm{mse}-2\mathrm{cov}(e_a,e_b)\). It adds no labels. Common bias makes \(L_\mathrm{pair}=0\). On this pairing, scope-swap is exactly \(4 L_\mathrm{mse}\); the independent updater on train is exactly \(2 L_\mathrm{mse}\). Fixture: 128 enumerated pairs, topology-type holdout, 60/128 with a nonzero unobserved-related gap (max 0.167). Showcase T04: observing A updates B only under \(\{A,B\}\) (\(d_B=0.025\) vs 0).

## Recommendation

**Simplify to ordinary posterior KL.** If extra weighting is wanted, use individual marginal MSE, not this pair difference. Do not retain this exact objective as the method contribution. Do not redesign it here.

## Evidence table

| This assignment establishes | It cannot establish | Baseline a future method must beat |
|---|---|---|
| The proposed pair loss is a covariance reweight of existing marginal errors; r cancels; common bias is invisible | That optimizing it would help or hurt a model | Ordinary joint KL on the same examples |
| A 4-component exact filter and 128-pair topology holdout exist and pass CPU checks | Learning, transfer, or novelty versus Bayesian Teaching / MemoPilot | KL + individual MSE, then those published comparators |
| Same evidence / different partition can move an unobserved related component | A natural-world event rate (evidence is enumerated) | Transition-only \(r\) as the drift control |

## Stop

No API batch. No training plan launched. Commander decides whether a distinct contribution remains after this algebra. Paid-harness recovery gaps from the task-006 review were not rebuilt (out of scope).
