# Task 008 review: verified preflight, learning decision still open

Author: Codex (commander), 14 September 2026. Reviewed worker feedback, data, model helpers, training/preflight scripts, configuration, checkpoints and tests. No API calls, downloads or optimizer updates in this review. Cached pretrained weights were loaded for read-only checks.

## Decision

**Accept the data and basic pretrained-model verification. The external learning package is incomplete.** Task 008 has not produced a learning or generalization result. In particular, KL 0.760 on four training rows is a forward-path check with an untrained head, not held-out performance.

Do one bounded ordinary-KL learning pilot after [task 009](../research_tasks/009_learning_pilot_release-commander.md) completes the actual training/evaluation path. Keep the six planned language-model runs and add one small numeric MLP control with the same three seeds. This control asks whether text processing and a pretrained language backbone add anything to this fully numerical problem. No new loss, paid prompt grid, larger backbone or full local training.

## Independently verified

- **164 tests passed** using `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`.
- All **2,816** previous/current posterior targets reproduce with an independent product-of-group transition kernel. Maximum joint error is 3.33e-16; log-likelihood error is 5.33e-15.
- Saved hashes and rendered inputs agree; all 2,816 texts are unique and no canonical evidence family crosses splits. Current targets, historical prefixes and bookkeeping labels are absent from model text.
- All inputs tokenize below the 1,024-token limit: observed range **423–558**, checked with the cached pinned tokenizer without a network forward pass on holdouts.
- Worker records show three executed verification updates per language-model control, unchanged frozen parameters and an exact same-minibatch continuation after helper-level checkpoint reload. Read-only inspection confirms all 56 saved LoRA B tensors are nonzero: the adapters did update.
- The cached checkpoint is Qwen3-0.6B-Base at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Worker verification used CPU float32 and four threads. Its approximately 22 GB peak RSS included multiple model copies and is not a measured single-run GPU requirement.

Evidence: [task008_audit-commander.json](task008_audit-commander.json). The test suite does not establish that the complete fit/resume/evaluation workflow works.

## What must change before external compute

1. **The real training command cannot resume.** `fit` has no resume argument or checkpoint-loading path. Checkpoints omit the scheduler, private shuffle RNG, current permutation/cursor and selection state. Metrics and the manifest are written only after training ends; only the best epoch checkpoint is saved. The successful preflight helper replay is narrower than interruption recovery in this loop.
2. **There is no held-out evaluator.** The CLI implements `probe` and `fit`; training records train/dev KL only. It cannot yet deliver the transfer, selective forecast or decision results that motivate the experiment.
3. **Initial heads are not matched.** With seed 0 reset before each construction, head weight maximum difference is **0.0622021** and bias difference **0.0453235**. Adapter initialization consumes RNG before the head is created. Consequently the worker's two initial losses are not a controlled comparison. Initialize/copy the same head explicitly and verify initial predictions match.
4. **The YAML is documentary, not executable configuration.** Neither command reads it. The manifest lacks full code/tokenizer/configuration and target identities; checkpoint loading also needs strict identity and tensor checks. A run must declare these before its first update.
5. **The proposed CUDA installation is inconsistent with the project lock.** Torch is sourced from the CPU index. Installing a CUDA wheel manually and then using ordinary `uv run --group train` can restore the locked CPU package. Prepare a consistent external uv environment/profile and check its actual backend. See uv's [PyTorch integration guidance](https://docs.astral.sh/uv/guides/integration/pytorch/) and [automatic synchronization behavior](https://docs.astral.sh/uv/concepts/projects/sync/). No driver changes here.
6. **The action query always selects an observed component.** In all 2,816 rows it is the first suffix component. Those action metrics cannot establish transfer to an unobserved related component. Evaluate all four components from the existing predicted joint; no new data are needed.
7. **The independent-component reference uses different information.** It replays the hidden historical prefix under a different model; the learner gets the supplied previous joint and suffix. Start all comparison filters from that same supplied state, explicitly specifying any marginal-product projection. Preserve the old number as a historical reference with a different interface.
8. **The renderer leaves transition timing implicit.** It lists hazards and says no extra transition at the end, but does not explicitly state independent group events occur before every subsequent observation. Add that sentence through a versioned rendering correction, preserving the sampled records and targets.

Support microbatches with accumulation to the fixed effective batch of 32 so memory constraints do not silently change the study. Measure training, evaluation and checkpoint costs on the eventual external machine before estimating the full job.

## What this pilot can tell us

The question is narrow: **can a learned updater use a supplied belief and dependency structure to transfer evidence to unobserved related components while preserving unrelated forecasts?** Training uses singleton and one-pair structures; structural tests use two pairs and triples. Structure is given, previous beliefs are exact, the state space has only 16 entries, and there is no recursive learned-state rollout. This does not test discovering mechanisms, persistent memory, language understanding or deployment adaptation.

Compare head-only, LoRA plus head, and a small numeric MLP, all with ordinary teacher-to-student joint KL. Include exact filtering and stale/transition-only/same-information independent references. Check joint KL, related and unrelated marginal errors, and priced decision regret. Separate h=0 from h=.03: group structure has no dynamical effect at h=0, so those examples cannot establish structural transfer. h=.03 permits resets; it is not a label that a reset actually happened.

Coverage is limited. Structural holdout contains 286 unobserved-related and 237 unrelated component instances across 256 records before hazard stratification. Safe is optimal in only 8 of its 1,024 component decisions. Report denominators and paired per-example results; do not manufacture an action-balanced population after seeing performance. This is a directional pilot, not a publication-sized significance claim.

## Where the project should go

The long-term target remains selective revision of reusable knowledge: evidence changes predictions for linked tasks, while unrelated tasks retain their performance. The present known-model filter is a controlled learning test on the way to that target, not the finished research contribution. The existing [novelty audit](../research-novelty-transfer.md) already requires this distinction. This review does not constitute a new exhaustive literature search.

There is **still no defensible novel method**. Ordinary posterior distillation is the current baseline; the paired loss remains retired. I recommend a limited investment to obtain actual learning evidence, followed by a decision:

| Pilot outcome | Research decision |
|---|---|
| Numeric MLP matches or exceeds the language models | Stop treating approximation of this supplied numerical filter as an LLM method contribution. Any continuation must introduce a justified need for language-based dependency attribution, not more training on this table. |
| Ordinary LoRA transfers well and preserves unrelated forecasts | A special revision objective is not motivated by this test. Use it as a baseline; consider a small language-based transfer test only with a distinct hypothesis and the existing literature comparisons. |
| Training and same-topology fit are good, but structural transfer fails consistently | Investigate compositional representation or update scope as a candidate inductive bias, first excluding topology coverage, serialization and capacity confounds. Failure alone neither establishes novelty nor licenses a new-method claim. |
| Training/development fit is poor | Investigate optimization or representation once with a bounded correction. Do not interpret failed fitting as a reasoning limit or automatically scale compute. |
| Effects differ across seeds or role coverage is too thin | Call the pilot inconclusive. Do not choose a preferred seed or silently expand runs until the desired result appears. |

Task 009 is a finite completion of this release plus the cheap numeric control. It should end with verified commands and a reviewable external job. We should then obtain the first learning result, rather than keep inventing objectives or adding diagnostic grids.
