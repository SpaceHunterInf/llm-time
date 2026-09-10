# Task 004: sequential selective revision with persisted state

Plan author: Codex (commander). Implementation worker: Grok. Start after task 003's integrity checks pass. Read the task-002 review and the original research notes for motivation; do not spend an implementation session repeating the literature review.

## Question and decision gate

Gemini with thinking solved the single-event worksheet. Does it still update linked beliefs appropriately when hidden changes recur, and does its own carried state help or hinder compared with reconstructing from the same observations?

Use thinking enabled for the primary conditions. Do not create the main failure by disabling it. No training, unknown-graph task, or ToolSandbox integration yet. This remains a diagnostic test, not a claimed method paper.

## Phase A: exact sequential reference and fixed tapes, CPU only

Build a separate three-bit filtering module rather than altering the accepted single-event reference. Support an arbitrary named pair of linked components and an unrelated component. Each bit's reward rate is .2 or .8; initial independent high probability .99.

Before each observed outcome, apply a transition. In the shared condition, a Bernoulli(h) event resets both paired bits, independently redrawing each with probability .5. In the independent condition, each paired bit has an independent Bernoulli(h) reset. The unrelated bit is static. Then emit a Bernoulli observation at the publicly scheduled component. The event and latent bits are private. Multiple resets across the sequence are possible; the model is never told their actual times. h is a known public model parameter, initially .03.

Maintain the full normalized joint posterior over 8 states. Three marginal probabilities alone are NOT generally a sufficient recursive state under shared resets. Implement the 8x8 transition matrix and Bayesian observation update. Treat these probabilities as an external state representation, not access to internal neural beliefs.

Prepare eight deterministic development tapes:
- Three seed pairs (0, 1, 2), each shared versus independent: six tapes.
- Two known-stationary h=0 controls, marked as distinct stated models: two tapes.
- Counterbalance which labels are observed/linked/unrelated across pairs; paired shared/independent tapes use common indexed random uniforms for comparable latent/outcome generation, with separate event/redraw/emission streams. Do not claim identical evidence where realizations differ.
- Schedule relative to roles: initially cycle observed/linked/unrelated four times (12 observations). Then append `[observed]*8 + [linked]*2 + [unrelated]*2` twice, followed by `[observed]*8`. Total 44 observations. Probe after observations 12, 20, 32, 44. At the last three probes, recent evidence concerns only the observed member of the pair.
- Freeze seeds and tapes before any model outputs. Include all generated tapes, even ones without a useful change; report their reference trajectories and where diagnosis actually has positive value. Do not tune h or select favorable seeds after inspecting LLM errors.

Private no-reset stretches arising under h>0 remain evaluated with that same stated h; never reveal the realization by changing the public hazard to zero. The h=0 tapes are a separate known-stationary calibration control, not a substitute for those stretches.

At each probe, ask for the full joint posterior and a decision on the linked component: direct, safe, or perfect-rate diagnostic. Use safe=.65 and diagnostic cost=.04 initially. Derive marginal forecasts from the joint and score them separately from the action. The question specifies an isolated one-decision utility; evaluation choices do not alter the fixed evidence tape or reveal a diagnostic result to future checkpoints. Explain this limitation. Actual information-seeking interaction is a later extension.

## Phase B: matched observation conditions

All conditions receive the same complete public observation prefix, generative model, joint-state ordering, output schema, and generation allowance. Use the same think-enabled model/configuration. Keep each checkpoint's API request self-contained; explicitly serialize the state instead of relying on an opaque chat session.

1. **Reconstruct:** observation prefix and an empty previous-state slot.
2. **Self-state:** same prefix plus this condition's actual previous joint-state output, labelled with its checkpoint time. No private evaluator corrections.
3. **Reference-state:** same prefix plus the exact observation-limited posterior at the previous checkpoint, using the same format/time label. It must not include current or future observations. This is an explicit computational-assistance intervention, not an information-free baseline.

At the first checkpoint, the two state conditions receive the common initial joint prior, timestamped before observation 1. The reconstruct condition receives the same initial prior in its generative description. All conditions output the same schema. Explain that history-plus-state tests the effect of redundant external state; it does not test memory compression or a strict context bottleneck.

For a malformed/invalid self-state, retain the last valid timestamped state with an explicit invalid-update marker; do not inject a correct replacement. Record the fallback and report a valid-state-only sensitivity analysis alongside all checkpoints. Resume from the actual saved condition-specific previous state, and include its hash in the next request identity. Processing order must respect each tape/condition dependency.

Eight tapes × four checkpoints × three conditions = **96 planned decisions**, plus the predeclared retry reserve. At the old 2,000-input / 10,240-total-output caps, 192 attempts reserve about $18.27 at the previously recorded prices; this is illustrative, not a current promise. Check real prompt counts, provider controls, pricing, and complete-batch affordability before execution. If adequate output/context caps require more, reduce the preregistered number of tape pairs or report the full revised estimate BEFORE the batch, rather than truncating it midway.

## Analysis and interpretation

Primary: paired expected decision loss of self-state minus reconstruct, within the same tape/checkpoint. Report reference-state separately to identify whether accumulated state error is correctable. Report all conditions, every checkpoint, per-tape totals, and action-class denominators. Only eight tapes are independent development units; checkpoints and conditions are not independent samples. Use descriptive results initially.

Secondary: posterior MSE, expected reward Brier, linked/unrelated forecast errors against the exact reference, maximum unrelated error, joint normalization, unrecovered-state errors, own-forecast/action inconsistency, invalid outputs, tokens, and latency. Do not judge a return to an old bit value as intrinsically wrong: resets can redraw it. Forecast errors on unobserved components are measured against conditional predictions, not omniscient latent truth.

If self-state is worse and reference-state reliably repairs it, design a targeted state-replacement intervention on separate tapes before training. If all three solve the task, accept that result and move toward identifiable learned structure or a small language-composition task. If all three fail, first localize arithmetic/transition use versus decision valuation; do not assume memory is the cause. If errors occur only in the tiny joint-output schema, redesign representation before claiming an adaptation failure.

The original .02 utility-loss screening number is not a significance test. Report magnitude, frequency, and tape dependence. Any promising effect needs new seeds and role permutations before a learning proposal. Bayesian teaching, generic memory learning, and multi-turn inference already have close prior work; sequentiality alone is not the novelty.

## Acceptance and return

First deliver Phase A/B implementation and dry-run, not a paid batch: stochastic transition rows; h=0 identity; h=1 independent redraw; one-step agreement with task 001; independent brute-force two/three-step path check; label permutation; chunked filtering equals replay; shared correlation retained; no future/hidden fields in prompts; exact previous-state timestamps; fake-client sequential resume without duplicate calls; known cases requiring each of the three decisions as unit fixtures (not cherry-picked experimental tapes).

Provide a small reference-trajectory plot, frozen tape/config/request hashes, prompt samples, and complete-batch cost estimate. Return `research_tasks/004_sequential_revision-worker.md`. The commander will review the generated fixture semantics and requests before supplying the live invocation; this is a scientific review checkpoint, not a request for the user to reauthorize ordinary small API experiments.
