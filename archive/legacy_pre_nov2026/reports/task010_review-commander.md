# Task 010 review and research decision

Author: Codex (commander), 15 September 2026. Reviewed worker results, training/evaluation code, raw predictions, identities and logs. Reran tests and independently scored saved predictions; no new research training, GPU work or paid calls.

## Decision

**Accept 010. Close the automatic numerical-training queue.** The longer fixed schedule helped the numeric model on both training and development data. This resolves the narrow diagnosis positively, but does not establish a publishable contribution or justify another language-model run.

This is not an application of 010's failure condition: all three seeds did beat transition-only on train and dev. Stopping here is a research-value decision. Repeatedly improving approximation of a known 16-state filter would still leave the main novelty and application questions unanswered.

## Verified result

- **183 tests passed**, including small numeric tests of recovery, identity rejection, explicit warmup and raw-prediction scoring.
- All three runs have 1,920 logged training updates and 30 development evaluations. All select step 1,920 by dev KL. Their first 192 batch sequences match the corresponding 009 runs.
- All recorded execution-source hashes match current files. The trainer and new configuration are now included. This is a check of the recorded list, not a claim that every transitive dependency is archived.
- Full keyed predicted joints are present for all six train/dev evaluations: **6,912 example predictions**. I independently computed KL, marginals and reference-utility regret from those joints and the saved targets. The audit records agreement and small float32 probability-normalization error.
- No structural/same-topology prediction files were produced for this diagnosis. The worker's 74.8-second fitting time, four threads and approximately 432 MB RSS are saved resource measurements, not independently retimed training.

Evidence: [task010_audit-commander.json](task010_audit-commander.json).

Mean KL across three seeds, lower is better:

| Predictor | Train | Development |
|---|---:|---:|
| 009 numeric MLP | 0.2854 | 0.2935 |
| 010 numeric MLP | **0.1048** | **0.1239** |
| Transition-only reference | 0.1202 | 0.1338 |
| Independent filter | **0.0168** | **0.0176** |

The roughly 58% reduction in development KL is real under this changed budget/schedule. The remaining development KL is about seven times the independent reference. Unrelated predictions still have appreciable error where the independent reference has essentially none. Better training/development fit supports the usefulness of this optimization change; it does not prove there is no overfitting or that further fitting would solve structural transfer. No new structural results were measured.

One report detail needs interpretation: logged `lr` is the scheduler rate **after** the optimizer update. Thus the reported step-one LR is the rate prepared for the next update, not the rate applied to step one. This does not alter the fixed 19-step warmup or the result. The 009 verification-count correction is correctly acknowledged in the 010 return.

## What we have, and what we do not

We have reusable research infrastructure: exact references, clean fixed-evidence data, meaningful statistical controls, checked decision scoring, a small learning pipeline, and negative findings that rule out some easy claims. These assets make a new hypothesis cheaper to test.

We do not have a distinct method, a demonstrated advantage over strong cheap alternatives, or evidence that the present numerical problem needs language-model capabilities. Structure, dynamics and an exact previous belief are supplied; the model is largely being asked to approximate a computation we already know how to perform. The current results do not establish learning dependency structure, maintaining uncertain knowledge over a long interaction, or transferring a semantic correction between tasks.

I should not treat another completed implementation brief as equivalent to progress toward a contribution. The current project is not on a demonstrated path to a method paper. More compute might improve its scores without changing that assessment.

## Literature constraint remains real

The focused primary-source recheck today confirms that generic memory revision is already occupied. WorldEvolver revises deployment-time context using episodic transitions, rules extracted from prediction–observation mismatches, and confidence-filtered foresight. Its current arXiv revision is v2. [WorldEvolver](https://arxiv.org/abs/2606.30639)

MemoPilot explicitly trains a memory updater to improve a frozen player across sequential interactions. A generic learned updater would therefore be insufficient as our novelty claim. [MemoPilot](https://arxiv.org/abs/2606.08656)

These are narrow primary-source checks, not a renewed exhaustive literature audit. The existing [novelty and transfer audit](../research-novelty-transfer.md) already warned against both a supplied-factor task and simply relabeling bandits as tools. That warning should now govern the decision.

## The only continuation I currently recommend considering

Return to the original question of **attributing a failure to a shared mechanism, then deciding how far a correction should propagate**. For example, a changed time-conversion convention affects reminders and date-filtered search, while contact lookup remains valid. A single surprising outcome may also be transient noise. The agent must infer the plausible cause and scope from permitted descriptions and experience, test competing explanations when necessary, and transfer a justified correction to an unobserved composition.

That example is a candidate research problem, not an implemented method or an assertion of novelty. A natural candidate would maintain competing mechanism explanations with evidence and uncertainty, then update dependent predictions selectively. Those ingredients overlap established methods; assembling them is not enough. We have not yet identified a defensible methodological difference.

Before issuing another worker implementation brief, the commander must make one concrete research case with:

1. **A specific method difference:** state what is represented, what evidence changes it, how update scope is inferred, and what existing method cannot obtain through the same interface and budget. Avoid merely adding a graph, confidence score or posterior loss.
2. **A comparison that could defeat the proposal:** strong generic memory revision, a simple statistical observer with inferred dependencies, and a supplied-structure reference. Match observable information and charge diagnostic calls.
3. **A discriminating example:** the same local failure warrants propagation in a shared-mechanism case and no propagation in an independent/transient case. The available evidence must make the distinction identifiable without leaking the answer.
4. **A reason for language and meaningful headroom:** require semantic dependency attribution or transformation composition, rather than a renamed arm or a numeric table. Measure first-attempt success on unobserved linked tasks and damage to unrelated tasks.

If this case cannot be made clearly against the literature, stop or choose a different research problem. Do not commission another benchmark or buy training time just to maintain momentum. There is no new worker experiment authorized by this review, and no need to procure compute now.
