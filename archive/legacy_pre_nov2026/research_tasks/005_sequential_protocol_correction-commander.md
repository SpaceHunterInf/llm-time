# Task 005: correct and repeat the sequential development experiment

Plan author: Codex (commander). 11 September 2026. Implementation worker: Grok. Read `auto_research_rules.md` and `reports/task004_review-commander.md` first. Preserve all task-004 v1 runs and worker reports. Do not edit commander plans or train a model.

## Why this task

The saved independent-reset requests omit the .5 redraw distribution, and the parser silently normalizes grossly invalid joints. The apparent memory penalty is dominated by one response. Correct the measurement before drawing a learning conclusion. This is one bounded protocol correction and development rerun, not an environment expansion.

## A. Correct the public model and state interface

Use one shared generative paragraph in EVERY condition:

> Initially each component is high with probability 0.99, independently. A reset is different from initialization: whenever a component resets, its new state is high with probability 0.5 and low with probability 0.5, independently of its previous state. Components reset together share only the event; their redrawn values are independent. The unrelated component never resets. Before every observed outcome, apply that time step's reset transition, then generate the outcome from the current rate.

Only the event-coupling paragraph differs between shared and independent structures. Retain h=.03 and the existing known-stationary h=0 controls. State the Bernoulli outcome law explicitly. State that the decision uses the state immediately after the final listed observation, with no extra transition before diagnosis or reward. Say “you pay a cost of .04,” not “diagnosis pays .04.” Define the safe utility .65 and diagnostic reveal of the latent RATE as before.

Use a keyed joint distribution to eliminate integer/positional ordering ambiguity:

```json
{
  "joint": {
    "000": 0.0, "001": 0.0, "010": 0.0, "011": 0.0,
    "100": 0.0, "101": 0.0, "110": 0.0, "111": 1.0
  },
  "action": "choose_linked"
}
```

Each key's characters are, left to right, observed, linked, unrelated; 1 means high and 0 low. Named labels may rotate, but key semantics never do. The same keyed format is used for previous_state. Do not use “least significant bit” or an implicit array order. Derive all marginal probabilities and forecasts in the evaluator from this joint; do not ask for redundant marginal/mean fields in v2. Keep exact references internally convertible to/from the existing eight-state order.

Explain temporal use of the state identically in all conditions: previous_state summarizes observations through its timestamp and is not independent evidence. A solver may reconstruct from the original prior and all observations, or start from the previous state and process only the subsequent observations. It must not apply already summarized observations twice. The state is an estimate, not additional observations. Preserve the three existing treatments, the actual self-state feedback, and previous-time reference-state (no current posterior leakage).

## B. Validate and preserve outputs honestly

- Require exactly all eight joint keys, finite non-boolean probabilities in [0,1], and sum within **1e-4** of 1. Reject larger deviations as schema failures. Within tolerance, normalization for roundoff is allowed, but log the original sum, original masses, normalized masses, and correction magnitude.
- Log first-attempt schema success, successful retry count, final schema success, decision loss with the predeclared safe fallback, and a valid-only sensitivity analysis. A failed state update retains the last valid timestamped state with an explicit invalid marker; it must not be replaced with evaluator truth.
- Retry at most once, with a neutral schema-error description and the same task. Never include reference answers in repair instructions. Save exact requests and responses as already supported.
- Add an offline v1 audit mode that reports raw sum violations and joint/marginal disagreements without rewriting old accepted scores. Match the commander's 15 sum deviations >1e-4 and 13 marginal disagreements >.001 on the retained v1 final outputs.

## C. Repair the sequential identity/recovery path

- Hash actual tape model parameters, role-label mapping, observation schedule/outcomes, probe times, schema/prompt version, effective generation settings, relevant shared harness/parser implementation, and exact previous-state content. Tape IDs alone are insufficient.
- Use a logical slot only to locate a completed job; verify the full expected request/state/scientific identity before reusing it. A mismatch must raise before a new generation, not silently skip.
- Construct initial self-state identity identically on first run and resume. Verify exact job/request/state identity when recovering a saved response. Apply the same schema and finish-status checks during recovery as during fresh scoring. Preserve cumulative cost and retry allowance across interruptions; never reissue a saved complete response or treat a previously charged attempt as free.
- Add real `run_sequential` regression tests for changed-tape-same-ID rejection, changed previous-state rejection, unchanged multi-checkpoint resume with no duplicates, and crash after response-save but before score-save. Cover a schema-failed first attempt followed by a valid retry. A simulated loop that bypasses the live runner is not sufficient.

## D. Dry-run, smoke, and corrected grid

Create versioned `sequential_v2` artifacts in new directories. Reuse all eight original tapes and four checkpoints; do not select seeds using errors or drop inconvenient tapes. The corrected comparison is 96 planned decisions across reconstruct/self_state/reference_state. It is development material, not a fresh holdout or an isolated prompt-ablation claim.

Before any paid run:
1. All existing tests and new protocol/identity tests pass.
2. Validate shared and independent base prompts contain the SAME complete redraw law. Add semantic parameter checks and manually inspect at least one standalone prompt of each structure at each checkpoint, rather than only checking that a particular phrase occurs.
3. Recheck official provider pricing/settings and count full requests. Reserve the entire grid plus retries, with an allowance for the state-format size; do not start a budget that cannot cover completion. Existing 4,000 input / 10,240 provider-output bounds imply about $18.85 for 192 attempts at the previously recorded rates. Keep the total corrected experiment, including smoke, near the already used ~$20 scale. If an adequate budget is materially larger, report the concrete estimate before launching instead of cutting off midway.
4. Use thinking-enabled `gemini-3.5-flash`, temperature 0, with the same 8192 configured thinking allowance unless provider support requires an explicitly reported change. Inspect a short schema smoke containing shared, independent, and stable inputs. Include those smoke jobs as the first jobs of the versioned run if settings remain unchanged, so they are not paid twice. Do not claim the smoke measures a memory effect.

After checks, complete ONE corrected development grid. This remains within the user's authorization for ordinary small API experiments; do not ask for a new $1-style cutoff. If smoke reveals a schema/protocol defect, fix it and restart under a new version rather than merging incompatible requests. Do not launch further grids or training automatically.

## E. Required result report and decision gates

Return `research_tasks/005_sequential_protocol_correction-worker.md` and `reports/selective_revision_sequential_v2_live.md`, with:
- Complete parameter/request/identity hashes, tests actually run, ledger costs, finish metadata, first/final attempt validity, retries, and remaining limitations.
- Per-tape and per-condition loss, all action-class denominators, linked/unrelated forecast errors and worst cases. Unrelated error is relative to the exact conditional reference, not to initial .99 after new evidence about the unrelated component.
- Own-forecast-optimal versus reference-optimal decisions. Do not label every missed diagnostic as a valuation error when the model's own forecast makes skipping it rational.
- Self-minus-reconstruct effects by tape and sensitivity excluding the largest single cell. n=12 self/reference prompts are identical initially; identify these as repeat-input checks, not different memory evidence.
- A clear statement that v1→v2 differences combine protocol/schema corrections and do not establish a learned intervention effect.

Stop after the report. Commander chooses the next branch:
1. Corrected task solved → no training; design identifiable structure learning or a language-composition test.
2. Persistent forecast error with valid schema → exact CURRENT posterior assistance on matched checkpoints to localize inference versus action use, then new-seed validation.
3. Persistent self-state penalty → new seeds/role rotations and controlled state replacement, checking whether the effect survives individual outliers and repeated identical requests.
4. Mostly formatting/state-encoding failures → simplify the representation; do not train under the label of selective revision yet.
