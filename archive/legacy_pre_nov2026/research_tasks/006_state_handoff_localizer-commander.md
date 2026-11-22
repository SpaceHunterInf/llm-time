# Task 006: isolate state continuation and action use

Author: Codex (commander), 11 September 2026. Worker: Grok. Read `auto_research_rules.md` and [the task-005 review](../reports/task005_review-commander.md). Preserve all earlier runs and worker reports. Do not edit commander plans. No training, new environment, or broad model sweep.

## Question and boundary

Task 005 found that an exact previous posterior plus full history performs worse than reconstruction. Does removing already summarized observations restore performance? Can the model act correctly when given the exact current posterior?

This is one small diagnostic experiment, not a novel method demonstration. It tests a cheap alternative before investing in the proposed training intervention. Do not automatically continue experiments after the report.

## A. Finish correctness work before paid execution

1. Fix own-forecast regret using ALL action utilities under the emitted linked probability: linked=.2+.6p; safe=.65; diagnose=.61+.15p. Preserve reference regret as a separate quantity. Add tests where (a) the action is reference-optimal but own-forecast-suboptimal, and (b) the own-optimal action is worse than the chosen action under the reference. Reproduce corrected v2 own-error counts 2/10/7, 19 overall, 18 among the 31 reference errors, in a separate offline audit. Do not overwrite historical rows.
2. Validate raw JSON mass types before float conversion; reject booleans, strings, null, nonfinite, out-of-range, missing/extra keys, and out-of-tolerance sums with a schema error, not an API error. Retain the 1e-4 sum tolerance and small normalization logging.
3. Recover attempts as an ordered persisted history keyed by scientific job identity. Verify each exact request hash against its own saved request text; separately verify the base job identity and that any repair is the permitted suffix for the saved preceding validation error. Require identity fields. Never compare a valid repair request's hash directly to the base prompt hash.
4. Resume must use a saved valid result without generation, continue a saved invalid first attempt with only its remaining repair allowance, and stop after exhausted attempts. Charge each attempt once, retain a conservative reserve for genuinely unknown usage, and retain cumulative attempt counts. Treat missing ledger/attempt evidence as an explicit bookkeeping error or reconstruct it from persisted evidence, not as free usage.
5. Include shared provider/harness, extraction/parser, pricing implementation and all relevant prompt/model parameters in identity. Keep the existing current-tape and previous-state checks. Reuse shared recovery machinery where practical rather than maintaining diverging copies.
6. Test the actual runner with: full multi-checkpoint self-state resume; changed previous state; changed tape with unchanged ID; crash after saving an invalid first response; crash after saving a valid repair response; incomplete finish recovery; exhausted retries; unchanged resume with zero duplicates and reconciled costs. Inject actual interruption points. The task-005 review describes two confirmed failures. Test identity rejection BEFORE any new generation.

Run the full suite with uv. These are corrections to accounting/measurement; do not rerun the completed 96-job v2 grid.

## B. Fixed new tapes and four interfaces

Use the existing generator with its default rates/prior/hazard, observation schedule and deterministic role rotations. Fix these six tapes now:

- shared_s3, independent_s3, shared_s4, independent_s4;
- stable_s12 and stable_s13 (explicit known h=0 controls).

One target checkpoint: **after observation 32**. Previous-state timestamp: **after observation 20**. These seeds were selected before checking reference outcomes. Commander CPU inspection gives four diagnose-optimal and two linked-optimal cells, with no safe-optimal cell; preserve this set and report that limitation. Do not search for seeds with larger effects. These are new-seed diagnostic observations, not structural holdouts or a powered evaluation.

| Interface | Supplied evidence | Required computation |
|---|---|---|
| R: reconstruct | Original prior and observations 1..32; no previous state | Reconstruct current joint and choose action |
| F: full history plus previous reference | Exact joint after observation 20 and observations 1..32 | Same v2 redundant-state interface |
| S: previous reference plus suffix | Identical exact joint after observation 20, only observations 21..32 | Continue filtering from timestamp 20 |
| C: current reference | Exact joint after observation 32; no observation history | Copy the supplied current joint unchanged and choose action |

For R and F retain the corrected v2 task wording as closely as possible; changes should be only those required by the repaired code. S states that the initial computation state is the supplied timestamped posterior and there are exactly 12 subsequent transitions/observations; never restart from .99 or add an extra transition. C explicitly says the supplied distribution already conditions on all observations through the decision time and no further inference or transition is needed. This is posterior assistance, not revealed simulator state.

All interfaces share the full correct generative model, role semantics, safe/diagnostic utilities, keyed joint/action output, provider settings, and schema repair policy. R/F/S must have mathematically identical exact final posteriors. The full joint is necessary: do not replace it with independent marginals. Test continuation from the exact n20 joint over rows 21..32 against reconstruction to 1e-12 for all six tapes.

Use **two independent requests per tape/interface: 6 × 4 × 2 = 48 planned decisions**, each with at most one schema repair. The two repeats must have byte-identical model-facing prompts; repeat IDs belong only in job identity/logs. No outputs feed into later jobs. This deliberately isolates correct-state use before reintroducing errors from self-generated state.

Randomize execution order reproducibly (seed 6006); preserve repeat identity and resume without collapsing identical prompts into one job. Report within-input variability separately from across-tape differences. Two repeats and six tapes are not enough for a general memory claim.

## C. Preflight and execution

- Save a versioned manifest, all 24 unique base prompts, references, expected action denominators, identities, and full retry-reserved budget before the first generation. Inspect all four interfaces on one shared, one independent, and one stable tape. Validate suffix boundaries and absence of hidden latents/future observations.
- Use `gemini-3.5-flash`, temperature 0, configured thinking 8192 and visible output cap 2048, unless a provider requirement needs an explicitly recorded change. Recheck official pricing/settings and count tokens. At the last recorded rates with 4000 input and 10240 provider-output bounds, 96 maximum attempts reserve **$9.42336**; this is a planning estimate, not a fresh price verification. Budget the complete experiment including repairs before launch. A materially larger requirement should be reported concretely rather than causing a mid-run cutoff.
- A three-job schema smoke may be included in the same 48-job manifest; do not pay twice for those jobs. Changing scientific settings or prompts requires a new identity/version. Never resume a prior experiment directory after code/prompt changes.
- Once acceptance checks and smoke pass, complete this ONE 48-job experiment in a new `runs/selective_revision_state_handoff_v1/` directory. Ordinary small API experiments are already authorized; do not ask for an arbitrary $1 cutoff. Stop after the results report.

## D. Report and interpretation

Return `research_tasks/006_state_handoff_localizer-worker.md` and `reports/selective_revision_state_handoff_v1_live.md` with reproducible offline commands and:

- Tests, exact hashes/settings, per-attempt finishes, first/final validity, repairs, costs, reservations, and recovery behavior.
- All per-tape/per-repeat actions, joint-derived forecasts, reference regret and own-forecast regret, joint error, linked/unrelated forecast errors, and every optimal-action denominator. Include safe fallback in all-job loss and show valid-only sensitivity. Mark missing safe-optimal coverage.
- Pair comparisons within tape and average repeats before calculating tape-level effects. Show all six tape differences for F−R, S−F, S−R, and C−R. Do not treat 48 jobs as 48 independent environments or add more calls to chase significance.
- Repeated-input disagreement and sensitivity excluding each tape. Do not cherry-pick the repeat that supports an explanation.
- Free offline action replacement: maximize the known utility using each emitted joint. Report its reference regret alongside the actual action. This uses model predictions, not a hidden-state oracle, but remains an offline diagnostic.
- C joint-copy accuracy separately from C action accuracy; if copying fails, do not label its action errors pure valuation errors.

Conditional conclusions:
1. F underperforms R but S approaches R/C: a state-continuation interface is the first remedy. This supports an interface explanation; shorter prompts/computational load remain alternative explanations. It does not prove internal double counting or justify a novel training claim.
2. R/S have accurate forecasts but action errors; C or deterministic choice removes them: use the simple action calculation as the baseline. Do not train a revision module to fix action arithmetic.
3. S retains material forecast error while C works: a recursive updating limitation remains a candidate, requiring a precise comparison against ordinary Bayesian teaching before training.
4. F−R does not recur on these tapes or is smaller than repeat variability: downgrade the v2 state-harm hypothesis and close this branch.
5. C itself fails: separate schema/copying from action use; report the limitation without another automatic experiment.

## E. Relation to the proposed method

No novel revision learner has been implemented. The research candidate remains supervision of the scope of an update using matched evidence under different dependency structures, including preservation targets and structural holdouts. Generic posterior distillation and learned memory are established comparators. This assignment tests whether cheaper interface/action fixes remove the numerical failure; it is not an implementation of that training proposal.

End the worker report with a brief evidence table: what this experiment establishes, what it cannot establish, and which simplest baseline a future method must beat. Commander will decide whether to close the numerical diagnostic and write a concrete structural-transfer/training comparison. Do not build an additional benchmark or start training speculatively.
