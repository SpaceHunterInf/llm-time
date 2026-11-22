# Task 002: build the fixed-history pilot, then stop at dry-run

Plan author: Codex (commander). Implementation assignment for Grok.

Read `auto_research_rules.md`, `AUTO_RESEARCH-commander.md`, and the accepted task-001 reference/report. This brief is queued: task 001 must pass research-lead review first. Implement the harness without running a paid experiment. The research lead supplies the actual model ID and budget for execution after checking availability/pricing.

Planning settings checked by the lead on 9 September 2026: retain `gemini-3.5-flash` for continuity. Google's [official pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash) lists standard text input at US$1.50/million tokens and output, including thinking, at US$9.00/million tokens. Recheck at execution time. The user authorized ordinary small experiments and rejected the lead's arbitrary US$1 cutoff. Preflight the full batch including retries and reserve enough to finish. For illustration, 72 primary calls plus up to 72 retries, each capped at 2,000 input and 2,048 total billed output tokens, cost at most US$3.086208 at these rates. This is conditional arithmetic, not a claim that the provider cap or prompt bound is already implemented. If thinking cannot be bounded as part of total billed output, this example is not a valid cap. Choose output allowance from the smoke test so truncation does not manufacture reasoning failures; budget accordingly before the complete experiment.

## Deliverables

Add a small module for public case construction/prompting/response validation, a CLI under `scripts/`, focused scientific-integrity and budget/resume tests, a versioned pilot config, and a dry-run report. Reuse the task-001 observer and existing Gemini client where safe. Do not change historical benchmark behavior. No new heavy dependencies, training, model downloads, DeepSeek use, or email. Use uv. No commits or subagents.

## Pilot design

Start with 24 public cases (8 per shared/independent/stable structure): pair each of four A evidence batches (0, 2, 4, 5 successes out of 5) with diagnostic costs .04 and .20. Use q=.99, h=.1, low=.2, high=.8, safe=.65 as in the worked example. Stable means explicitly stated h=0. A/B reset values are independent conditional on a shared event. C does not reset. No fresh evidence on B or C.

Create three input treatments per case, for 72 initial calls:
1. Raw A evidence with prior and full generative description.
2. Verified A success/failure counts with otherwise identical information. Since there is no within-batch change, counts preserve the likelihood; this equivalence would not hold for arbitrary changing histories.
3. Counts plus the observation-limited reference probabilities for A/B/C being high. Supply these as trusted predictions, not as a recommended action. This is an explicit privileged-computation intervention using the same public evidence, not an ordinary baseline.

All treatments request the same JSON: probabilities that each rate is high, predictive expected reward for each component, and one choice from choose_B/choose_safe/diagnose_B. Define each numerical field unambiguously. No rationale required. A diagnose choice commits to using the revealed latent rate optimally in this first probe; we are testing valuation of information, not interpretation of diagnostic results yet. State this limitation visibly.

Pair the same public case across treatments. Counts/raw ordering, prior, decision stakes, and question wording must not change across treatments. Use neutral case IDs; no hidden-event labels or target answers in raw/counts prompts. Give supplied structure faithfully; do not tell the agent an event actually occurred. Use a shared template for reset semantics to prevent prompt differences from creating different probability models.

This first grid holds rates/roles fixed and is development material. It is not final structural-transfer evidence. Subsequent validation must vary q/h/stakes, counterbalance labels/change direction, and include longer histories. When sampling no-event control realizations under h>0 later, keep the same stated h as changed realizations; never leak the private event label by telling no-event agents h=0.

## Scoring

Primary: expected utility of chosen action under the exact permitted-information posterior; decision loss = optimal reference expected utility minus chosen-action utility. Include diagnostic cost exactly once. Invalid/missing responses receive a separately reported failure count and a predeclared safe-action fallback for intention-to-treat utility; never silently drop them. Report both attempted cases and valid cases. Distinguish API failure, parse failure, and scientific decision error.

Secondary: component-wise probability error, expected proper score under the reference predictive distribution, linked-B decision accuracy, unrelated-C forecast drift from .794, and frequency of buying overpriced diagnostics. Forecasts and actions may disagree: score them separately. Do not use emitted confidence as internal-belief evidence.

Report every case, treatment, decision value, and paired difference. Summaries are descriptive macro-averages over this chosen grid, not estimated natural-world return. Do not compute confidence intervals pretending repeated treatments/cost variants are independent episodes. No claim of improved performance without actual model outputs.

## Run integrity and budget requirements

- Dry-run is default and must work without reading `.env`, constructing an API client, or network access. Save public prompts, separate evaluator references, hashes of config/prompt/case, requested model ID, and code revision plus dirty-file hashes. Never send the evaluator object to the provider.
- Live mode requires explicit model, max calls (including retries), input/output token limits, verified pricing, and maximum USD spend. Preflight that the configured budget covers the full requested experiment and retry reserve; reject underfunded configurations before making any calls. Reject missing/invalid budget inputs. Check prompt-size bounds before a call. Bound generated output via provider configuration; account for thinking/output charges where applicable. If a reliable cost bound cannot be implemented, stop with a clear limitation instead of promising a hard dollar ceiling.
- Start live execution with a smoke of at most 3 calls. At most one in-flight request until caps/retries work. Stop on authentication/quota errors; no silent model substitution or recursive repair loop.
- Log and flush each attempt (including failed parses/retries), actual model metadata, input/output usage, cost estimate, latency, and exception category. Retry at most once and only within the total cap; do not give reference answers in repair prompts. No keys/headers in logs.
- Resume by public-case + treatment + model + prompt/config hashes; preserve completed attempts and cumulative spending across resume. Never overwrite old runs or reset the spend counter on restart. Unknown usage after a interrupted request is conservatively reserved against the cap, not assumed free.
- Frozen-call responses become evidence only after the lead inspects representative prompts and outputs.

## Required checks

Fake-client end-to-end run with known reference answers; raw/count likelihood equivalence; supplied predictions give enough information to compute diagnostic utility; no hidden evaluator fields in ordinary prompts; invalid JSON/probabilities rejected; API failure remains visible; strict pre-call cap; retry counted; interrupted attempt reservation; resume idempotence and cumulative cap; deterministic dry-run artifact hashes.

Finish with the dry-run command, files, tests actually executed, total planned calls, conservative cost-bound mechanism, and limitations. Do not run Gemini until the lead supplies execution settings in a follow-up.
