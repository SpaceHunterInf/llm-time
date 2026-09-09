# Task 001: selective-revision reference and offline fixtures

Plan author: Codex (commander). Implementation assignment for Grok.

Read `auto_research_rules.md` first. You are the implementation agent; the research lead reviews your diff and results. Implement this bounded task now, without expanding the research programme.

## Scope and constraints

CPU only, Python standard library for the new reference, use uv. No LLM/API experiments, training, model downloads, email, web research, sudo, git commits, or subagents. Never read or print `.env` or credentials. Do not alter existing engine behaviour, existing reports, dependencies, or the user's research notes. The lead is setting up the uv environment; do not install dependencies yourself. Use `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync ...` once available; report a blocked test honestly if needed.

Create only:
- `llm_rl_engine/analysis/selective_revision.py`
- `scripts/selective_revision_sanity.py`
- `tests/test_selective_revision.py`
- `reports/selective_revision_reference.json`
- `reports/selective_revision_reference.md`

## Scientific question

Can identical evidence on A rationally change a decision on unobserved B differently when A/B share a reset event versus when their reset events are independent, while leaving C intact? This is a mathematical diagnostic, not evidence of LLM capability or training benefit.

## Exact generative model

Three binary latent rates for A/B/C, each independently high with prior probability q=0.99; low=0.2, high=0.8. There is ONE possible event before a batch of observations, with hazard h=0.1. No transitions inside the batch.

- shared: a Bernoulli(h) event simultaneously resets A and B. On reset, each affected latent rate is independently redrawn high/low with probability 0.5. Sharing an event does NOT mean sharing the redrawn value. C cannot reset in this interval.
- independent: independent Bernoulli(h) events reset A and B with the same redraw rule. C cannot reset.
- stable: no component can reset (effective h=0); this is explicitly a different stated generative model, not a hidden no-event label.

Observations are Bernoulli draws conditional on the post-event rate. The primary worked case is five failures on A, with no observations on B/C. The prior q summarizes earlier evidence; do not invent raw prehistory to justify it.

Implement exact finite-state enumeration of events and latent rates, likelihood weighting, normalization, predictive means for each component, and posterior event probabilities. Accept observations by component so future diagnostic observations are possible. Validate inputs and handle zero-likelihood observations explicitly. No simulator ground truth or future outcome input belongs in the observer.

## Decision and costs

One choice on B, versus a known alternative of expected utility s=0.65. Optional perfect diagnostic reveals B's latent RATE (not a future reward) for cost c=0.04, after which the agent selects the higher expected-value option. The cost is charged exactly once, even when the safe option is chosen afterward.

With z=P(B high | observations), report:
- choose_B = z*high + (1-z)*low
- choose_safe = s
- diagnose_B = z*max(high,s) + (1-z)*max(low,s) - c
- optimal action with documented deterministic tie handling
- gross value of information and net diagnostic advantage against best direct action

Reference values approximately: shared A=.209, B=.548, C=.794, buy diagnostic; independent A=.209, B=.765, C=.794, choose B. Derive exact values; do not hardcode them into implementation. A reset may redraw the same state.

## Fixtures and output

Provide a CLI (`uv run python scripts/selective_revision_sanity.py`) that regenerates both reports deterministically, without API/network calls. Include the worked example and a small diagnostic grid: A batch length 5 with success counts 0..5, shared/independent/stable structures, diagnostic costs 0/.04/.20. Clearly call this an enumerated diagnostic grid, not independent sampled episodes or a performance benchmark. Raw observation lists and success/failure counts must encode the same evidence. Include public case inputs separately from reference targets; do not accidentally imply reference targets are ordinary agent inputs. Include exact expected action values, recommended decisions, posterior event probabilities, and forecasts. JSON must contain no NaN/Infinity. Markdown should explain assumptions, interpretation, and reproduction command.

## Acceptance tests (scientific invariants)

1. Normalization, probabilities in range, analytical one-component posterior agreement using a separately derived formula.
2. A marginals identical in shared/independent given only A evidence; B differs; C remains at its prior predictive.
3. h=0 gives shared=independent=stable. h=1 gives identical shared/independent latent distributions because redraws are independent.
4. No evidence gives equal component marginals in shared/independent (though their joints can differ).
5. Worked-example actions differ as specified; a sufficiently high diagnostic cost suppresses diagnosis; perfect information's gross value is nonnegative; verify branch costs and ties.
6. A label-permuted mathematical equivalent preserves results if the API supports label permutations; otherwise document this as follow-up, not tested.
7. Impossible inputs and invalid probabilities/costs are rejected; extreme permitted priors/hazards do not silently produce invalid values.

Run focused tests and regenerate reports. Finish with files changed, commands and actual results, any implementation limitations. Do not claim model experiments were run. Stop at this deliverable; the lead will specify the Gemini pilot separately.
