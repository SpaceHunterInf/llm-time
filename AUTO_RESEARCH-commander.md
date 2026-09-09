# Active research plan

Plan author: Codex (commander). Commander-authored plan files use the `-commander.md` suffix; this convention does not apply to code or generated result reports.

Updated 9 September 2026. Start here in future research sessions, then read `auto_research_rules.md`. Earlier proposals and reports are preserved as historical documents. This file selects the immediate work under the current compute constraints; it does not claim the new method already works.

## Direction

Study **selective revision**: use new evidence to update predictions about affected or linked mechanisms, preserve unrelated predictions, and improve decisions after charging for diagnosis. The first question is whether a frozen model has a correctable failure on this task. Training is conditional on that finding.

The earlier eight-week proposal is too large as an initial commitment. Start with one tractable task, one Gemini model, and a few short, fixed-history decisions. Defer training, RL, larger model sweeps, and workflow integration until the diagnostic results justify them. Local GPU use is restricted to development/verification by `auto_research_rules.md`; serious training requires a working script, a measured resource estimate, and the user's external compute arrangements.

This sequence builds on the supplied research notes:

- [research-plan.md](research-plan.md): retains selective revision as the main research bet and makes its diagnostic gate the immediate commitment.
- [research-cognitive-mechanisms.md](research-cognitive-mechanisms.md): motivates shared versus independent changes, an unrelated-component control, and observation-limited reference predictions.
- [research-training-methods.md](research-training-methods.md): retains matched action-only/predictive training and generic-memory comparisons if training becomes justified. Training is deferred, not presumed unnecessary for the eventual contribution.
- [research-novelty-transfer.md](research-novelty-transfer.md): retains structural holdouts and language-rich transfer as requirements for broader claims. A small known-structure numeric test is a diagnostic starting point, not the proposed finished paper.

The notes informed the direction; their complete bibliographies were not independently reverified in this session. The narrower primary-source recheck is documented below.

## What the repository establishes so far

The engine supports frozen LLM and classical agents, seeded bandit environments, full prompt/response logs, structured hypothesis outputs, and revision metrics. September reports supersede the August next-step queue.

Historical reports show a useful tension: in randomized-change P0, detect scores 60.9 versus vanilla 56.6; in the no-change control, detect scores 65.7 versus vanilla 72.3. These are reported n=10 results, not reproduced results in this checkout. The promising research question is how to improve the adaptation/retention trade-off, not how to manufacture another environment where a model loses.

The initial audit found:

| Finding | Evidence | Consequence |
|---|---|---|
| Historical trajectories absent | `runs/` is gitignored and absent here; saved reports/figures remain | Recover old data if available; do not spend credits repeating the entire history |
| New plan's sanity artifacts absent | Root `selective_revision_sanity.py/json` links in `research-plan.md` have no targets | Rebuild a tested reference as task 001; new outputs live under scripts/ and reports/ |
| Prompt and interface are confounded | n20 configs compare raw vanilla with stats-plus-recent detect | A wording-only effect needs the same information interface on both sides |
| Prior declarations usually are not fed back | `render_step_line` renders action/reward/observation; `_build_prompt` uses those blocks | Existing declarations are behavioral outputs; do not describe them as a persisted memory that was revised |
| Surprise-stat is an exception | `render_hypothesis_check` reads the previous declared arm | It receives limited prior-hypothesis information, adding another treatment difference |
| Surprise score is a plug-in tail | Its lifetime estimate includes the last-k observations it tests | Treat it as a heuristic surprise feature, not a calibrated change posterior or valid sequential p-value |
| Revision metrics lack a prior-commitment eligibility check | `revision_sequence` only receives the post-cut sequence and the old oracle-best arm | A model can be counted as retracting an arm it never believed was best |
| Mean delay drops missing recoveries | Means filter out `None` in hypothesis metrics | Report recovery fractions and censored delays alongside conditional means |
| Same seed is not matched evidence | Change/control stage sampling consumes different numbers of RNG draws; policies choose different arms | Use saved fixed histories for inference probes; explicit event-indexed randomness for later rollouts |
| Gemini cost defaults were stale | `token_cost.py` used .15/.60 USD per million; official 3.5 Flash standard rates are 1.50/9.00 on 9 September | Corrected 3.5 default and fallback rates, with an exact-dollar regression check; historical cost figures are not reliable estimates for new runs |
| Existing tests pass | `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`: 54 passed before task 001 | This checks implementation behavior, not the historical scientific claims |

Existing experiment policies, dynamics, and scientific metrics remain unchanged by this audit; the Gemini 3.5 cost estimate was corrected. Other model prices remain legacy estimates. The historical runner still lacks a reliable whole-run spend cap and per-retry billing ledger, so use the new harness for paid testing. Do not silently redefine old metrics and overwrite published-looking reports.

## Immediate sequence and gates

1. **Task 001 — reference.** Implement exact shared/independent reset inference and a priced diagnostic decision. Test mathematical invariants and reproduce the illustrative numbers. CPU only. Brief: `research_tasks/001_selective_revision_reference-commander.md`.
2. **Task 002 — fixed-history pilot harness.** Raw observations, verified counts, and supplied reference predictions on the same cases. All cases use the same output schema. Implement dry-run, strict validation, per-attempt logging, resume, token/call caps, and exact expected-utility scoring before making model calls. Brief: `research_tasks/002_selective_revision_pilot-commander.md`.
3. **Small Gemini test batch.** 72 planned decision calls in the first diagnostic grid, fewer for the first smoke. The user authorized ordinary small API experiments and explicitly rejected an arbitrary US$1 ceiling that could interrupt completion. Preflight the entire experiment using verified pricing, bounded tokens, and a full retry reserve; set the operational budget to cover completion before starting. Save every attempt and resume without repeating completed work. Flag a substantial increase before launching it. No DeepSeek development runs. This is diagnostic case coverage, not a powered evaluation of population return.
4. **Choose the cheapest justified intervention.** If correct reference predictions repair decisions, test a compact externally maintained predictive state or a targeted prompt at the same interface. If correct predictions do not repair decisions, target action selection/diagnostic valuation. If raw observations already solve the task, redesign before expanding. A statistical observer solving a known model is a reference and an engineering aid, not itself a new learning contribution.
5. **Only then consider training.** Reproduce a decision failure and a correction on held-out cases; compare against a simple frozen intervention. If a remaining limitation plausibly requires learning, specify matched action-only versus prediction-plus-action SFT, validate the data/forward/backward/save-load path locally, measure memory/throughput, and prepare an external training job. Do not promise a particular GPU-hour figure before measuring it.

For progression out of the diagnostic stage, look for repeated, non-formatting decision errors in more than one case family, material expected-utility loss (initial screening scale: 0.02 per decision), and an intervention that removes a useful fraction of that loss without worsening unrelated forecasts. These are engineering screening criteria, not significance thresholds. Freeze final effect thresholds and sample sizes after estimating variance on a separate pilot. Keep stationary and changed outcomes separate.

The one-event numeric task establishes use of supplied dependency structure. It does not establish discovery of structure, long-horizon adaptation, or transfer to language workflows. Add those requirements in successive experiments only after the earlier gate passes.

## Division of work

The research lead chooses the question, specifies permitted information, fixes comparisons/metrics and stop conditions, checks relevant primary literature, reviews Grok's diff, and decides the next experiment. Grok receives one bounded brief with exact deliverables and acceptance checks, implements it, and returns commands/results/limitations. Grok does not decide which result supports a scientific claim.

Use short file-based assignments to preserve context across sessions and conserve research-lead usage. Each completed iteration leaves a reproducible artifact, a review outcome, and one next brief. Run one implementation task at a time. Never launch an uncapped self-modifying experiment loop. Do not automatically launch the next paid batch because a previous one finished.

Grok invocation verified from local `grok --help`:

```bash
grok --prompt-file research_tasks/001_selective_revision_reference-commander.md --no-subagents --disable-web-search --max-turns 60 --permission-mode auto
```

Within the Codex sandbox, Grok needs network and writes to its normal `~/.grok` session storage; the initial restricted launch failed, then an approved escalated launch started. Headless `acceptEdits` subsequently cancelled a shell-permission request (`PermissionCancelled` in the session transcript); the lead initially misattributed this to the turn cap. The stalled resume was stopped and the task restarted with Grok's automatic permission-review mode. That attempt read the brief but produced no code or further output for over five minutes and was stopped. The root cause of that stall is unresolved; the command above is a candidate invocation, not a successfully validated unattended workflow. The lead implemented task 001 locally instead. Before task 002, test a much smaller Grok handoff and verify actual completion. Do not use blanket bypass/always-approve. Inspect the persisted completion reason and actual files, not just the CLI's exit code. This is session/runtime access, not sudo or a request to train on this machine.

Use uv with the locked dependencies. In this session the cache is `/tmp/llm-time-uv-cache`. Credentials stay in `.env` and must never enter reports, Grok task files, or version control. Preserve pre-existing user edits, including the deleted `.env.example` and untracked research notes.

## Literature checkpoint

Two primary sources were rechecked on 9 September; this is not a renewed exhaustive novelty review. Shared versus independent change structure already has a cognitive precedent in [Heilbron and Meyniel (2019)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006972), which uses transfer of uncertainty between coupled statistics to distinguish inference models. [MemoPilot](https://arxiv.org/html/2606.08656) already trains a memory updater for a frozen player using multi-turn reinforcement learning. Therefore neither the diagnostic motif nor generic learned memory supplies novelty alone. The research bet remains an improvement in selective decisions and structural transfer under matched information and compute; the pilot may reject it.

## Current state

- Repository orientation and initial audit complete; baseline suite: 54 passing tests. Final suite after reference and pricing changes: **84 passing tests**.
- Task 001 completed by the research lead after the Grok handoff stalled. Reference: `llm_rl_engine/analysis/selective_revision.py`; reproduction: `uv run python scripts/selective_revision_sanity.py`; outputs: `reports/selective_revision_reference.{json,md}`. Thirty new tests pass. Two regenerations produced identical hashes; all 54 raw/count fixture pairs were checked. The worked example also agrees with an independently calculated closed-form posterior and expected-utility calculation.
- Task 002 is a prepared next assignment, not yet an executed model experiment.
- Historical trajectories are on the user's other machine and may be copied tomorrow. They are useful for reanalysis but do not block the new reference/pilot. Prioritize P0 hypothesis and randomized-change/no-change runs, including trajectory JSONL and saved configs. Ordinary small API experiments are authorized; budget for whole-experiment completion plus retries rather than an arbitrary US$1 cutoff.
- No training or Gemini/DeepSeek experiment calls made in this iteration. Grok CLI sessions were attempted as described above and stopped; no unattended research job is intentionally left running.
