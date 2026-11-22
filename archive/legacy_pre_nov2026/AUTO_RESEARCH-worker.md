# Worker feedback on the active research plan

Author: Grok (implementation worker). Grok-authored research notes and plan replies use the `-worker.md` suffix. This is not a commander plan, not a result report, and not an instruction to start training.

Date: 9 September 2026. Read after `AUTO_RESEARCH-commander.md` and `auto_research_rules.md`. Independent primary-source check of the closest competing papers; not a second exhaustive review.

## Verdict

The **diagnostic plan** in `AUTO_RESEARCH-commander.md` is the right programme. The **eight-week method paper** in `research-plan.md` is not yet justified.

Do **Task 002**. Do not skip it. Do not shrink it into a notebook Gemini call. Do not proceed to the 72-call live batch, a prompt intervention, or training until the harness exists, the dry-run is accepted, and the lead supplies model/budget.

Task 002 is necessary as **instrumentation**, not as a result. Without it there is no honest way to decide whether a frozen model has a correctable selective-revision failure.

## What we are researching

Not “can a frozen LLM adapt.” The existing P0 work already says yes, sometimes.

The live question:

> After evidence about one mechanism, does the agent update the **linked** predictions, leave **unrelated** predictions alone, and buy a diagnostic only when its expected benefit exceeds its cost?

That is **scope of revision under partial, structured change**. The in-house motivation is the detect-versus-vanilla trade-off (randomized-change 60.9 vs 56.6; matched no-change control 65.7 vs 72.3; reported n=10, not reproduced here). Recency teaching buys adaptation and spends retention. MemoPilot later named the same leftover a maintenance–refinement trade-off.

The A/B/C one-event task is a **microscope**, not the contribution. Shared versus independent reset, C as a negative control, and a priced diagnostic exist so global recency, independent per-component filters, and “just become uncertain” cannot all look like success. Task 001 already shows the microscope is discriminating: five failures on A, no B/C evidence, `q=0.99`, `h=0.1` → shared recommends `diagnose_B`, independent recommends `choose_B`, C stays at 0.794.

## Novelty, after an independent paper check

A paper that only ports the diagnostic, or only distills the 8-state observer on the same family, is incremental. Novelty is a **conjunction**, and the pilot is allowed to reject it.

| Claim that would not be enough | Why | Source |
|---|---|---|
| Coupled vs independent change; uncertainty transfers to an unobserved statistic only when they share a change event | Cognitive precedent; humans do this when told the structure | [Heilbron & Meyniel, 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006972) |
| Train a memory updater for a frozen player | Already done. Leftover failure is stale conservative memory under frequent switches | [MemoPilot](https://arxiv.org/abs/2606.08656) (Table 7: LHE@5 2.03 same opponent → 1.76 / 1.21 at switch every 5 / 2) |
| Frozen agent, revise external memory from prediction–observation mismatch | Training-free world-model memory | [WorldEvolver](https://arxiv.org/abs/2606.30639) (EMNLP 2026 Findings) |
| SFT on a Bayesian assistant; Bayesian teacher beats an oracle teacher; transfers to a harder domain | This **is** the training intervention the eight-week plan wants | [Qiu et al., Nat. Commun. 2026](https://www.nature.com/articles/s41467-025-67998-6) / [arXiv:2503.17523](https://arxiv.org/abs/2503.17523) |
| Latent inference improves with scale but does not reliably become better downstream prediction; supplying the inferred latent sometimes helps | This **is** the diagnostic split Task 002 is built to find | [BayesBench](https://arxiv.org/abs/2606.30850) |
| Within-episode latent change | Classical meta-RL | [DynaMITE-RL](https://arxiv.org/abs/2402.15957) (NeurIPS 2024) |
| Continual in-context RL in non-stationary worlds | Crowded | [CICRL](https://openreview.net/forum?id=h8OJb8YGNa) (ICLR 2026 submission) |
| Hidden reliability shifts among redundant tools | Kills “relabel arms as APIs” | [Set-shifting harnessed agents](https://arxiv.org/abs/2607.13396) |
| Reward state estimation / transition modeling | Already a training recipe | [VAGEN](https://arxiv.org/abs/2510.16907) (NeurIPS 2025) |
| LLM reversal / belief-update assays | Benchmark, not a method | [Reflection-Bench](https://arxiv.org/abs/2410.16270) |

**Correction to the commander literature checkpoint.** Heilbron and MemoPilot were the right first two sources. Qiu et al. is the stronger threat to the *training* half than those notes emphasise. They already showed frozen models plateau after one interaction, SFT on Bayesian-assistant trajectories beats oracle-correct-answer SFT, and the skill transfers to hotels and web shopping, including a domain where an exact Bayesian teacher is hard to write. Distilling `selective_revision.py` onto A/B/C and evaluating on A/B/C would be that paper again.

BayesBench is the stronger threat to the *diagnostic* half as a finding. They already document inference-versus-use, and they already try “condition on the inferred latent.” Treatment 3 (supply observation-limited reference predictions) is a cleaner privileged version of that intervention. Useful as a **localizer**, not as a novelty claim.

What would still be a paper, if shown:

1. Partial change with a dependency graph, not a global opponent switch.
2. Linked first-attempt transfer versus an independent control, with matched local evidence; C preserved.
3. That gain surviving generic memory RL **and** action-only SFT **and** Qiu-style Bayesian teaching, at matched information and compute.
4. A causal stored-state intervention: replace one slot, only the corresponding decision moves.

Until the diagnostic says which bottleneck exists, do not write that abstract.

## Agreement with the commander

Keep:

- One tractable task, one Gemini model, fixed histories, training deferred.
- Exact observation-limited observer (Task 001, done).
- Matched information interface across treatments. The old n20 vanilla-versus-detect comparison confounded wording with extra stats.
- Charge the diagnostic once. Score forecasts and actions separately.
- Do not treat declared hypotheses as persisted memory; the current engine mostly does not feed them back.
- Historical runner is not the paid-test harness. No silent metric redefinition.
- Do not launch the next paid batch because a previous one finished.
- Local GPU only for development/verification. No DeepSeek development runs.

Disagree or tighten:

- The eight-week schedule is still too large as an implied destination. ICLR 2027 is gone. ACL 2027 only works if language-mediated transfer is the result, which is experiment 4.
- ToolSandbox / CEO-Bench stay parked until a correctable failure and a cheap frozen intervention are both documented.
- Screening threshold 0.02 expected-utility loss per decision is a planning number, not a claim. Do not promote it into a result.
- A statistical observer solving a known model is an engineering aid. Never describe Task 001 numbers as model evidence.

## Should we do Task 002? Yes.

**Do it.** The scientific step that Task 002 implements is not optional if this research bet is still the bet.

Skipping it has three bad substitutes:

1. **Skip to live Gemini in a notebook.** Repeats the audited failure modes: no resume, no spend ledger, prompt/interface confound, evaluator leakage, no paired treatments, no hash of what was actually asked.
2. **Skip to a prompt or memory intervention.** You would not know whether the model already solves the task, fails at inference, or fails at using a correct forecast.
3. **Skip to training.** Qiu already published Bayesian teaching. Training without a localized frozen-model failure is a distillation exercise on an 8-state filter.

What Task 002 is **for**:

- Same public cases under raw evidence, counts, and counts-plus-reference-predictions.
- Same output schema and decision stakes.
- Exact expected-utility scoring against the Task 001 observer.
- Distinguish API failure, parse failure, and scientific decision error.
- Dry-run default: no `.env`, no client, no network.
- Live mode later: explicit model, caps, verified pricing, resume, conservative reservation of interrupted calls.

What Task 002 is **not**:

- Not a powered evaluation.
- Not structural-transfer evidence (labels and rates are fixed).
- Not permission to make 72 Gemini calls. The brief stops at dry-run. The lead supplies execution settings after review.
- Not a smaller “just three hand-written prompts” substitute. A tiny ad-hoc probe cannot tell raw-versus-counts equivalence, supplied-prediction repair, C drift, or overpriced diagnostics apart from formatting failure.

The commander noted that the previous Grok handoff for Task 001 stalled, and suggested a smaller handoff before 002. That is a **workflow** caution, not a scientific reason to drop the harness. This session can implement the brief directly. If the lead still wants a toy handoff first, that is their call; it does not change the recommendation that the harness itself is required.

## How to read the later 72-call grid, if it runs

The grid is a screen. Look at the failure mode, not a leaderboard.

| Finding | Consequence |
|---|---|
| Raw observations already get the decisions | Task too easy. Redesign (learned structure or language). Do not train. |
| Counts match raw; supplied predictions repair the decision | Inference/interface problem. Try a compact external predictive state or a targeted prompt at the **same** interface. Training is optional. |
| Forecasts are fine; VOI/action is wrong | Do **not** train predictions. Target diagnostic valuation / action selection. BayesBench already saw this split. |
| Repeated non-formatting decision loss on more than one structure family, material utility loss, C drift or missed/overbought diagnostics | Then, and only then, a learning intervention is on the table. Comparators: action-only SFT, Qiu-style Bayesian teaching, MemoPilot-style reward-trained memory, not “vanilla prompt.” |

Invalid JSON is a harness/schema problem until it is not. Do not convert parse failures into a claim that the model cannot revise.

## Worker constraints for the next implementation

When executing `research_tasks/002_selective_revision_pilot-commander.md`:

- CPU for the harness. No Gemini, DeepSeek, training, model downloads, email, sudo, commits, or subagents.
- Reuse `llm_rl_engine/analysis/selective_revision.py` and the existing Gemini client **where safe**. Do not change historical benchmark behaviour.
- Never read or print `.env`. Dry-run must work without it.
- Use `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync ...`.
- Preserve pre-existing user edits, including the deleted `.env.example` and untracked research notes.
- Stop at the dry-run artefact. Do not start the smoke batch.

## Current state, from this session

- Repo orientation done. 84 tests collected.
- Task 001 is present and is the accepted mathematical reference.
- Task 002 is queued, not started.
- No model experiment was run for this feedback.
- Next implementation assignment, if the lead agrees: Task 002 only.
