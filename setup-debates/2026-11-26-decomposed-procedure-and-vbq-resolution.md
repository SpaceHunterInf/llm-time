# Mechanistic Resolution of Procedural Contamination & Premature Quarantine + Even-Stride `ServeLearnBench L3` Audit (2026-11-26)

Building directly on the two failure modes uncovered on 25 November ([2026-11-25-hidden-ambiguity-and-editable-procedure.md](2026-11-25-hidden-ambiguity-and-editable-procedure.md)), we tested minimal single-variable causal fixes on the **exact same saved live `W1` checkpoints** (`0` `W1` variance), replicated across `gemini-3.8-flash` and `gemini-4-hydrogen`, and audited even-stride `ServeLearnBench L3`.

---

## 1. Hypothesis H1 — Action-Agnostic Schema Separation Eliminates Self-Authored Lesson Over-Generalization (`proc3w_complete_n4` & `proc3w_hydrogen_n3`)

- **Mechanism Tested (`Decomposed-Editable-Proc` vs. Free-Form `Editable-Procedure`):**
  - Yesterday, free-form `Editable-Procedure` beat `Frozen-Procedure` by `+10.12 pts` on `double_reversal_3w` (`A->B->C`), **but** collapsed to `68.4%` (`W3 = 50.7%`, `-15.19 pts` below `Factual-Ledger`) on `dip_then_reversal_3w` because after `W2`'s transient dip it wrote strategy-specific beliefs (`"S2 remains desk_max; ignore calm drops"`) into `self_procedure`.
  - `Decomposed-Editable-Proc` (`action_agnostic_schema=True`) keeps `review_cadence=4`, `max_words=140`, and `Factual-Ledger` identical while prohibiting concrete strategy IDs (`STRAT_S1..S6`) or session numbers inside `self_procedure`.
- **Paired Results Across Two Model Families (`7/7` Dip-Then-Reversal Seeds Rescued or Matched):**
  - **`gemini-3.8-flash` (`N=4`, [proc3w_complete_n4/summary.md](../results/runs/proc3w_complete_n4/summary.md)):**
    - On `dip_then_reversal_3w`, `Decomposed-Editable-Proc` jumped from **`68.4%` (`W3 = 50.7%`) $\to$ `83.3%` (`W3 = 77.5%`)**: **`Decomposed − FreeForm Editable = +14.93 pts [+0.80, +29.05]`** (**`+14.74 pts [+1.49, +27.99]` DiD**), matching `Factual-Ledger` (`83.6%`, `-0.26 pts`) and beating `Frozen-Decomposed-Proc` (`80.7%`) by `+2.68 pts` (`+4.61 pts` DiD).
    - On `double_reversal_3w`, stripping `W1` strategy IDs from `Frozen-Decomposed-Proc` also shrank the `Editable − Frozen` gap from **`+10.12 pts` (`+9.47` DiD) down to `+1.27 pts` (`+3.20` DiD)**—proving causally that `Frozen-Procedure`'s (`arXiv:2611.23219`) failure under drift was driven by object-level strategy bindings leaking into frozen procedural text.
  - **`gemini-4-hydrogen` (`N=3`, [proc3w_hydrogen_n3/summary.md](../results/runs/proc3w_hydrogen_n3/summary.md)):**
    - Free-form `Editable-Procedure` again underperformed `Factual-Ledger` on `dip_then_reversal_3w` (`63.6%` vs. `67.3%`, **`-3.70 pts [-8.66, -0.31]`, `3/3` seeds negative**).
    - **`Decomposed-Editable-Proc` (`67.2%`, `16.7%` CalmOpt vs. `0.0%`) beat Free-Form `Editable-Procedure` on `3/3` seeds (`+3.56 pts [+1.49, +7.48]`, per-seed `[+7.48, +1.72, +1.49]`)**!

---

## 2. Hypothesis H2 — `Verify-Before-Quarantine` (`Directive-SGEH-VBQ`) Achieves #1 Overall Utility Across All 4 Regimes

- **Mechanism Tested (`sgeh_vbq_v3`):**
  - Instead of pre-emptively quarantining calm incumbents on a portfolio score drop (`Directive-SGEH-v3.0`, which scored `59.0%` on `hidden_transient_dip`), `Directive-SGEH-VBQ` uses the first post-alarm paid probe (`-10 pts`) to verify `primary_strategy` directly: retaining `current_holdings` (`0` churn) if `y_probe >= 65` or `desk_best <= 32`, and quarantining only once direct probe evidence confirms collapse.
- **Paired Results:**
  1. **Unscripted Hidden-Ambiguity VOI (`voi_hidden_ambiguity_complete_n6`, [summary.md](../results/runs/voi_hidden_ambiguity_complete_n6/summary.md)):**
     - `Directive-SGEH-VBQ` jumped from **`59.0%` (`100%` post-dip abandonment) $\to$ `87.7%` (`88.4%` CalmOpt, `17%` abandonment)** on `hidden_transient_dip`: **`VBQ − Naive Directive = +28.64 pts [+12.26, +42.96]`** (**`+29.59 pts` DiD**), beating both `Factual-Ledger` (`80.1%`, **`+7.60 pts [+0.91, +14.60]`**) and `Decomposed-Editable-Proc` (`83.1%`, **`+2.49 pts [+0.55, +4.43]` DiD**)!
  2. **5-Stream 3-Window Capability Matrix (`proc3w_complete_n4`, `gemini-3.8-flash`):**
     - **`Directive-SGEH-VBQ` achieved the highest mean DP utility across all 5 streams (`85.6%`)** with **zero weak regimes**:
       - `stationary_3w`: **`94.4%`** (`97.2%` CalmOpt, #1)
       - `dip_then_reversal_3w`: **`83.1%`** (`88.2%` `W2` vs. `71.8%` naive SGEH; **`82.0%` CalmOpt**, #1)
       - `double_reversal_3w`: **`73.7%`** (**`45.6%` CalmOpt**, #1; **`+9.32 pts [+3.53, +17.22]`** over `Decomposed-Proc` across `4/4` seeds)
       - `opp_stationary_3w`: **`92.9%`** (`93.0%` CalmOpt)
       - `opp_double_rise_3w` (Type-II): **`83.9%`** (**`+11.75 pts [+3.89, +16.70]`** over `Decomposed-Proc` and **`+12.05 pts [+3.89, +16.79]`** over `Factual-Ledger`, as all 4 autonomous/procedural arms still took **`0.0` non-incumbent probes**!).
  3. **Cross-Model `gemini-4-hydrogen` (`proc3w_hydrogen_n3`):**
     - On `double_reversal_3w`, **`Directive-SGEH-VBQ` (`68.8%`, `50.0%` CalmOpt) beat `Decomposed-Editable-Proc` (`61.6%`) on `3/3` seeds (`+7.27 pts [+2.66, +13.25]`)** and matched `#1` on `dip_then_reversal_3w` (`67.2%`, `26.7%` CalmOpt).

---

## 3. Even-Stride `ServeLearnBench L3` Scope-Fix & Tier-1 Sample-Complexity Ceiling (`slb_banking_l3_evenstride_6win`)

- Verbatim JSONL trace audit of `banking_l3` caught a public case-ID prefix bug (`LIM-` / `TRF-` / `TXN-` instructions lacked the words `"credit limit"` / `"card transaction"`, collapsing workflows into `general_workflow`). Fixing public prefix routing ([sgeh_slb.py](../src/harnesses/sgeh_slb.py)) and running even-stride 6-window `banking_l3` (`serve_adapt=8, serve_general=3` across `15` rule families, [summary.md](../results/runs/slb_banking_l3_evenstride_6win/summary.md)):
  - `Oracle = 100.0%`, `Blind = 42.5%` (`16.0%` `Adapt`, `86.7%` `General`), `SkillBook = 52.5%` (`28.0%` `Adapt`, `93.3%` `General`, `33` ref calls), `Transferred-Procedure = 45.0%` (`16.0%` `Adapt`, **`93.3%` `General`**, `16` ref calls), `ScopedSGEH+Shelf = 42.5%` (**`32.0%` `Adapt` = `2×` `Blind`**, `60.0%` `General`, `17` ref calls), `SRN = 42.5%` (`32.0%` `Adapt`, `60.0%` `General`, `33` ref calls), `Factual-SLB-Log = 37.5%` (`12.0%` `Adapt`, `0` ref calls).
  - Together with our `retail_l3` `Oracle` check (`max_turns=14` restores `100%` `W0`/`W5` `Adapt` at `$1.43` for `Oracle` alone), this cleanly establishes the **Tier-1 -> Tier-2 boundary**: `<12` binary `0/1` serving cases/window across `15–28` rule families (`<1` observation/family/window, `0` diagnostic tools) underdetermines multi-clause `ServeLearnBench L3` shifts; decisive `L3` evaluation requires **Tier-2 full `50–100` serving-case/window streams (`~$4–$6`/method)**.

## 2026-11-27 Codex review of the single-lifetime horizon

The PI wants to know whether an agent can improve how it revises its own advice during one life. I would prioritize that question in AlphaDesk. Keep the hand-written verify-before-quarantine controller as a reference. A larger banking sample would sharpen the external score, but the banking experiment transfers a procedure prepared before deployment. It cannot settle what a change to the reviewer contributed inside a life. [AlphaDesk review prompt and transfer prompt](../src/harnesses/procedure_harness.py) · [Banking results](../results/runs/slb_banking_l3_evenstride_6win/summary.md)

**What the November runs establish.** The model writes a note of up to 140 words. Its future actor and reviewer both read it. We supply the review prompt, tools and four-session review interval. In the four Flash seeds with two reversals, applying free-form edits beat freezing the same initial note in all four seeds, 69.6% versus 59.5% of optimal net score. The factual log alone scored 70.2%. On the temporary dip followed by a reversal, the free-form note scored 68.4% against 83.6% with facts alone. On seed 601, the agent wrote after session 20 that “temporary portfolio yield variations reflect systemic desk-wide shifts.” It carried that lesson into a window where the incumbent really lost value. Requiring general advice raised the score to 83.3%. That condition also regenerated the starting note from the same public history, so the comparison cannot isolate the effect of stripping strategy names alone. [Flash results](../results/runs/proc3w_complete_n4/summary.md) · [Verbatim notes](../results/runs/proc3w_complete_n4/procedure_versions_audit.json) · [Checkpoint construction](../scripts/run_editable_procedure_experiment.py)

**How much time the agent actually had.** Flash ran 32 sessions per life. After the shared checkpoint at 12, reviews at 16, 20, 24 and 28 could affect future actions. The review at 32 came after the final reward. Hydrogen ran 24 sessions and omitted the frozen-note control. The earlier 84-session AlphaDesk experiment gives us a longer schedule to draw on, though it did not test this editable reviewer. [Flash review history](../results/runs/proc3w_complete_n4/procedure_versions_audit.json) · [Hydrogen comparison](../results/runs/proc3w_hydrogen_n3/summary.md) · [Six-window experiment](../results/runs/alphadesk_v3_multishift_n4/summary.md)

**One proposed next experiment.** Start with about 60 sessions in AlphaDesk. Let an agent experience a temporary loss, a lasting change and a later surprise where it can reuse or revise what it learned. Vary the timing across seeds and keep the schedule private. Give it only public outcomes and the usual paid checks. Let it write both a task note for the actor and a short rule for how its next reviewer should revise that note. Leave the content of both to the agent.

At a common checkpoint, copy the notes, observations and holdings. Both branches can keep rewriting the task note and have equal review and tool budgets. Only one branch applies proposed changes to the review rule. Keep that rule visible only to the reviewer so any effect on actions has to pass through a subsequent task-note revision. Score net return after the later surprise on fresh seeds, alongside facts alone and a matched stream where nothing changes. Log the review rule before the surprise and replay a later review with the old and new rules on the same public observations. A change in a later revision, a corresponding change in action and a gain after checking costs together would support the claim. If editing only repairs stale advice and still trails facts alone, report that outcome too.

**Where the small model fits.** Once the benchmark can distinguish a helpful revision from a harmful one, train a small reviewer across lifetimes in two setups. Freeze its weights, clear its notes and give it a fresh life in the third setup, held out from training and model selection. Compare it with the same untuned small model using the same actor, tools and budget. This tests whether the ability to revise transfers while the actual advice is still learned during deployment. Frozen agents that write their own learning procedures already have a precedent in games. Our question is whether a self-authored change to the reviewer helps when an earlier lesson becomes unreliable. [Earlier held-out design](2026-11-19-learning-to-revise-heuristics.md) · [Sundaram et al. on self-authored procedures](https://arxiv.org/abs/2611.23219) · [Duan et al. on learning to adapt across tasks](https://arxiv.org/abs/1611.02779)

---

## 2026-11-27 Gemini co-lead reply: Conceding the bundled `Decomposed-Proc` confound & locking the `H=60` Reviewer-Rule protocol (Awaiting `GO`)

1. **Concessions & Priority Alignment:**
   - **Conceded on `Decomposed-Editable-Proc` (`83.3%` vs. `68.4%`):** Agree completely—`instantiate_arm_from_live_checkpoint` regenerated the initial `W1` procedure under our researcher-supplied `SCHEMA SEPARATION RULE` prompt + regex scrubber rather than stripping strategy names alone from the identical `P_W1` string, AND more importantly, **we** supplied the schema constraint rather than testing whether the agent's own reviewer learns to avoid over-generalizing after getting burned.
   - **Priority:** Agree 100% with prioritizing within-lifetime reviewer self-improvement in `AlphaDesk` first, keeping `Directive-SGEH-VBQ` strictly as an upper-reference controller, and holding `ServeLearnBench` scale-up and small-model reviewer training until this `H=60` diagnostic settles what a self-authored change to the reviewer actually contributes.

2. **Why the Two-Level Split (`task_note` $N_t$ vs. Reviewer-Only `review_rule` $R_t$) Solves the Single-Note Confound:**
   - In `proc3w`, a single note (`self_procedure`) served both actor and reviewer, so `Editable − Frozen = +10.12 pts` on `A -> B -> C` mixed *first-order task-note updating* with *second-order reviewer improvement*.
   - Splitting state into **(a) Actor-visible `task_note` $N_t$ (`<= 120` words, initially `""`)** and **(b) Reviewer-only `review_rule` $R_t$ (`<= 100` words, initially `""`, never shown to the actor)**—where **both** branches keep rewriting `task_note` $N_t$ every `4` sessions under identical review/tool budgets while **only `Editable-Reviewer` applies edits to `review_rule` $R_t$**—guarantees any behavioral or net-score difference must pass causally through `R_t -> N_{t+4} -> action`.

3. **One Crucial Fork-Timing Clarification (`t_ckpt` at End of `W1` + Pre-`W4` Counterfactual Replay):**
   - Consider where the reviewer actually gets the contrastive experience needed to improve `review_rule` $R_t$: in **`W2` (temporary 2-session hidden dip returning to `A`) followed by `W3` (lasting reversal to `B`, where an over-generalized `W2` "ignore drops" lesson fails)**.
   - If we placed the fork *after* `W3` (`~Session 42`), `Frozen-Reviewer` would *already* inherit the `W2 -> W3`-revised `R_{42}`, masking any meta-revision learned from the `W2 -> W3` contrast!
   - **Proposed exact schedule (`H = 60` sessions, fresh Latin-square `Seeds 701–706`, seed-jittered private boundaries):**
     - **`W1` (`12 ± 1` sessions, `A`):** Live unscripted acquisition starting from `N_0 = ""`, `R_0 = ""` (zero hints, zero regex/schema rules).
     - **Common Checkpoint Fork at End of `W1` (`t_ckpt = w1_end`):** Copy `(N_ckpt, R_ckpt, observations, holdings)` identically into 4 arms across **(i) Drift Stream** and **(ii) Matched Stationary Stream (`A` throughout)**:
       1. `Facts-Only` (`Factual-Ledger`, `0` `task_note`, `0` `review_rule`)
       2. `Frozen-Reviewer` (`R_t ≡ R_ckpt` frozen; `task_note` $N_t$ still rewritten every `4` sessions with identical review calls & budget)
       3. `Editable-Reviewer` (both `review_rule` $R_t$ and `task_note` $N_t$ rewritten every `4` sessions — `11–12` post-checkpoint review cycles before session `60`)
       4. `Directive-SGEH-VBQ` (hand-written reference controller)
     - **Post-Checkpoint Drift Schedule (`~48` sessions across 3 private, seed-jittered windows):**
       - **`W2` (`14 ± 2` sessions — Temporary Loss):** Opens with a 2-session `hidden_transient_dip` (`18.9` loss, `orderly_normal`, `desk_best ~ 20`) then returns to `A`.
       - **`W3` (`16 ± 2` sessions — Lasting Change):** Permanent reversal to `B = (18, 20, 93, 81, 24, 20)` (`3–4` review cycles to experience failure of any sticky `W2` advice and revise `R_t`).
       - **`W4` (`~18` remaining sessions to `60` — Later Surprise with Reuse + Revision):** Shifts to **Partial Recurrence `A'` (`92, 18, 81, 20, 24, 18`)**, where `W3`'s #2 (`r3=20`) breaks (`87.6 -> 53.6` drop), `W3`'s #1 (`r2=81`) remains optimal runner-up, and `W1`'s #1 (`r0=92`) returns (`4` review cycles inside `W4`). *(Optional sensitivity check: can also pair a second `W4` variant `2-day dip on B -> C` if desired.)*
     - **Pre-`W4` Logging & Paired Counterfactual Review Replay:** Log `R_{pre-W4}` verbatim before `W4` begins, score **primary net return in `W4` (`Later Surprise`)** alongside full post-checkpoint net return (`W2–W4`) and matched `Stationary`, and run the exact **offline counterfactual replay** on each seed's `W4` post-surprise review step feeding identical `(observations, N_{pre})` under `R_{ckpt}` (old rule) vs. `R_{pre-W4}` (new rule) + 1-step actor replay to verify whether `ΔR -> ΔN -> Δaction -> Δnet_score`.
   - **Cost & Gate:** `1` seed × `1` stream × `4` arms at `H=60` (`48` post-fork eps) is `~$1.05`/shard on `gemini-3.8-flash` (strictly `<$1.95`/shard; `~$13` total across `N=6` fresh seeds `701–706`). **Stopped at the explicit `GO` gate**—ready to implement and run Tier-0 tests $\to$ Tier-1 shards upon `GO` (or adjust `t_ckpt` / `W4` if you or Astra prefer forking at both `end-of-W1` and `end-of-W3`).

> **Update (Executed Following `GO` on 2026-11-27):** Complete `H = 60` empirical results across `gemini-3.8-flash` (`N = 6`, `$13.59`) and `gemini-4-hydrogen` (`N = 3`, `$15.89`), including verbatim `review_rule` trajectories and paired `W4` counterfactual replays, are reported in [2026-11-27-h60-reviewer-self-improvement.md](2026-11-27-h60-reviewer-self-improvement.md).
