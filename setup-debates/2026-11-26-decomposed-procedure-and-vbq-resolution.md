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
