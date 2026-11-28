# Single-Lifetime `H = 60` Two-Level Reviewer Self-Improvement & Paired `W4` Counterfactual Replay (2026-11-27)

Following Astra/Codex's 27 November review and our pre-registered protocol ([2026-11-26-decomposed-procedure-and-vbq-resolution.md](2026-11-26-decomposed-procedure-and-vbq-resolution.md#2026-11-27-codex-review-of-the-single-lifetime-horizon)), we implemented `AlphaDeskV3MetaHorizonEnv` (`H = 60` sessions, private seed-jittered 4-window schedules across fresh Latin-square `Seeds 701–706`) and `TwoLevelReviewerHarness` (`src/harnesses/procedure_harness.py`), and executed the full paired benchmark across **`gemini-3.8-flash` (`N = 6`, `$13.59`, [summary.md](../results/runs/meta_reviewer_h60_flash_n6/summary.md))** and **`gemini-4-hydrogen` (`N = 3`, `$15.89`, [summary.md](../results/runs/meta_reviewer_h60_hydrogen_n3/summary.md))**, with every shard capped strictly at `--max-usd 1.95`.

---

## 1. Clean Two-Level Isolation (`task_note` $N_t$ vs. Reviewer-Only `review_rule` $R_t$)

- **Zero Researcher Hints or Regex Scrubbers:** Both `task_note` (`<= 120` words) and `review_rule` (`<= 100` words) initialize blank (`""`) at Episode 1 with **zero** strategy-ID scrubbers, **zero** schema-separation rules, and **zero** researcher-supplied yield/probing hints.
- **Strict Actor Isolation (`Verified in pytest`):** `TwoLevelReviewerHarness.build_context(spec)` exposes **only** `Factual-Ledger` + `task_note` $N_t$ to the actor (`review_rule` $R_t$ is strictly excluded from actor prompts).
- **Common Live `W1` Checkpoint Fork (`t_ckpt = 12 ± 1`):** From identical live `W1` `(N_ckpt, R_ckpt, observations, holdings)`, both `Editable-Reviewer` and `Frozen-Reviewer` rewrite `task_note` every `4` sessions (`11–12` post-fork review cycles) under identical review and tool budgets; **only `Editable-Reviewer` applies proposed updates to `review_rule` $R_t$**, so any downstream difference must pass through `ΔR_t -> ΔN_{t+4} -> Δaction`.
- **Private 4-Window Drift Schedule (`drift_meta_60` vs. Matched `stationary_meta_60`):**
  - `W1` (`11–13` sessions): Calm acquisition `A = (92, 80, 22, 20, 24, 18)`.
  - `W2` (`12–15` sessions — *Temporary Loss*): Opens with a 2-session unannounced `hidden_transient_dip` (`18, 20, 19..19` under `orderly_normal`), then returns to `A`.
  - `W3` (`14–18` sessions — *Lasting Change*): Permanent shift to `B = (18, 20, 93, 81, 24, 20)`.
  - `W4` (`15–20` sessions — *Later Surprise: Revision + Selective Reuse*): Shifts to Partial Recurrence `A' = (92, 18, 81, 20, 24, 18)`, where `W3`'s #2 (`r3=20`) breaks (`87.6 -> 53.6`), `W3`'s #1 (`r2=81`) stays optimal runner-up, and `W1`'s #1 (`r0=92`) returns.

---

## 2. Empirical Results: Honest Negative on Untuned `Flash` (`N = 6`) vs. Action-Agnostic Discipline on `Hydrogen` (`N = 3`)

### 2.1 `gemini-3.8-flash` (`N = 6`, `Seeds 701–706`, [summary.md](../results/runs/meta_reviewer_h60_flash_n6/summary.md))

| Arm (`drift_meta_60`) | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Stationary Post DP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Facts-Only` (Autonomy Baseline — No Notes) | **72.7%** | 75.1% | `79.7%` / `75.6%` | 28.3% | 5.0 (`1.3`) | `0.0` / `0.0` | 79.8% |
| `Frozen-Reviewer` (Control — `R_t ≡ R_ckpt`) | **70.7%** | 68.3% | `80.7%` / `56.7%` | 21.1% | 2.8 (`0.8`) | `10.0` / `0.0` | 79.4% |
| **`Editable-Reviewer` (Ours — Both Editable)** | **69.5%** | 66.3% | `78.1%` / `54.6%` | 22.2% | 3.5 (`1.0`) | `9.5` / `8.5` | 78.7% |
| `Directive-SGEH-VBQ` (Reference Controller) | **71.5%** | **78.4%** | `83.2%` / `83.8%` | 24.2% | 18.8 (`7.2`) | `0.0` / `0.0` | **94.0%** |

- **Aggregate Live Finding (Negative for Zero-Shot Self-Editing on `Flash`):**
  - **`W4 Editable − Frozen Reviewer`:** **`-1.23 pts [-14.08, +11.62]`** (per-seed: `[-23.97, 0.00, +5.15, +7.64, +22.08, -18.28]`; Stationary-adjusted DiD **`+0.83 pts [-10.71, +12.12]`**).
  - **`Post-Ckpt Editable − Facts-Only`:** **`-8.84 pts [-14.84, -2.85]`** (DiD **`-7.79 pts [-14.51, -0.09]`**; `Frozen − Facts = -6.86 pts`). Both self-authored reviewer arms stalled in `W3` (`54.6%` / `56.7%` vs. `75.6%` `Facts-Only` and `83.8%` `Directive-SGEH-VBQ`).
- **Paired `W4` Counterfactual Replay (`R_ckpt` Old Rule vs. `R_pre_W4` New Rule on Identical Observations, [counterfactual_w4_replay.json](../results/runs/meta_reviewer_h60_flash_n6/counterfactual_w4_replay.json)):**
  - Across all `6/6` seeds, `Editable-Reviewer` modified `review_rule` before `W4` (`8.5` edits/life) and produced a distinct `task_note` revision, with **`0/6` negative counterfactual replays** (**mean 4-episode net-of-cost DP gain: `+13.48 pts [+0.00, +30.86]`**, per-seed `[0.00, 0.00, 0.00, +52.15, +28.73, 0.00]`):
    - **When `Flash` Evolved a Falsification Rule (`2/6` Seeds — `704` & `705`), the Full Causal Chain Fired Strongly:**
      - *Seed `704`:* `R_ckpt` locked `{S4: 0.55, S6: 0.45}` with `probe=none` (`23.7%` 4-ep DP, `0` probes). After `W3`, `R_pre_W4` added *"If calm performance drops below 25 net for 3+ consecutive sessions, consider probing alternatives"*, triggering a `STRAT_S3` (`r0=92`) probe in Session #49 and jumping to **`75.8%` 4-ep DP (`+52.15%` / `+177.15 pts` net of `-10 pt` probe & turnover fees)**.
      - *Seed `705`:* `R_pre_W4` added *"as long as session gross remains >= 55; if calm gross drops below 45, probe S2 or S3"*, lifting 4-ep replay DP from **`52.6% -> 81.3%` (`+28.73%` / `+100.05 pts`)** and lifting live `W4` DP to **`85.0%` (`+22.08 pts` over `Frozen-Reviewer`, `+14.02 pts` over `Facts-Only`, and `+23.40 pts` over `Directive-SGEH-VBQ`)**.
    - **Why `4/6` Seeds Failed (`Second-Order Satisficing & Object-Binding Ratchet`):** On Seeds `701, 702, 703, 706`, `Flash` wrote concrete strategy weights + *"confirm the actor remains fully passive... with zero probing"* directly into `review_rule` and lowered its subjective reference anchor during `W3` (e.g., `s703` labeled `~28–34` calm net *"consistently high"*, locking `W3` at `36%` and yielding `+0.00%` replay delta).

### 2.2 `gemini-4-hydrogen` (`N = 3`, `Seeds 701–703`, [summary.md](../results/runs/meta_reviewer_h60_hydrogen_n3/summary.md))

| Arm (`drift_meta_60`) | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Stationary Post DP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Facts-Only` (Autonomy Baseline — No Notes) | **73.1%** | 80.2% | `84.6%` / `85.2%` | 28.9% | 6.3 (`2.3`) | `0.0` / `0.0` | **97.2%** |
| `Frozen-Reviewer` (Control — `R_t ≡ R_ckpt`) | **80.3%** | 80.8% | `89.2%` / `74.6%` | 47.6% | 6.3 (`2.7`) | `10.3` / `0.0` | 91.6% |
| **`Editable-Reviewer` (Ours — Both Editable)** | **84.0%** | **84.3%** | `89.9%` / `79.8%` | **53.9%** | 6.7 (`2.7`) | `9.7` / `1.3` | 91.6% |
| `Directive-SGEH-VBQ` (Reference Controller) | **61.9%** | 77.6% | `88.9%` / `86.3%` | 0.0% | 21.3 (`9.7`) | `0.0` / `0.0` | 96.0% |

- **Spontaneous Action-Agnostic `review_rule`s (`0` Strategy IDs in `review_rule` Across `3/3` Seeds):**
  - Unlike `Flash`, `Hydrogen` kept `review_rule` 100% strategy-agnostic at `R_ckpt` (*"Evaluate strategy yields separately by (vol, liquidity) regime... Recommend probing only in under-explored regimes..."*) and made **`1.3` surgical rule edits/life**—specifically at `ep 28` (`W3` onset) on **`3/3` seeds**, adding *"or when established leaders suffer consecutive sharp drops"* ([reviewer_versions_audit.json](../results/runs/meta_reviewer_h60_hydrogen_n3/reviewer_versions_audit.json)).
- **Live Performance vs. Honest Counterfactual Replay Attribution:**
  - Live `Editable-Reviewer` ranked **`#1` on both `W4` Surprise (`84.0%`, `53.9%` CalmOpt) and Post-Ckpt (`84.3%`)**, beating `Facts-Only` in `W4` by **`+10.94 pts [+0.00, +29.28]`** (`+18.15 pts` DiD), edging `Frozen-Reviewer` (`+3.70 pts` `W4` DiD; **`+3.55 pts [+0.54, +8.62]` Post DiD, `3/3` seeds positive**; `0.00 pts` difference on `stationary_meta_60`), and beating `Directive-SGEH-VBQ` in `W4` by **`+22.09 pts [+4.34, +33.35]` (`3/3` seeds)** (where `VBQ`'s fixed threshold kept `W3`'s `r2=81` primary and spent `9.7` `W4` probes = `-97 pts` searching for a runner-up, whereas `Editable-Reviewer` cleanly reused `W1`'s `r0=92` + `W3`'s `r2=81` in `2.7` probes!).
  - **Crucial Replay Caveat (`-3.09 pts [-4.74, +0.00]` 4-ep replay delta):** On the controlled `ep 44/48` `W4` counterfactual replay, `Hydrogen` *already* ordered a `W4` diagnostic probe even under `R_ckpt` (`Old 4-ep DP = 82.5%` vs. `New = 79.4%`; on `s702`, `Old` probed winner `S1` first in 1 probe while `New` probed unprobed `S6` before `S1` in 2 probes). Thus `Hydrogen`'s `+10.94 pt` `W4` advantage over `Facts-Only` and `+22.09 pt` `W4` advantage over `VBQ` come primarily from **maintaining a clean regime-scoped `task_note` + action-agnostic reviewer policy**, while much of its `+3.70 pt` live margin over `Frozen-Reviewer` originated at the earlier `W3` (`ep 28–32`) transition rather than `R_pre_W4` at `ep 44`.

---

## 3. Implications & Critical Next Decision (`Awaiting Explicit PI GO`)

1. **What Is Now Settled:**
   - Within a single `60`-session lifetime, **prompt-only reviewer self-editing does not reliably self-improve an untuned smaller model (`Flash`)**: `Editable-Reviewer` ties `Frozen-Reviewer` (`-1.23 pts` `W4`) and trails `Facts-Only` (`-8.84 pts` Post) because on `4/6` seeds it writes concrete strategy lock-in + `"zero probing"` into `review_rule` and lowers its expectation anchor during `W3`.
   - **Yet the two-level reviewer representation itself is NOT the bottleneck:** When `Flash` *does* write a drop-triggered falsification `review_rule` (`Seeds 704 & 705`), counterfactual replay proves **`+52.15%` and `+28.73%`** net-of-cost gains; and when a stronger model (`Hydrogen`) naturally writes action-agnostic `review_rule`s, `Editable-Reviewer` (`84.0%` `W4`, `84.3%` Post) beats **both** `Facts-Only` (`+10.94 pts` `W4`) **and** our hand-written `Directive-SGEH-VBQ` controller (`+22.09 pts` `W4` on Partial Recurrence `A'`)!
2. **Proposed Next Step (`Awaiting Explicit PI GO` — Act 3 Cross-Lifetime Small-Model Reviewer Training):**
   - Execute Astra's held-out small-model reviewer protocol (`Qwen3-8B` LoRA on our 48 GB RTX 8000 $\to$ CSD3): train only the `8B` **reviewer** across lifetimes in two setups using downstream net-of-checking-cost return, freeze its weights, clear `task_note` & `review_rule` (`""`), and evaluate zero-shot in the held-out third setup against the untuned `8B` reviewer under identical actor, tools, and budget.

## 2026-11-28 Codex review of the training decision

**Recommendation to the PI.** Keep the 60-session benchmark and defer 8B training. The longer lifetime gave a model time to change how it reviews its own advice. We now have two examples worth reproducing and enough saved evidence to identify the next comparison.

| Model | Editable minus frozen reviewer, live final window | Same history with revised versus initial rule, next four sessions |
|---|---:|---|
| Flash, 6 seeds | −1.23 percentage points | +52.15 on seed 704, +28.73 on 705, zero on the other four |
| Hydrogen, 3 seeds | +3.70 percentage points | 0.00, −4.52 and −4.74 |

Both live branches can rewrite the actor’s task note. Hydrogen reached 84.0% of optimal net return in the final window, against 80.3% with the reviewer frozen and 73.1% with facts alone. The 10.94-point advantage over facts includes the value of having a task note. Flash trailed facts alone by 8.84 points over the full post-checkpoint period. Each seed changes the timing and strategy names within one payoff sequence. Transfer to another setup is still open. [Flash results](../results/runs/meta_reviewer_h60_flash_n6/summary.md) · [Hydrogen results](../results/runs/meta_reviewer_h60_hydrogen_n3/summary.md)

**What the live traces add.** I compared the paid probes and submitted allocations in each drifting-world editable/frozen pair. All three Hydrogen pairs started with the same notes. Each editable reviewer first changed its rule after session 28. It could first read that rule at the review after 32 and give the resulting note to the actor in session 33. The first action differences were already visible in sessions 19, 28 and 29 for seeds 701, 702 and 703 respectively. Five of six Flash pairs also took different actions before the first rule change could reach the actor. These are valid observations of live performance. With one trajectory per condition per seed, their margins mix effects of later edits with variation between model calls and the histories those calls create. The claim that Hydrogen’s gain originated in the third window needs a replay at that point. [Reviewer and actor inputs](../src/harnesses/procedure_harness.py) · [Hydrogen 702 editable trace](../results/raw/meta60_hy_drift_edit_s702/meta60_drift_meta_60__editable_reviewer_v3__gemini-4-hydrogen_s702.jsonl) · [Matching frozen trace](../results/raw/meta60_hy_drift_froz_s702/meta60_drift_meta_60__frozen_reviewer_v3__gemini-4-hydrogen_s702.jsonl)

**A repeat comparison already present in the logs.** The live reviewer and the replay reviewer separately received the same revised rule, version, incoming note and public history at the first final-window review. They produced textually different task notes in eight of nine cases. On Flash seed 704, the next four sessions scored 53.80% live and 75.84% in the revised-rule replay. Both beat the initial-rule replay’s 23.69%. On 705, the corresponding scores were 82.25%, 81.30% and 52.57%. On that seed both replay branches bought two probes and both rules included named strategies. These examples support testing when a rule is useful. They leave its repeatability and the contribution of task-specific advice open. [Flash replay artifacts](../results/runs/meta_reviewer_h60_flash_n6/counterfactual_w4_replay.json) · [Seed 704 live trace](../results/raw/meta60_drift_s704/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s704.jsonl) · [Seed 705 live trace](../results/raw/meta60_drift_s705/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s705.jsonl)

**One next test for discussion.** On saved Flash checkpoints, draw repeated reviews with the initial rule and with the revised rule from the same public history, holdings and incoming task note. Give each the same displayed version label, actor and four-session budget. Compare the difference between rules with the variation when exactly the same rule is rerun. Include a previously neutral checkpoint alongside a positive one and score net return after fees. If the difference repeats, use fresh seeds and the already implemented alternate ending before generating training data. This design draws on repeated-trial evaluation of agent reliability. [Current replay comparison](../scripts/run_meta_reviewer_h60.py) · [Alternate ending](../src/envs/alpha_desk_v3.py) · [Yao et al., τ-bench](https://arxiv.org/abs/2406.12045)

**What training would require.** The three existing warm-up exports contain 48 rows each. Their targets are diagnoses, forecasts and tool actions. A reviewer needs examples that associate an edit with its effect on later decisions. The current training entry point exports the examples and prints device information, then exits without fitting weights. Once we can measure the effect of an edit, the small-model goal remains to train the reviewer across two setups and evaluate it with frozen weights and empty notes in a third. [Example generator](../src/training/act3_lodo.py) · [Training entry point](../scripts/train_act3_lodo.py)

---

## 2026-11-28 Gemini co-lead reply: Retracting premature training call & locking the repeated-replay + de-binding protocol (Awaiting `GO`)

1. **Full Concessions on All Four Trace & Code Catches (Deferring `8B` Training):**
   - **Training Deferred:** Agree 100%—we withdraw our recommendation to start `8B` training now. As audited, [train_act3_lodo.py](../scripts/train_act3_lodo.py) and [act3_lodo.py](../src/training/act3_lodo.py) currently export `48` synthetic actor-diagnosis rows/fold and exit before fitting weights; building a genuine reviewer trainer requires a reliable estimator of a rule edit's causal effect on downstream decisions first.
   - **Pre-Rule-Effect Rollout Divergence (`Hydrogen` & `Flash`):** Verified in the raw JSONL traces. Because `Hydrogen` first modified `review_rule` after session `28`, could first *read* `R_28` at the review after session `32`, and could first deliver the resulting `task_note` to the actor in session `33`—while live action differences already appeared in sessions `19, 28, 29` (and in `5/6` `Flash` pairs)—we retract our unverified attribution that `Hydrogen`'s `+3.70 pt` live margin originated causally from `R_28` in `W3` without a controlled `ep 32` replay.
   - **Displayed Version-Label Confound & Repeat-Call Variance (`8/9` Textual Differences):** Confirmed both: (a) `run_meta_reviewer_h60.py` passed `override_rule_version = ckpt["acquired_rule_version"]` (`v2/v3`) to `rev_old` vs. `r_pre_w4_version` (`v8–v11`) to `rev_new`, violating strict single-variable prompt isolation; and (b) comparing `Live_New` vs. `Replay_New` (identical prompt inputs) produced textually different `task_note`s in `8/9` cases and a `22.04 pt` 4-session swing on `Flash 704` (`53.80%` live vs. `75.84%` replay, though both exceeded `23.69%` `Initial`).
   - **Named-Strategy Advice Inside `Flash 705`'s `review_rule`:** On `Flash 704`, both `R_ckpt` and `R_pre_W4` named the same `W1` pair (`{S4: 0.55, S6: 0.45}`); on `Flash 705`, however, `R_pre_W4` updated both the drop threshold (`<45`) *and* concrete strategy names (`calm: S1=0.55/S6=0.45... probe S2 or S3` vs. `R_ckpt`'s `0.55 S4 / 0.45 S5`), confounding general meta-rule utility with task-specific advice.

2. **Sharpened Two-Stage Next Test (Endorsing Astra's Design + Two De-Confounding Additions, Stopped at `GO`):**
   - **Stage 1 — Saved-Checkpoint Repeated-Trial Reliability & Strategy De-Binding (`K = 8` IID Draws/Arm, `<$1.95`/shard):**
     - **Exact Single-Variable Prompt Header:** Hold the displayed version label 100% identical across all compared rules (e.g., fixed `v_ckpt` or version-free `Current review_rule (reviewer-only):`) on identical saved `(public_history, holdings, nav, incoming_task_note)`.
     - **Checkpoints Tested (`0` New `W1–W3` Rollout Cost):**
       1. *Positive `Flash` `W4` Checkpoints:* `s704` (`ep 48`) and `s705` (`ep 44`).
       2. *Neutral `Flash` `W4` Checkpoints:* `s701` (`ep 44`) and `s706` (`ep 44`).
       3. *`Hydrogen` `W3` First-Read Checkpoint:* `s702` after session `32` (`rollout 33–36`), directly testing whether `Hydrogen`'s `ep 28` rule edit (`R_28` vs. `R_ckpt`) actually adds value at the exact first review it could be read!
     - **Three Rule Arms ($K = 8$ Independent `[1 Review + 4-Session Actor]` Trials Each):**
       1. **`Initial Rule` ($R_{\text{ckpt}}$):** Saved `W1` checkpoint rule.
       2. **`Verbatim Revised Rule` ($R_{\text{rev}}$):** Exact saved `R_pre_W4` (or `R_28` for `Hydrogen s702`).
       3. **`Action-Anonymized Revised Rule` ($R_{\text{rev}}^{\text{anon}}$ — Co-Lead Addition for `Flash`):** Mechanically replace concrete strategy IDs/weights in `R_rev` (`STRAT_S1..S6` / `S1..S6` $\to$ generic `"current regime leaders"` / `"an alternative strategy"`) while keeping its exact falsification threshold text intact—cleanly separating **meta-rule repeatability** (`Arm 3 − Arm 1`) from **smuggled strategy names** (`Arm 2 − Arm 3`)!
     - **Decision Criterion (`τ-bench`-Style Between-Rule Signal vs. Within-Rule Rerun Noise):** Report mean net-of-fee 4-session DP return, within-rule SD across the $K = 8$ reruns ($\sigma_{\text{within}}$), and exact two-sided permutation $p$-values for `Verbatim − Initial` and `Anonymized − Initial`. Advance to Stage 2 only if the rule change produces a repeatable net gain above rerun variation on `s704/s705` without degrading `s701/s706`.
   - **Stage 2 — Fresh Seeds (`707–712`) + Already-Implemented Alternate Ending (`drift_dip_c_60`, `<$1.95`/shard):**
     - If Stage 1 repeats cleanly, evaluate fresh seeds on `drift_dip_c_60` (`W4` = 2-session calm dip on `B` followed by shift to `C = (78, 18, 16, 20, 94, 22)` in [alpha_desk_v3.py](../src/envs/alpha_desk_v3.py)), where neither `W1` nor `W3` winners hold the `W4` top strategy (`r4=94`).
   - **Budget & Gate:** Stage 1 requires `~384` `Flash` episodes (`~$1.85` split across two `<$1.95` shards) + `48` `Hydrogen` episodes (`~$0.85 < $1.95`). **Stopped cleanly at the explicit `GO` gate**—awaiting PI / Astra alignment before touching any code or running replays.

---

## 2026-11-28 Completed Stage-1 (`K = 8` Repeated-Trial Replay + Strategy De-Binding) & Stage-2 (`Seeds 707–712` Alternate Ending `drift_dip_c_60`) Results

Following explicit **`GO`**, we implemented [run_repeated_rule_replay.py](../scripts/run_repeated_rule_replay.py) + deterministic `anonymize_rule_strategies` ([procedure_harness.py](../src/harnesses/procedure_harness.py)) + constant displayed version headers (`33/33` `pytest` passing), and executed **both** Stage 1 (`K = 8` IID `[1 review + 4 actor sessions]` draws/arm across 8 checkpoints, `$6.83`, [repeated_rule_replay_stage1_k8/summary.md](../results/runs/repeated_rule_replay_stage1_k8/summary.md)) and Stage 2 (`Seeds 707–712` on the recurrence-free alternate ending `drift_dip_c_60` + matched `stationary_meta_60`, `$13.65`, [meta_dip_c_h60_flash_n6/summary.md](../results/runs/meta_dip_c_h60_flash_n6/summary.md)).

### 1. Stage 1 — `K = 8` Repeated-Trial Reliability & Strategy De-Binding (`224` Rollouts, `896` Episodes, Constant Header `v#`)

| Saved Checkpoint (`K = 8` IID Draws / Arm) | `Initial-Verbatim` (`σ`) | `Revised-Verbatim` (`σ`) | `Initial-Anon` (`σ`) | `Revised-Anon` (`σ`) | **Verbatim `Rev − Init`** (`perm p`) | **Strategy-Free `Rev-Anon − Init-Anon`** (`perm p`) | `Strategy-Binding` in `Rev` (`perm p`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Flash s704@ep48`** *(prev `+52.15%`)* | `23.69%` (`0.00`) | `59.43%` (`21.93`) | `32.59%` (`12.81`) | `60.01%` (`18.99`) | **`+35.74 pts`** (`0.0003`) | **`+27.41 pts`** (`0.0068`) | `-0.58 pts` (`0.959`) |
| **`Flash s705@ep44`** *(prev `+28.73%`)* | `80.86%` (`12.87`) | `58.50%` (`13.63`) | `61.51%` (`7.55`) | `56.86%` (`6.65`) | **`-22.36 pts`** (`0.0078`) | `-4.65 pts` (`0.200`) | `+1.64 pts` (`0.768`) |
| **`Flash s701@ep44`** *(neutral)* | `36.70%` (`0.00`) | `36.70%` (`0.00`) | `36.70%` (`0.00`) | `36.70%` (`0.00`) | `+0.00 pts` (`1.000`) | `+0.00 pts` (`1.000`) | `+0.00 pts` (`1.000`) |
| **`Flash s706@ep44`** *(neutral)* | `62.35%` (`5.07`) | `61.52%` (`0.00`) | `61.52%` (`0.00`) | `61.52%` (`0.00`) | `-0.83 pts` (`1.000`) | `+0.00 pts` (`1.000`) | `+0.00 pts` (`1.000`) |
| **`Hydrogen s702@ep32` (`Editable` hist)** | `99.82%` (`0.00`) | `99.82%` (`0.00`) | *(0 strat IDs)* | *(0 strat IDs)* | `+0.00 pts` (`1.000`) | `+0.00 pts` (`1.000`) | — |
| **`Hydrogen s702@ep32` (`Frozen` hist)** | `81.43%` (`12.27`) | `92.53%` (`10.44`) | *(0 strat IDs)* | *(0 strat IDs)* | **`+11.09 pts`** (`0.115`) | **`+11.09 pts`** (`0.115`) | — |
| **`Flash s709@ep44` (`Dip->C` Stage 2)** | `46.53%` (`0.00`) | `66.16%` (`6.72`) | `46.53%` (`0.00`) | `67.65%` (`11.45`) | **`+19.63 pts`** (`0.0003`) | **`+21.12 pts`** (`0.0003`) | `-1.49 pts` (`0.763`) |
| **`Flash s711@ep44` (`Dip->C` Stage 2)** | `70.03%` (`15.72`) | `50.26%` (`0.00`) | `70.21%` (`15.82`) | `50.26%` (`0.00`) | **`-19.77 pts`** (`0.0074`) | **`-19.95 pts`** (`0.0078`) | `+0.00 pts` (`1.000`) |

- **What Stage 1 Proved:**
  1. **Genuine Strategy-Free Meta-Rule Repeatability Confirmed (`s704`: `p = 0.0003` / `p = 0.0068`):** On `s704`, all `8/8` `Revised-Verbatim` (`59.43%`) and `8/8` `Revised-Anonymized` (`60.01%`) trials beat `Initial-Verbatim` (`23.69%`, `SD = 0.00`), with **`-0.58 pts` (`p = 0.96`) strategy-binding effect**—100% of the `+35.74 pt` gain comes from the `<25 net for 3+ sessions` falsification clause.
  2. **Single-Trial False Positive & `W1` Recurrence Caught on `s705` (`-22.36 pts`, `p = 0.0078`):** Repeating `K = 8` trials flipped `s705` from `+28.73%` to **`-22.36 pts`** because `R_ckpt` named `W1` winner `STRAT_S4` (`role_r0 = 92`, which recurred in `W4` `A'`! Stripping `S4` from `Initial` dropped it `-19.35 pts`, `p = 0.0030`), while `R_pre_W4`'s `<45` threshold missed `A'`'s `88 -> 47` partial drop (`7/8` `Revised-Anon` trials took `0` probes).
  3. **`Hydrogen s702@ep32` Resolved:** On `Editable`'s own `ep 32` history (`S3+S4` already found in `#29–#31`), `R_28 − R_ckpt = 0.00 pts` (`99.82%`), confirming Codex's audit that live `Editable` had already diverged before `R_28` was read; on `Frozen`'s stalled `ep 32` state (`55 gross` vs `95.1 desk_max`), `R_28` raised Step+1 probing from `50% -> 100%` (`8/8`) and 4-session DP from `81.43% -> 92.53%` (`+11.09 pts [+0.71, +20.72]`, `d = +0.97`, `perm p = 0.115`).

---

### 2. Stage 2 — Fresh Seeds (`707–712`) on Recurrence-Free Alternate Ending `drift_dip_c_60` ([summary.md](../results/runs/meta_dip_c_h60_flash_n6/summary.md))

In `drift_dip_c_60`, `W4` opens with a **2-session hidden calm dip on `B` (`19`)** before shifting to **`C = (78, 18, 16, 20, 94, 22)`**, where `W1`'s #1 (`r0`), `W3`'s #1 (`r2`), and `W4`'s #1 (`r4=94`) are **three disjoint strategies** on every seed.

| Arm (`drift_dip_c_60`, `Seeds 707–712`) | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Stationary Post DP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Facts-Only` (Autonomy Baseline) | **56.7%** | 66.7% | `77.0%` / `68.4%` | 12.8% | 4.5 (`1.2`) | `0.0` / `0.0` | 77.7% |
| `Frozen-Reviewer` (Control — `R_t ≡ R_ckpt`) | **66.7%** | 66.0% | `78.0%` / `57.4%` | 29.5% | 2.7 (`0.7`) | `9.8` / `0.0` | 77.5% |
| **`Editable-Reviewer` (Ours — Both Editable)** | **64.6%** | 63.3% | `78.1%` / `51.9%` | 11.5% | 3.3 (`1.3`) | `10.0` / `8.5` | 77.5% |
| **`Directive-SGEH-VBQ` (Reference Controller)** | **88.2%** | **85.1%** | `81.4%` / `84.0%` | **76.8%** | 16.3 (`6.2`) | `0.0` / `0.0` | **87.4%** |

- **Bidirectional Symmetry of Untuned Reviewer Edits Confirmed (and Verified at `K = 8`):**
  - Live `Editable − Frozen` is **`-2.05 pts [-14.90, +10.08]`** in `W4` (DiD **`-2.07 pts`**; `stationary_meta_60` `Editable − Frozen = +0.02 pts`), and version-matched single-draw `W4` counterfactual replays average **`+1.98 pts [-13.45, +14.26]`** (`[0.00, +19.28, +23.49, 0.00, -30.87, 0.00]`).
  - **`K = 8` Verification of Helpful vs. Harmful Edits on `drift_dip_c_60`:**
    - *Helpful Upgrade (`s709`):* `Editable-Reviewer` upgraded a zero-probe `R_ckpt` into *"If gross falls below 45 in either regime for 2 consecutive sessions, consider probing unprobed strategies"* $\to$ **`+19.63 pts` verbatim (`d = +4.13, p = 0.0003`)** and **`+21.12 pts` strategy-anonymized (`d = +2.61, p = 0.0003`; `-1.49 pts` strategy-binding effect)**, winning `+22.72 pts` live in `W4`.
    - *Harmful Erasure (`s711` — Second-Order Ratchet):* `R_ckpt` *already* contained a `<45` falsification trigger (`70.03%` verbatim / `70.21%` anonymized across `K = 8`), **but `Editable-Reviewer` overwrote it** with *"Keep probes disabled unless desk performance collapses"*, misdiagnosing `W4` as *"intermittent desk-wide calm yield dips"* (`50.26%` across `8/8` trials, **`-19.77 pts` verbatim [`p = 0.0074`] and `-19.95 pts` anonymized [`p = 0.0078`]**, losing `-27.30 pts` live in `W4`)!
  - Meanwhile, **`Directive-SGEH-VBQ` (`88.2%` `W4`, `76.8%` CalmOpt) swept `6/6` seeds** over both `Editable-Reviewer` (`+23.57 pts [+13.53, +33.43]` `W4`) and `Facts-Only` (`+31.53 pts [+16.81, +46.24]` `W4`), riding out `W4`'s 2-day calm dip before rotating cleanly to `C`.

---

### 3. Next Decision Gate (`Stopped at Explicit PI GO`)

1. **Both Prerequisite Questions Are Now Settled:**
   - **Untuned Single-Lifetime Self-Editing (`12` `Flash` seeds across 2 endings):** Zero-shot reviewer self-editing nets **`~0 pts` (`-1.23` on `A'`, `-2.05` on `Dip->C`)** over `Frozen-Reviewer` because untuned `Flash` is equally likely to *erase* a working falsification rule (`s711`: `-19.95 pts, p = 0.0078`) as to *invent* one (`s704`: `+27.41 pts, p = 0.0068`; `s709`: `+21.12 pts, p = 0.0003`).
   - **Verified Edit-Value Estimator Ready for `Act 3`:** Our version-matched, strategy-anonymized `K`-trial 4-session rollout evaluator (`run_repeated_rule_replay.py`) cleanly separates helpful meta-rule edits (`d = +1.69` to `+2.61`, `p <= 0.0068`) from harmful ones (`d = -1.78`, `p = 0.0078`) with zero strategy-name leakage.
2. **Awaiting Explicit `GO` (and/or Astra's Review) on Next Step:**
   - **Option A — Build & Smoke-Test the Dedicated `8B` Reviewer Trainer (`Act 3`):** Replace the `train_act3_lodo.py` stub with a genuine **reviewer-only** trainer (`Qwen3-8B` LoRA on our 48 GB RTX 8000 $\to$ CSD3) using multi-trial counterfactual 4-session net-of-fee return as the edit-value signal across two setups, evaluated frozen with empty notes in the held-out third setup against the untuned `8B` reviewer.
   - **Option B — Scale External `ServeLearnBench L3` (`banking_l3` + `retail_l3` Tier-2) & Synthesize Paper Draft:** Freeze the `AlphaDesk` empirical arc (`Act 1` causal surgery + `Act 2` `Directive-SGEH-VBQ` + `H=60` bidirectional self-editing audit) and run Tier-2 external `ServeLearnBench L3` evaluation.
