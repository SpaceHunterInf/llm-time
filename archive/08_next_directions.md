# Next Directions: Mechanism, Hardness, Logging

**Status:** Logged 2026-07-31 · **superseded as the live log** by `09_research_roadmap.md` §11 (2026-08-23: n=20 lock, surprise teach, next probes)  
**This file:** P0-long / ExploitDecay session notes. Do not treat A/B here as the current queue.

**Related:** `07_staged_mab_design.md`, `09_research_roadmap.md`, prompt ablations P0–P2, `reports/benchmark_50turns_gemini35_v3.md`

---

## 0. Context (where we left off)

Frozen Gemini (`gemini-3.5-flash`, raw history) on staged bandits:

| Finding | Evidence |
|---|---|
| Strong on discrete best-arm flips | P0: LLM >> UCB even under **blind** prompts |
| Modest cross-family edge | P1 blind still beats UCB slightly |
| Soft fail on pure multi-sinusoid | P2 ≈ SW-UCB under all prompts |
| Regime-change **coaching** | Speeds recovery / cuts overcommit; **not** required for P0 win |
| Prior P0–P2 totals | Originally **aware**-contaminated; ablations correct the map |

Open worries:

1. Is the LLM forming a real **pattern/alpha**, or shallow history heuristics?
2. Can we find a clean **hard fail** (not just soft P2)?
3. We lack proprietary **thinking** traces — only JSON rationales.

---

## 1. Mechanism: pattern / alpha vs “feeling”

### 1.1 What we already log

| Field | Logged? | Notes |
|---|---|---|
| `prompt` | Yes | Full raw history each step |
| `raw_response` / `parsed_response` | Yes | Includes `rationale`, `confidence` |
| Gemini internal thinking tokens | **No** | Proprietary / not exposed for our API path |

Treat `rationale` as **weak process evidence** (can be post-hoc), not true CoT.

### 1.2 Informal evidence (offline sample)

P0 ablation `llm_blind` seed 1 rationales show:

- Explicit **success-rate counting** (“22/25 ≈ 88%”)
- **Sticky full-history means** right after stage cuts
- Occasional **hypothesis language** (“non-stationary or lower true p”)
- Failure mode: switch back using **lifetime** means (overcommit)

**Hypothesis:** shallow alpha = empirical means + recency, **not** latent stage/period model.

### 1.3 Probes (not yet run)

| Probe | Goal | Cost |
|---|---|---|
| **Rationale coding** at boundaries | Categories: means / recency / “env changed” / explicit rule | $0 (use existing trajs) |
| **Forced belief schema** | `belief: {best_arm, p_est, regime_stable}` + score vs oracle | Small re-run |
| **History masking** | Hide last-k rewards post-change — does switch still happen? | Medium |
| **Cross-episode transfer** | Fresh context each ep — no multi-ep learning expected | Low |

---

## 2. Hardness: scenarios where LLM should struggle

Task structure dominates prompt. Soft fail already: **P2 pure sinusoids**.

| ID | Scenario | Why it should hurt | Priority |
|---|---|---|---|
| **A. P0-long** | More stages, longer H (e.g. H=180, L=30, 6 stages) | Context bloat; old counts dominate; recovery degrades late | **DO FIRST** |
| **B. ExploitDecay** | Pulling good arm lowers its p; new alpha spawns | Self-caused death; credit assignment | Next env |
| **C. Delayed / sparse reward** | Feedback lag or batch reward | Breaks step-wise counting | Later |
| **D. Large K** | 20–50 arms | Exploration budget | Later |
| **E. Compositional rule** | Best arm = f(t, last r); rule edits mid-ep | Needs real hypothesis class | Later |
| **F. Adversarial flip** | Flip when agent locks in | Punishes sticky exploit | Later |

**Avoid:** more full-observation timing toys (Pendulum/Traffic ceiling).  
**Always:** `description_mode: blind` for fair hardness claims.

---

## 3. Logging / CoT policy

- Keep logging JSON `rationale` + prompts.
- Document: **no proprietary CoT** available via current Gemini path.
- Future: optional structured `belief` field + metrics; do not block on hidden thinking APIs.
- Primary science = **behavior** (reward, regret, recovery, overcommit); rationales secondary.

---

## 4. Recommended proceed order

```text
DONE     Staged P0/P1/P2 + prompt ablations (blind/aware/minimal)
DONE     A. P0-long under blind (still strong LLM)
DONE     B. ExploitDecayBandit + blind (LLM ≈ SW-UCB, high overmine)
NEXT     Offline rationale audit (existing trajs)
LATER    Structured belief prompt ablation (P0 short)
OPTIONAL Harder decay, delayed reward, large K
```

---

## 5. Experiment A — P0-long (this session)

### Goal

Stress-test discrete best-arm falsification when:

- More regime changes (more chances to fail recovery)
- Longer episode (longer history → mean contamination)

### Spec (pilot)

| Knob | Value | Rationale |
|---|---|---|
| Family | 6× `constant`, distinct best arms | Same as P0, more flips |
| `stage_length` L | 30 | Match P0 stage length |
| Horizon H | **180** | 6×30; 2× original staged H |
| Seeds | 0–4 | Match suite |
| Prompt | **blind only** | Fair hardness |
| Agents | random, greedy, ucb, thompson, SW-UCB(w=10), oracle, `llm_raw_history` | Standard |

**Config:** `llm_rl_engine/configs/experiment_staged_p0_long.yaml`  
**Success / fail criteria (descriptive):**

| Outcome | Interpretation |
|---|---|
| LLM still >> UCB on total reward + recovers most stages | Adaptation scales to 6 flips @ H=180 |
| Edge shrinks; late-stage overcommit / recovery fails | Length/context is the cliff |
| LLM ≈ UCB or worse late | Hard fail on stretched P0 |

### Cost estimate

- 1 LLM × 5 seeds × 180 steps ≈ **900** API steps (~2× P0 single-condition)
- Roughly ~$0.25–0.35 depending on tokens/step growth with history

### After A (resume later)

1. Analyze stage-wise reward / recovery / overcommit for early vs late stages  
2. Rationale coding on boundary steps  
3. Design B (ExploitDecay) if A still “too easy”

---

## 6. Experiment B — ExploitDecay (**implemented + run**)

**Idea:** Toy alpha mining + decay.

- One arm starts strong; each pull of the **active alpha** lowers its \(p\) by `decay_delta`  
- When \(p <\) `death_threshold`, spawn a new alpha on a different arm  
- Public obs: arm + reward only; **blind** task text  

**Code:** `llm_rl_engine/envs/exploit_decay_bandit.py`  
**Config:** `llm_rl_engine/configs/experiment_exploit_decay.yaml`  
**Run:** `runs/2026-08-01_benchmark_exploit_decay/`  

**Defaults:** `strong_p=0.85`, `weak_p=0.20`, `decay_delta=0.08`, `death_threshold=0.35`, `p_floor=0.15`, H=90, 5 seeds  

### 6.1 B results snapshot (2026-08-01)

| Agent | Reward | Regret | α deaths | Overmine | Recovery delay |
|---|---:|---:|---:|---:|---:|
| oracle | **52.4** | 0.0 | 12.0 | 12% | 1.0 |
| **sliding_window_ucb** | **29.6** | **24.1** | 3.8 | **16%** | 9.4 |
| **llm_raw_history** | **28.4** | 34.2 | 4.8 | **49%** | 12.7 |
| random | 24.8 | 27.5 | 3.2 | 21% | 20.2 |
| ucb | 22.6 | 42.6 | 2.2 | 73% | 23.9 |
| thompson | 16.6 | 52.1 | 1.8 | 81% | 13.2 |
| greedy | **13.6** | 58.4 | 0.2 | **100%** | — |

| Metric | Value |
|---|---|
| LLM / oracle | **54.2%** (much worse than P0 ~74%) |
| LLM − SW-UCB | **−1.2** (tie / slight lose) |
| LLM − UCB | **+5.8** |
| Cost | ~450 LLM steps, blind |

**Reading:** This is the **hard fail / soft cliff** we wanted for *use-dependent* change. Full-history greedy/UCB overmine dead veins. LLM rematches SW-UCB on total reward but **overmines far more** (49% vs 16%) and recovers slower than the window baseline. Not the P0-style dominant win.

**Question answered:** LLM partially remine, but **over-mines** more than forgetful classical methods — shallow sticky means still dominate after self-caused death.

---

## 7. Prompt hygiene (locked)

| Mode | Use |
|---|---|
| `blind` | Default fair experiments going forward |
| `aware` | Ablation only (coaching) |
| `minimal` | Ablation only |

Original staged P0–P2 totals = aware-era; cite ablations for fair claims.

---

## 8. Session log

| Date | Action |
|---|---|
| 2026-07-31 | Ideas logged in this file |
| 2026-07-31 | **A (P0-long) config + run completed** |
| 2026-08-01 | **B (ExploitDecay) implemented + blind benchmark completed** |
| — | Resume: rationale audit; optional harder ExploitDecay / delayed reward |

### 8.1 A results snapshot (2026-07-31)

**Run:** `runs/2026-07-31_benchmark_staged_p0_long/`  
**Config:** `llm_rl_engine/configs/experiment_staged_p0_long.yaml`  
**Setup:** H=180, 6× constant, L=30, `description_mode: blind`, 5 seeds  

| Agent | Mean reward | Mean regret |
|---|---:|---:|
| oracle | 151.0 | 0.0 |
| **llm_raw_history** | **112.0** | **38.2** |
| sliding_window_ucb | 78.0 | 73.4 |
| ucb | 66.8 | 83.6 |
| thompson / greedy / random | ~60–61 | ~89 |

| Metric | Value |
|---|---|
| LLM / oracle | **74.2%** (similar to short P0 blind ~71%) |
| LLM − UCB | **+45.2** |
| LLM − SW-UCB | **+34.0** |
| Cost | ~$0.47 (900 steps; ~3.1k input tok/step avg) |

**Stage means (LLM):** R0–R5 ≈ 18.6, 18.0, **12.4**, 18.6, 20.6, **23.8**  
**Early vs late (sum of 3 stages):** LLM early 49.0 → late **63.0** (improves late); UCB flat ~33.

**Reading (session end):** Stretching to 6 stages / H=180 did **not** create a hard fail. LLM remains clearly best non-oracle under blind (~74% oracle, +45 vs UCB). SW-UCB improves vs short P0 (78 vs ~36 on H=90) but still far behind LLM. Mid-episode dip (stage 2) + high overcommit exist, but recovery continues through stage 5. **Cliff not found on A alone.**

**Resume later:**
1. Optional deeper recovery/overcommit plots for A  
2. Offline rationale audit (existing trajs)  
3. **B. ExploitDecayBandit** under blind  
4. Only if needed: more stages / larger K / delayed reward
