# Research Roadmap: Hypothesis, Falsification, Confidence, Heuristics

**Status:** Active north star (2026-08)  
**Supersedes as *question*:** env zoo / “LLM vs UCB scoreboard”  
**Does not replace:** engine docs (`01`–`04`), staged design (`07`), session log (`08`), early heuristic notes (`05`)

Bandits are the **first microscope**, not the identity of the project. Later: other long-horizon tasks with the **same loop**.

---

## 1. Core question

Can a **frozen** LLM, over many interaction turns, with only feedback (no weight updates, no gold labels):

1. **Form a hypothesis** about the world (reward law, best action, rule, …)
2. **Assign confidence** to that hypothesis
3. **Choose actions** via a **heuristic** (policy sketch) — either **self-evolved** or **taught**
4. **Falsify** the hypothesis (and/or heuristic) when the world **changes**
5. **Re-form** a new hypothesis and adapt

If **no** → we **teach** (prompt scaffolds, ε-greedy, change-detection rules, summaries).  
If **yes** → we **quantify**: lock time, falsification delay, re-lock time, confidence calibration.

This is **in-context hypothesis revision under non-stationarity**, not “build a better UCB.”

---

## 2. What we already have (setup, not a paper)

Engine: frozen Gemini (3.5 Flash), raw history, JSON `{action, confidence, rationale}`, classical baselines, plots, prompt **blind / aware / minimal**.

| Cell | Result (actions / regret) |
|---|---|
| Stationary / NumberGuess | Competent |
| Reversal | Strong (prompt partly leaked “may change”) |
| Periodic / PhaseLocked / Drift | Mixed; periodic weak |
| Pendulum / Traffic | Too easy (full state) |
| **Staged P0** (calendar best-arm flips) | LLM **>>** full-history UCB even **blind** |
| **P1 / P2** | Modest / **no** edge on pure sinusoids |
| **P0-long** (H=180) | Edge **held** |
| **ExploitDecay** | LLM **≈ SW-UCB**, **overmines** — first hardness cell |

**Caveat:** we measured **policy** (pulls, regret, recovery delay), not **declared hypotheses**. Rationales look like empirical means + recency. `confidence` is logged, unused.

**Literature (not scooped if we measure the loop):**

- LLM-as-stationary-bandit / exploration: Krishnamurthy et al. (NeurIPS 2024) — raw history often fails; needs CoT + summaries
- ICRL from rewards: Monea et al. (COLM 2025); Song et al. (ICLR 2026)
- Thin non-stationary LLM vs humans (E&E); TextBandit leaves non-stationary as future work
- Classical piecewise-stationary bandits detect+restart (CUSUM/GLR) — algorithms, not LLM hypothesis lifecycle

**Our slot:** lock / falsify / re-lock + confidence + **self-evolved vs taught heuristic**, first in a toy, then other long-horizon tasks.

---

## 3. Phases

```text
Phase 0   INSTRUMENT the loop on P0          DONE (n=20)
Phase 1   QUANTIFY delays vs signal            next (L, Δp)
Phase 2   TEACH vs SELF-EVOLVE heuristics      started (detect vs surprise)
Phase 3   LEAVE THE TOY (one other long-horizon task)
```

Park: more physics toys, more sinusoid variants, proprietary CoT, “P0 with even more stages” as a main line (A already showed length doesn’t break action-level adaptation), CEO-Bench clone, tool-calling harness.

---

## 4. Phase 0 — Imminent (instrument)

**Goal:** Make the setup **ask the core question**, not add environments.

**Env:** StagedBandit P0 — 3× constant, distinct best arms, \(L=30\), \(H=90\), **`description_mode: blind`**.  
Hypothesis is checkable: “arm \(i^\star\) is best.”

**LLM output every step (forced JSON):**

```json
{
  "hypothesis": {
    "type": "best_arm",
    "best_arm": 2,
    "stable": true
  },
  "confidence": 0.8,
  "heuristic": "exploit_best | epsilon_greedy | round_robin | custom",
  "heuristic_text": "optional short rule",
  "action": {"arm": 2}
}
```

Start with **best_arm** only (not full \(p\) vector). Keep `action` as what the env executes.

**Primary metrics:**

| Metric | Definition (sketch) |
|---|---|
| **Lock time** | Steps until `hypothesis.best_arm` matches oracle best for \(K\) consecutive steps (per stage) |
| **Falsification delay** | After stage cut \(\tau\): steps until hypothesis **leaves** the old best (or `stable=false`) |
| **Re-lock time** | After falsify: steps until hypothesis matches **new** best for \(K\) steps |
| **Confidence calibration** | Among steps with confidence in bin \(c\), fraction where hypothesis is correct — especially in a window around \(\tau\) |

Secondary: regret, overcommit, action recovery delay (already have).

**Seeds:** ≥20 for this one cell (5 was a smoke test). One model (`gemini-3.5-flash`) first.

**Baselines:** random, greedy, UCB, SW-UCB, oracle (no hypothesis fields; action-only). LLM conditions in Phase 0: **free** structured output, still **blind** task text.

**Success of Phase 0:** plots of hypothesis vs oracle over time; three delay histograms; reliability diagram for confidence. Then we know whether to **quantify** (Phase 1) or **teach** (Phase 2) first.

### 4.1 Observation interface (raw history vs stats vs tools)

**Decision (2026-08):** Do **not** start with a tool-calling harness. Tools add a second skill (whether the model *queries* stats) on top of hypothesis revision.

Treat **what the model sees** as an experimental factor. Same `{hypothesis, confidence, heuristic, action}` schema:

| ID | Prompt contents | Tests |
|---|---|---|
| **R** `raw` | Full action–reward transcript | Pure ICRL; long-context counting |
| **S** `stats` | Engine-computed per-arm \(n\), mean, last-\(k\) rewards | Revision given sufficient statistics |
| **S+R** `stats_plus_recent` | Stats table + last \(k\) raw steps only | Default for Phase 0 |

Stats are **what a human could tally from the same rewards** — not \(p_t\), not `best_arm`, not “the regime changed.” Krishnamurthy et al. (2024) found raw history often fails exploration unless sufficient stats are provided; we **ablate** that instead of picking one vibe.

**Default first run:** `stats_plus_recent` with `history_tail: 10`, **blind**. Then add **R vs S** before claiming they can/can’t falsify.

**Later (not Phase 0):** optional tools (`mean(arm)`, `counts()`) as a *self-evolved tooling* condition in Phase 2/3.

Config knobs: `interface: raw | stats | stats_plus_recent`, `history_tail: 10`.

---

## 5. Phase 1 — Quantify

If hypotheses actually track / falsify:

- Vary gap \(\Delta p\), stage length \(L\), number of flips, \(H\)
- Plot delay vs signal strength
- Optional second cell: ExploitDecay (self-caused death) with the same schema

Do **not** broaden task class until this is clean.

---

## 6. Phase 2 — Teach vs self-evolve

Same P0 env and schema. Prompt conditions:

| ID | What we install |
|---|---|
| **Free** | Schema only; invent heuristic in `heuristic` / `heuristic_text` |
| **Taught explore** | ε-greedy or round-robin in the prompt |
| **Taught detect** | “If recent rewards on hypothesized arm collapse, set stable=false and re-explore” |
| **Taught surprise** | H is a model; if recent pulls of that arm are surprising, drop H. Does **not** name `mean_w`. |

Compare lock / falsify / re-lock / calibration.  
n=20 (2026-08-23): detect speeds falsify **and** reward; surprise speeds falsify only. Exploration teach is lower priority (not the bottleneck).

`05_heuristic_memory_experiments.md` is an earlier sketch of heuristic memory; Phase 2 is the **self vs taught** contrast aligned with the new question.

---

## 7. Phase 3 — Leave the toy

Same JSON loop on **one** other long-horizon task (delayed reward, hidden-rule tool use, changing mechanic). Claim: the **loop** transfers, not the bandit.

---

## 8. Prompt hygiene (locked)

| Mode | Use |
|---|---|
| `blind` | Default fair measurement |
| `aware` | Ablation only |
| `minimal` | Ablation only |

Fair claims = blind. Teaching heuristics is an **explicit condition**, not hidden in the task description.

No proprietary Gemini thinking traces. Structured `hypothesis` / `heuristic` is the process evidence.

---

## 9. Related files

| File | Role |
|---|---|
| `07_staged_mab_design.md` | Staged env spec |
| `08_next_directions.md` | P0-long / ExploitDecay session log |
| `05_heuristic_memory_experiments.md` | Earlier heuristic-memory conditions |
| `03_agent_protocol.md` | Current JSON (action + confidence + rationale) |
| This file | **Question + phases** |

---

## 10. Imminent work (Phase 0 — implemented)

| Piece | Location |
|---|---|
| Interface helpers | `llm_rl_engine/agents/history_views.py` |
| Prompt | `llm_rl_engine/prompts/hypothesis_loop.txt` |
| Agent | `llm_hypothesis` / `prompt_mode: hypothesis_loop` |
| Metrics | `lock_time`, `falsification_delay`, `relock_time`, `confidence_calibration` |
| Plot | `plots/hypothesis_loop.png` |
| Config | `llm_rl_engine/configs/experiment_p0_hypothesis.yaml` |

First smoke (`experiment_p0_hypothesis.yaml`, 2026-08-21): H locks in stage 0, **fails to falsify** late; overconfident. Next run ablates interface + taught detect.

```powershell
uv run llm-rl-benchmark --config llm_rl_engine/configs/experiment_p0_hypothesis.yaml
uv run llm-rl-benchmark --config llm_rl_engine/configs/experiment_p0_hypothesis_ablate.yaml
```

**Ablation agents:** `llm_hyp_raw` (R), `llm_hyp_stats` (S), `llm_hyp_sr` (S+R + recency means), `llm_hyp_taught_detect` (S+R + recency + “drop H if recent mean collapses”).

---

## 11. Session 2026-08-23 — n=20 lock + surprise teach

Blind P0, Gemini 3.5 Flash, T=0, H=90, L=30, 3 constant stages. Hypothesis JSON every step. Episodes now flush as they finish (`run_benchmark.py`) so a killed run does not lose completed seeds. Long jobs are 10-seed shards.

CEO-Bench (Chen et al., arXiv:2606.18543) is a fat SaaS sim, not this loop. **Not a scoop** if we keep lock / falsify / re-lock + taught vs self-evolved as the object.

### 11.1 n=20 vanilla raw vs taught detect

Smoke (n=5, S+R free) overstated “fails to falsify.” At n=20, vanilla **does** leave the old H on every cut (40/40) and usually re-locks. Teaching does not create the loop; it **speeds the kill**.

| Agent | Reward | H correct | Stage-0 lock | Falsify delay | Re-lock cut 1 / 2 |
|---|---:|---:|---:|---:|---|
| Oracle | 75.8 ± 4.4 | — | — | — | — |
| Taught detect | **61.9 ± 6.9** | **73%** | 4.7 | **3.2** | **9.9 / 11.1** |
| Vanilla raw | 57.4 ± 7.4 | 65% | 4.7 | 6.7 | 11.6 / 15.1 |
| UCB | 39.0 ± 7.0 | — | — | — | — |
| SW-UCB | 38.7 ± 4.1 | — | — | — | — |

Paired detect − raw: **+4.6** (se 1.7; detect wins 14/20). Stage-0 lock is identical. Both still crush full-history UCB. Vanilla is more overconfident (conf 0.74 vs 65% correct; detect 0.70 vs 73%).

Runs: `runs/2026-08-23_benchmark_p0_hypothesis_n20_{a,b}` (gitignored).  
Configs: `experiment_p0_hypothesis_n20.yaml` and shards `_a` / `_b`.

### 11.2 Surprise-meta teach (n=20, new agent only)

Prompt `hypothesis_taught_surprise.txt`: H is a model of which arm is best *right now*; confidence tracks recent pulls of that arm, not n; if those pulls are surprising, `stable=false` and explore. **Does not** name `mean_w`, collapse, or regimes. Same S+R interface as detect.

| Agent | Reward | H correct | Falsify delay | Re-lock cut 1 / 2 |
|---|---:|---:|---:|---|
| Taught detect | **61.9** | **73%** | 3.2 | **9.9 / 11.1** |
| Taught surprise | 57.7 | 66% | **3.6** | 12.5 / 12.7 |
| Vanilla raw | 57.4 | 65% | 6.7 | 11.6 / 15.1 |

Paired vs raw: detect +4.6 (14/20); surprise **+0.4** (8/20). Surprise **matches detect on killing H** and **matches vanilla on score**.

Mechanism (seed 0, first cut): detect stays `stable=false` and follows `mean_w` until the new arm locks. Surprise *notices* (“three 0s, surprising (p ~ 0.03)”) then **restores `stable=true`** when other arms also return 0 on tiny n — lifetime mean wins again. Raw never computes surprise.

**Claim to keep:** belief revision is teachable in English without a P0 cheat sheet. Harvesting the new arm is not; that still needs the recency detector (or something that stops the revert to high-n means).

Runs: `runs/2026-08-23_benchmark_p0_hypothesis_surprise_{a,b}`.  
Analyze: `uv run python scripts/analyze_hypothesis_teach.py`

### 11.3 Next (resume here)

Do **not** add more P0 English variants that name `mean_w`. The detect vs surprise split is enough on this cell.

1. **Stop the revert (preferred next).** Keep surprise-style English (or none). Put an engine **surprise / likelihood of last-k under the declared H** in the observation — not a tool the model must call. Question: if surprise is computed for them, do they keep `stable=false`, or still restore from n?
2. **Timescale axis.** Same two poles (vanilla raw, taught detect), L=10 vs L=30 (optional L=5 as negative control). Expect: L=30 re-lock; L=5 only a confidence / `stable` dip, no pretty new H.
3. **Transfer.** Same hypothesis schema on **ExploitDecay** (already a hardness cell at the action level). Did we teach “prefer recent mean” or “revise H”? Recency-greedy should **hurt** here.
4. Park: ε-greedy teach (exploration was not the bottleneck), tools / code, CEO-Bench-like fat worlds, more sinusoids.

Vanilla already has a heuristic: lifetime greedy, stationary prior, in a non-stationary world. That is the diagnosis, not the research goal.
