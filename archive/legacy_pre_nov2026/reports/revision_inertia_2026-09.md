# Session log: belief revision, controls, and inertia (Sept 2026)

**Question.** How do belief revision and behavioural adaptation interact when the world stops satisfying the agent’s previous hypothesis?

**Setup (unchanged).** Frozen Gemini 3.5 Flash, T=0, blind task text, hypothesis JSON every step (`best_arm`, `stable`, `confidence`, `heuristic`, `action`). Bandits are the microscope. Fair claims use blind prompts.

This note covers work after the 24 Aug group meeting (`reports/group_meeting_2026-08-24.md`). Raw run logs stay in `runs/` (gitignored).

---

## 1. What we already had (n=20, fixed P0 cuts at 30 and 60)

Four prompts on the same piecewise-constant bandit:

| Prompt | What we installed |
|---|---|
| Vanilla raw | Schema only; full history |
| Taught surprise | H is a model; drop it if recent pulls are surprising |
| Surprise + p-value | Same + engine binomial tail under lifetime mean |
| Taught detect | Prefer last-10 mean; collapse ⇒ `stable=false`; do not restore from large n |

Detect was best on **reward** when the world actually flipped. Surprise matched detect on **killing H quickly** but not on score (they noticed surprise, then put H back). The p-value made revision stickier, not faster.

That still mixed *doubt* with *durable abandonment*.

---

## 2. Split the loop: doubt / retract / resurrect

New metrics (also used on the old n=20 traces, no extra API):

- **Doubt:** first `stable=false` or declared `best_arm ≠` old best
- **Retract:** first 3 consecutive steps off the old H
- **Resurrect:** return to the old H after having left it
- **Search:** retract → re-lock on the new best
- **Belief–action consistency:** whether pulls match declared H

On the original n=20 P0 cuts (40 events):

| | Doubt | Retract | Retract − doubt | Resurrect | Declare new, still pull old |
|---|---:|---:|---:|---:|---:|
| Vanilla | 6.7 | 11.3 | 4.5 | **68%** | ~0 |
| Surprise | **3.6** | 12.1 | **8.5** | 45% | 0.02 |
| Surprise + p-value | 4.5 | 11.0 | 6.5 | 55% | 0.01 |
| Detect | **3.2** | **9.5** | 6.2 | 45% | 0.01 |

Everyone **doubts**. Surprise is fastest to doubt and slowest to **stay off**. Consistency is high (~0.8): they pull what they declare. The failure is **putting the old H back on**, not secretly milking a corpse while naming a new arm. Once they really retract, search is short (~1 step).

---

## 3. Random cuts and matched no-change controls (n=10)

Same four prompts. Cuts sampled (min stage 15). Control: **same seeds, same ghost times, best arm never flips.** False doubt at a ghost cut is a false revision.

**When the world flips**

| | Reward | Doubt | Retract | Resurrect |
|---|---:|---:|---:|---:|
| Vanilla | 56.6 | 100% at 7.0 | 85% | 65% |
| Surprise | 59.2 | 100% at 3.6 | 95% | 50% |
| Surprise + p-value | 60.4 | 100% at 4.5 | 100% | 45% |
| Detect | **60.9** | 100% at **3.0** | 90% | 45% |

**When nothing flips (control)**

| | Reward | False doubt | False retract |
|---|---:|---:|---:|
| Vanilla | **72.3** | **0%** | 0% |
| Surprise | 71.1 | 20% | 0% |
| Surprise + p-value | 70.4 | 30% | 0% |
| Detect | 65.7 | **55%** | **0%** |

Vanilla is already near oracle (~76) on the control. Every “addition” **hurts** the stationary world. Detect’s leak is false **doubt** (extra exploration), not false **retract**. The cookbook is a hair-trigger change-point prior, not a worse estimate of `p`.

If change and no-change are mixed 50/50, detect **loses the average** (63.3 vs vanilla 64.5) because the control penalty is larger than the change gain. Surprise / p-value are a tiny mixture edge and still lose on the control, so they do not Pareto-dominate vanilla.

A global winner would need **conservative doubt + committed retract**. We have not found that prompt. Vanilla ≈ P(change)=0; detect ≈ too high P(change) and incomplete retract.

---

## 4. Belief inertia: one flip after L ∈ {10, 30, 60} (n=10)

Pre-change length L, then 60 steps to recover. Same four prompts.

**Vanilla (clean curve)**

| L | Doubt | Retract | Policy leaves old | Resurrect |
|---|---:|---:|---:|---:|
| 10 | 7.7 | 9.1 | 4.3 | 10% |
| 30 | 10.5 | 17.1 | 8.9 | 30% |
| 60 | 10.6 | 17.9 | 10.0 | **50%** |

More supporting evidence → slower leave, more resurrection. Retract saturates after L=30; resurrection keeps climbing. At L=10 the **policy moves before H** (leave 4.3, doubt 7.7). At L=60 they couple (~10). Short lock-in: behaviour more plastic than belief. Long lock-in: both stick.

**Taught prompts.** Doubt stays fast at every L (detect ~3). Resurrection is **not** a monotone L-curve at n=10 (detect 60% / 90% / 50%). Teaching buys twitchy doubt, not proven protection against bringing H back. Do not over-read n=10 wiggles.

Reward is not comparable across L (longer good first stage).

---

## 5. What we think this means

1. **Doubt ≠ adaptation.** The coupling is retract-and-stay-off.
2. **Teaching recency is a prior on change**, paid for whenever the world is still.
3. **n is vanilla’s inertia.** Lifetime counts make H hard to kill and easy to restore.
4. **No global P0 prompt yet** that beats vanilla on both worlds.
5. Next test should **change the violation**, not add more P0 English.

---

## 6. Next: ExploitDecay

Same JSON loop, same four prompts, **blind**. Pulling the current winner **lowers its p**; when it dies a new alpha spawns. The agent’s own behaviour makes H false.

Predictions:

- Detect (recency collapse) should **overmine** or churn — P0 cookbook.
- If surprise / p-value cut overmining and vanilla does not, revision is doing the work.
- If everyone doubts then resurrects the dying arm, the coupling failure survived a different world.

Metrics: existing overmine / recovery delay, plus doubt / retract / resurrect **at each alpha death**.

Configs: `experiment_exploit_decay_hyp_ad.yaml`, `experiment_exploit_decay_hyp_ss.yaml` (n=10, H=90). Analyze: `scripts/analyze_exploit_decay_hyp.py`.

---

## 7. Code map

| Piece | Location |
|---|---|
| Doubt / retract / resurrect | `llm_rl_engine/analysis/metrics.py` (`revision_sequence`) |
| Random cuts, no-change, unequal L | `llm_rl_engine/envs/staged_bandit.py` |
| Protocol configs | `experiment_p0_revision_{change,control}_{ad,ss}.yaml` |
| Inertia configs | `experiment_p0_inertia_L{10,30,60}_{ad,ss}.yaml` |
| Analyze | `scripts/analyze_revision.py`, `analyze_revision_protocol.py`, `analyze_inertia.py` |
| Plan note | `next_steps_exploration` |
