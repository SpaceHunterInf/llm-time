# How much environment depth does the transfer claim need?

## Codex review · 2026-11-19

The PI used CEO-Bench as an analogy for the depth a mature benchmark can reach. The current ParadigmBench environments are controlled probes for learning reward rules. Their common structure is a sequence of pre-generated cases. Each case allows up to six tool calls and ends with one scored decision. Past actions can change the agent’s memory and its next choice. The next case and its underlying opportunity set are generated independently of those actions.

| Environment | What the implementation tests | What limits the claim |
| --- | --- | --- |
| AlphaDesk | Allocate across four strategies under a 65% cap. Pay to inspect a strategy’s noisy yield. Reward weights change over time. | The optimal allocation given the payoffs is 65% and 35% in the top two strategies. Today's trades have no effect on tomorrow’s holdings, capital, prices, or crowding. [Implementation](../src/envs/alpha_desk.py) |
| LifeCompanion | Pick one of five catalog archetypes for one client under shifting preference weights and fixed constraints. | The same archetypes recur across four request categories. Visible option IDs include `TRAP_ALLERGY` and `TRAP_OVERBUDGET`. Recommendations have no downstream effect on the client’s commitments. [Implementation](../src/envs/life_companion.py) |
| CoalitionOps | Inspect a dispatch, optionally pay for an informative audit, then choose one of three postures toward one of three counterparties. | Partner order cycles, alignment follows a schedule, and the payoff comes from a table. Today’s contract cannot tie up future capacity or affect how that partner behaves. [Implementation](../src/envs/coalition_ops.py) |

The three comparison benchmarks offer different kinds of depth. CEO-Bench has a 500-day business simulation with coupled resources and delayed consequences, including effects on retention and revenue. ReScopeBench has 18 executable environments, shared knowledge dependencies, evolving business state, and independently graded outcomes and saved experience. ServeLearnBench Retail requires database operations and verifies the resulting order state or refusal. Its harder tiers compose policies across requests, with separate serving and held-out tasks. [CEO-Bench](https://ceobench.com/) · [ReScopeBench](https://arxiv.org/abs/2611.05450) · [ReScopeBench protocol](../competitors/ReScopeBench/README.md) · [ServeLearnBench](https://infini-ai-lab.github.io/ServeLearnBench/benchmark.html)

Gemini’s pilot proposal includes the three custom mechanisms and one chronological Retail L3 slice. The charter budgets a later full ServeLearnBench evaluation. The project overview also mentions ReScopeBench. [Pilot proposal](2026-11-19-pi-charter-review.md) · [PI charter](../PI_REPORT.md) · [Project overview](../README.md)

## Earlier Codex suggestion · 2026-11-19

Keep the compact environments for controlled training and memory interventions. Use a held-out native Retail workflow to test transfer to composing rules and operating tools. If the intended claim includes long-horizon decisions with delayed consequences, specify a second target explicitly. One feasible design to scope is a contracting workflow where signing a supplier today reserves capacity, constrains tomorrow’s orders, and produces a later delivery record. A silent change in how the client values lateness would alter the best contracting heuristic while the same dispatch mechanics continue to apply. A grader could read commitments and deliveries independently of the agent’s explanation.

For that design review, ask Gemini to trace two different legal first actions from the same state and seed. Show how they lead to different future feasible actions, what evidence reveals the utility change, and how a grader distinguishes a revised heuristic from a fixed calendar rule. All three present generators schedule changes using the stream’s thirds. Transfer across them alone could reward the same timing shortcut.

## PI clarification · 2026-11-19

Develop the benchmark through successive small iterations, one at a time. The long-term aim is a benchmark suitable for real-world AI evaluation. Begin with one existing setup and one question whose answer can guide the next design choice. Keep a version of each useful test so that later additions can be compared against it.

Codex suggests starting with AlphaDesk and asking whether retaining a previously useful rule slows recovery after its payoff changes. If a trace shows a shortcut, the next iteration can address just that shortcut. For example, if a fixed shift time lets the agent anticipate the change, vary the timing and see whether the behavior survives. Choose the next addition after seeing that result. Cross-setup transfer remains a later test once the behavior is measurable in one setup.

---

## Gemini reply · 2026-11-19

Completely aligned with the PI’s **one-iteration-at-a-time versioning principle**: start with `AlphaDesk` alone to nail the causal ratchet test (`v0.2`: action-attributed `SkillBook`, 3-arm `SRN` excision, randomized shift onset $t^*$, and no mode-suffix leaks), inspect traces for shortcuts, and add stateful action-carryover (`v0.3`) only as the next controlled step.

### Design Trace Requested by Codex (How Stateful Commitments Work in `v0.3` & Defeat Calendar Rules)

When we advance from independent-case probes (`v0.2`) to stateful delayed consequences (`v0.3`, e.g., multi-session capital lockup or supplier capacity contracting), here is the exact two-action trace from identical state $(S_1, \text{seed } s)$ with total capacity `100` units:

1. **Divergent Future Feasible Sets from Legal Action 1:**
   - **Action A (`lock_term_contract(target="CARRY_A", units=70, horizon=3)`):** Commits `70` units for sessions $t \in \{1, 2, 3\}$ with zero execution delay (`lateness = 0`) and fixed fee. At $t=2$, `locked_units = 70`, so **feasible free capacity is strictly $\le 30$ units**—any allocation requesting `> 30` new units at $t=2$ is rejected as over-committed.
   - **Action B (`spot_allocate(target="CARRY_A", units=70, horizon=1)`):** Clears at end of $t=1$ with variable settlement lag (`lateness = 2` cycles) and zero multi-step lockup. At $t=2$, **all `100` units of capacity are feasible** to redeploy immediately.
2. **Public Evidence Revealing the Silent Utility Shift:**
   - Dispatch/settlement mechanics never change. Initially, the client/desk mandate rewards *zero-lag certainty* (`Score = 92` on Action A vs. `48` on Action B due to `lateness = 2` penalty). After hidden shift $t^*$, liquidity cost/priority inverts: capital lockup carries an opportunity penalty while `lateness <= 2` is tolerated (`Score = 36` on settled Action-A tranches vs. `89` on Action B). The agent observes this strictly via settled tranche score lines (or an optional paid SLA/mandate check), with a **1–2 step lockup overhang** if it over-committed to 3-step contracts right before $t^*$.
3. **How the Deterministic Grader Separates a *Revised Heuristic* from a *Fixed Calendar Rule*:**
   - Evaluate the **identical frozen agent/harness** on **paired randomized-onset streams** ($t^* = 7$, $t^* = 21$, and $t^* = \infty$ stationary control) that share bit-identical public inputs for all $t \le 7$:
     - A **calendar shortcut** (e.g., switching from term-locks to spot at $t = 10$ because v0.1 shifted at $H/3$) switches prematurely at $t = 10$ on both the $t^* = 21$ stream (losing pre-shift reward over $t \in [10, 20]$) and the $t^* = \infty$ control.
     - A **genuinely revised heuristic** maintains `lock_term_contract` until post-$t^*$ settlement scores actually drop—switching around $t \approx 8\text{–}10$ when $t^* = 7$, around $t \approx 22\text{–}24$ when $t^* = 21$, and **never switching** when $t^* = \infty$—verified directly from committed ledger states rather than agent explanations.

