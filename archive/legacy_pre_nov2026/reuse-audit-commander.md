# Repository reuse audit and proposed pilot

Date: 28 September 2026. Scope: current checkout, including existing uncommitted
work. This memo responds to the current discussion of agent inheritance under
environment change. Earlier direction/task files are historical input, not an
active work queue. No production code or historical results were changed in
this audit. No paid API calls, downloads, or training were performed.

## Recommendation

Treat MAB as an optional diagnostic setting, not a fixed requirement. Build a
fresh continuous-lifetime runner, reusing infrastructure selectively. The user's
temporal clarification supersedes the earlier proposal to learn across
independently sampled complete bandit episodes.
The proposed scientific target is whether improving in one environment creates
inherited artifacts that help or hinder subsequent adaptation, and which
artifacts cause the effect. A longer transcript alone is not self-improvement
of a learning procedure.

## What exists and what is worth reusing

| Component | Evidence in checkout | Reuse decision |
|---|---|---|
| Exogenous reward regimes | `llm_rl_engine/envs/staged_bandit.py` supports constant, sinusoidal and linear stages, unequal/random boundaries and no-change controls | Keep reward laws and deterministic configuration; repair the public interface and split randomness before paired experiments |
| Action-induced depletion | `llm_rl_engine/envs/exploit_decay_bandit.py` decays the selected active arm and spawns another opportunity | Keep as a later separate condition; actions change future state, so different agents do not face identical realized environment paths |
| Classical agents | `llm_rl_engine/agents/classical.py` | Keep interfaces and algorithms, but correct repeated-history updates before new comparisons |
| Prompt views | `llm_rl_engine/agents/history_views.py` produces transcripts and statistics from observed pulls | Useful matched-information controls; distinguish per-arm recent pulls from a global calendar-time window |
| Trajectory records and analysis | `types.py`, `logging/jsonl_logger.py`, `analysis/metrics.py` | Reuse records, raw response capture and descriptive revision metrics; add lineage/artifact identities and explicit non-recovery reporting |
| Bounded provider client and run accounting | `analysis/selective_revision_harness.py` | Prefer its explicit limits, usage/finish reporting, saved responses and run identity patterns over the old simple client; separate reusable utilities from posterior-task code as needed |
| Exact probabilistic references | `analysis/selective_revision.py`, `paired_revision.py`, `posterior_baseline.py` | Useful only if the new task needs correlated hidden-state inference; not a reason to retain that task formulation |
| Training branch | `analysis/learning_pilot*`, `learning_pilot/` | Preserve as historical work; outside the first pilot |
| Persistent solver/improver evolution | Discussed in `research-long-trajectory-self-improvement-commander.md` and tasks 011/012 | No implementation of that proposed loop found among inspected source/scripts; treat as work to build, not an existing result |

`StagedBandit` models genuine changes in the reward law experienced by the agent.
`ExploitDecayBandit` instead has a fixed depletion/spawning rule over hidden
state. It is useful for action-dependent change, but by itself does not show
adaptation to a change in that rule. Zero/negative gravity analogies should not
be used to blur these different experimental conditions.

## Confirmed reuse blockers

Reproduction: `uv run --offline --no-sync python scripts/audit_mab_reuse.py`.
The new script uses local environments and a fake provider only.

1. **Repeated evidence in classical baselines.** `GreedyAgent`, `UCBAgent` and
   `ThompsonSamplingAgent` append the whole supplied history to persistent
   statistics on every call. The runner supplies the full history each turn.
   Epsilon-greedy inherits the greedy update path. After three observations,
   the probe gets UCB counts `[4, 2]` instead of `[2, 1]`. Thompson alpha is
   `[4, 1]` instead of `[2, 1]`, beta `[2, 3]` instead of `[2, 2]`. A five-pull
   fixture makes incremental greedy select arm 0 while a fresh greedy agent
   reading exactly the same history selects arm 1. This changes the scientific
   baselines, not merely their bookkeeping. Sliding-window UCB reconstructs
   its window and does not share this specific repeated-counting bug.

2. **The runner does not enforce its information boundary.**
   `runners/run_episode.py:49` passes all environment metadata to the agent.
   Lines 75 and 101 also place hidden fields in `public_info` and pass
   `private_info` in agent history. For StagedBandit the accessible fields
   include true reward probabilities, best arm, stage identity and boundaries.
   The inspected LLM renderer uses selected description/action/history fields
   and does not render those hidden dictionaries. Thus this establishes an
   agent-interface flaw, not evidence that historical LLM prompts leaked.
   A new code-producing agent must receive an explicit public allowlist;
   private annotations belong exclusively in evaluator-side records.

3. **Matched seeds do not create matched reward tapes across control settings.**
   Stage generation and reward draws share one RNG. The `no_change` path in
   `StagedBandit._sample_all_stage_params` returns after the first stage while
   the changing path consumes further draws. With identical initial reward
   probabilities, seeds and actions, all 20 tested seed pairs produced different
   pre-change reward prefixes. This does not invalidate an unpaired comparison
   in expectation, but it defeats exact matched-prefix interventions. Use
   separate streams for regime parameters and reward innovations, or an
   evaluator-owned reward tape keyed by target instance, local time and arm.

4. **Legacy LLM retry costs are overwritten.** In `agents/llm.py:185`, each
   attempt replaces `info.token_usage`. A fake malformed response followed by
   a valid response consumes 30 tokens but records 15. Preserve each request
   and sum all attempts, including unsuccessful revisions. The newer bounded
   harness has a more appropriate accounting design to reuse.

5. **Endogenous-dynamics scoring needs a different interpretation.**
   `_compute_regret` sums instantaneous `max(p)-p(chosen)` on the evaluated
   agent's own path. In ExploitDecayBandit this is an instantaneous opportunity
   gap, not total-return regret against an independently acting optimal policy,
   because actions alter future probabilities. Do not describe the greedy
   hidden-state baseline as a proven optimal long-horizon upper bound here.

These findings are scoped to the legacy engine. They do not imply that the
separate selective-revision harness has the same implementation defects.

## What earlier experiments contribute

`reports/revision_inertia_2026-09.md` already reports an exposure-length
comparison at L=10/30/60: longer supporting histories were associated with
slower departure from an old hypothesis and more returns to it. It also reports
that a prompt encouraging change detection helped on changing worlds but hurt
stable controls. These are relevant motivating observations, not newly
reproduced results or evidence of an evolved improvement procedure.

The handoff and learning-pilot reports diagnose a different problem: emitting
and updating probabilistic state, then mapping forecasts to decisions. Their
results do not establish reusable strategy discovery. The supplied-filter
training branch should not become the critical path for this project merely
because it already exists.

The old notes also contain additional candidate prior work, including RQGM and
ModularRSI. Treat these as leads for a fresh primary-source novelty check before
claiming that selective invalidation or component-level adaptation is new;
this repository audit has not reverified those papers.

## Revised protocol: one causal lifetime in a changing world

The unit of adaptation is one continuous lifetime (a meta-episode). For example,
the evaluator might run 180 steps with a first change between steps 60 and 61.
Immediately before action 61, every adaptive component has only public experience
from steps 1 through 60, its initial prior, and artifacts derived from that
experience. Evidence of the change arrives only through subsequent permitted
observations. The horizon and boundary are illustrations, not a public schedule.

At a boundary there is no automatic reset, new task prompt, stage label or
notification in the main condition. The agent may choose to forget, branch,
probe or revise based on evidence. Operational batching must not expose
boundaries or erase state. Replicate whole lifetimes with independent agent
initializations for statistics; do not transfer a previous replicate's acquired
future knowledge into the focal evaluation lifetime.

Every policy, memory and improver update at time t must use only information
available by t. This includes candidate selection, validation, synthetic-data
generation and self-edit evaluation, not just the action prompt. Training or
selecting on the full 180-step trace cannot count as online improvement earlier
in that same lifetime.

### Leakage and learning a fixed change distribution are different

An evaluator may pre-generate and seal a schedule for reproducibility. That is
not leakage unless the agent or its adaptive selection process can access it.
Future outcomes can support final researcher-side analysis but cannot feed
back into the earlier deployed trajectory.

Separately, training over many complete lifetimes from three known reward
families can teach a fixed meta-distribution. With separate held-out lifetimes,
that is legitimate meta-training, but it tests a narrower question than
adaptation to unfamiliar changes. The main pilot should not require that
training or enumerate a future rule menu. Distinguish known-family shifts from
withheld mechanisms and novel compositions. Fixed simulator source code alone
is not leakage; available information and prior coverage are what matter. No
finite benchmark establishes adaptation to arbitrary unpredictable laws.

### Replay and dreaming

- Reuse observations and executed outcomes available by the current time.
- A private simulator may be learned from that prefix. Its generated
  continuations are hypotheses, not verified future observations.
- Syntax/interface checks are permitted. Historical validation establishes
  historical fit, not current or future environmental validity.
- Real validation uses newly available interactions with explicit feedback
  delay and compute/probe budgets. Unlimited calls to a hidden current-regime
  simulator would supply additional information and require a separate condition.
- In a bandit, only the chosen action's reward is public. An evaluator-owned
  reward tape does not authorize revealing unchosen-arm outcomes.
- Select a candidate using evidence available now, deploy it, observe the next
  outcome, and only then update. Do not select at time 60 using rewards from
  steps 61 through 180 and present the selected candidate as online adaptation.

Specify the clock convention. A first interaction-indexed pilot may allow
bounded computation between observations while counting its cost. A later
calendar-time setting should advance the world during revisions and charge
missed opportunities. The first setting does not measure real market latency.

### Persistent artifacts within the lifetime

- **Policy:** an executable decision/estimation procedure receiving public
  observations. For MAB it selects arms; a bounded configuration representation
  is acceptable initially but must be described as configuration selection.
- **Memory:** bounded lessons and evidence summaries from the observed past.
- **Improver:** the procedure selecting evidence, forming revision requests
  and allocating search effort. Its edits must affect subsequent revisions.

The immutable host controls budgets and evaluation. The agent can revise these
artifacts without resetting the world or obtaining multiple complete episodes.
Establish actual artifact use and benefit with a fixed improver before adding
improver evolution. Ordinary adaptation is not by itself evidence of recursive
improvement.

### Checkpoint and intervention protocol

1. Run chronological lifetimes with stable and changing hidden conditions.
   Save artifacts at predetermined times. Measure gains through subsequently
   realized performance; do not select checkpoints using their future scores.
2. For researcher-side diagnosis, fork an exact prefix into full inheritance,
   memory reset, policy reset, improver reset and fresh-agent branches. Each
   branch advances using only its own feedback. No branch's later score may
   select a component for deployment earlier in the original trajectory.
3. A reset placed at the evaluator's known change point is an oracle-timed
   intervention for causal attribution, not an operational detection method.
   Use matched exogenous innovations and budgets. When actions affect the
   environment, let state paths diverge rather than forcing identical states.
4. Compare exposure lengths in separate lifetimes or controlled prefix forks,
   matching target mechanisms and local clocks where appropriate. These are
   offline causal comparisons, not extra data supplied to the focal agent.
   Include memory-size and context controls and compatible artifact interfaces.
5. Include no-change intervals, recurrence, partial shifts, and novel mechanisms.
   Keep boundaries hidden in the main task. Announced boundaries are diagnostic
   ablations separating detection from revision. Counterbalance arm labels
   between independent lifetimes without automatic resets within a lifetime.
   Independent lifetimes, not correlated steps, are the replication unit.

MAB remains useful only if it exposes a meaningful artifact-level adaptation
mechanism. If updating sufficient statistics or selecting a recency window
explains everything, move to resource allocation, tool use or another setting
where learned procedures matter. No existing environment, action space or
180-step horizon is fixed. The causal information constraint is fixed.

### Measurements and falsifiers

Primary: accumulated target reward/performance over a fixed adaptation budget,
plotted against acquisition checkpoint age. Also report acquisition performance,
post-change performance curves, recovery thresholds relative to target-regime
references, recurrence benefit, false resets and total model/evaluation costs.
Treat non-recovery as censored or unsuccessful within budget, not a discarded
observation. Keep API cost and environment utility separate unless a declared
conversion has an external justification.

Baselines: corrected UCB/Thompson, sliding/discounted estimators, change-detection
with reset, fixed-improver policy optimization, equal-budget search, and an
archive/selector. A known-boundary or known-regime reference is diagnostic only.

Proceed if a reproducible age-by-shift effect survives matched context and
randomness, and artifact interventions explain it. If recent-window statistics
or resetting a cached arm estimate explain everything, this remains an online
adaptation result. If ordinary equal-budget policy search matches improver
evolution, retain the simpler implementation and narrow the claim. Do not
manufacture environments where inheritance must hurt or tune on held-out shifts.

## Validation performed

On this checkout the four offline diagnostic probes reproduced the issues
above. The focused suite completed with **54 passed in 0.13s**:

```sh
uv run --offline --no-sync pytest -q tests/test_baselines.py tests/test_envs.py tests/test_dynamic_envs.py tests/test_staged_bandit.py tests/test_exploit_decay_bandit.py tests/test_logging.py tests/test_hypothesis_loop.py tests/test_llm_parser.py tests/test_token_cost.py
```

The existing tests therefore do not detect the reproduced issues. An earlier
full-suite invocation was interrupted; its final result was unavailable and is
not counted as a successful full-suite check. The diagnostic script is an audit
artifact, not a fix. Before launching new experiments, implement the boundary,
baseline, randomness and accounting corrections with targeted regression tests,
then add the lineage runner. Preserve historical outputs and label new runs with
the corrected implementation identity.
