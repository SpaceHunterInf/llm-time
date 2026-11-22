# Environment feasibility dossier

Status: ready for commander review. Date: 29 September 2026. Task 004.

This is a design note with exact illustrative arithmetic. It is not evidence that a learner adapts, that an LLM helps, or that any method is better than another. The hand-specified rules below were written for these examples. Oracle and wrong-factor rows use the hidden case table. The other rows use only the request type and scalar rewards already observed.

No new related-work claim is added. The objection already recorded in `research/related-work-commander.md` is the one checked here: a competent recency or factorized estimator may remove the reason for a special revision mechanism.

## Recommendation

Use a factored desk, not the three-arm bandit and not a job-service simulator.

The desk is the smallest setting in which an acquired two-part procedure can earn value, a hidden change can spoil one part, a wrong explanation is costly, and a whole reset can spend the part that was still valid. The same desk lets a per-type recency rule win. On an equal-weight illustration over a shared change, a type-local change, a stable world, and one non-additive trap, per-type recency scores `59/5` over the response window and selective cross-type transfer scores `87/8`. That gap is weight-dependent and is not a population result. It is enough to refuse a larger simulator whose extra machinery would be justified only by defeating this baseline.

The plausible negative outcome is: selective revision does not beat per-type structured recency unless the hidden changes are concentrated on a shared additive factor. Choosing that concentration to protect the unbuilt method would tune the environment to the hypothesis.

## Candidates compared

| Setting | What experience can acquire | What a partial change can do | What a simple baseline can already do | Role in this dossier |
|---|---|---|---|---|
| Dev bandit fixture, `configs/dev_continuing.json` | A preference over three atomic arms | Replace one or more Bernoulli means at a hidden index | `SlidingWindowUCB` updates the pulled arm's recent mean. Other arms stay as their own estimates. Pulling an arm is the observation. | Keep as the causal fixture. It does not separate procedure revision from recency. |
| Factored desk (this note) | A server choice per request type and a shared mode, `single` or `pair` | Spoil the shared mode, or one type's server, without labeling which | Per-type coordinate search repairs a local change. A one-line cross-type copy repairs a shared additive mode change and fails on a non-additive trap. | Recommended sketch. Do not build it until Codex asks. |
| Job service with a queue, deadlines, setup time, and size-dependent delay | The same routing and batching ideas, plus policies that depend on congestion | The question's batching example: setup savings can become a size penalty while routes stay valid | A myopic recency rule is misspecified once actions change the queue. A utility gap can come from congestion accounting rather than from procedure structure. | Not the next build. It adds a new clock and endogenous state before the baseline question above is settled. |

The desk is that batching example with the queue removed. One request arrives per tick. The action is still a route plus a mode. The hidden shared term is the pair effect. Routing stays type-specific. Nothing in the public view names those factors.

## 1. Interaction contract

Name for this sketch: factored desk. It is not a protocol version and it is not implemented in `src/single_life`.

### Clock and persistence

Ticks are one-based. Tick `t` brings exactly one request. The agent chooses one action. The scalar reward is revealed after that choice. The next request is tick `t+1`. Host computation between ticks does not advance the clock. This is the current `interaction-indexed-v0` convention. It does not measure service latency, queue growth, or market time.

There is one lifetime. The change at tick 13 does not reset the agent's rule, does not clear history, and does not emit a boundary mark. A bad probe is not refunded. Unchosen actions stay unobserved. A sealed schedule may be fixed by the evaluator; this witness uses the repeating cycle `T1, T2, T3`, which becomes predictable from public history after the cycle has been seen. Future request types are not shown before they arrive. Predictability of this particular cycle is not a hidden mechanism.

### Information

At decision `t`, before the reward, a fair method may see:

- the current request type, one of `T1`, `T2`, `T3`
- the four actions `A|single`, `A|pair`, `B|single`, `B|pair`
- completed events through `t-1`, each a type, an action, and a scalar reward
- remaining tick budget

It may not see the preferred server, the pair effect, the reward table, a regime id, a dependency graph, the reward of an untaken action, or a current-regime oracle.

Hidden evaluator state in the additive cases:

- preferred server: initially `T1=A`, `T2=B`, `T3=A`
- pair effect: `+3/10` before tick 13
- from tick 13, either the pair effect becomes `-2/5` for every type, or only `T1`'s preferred server becomes `B`, or nothing changes
- the trap replaces the additive formula from tick 13 with the table in `reference_witness.py`

The trap is a researcher-built counterexample, not a fitted world model.

### Reward

Units are utility points per tick. Cumulative utility is the sum of observed rewards. These points are not currency and are not API dollars. A later integer encoding is `reward_tenths = reward * 10` (`4/5` is 8, `1/10` is 1, `-2/5` is -4, `1/2` is 5). Every value in this dossier is a multiple of `1/10`.

For every additive case:

```text
reward(type, server, mode) = route(type, server) + batch(mode)
route = 1/2 if server is the hidden preferred server, else 0
batch(single) = 0
batch(pair) = pair effect
```

Incumbent procedure, used for acquisition and for the stale window:

| Type | Action | Reward before tick 13 |
|---|---|---|
| T1 | `A\|pair` | 1/2 + 3/10 = 4/5 |
| T2 | `B\|pair` | 4/5 |
| T3 | `A\|pair` | 4/5 |

After the shared pair change, the same actions pay `1/2 + (-2/5) = 1/10`. Switching only the mode, and keeping the server, pays `1/2`. Flipping only the server, and keeping pair, pays `-2/5`.

Investigation is not a side channel. A probe is one of the four actions, played on the request that arrived, and it returns the same scalar. While the rule waits for a type to recur, other types still arrive and must be acted on.

## 2. Worked trajectory

Case `shared_batch`. Rule `one_type_then_transfer`. The rule is hand-specified. It reads public history only. Annotations in prose are researcher notes and are not given to the rule.

The witness does not show a learner discovering the incumbent. It starts with that procedure already selected, then shows what the procedure is worth and what happens after the change.

Acquisition, ticks 1–12. The cycle is `T1 A|pair`, `T2 B|pair`, `T3 A|pair`, each paying `4/5`. Twelve ticks sum to `48/5`. Runtime observation at tick 1: `request_type=T1, action=A|pair, reward=4/5`. The same pattern, with the type and server cycling, holds through tick 12. No factor split is shown.

The hidden pair effect changes at tick 13. Tick 12's action cannot depend on that change. There is no boundary mark.

| t | Type | Action | Reward | What the rule can see | Researcher note |
|---|---|---|---|---|---|
| 12 | T3 | `A\|pair` | 4/5 | type, action, scalar | Last pre-change tick. |
| 13 | T1 | `A\|pair` | 1/10 | same fields; reward fell | Hidden pair effect is now `-2/5`. Not labeled. |
| 14 | T2 | `B\|pair` | 1/10 | scalar only | Same hidden change. |
| 15 | T3 | `A\|pair` | 1/10 | scalar only | Same hidden change. |
| 16 | T1 | `A\|pair` | 1/10 | scalar only | Same hidden change. |
| 17 | T2 | `B\|pair` | 1/10 | scalar only | Same hidden change. |
| 18 | T3 | `A\|pair` | 1/10 | scalar only | Exploited cells all pay `1/10`. Two explanations still fit. |
| 19 | T1 | `B\|pair` | -2/5 | this probe's scalar | Route-first probe. Worse than the stale `1/10` and worse than the untried single action. The rule is not told why. |
| 20 | T2 | `B\|pair` | 1/10 | stale scalar | T2 waits while the rule holds the next probe for T1. |
| 21 | T3 | `A\|pair` | 1/10 | stale scalar | Same wait. |
| 22 | T1 | `A\|single` | 1/2 | mode probe beat the stale scalar | The rule copies `single` onto every dropped type. That copy is a hypothesis. |
| 23–36 | cycle | original server, `single` | 1/2 each | scalar `1/2` | Servers chosen before the change stay in use. Fourteen ticks sum to `7`. |

Response window, ticks 19–36, sums to `-2/5 + 1/10 + 1/10 + 1/2 + 7 = 73/10`.

Stale window, ticks 13–18, sums to `3/5`. Lifetime under this rule sums to `48/5 + 3/5 + 73/10 = 35/2`.

### Wrong explanation still available at tick 18

The rule has observed only the incumbent cells, each now paying `1/10` instead of `4/5`. These two stories match that observation and disagree on untried actions:

| Story | Predicted `original server \| single` | Predicted `other server \| pair` |
|---|---|---|
| Pair effect fell to `-2/5`; routes unchanged (this case's hidden state) | 1/2 | -2/5 |
| Pair effect stayed `3/10`; match bonus fell by `7/10` | -1/5 | 3/10 |

The tick-19 probe returns `-2/5`, which rejects the second story for that cell and does not yet prove the first. The tick-22 probe returns `1/2`, which matches the first story on that cell. Copying `single` to T2 and T3 is still an extra assumption. Section 3's trap is the case where that copy is false. The full tick list, including empty annotation fields, is `worked_trajectory_shared_batch_one_type_then_transfer` in `witness.json`.

## 3. Feasibility witnesses

Fractions were chosen so each qualitative contrast is an exact nonzero rational. They were not fit to a method score. No method exists. All strategies share ticks 1–18. The table is the response window, ticks 19–36, in utility points.

Rules:

- `keep` — leave the acquired procedure in place.
- `oracle_repair` — play the hidden best action from tick 19. Evaluator knowledge. Not a method.
- `wrong_factor` — apply the other factor's repair. Evaluator knowledge of which story is wrong. Not a method.
- `per_type_stop` — if a type's post-change reward is at least `1/5` below its pre-change reward, try that type's other three actions in the order other-server/same-mode, same-server/other-mode, both changed. Stop that type at the first strict improvement. Do not copy a conclusion to another type.
- `one_type_then_transfer` — same drop test. Probe only the earliest dropped type, route first. A server improvement stays on that type. A mode improvement is copied onto every dropped type's original server.
- `whole_reset` — run the per-type probe list on every type, including types whose reward did not fall.
- `unnecessary_probe` — at tick 19 play `single` on the incumbent server, then revert. Interpreted on the stable case.

| Case | keep | oracle | wrong factor | per-type | transfer | whole reset |
|---|---|---|---|---|---|---|
| Shared pair change | 9/5 | 9/1 | -36/5 | 63/10 | 73/10 | 63/10 |
| Type-local route change | 57/5 | 72/5 | 6/1 | 72/5 | 72/5 | 56/5 |
| Stable | 72/5 | 72/5 | 9/1 | 72/5 | 72/5 | 48/5 |
| Non-additive trap | 6/1 | 63/5 | 6/1 | 121/10 | 37/5 | 21/2 |

Checked contrasts:

- The old procedure earns `48/5` over ticks 1–12 in every case (`4/5` per tick).
- After the shared change it pays `1/10` per tick. Keeping it through the response window scores `9/5`. The illustrated wrong route repair scores `-36/5`.
- Transfer on the shared change scores `73/10`. That is `1` above per-type search and `17/10` below the oracle. The shortfall is the wrong route probe (`-2/5` instead of `1/2`) plus two stale ticks while waiting for T1. Servers are not kept flipped: only T1's tick 19 uses the other server.
- On the type-local change, per-type search and transfer both match the oracle at `72/5`. T2 and T3 never change server. Whole reset does change those servers for two ticks each and scores `56/5`, which is `1/5` below keep.
- On the stable case, transfer and per-type search do not probe and score `72/5`. Whole reset scores `48/5`. One forced probe that reverts scores `141/10`, a loss of `3/10` against doing nothing. Adopting `single` anyway, the wrong-factor row, scores `9/1`.
- On the trap, T1 is repaired by `A|single` and T2 is repaired by `A|pair`. Transfer copies T1's mode and scores `37/5`. Per-type search scores `121/10`. The oracle scores `63/5`. T3's server is untouched by per-type search and by transfer.

Equal-weight mean of the four response scores, labeled as an illustration and not as a prior: per-type `59/5`, transfer `87/8`, whole reset `47/5`, keep `42/5`, oracle `63/5`. Per-type is ahead of transfer by `37/40` under these weights. A mixture that contained only the shared additive change would put transfer ahead by `1`. The sign is the mixture.

## 4. Baseline challenge

The strong simple reference is `per_type_stop`. It uses the same actions and the same scalars as every other fair rule. With deterministic rewards, one new sample identifies an action. On a type-local server change it matches the oracle. On a stable world it correctly does nothing. A flat contextual bandit over the twelve type-action pairs, with a stop-at-first-improvement rule, is this reference. An unstructured bandit that re-explores healthy types is the weaker `whole_reset` comparator, included because the question asks for a reset baseline, not because it is the strongest available rule.

`one_type_then_transfer` is the selective-revision stand-in. Its only extra act is copying a mode that helped one dropped type. On the shared additive change that copy is worth `1` utility point after probe cost, against per-type search, and it avoids flipping T2 and T3 onto the wrong server. That is the best case for the candidate mechanism in this sketch, and it is already achieved by a coordinate rule with one transfer line. It is not a residual that requires assumption records, a diagnostic oracle, or an LLM.

Factorized estimation suffices whenever the hidden change really is one visible coordinate, shared or local, and the reward is the sum of those coordinates. The action names already expose the coordinates. Giving a method the equation would be unnecessary; trying the other value of one coordinate reveals it. The trap is where that factorization is the wrong model. There the selective copy loses to per-type search. Making the action space larger, or adding a queue, so that this particular copy looks better would be increasing complexity to defeat the baseline.

A Bernoulli reward layer would add estimation error to every rule. It is not a reason to introduce a richer mechanism before the deterministic comparison is accepted or rejected. The dev bandit remains the place where stochastic arm means and a windowed mean are already implemented. In that fixture a partial mean change is what the windowed mean is for. There is no second factor to preserve.

## 5. Fairness and generalization

Every fair rule in the witness faces the same four actions, the same current request type, the same scalar history, and the same 36 ticks. None of them receives the preferred server, the pair effect, the trap table, or a regime label. `oracle_repair` and `wrong_factor` are excluded from that claim; they exist to show the value of a correct local edit and the cost of the other edit.

Methods may spend the tick budget differently. They may not receive a larger action set, a free probe bit, or a dependency hint. The transfer rule's cross-type copy is an ordinary decision from public rewards. The trap shows it can be the wrong decision. It is not privileged access.

These illustrations cover three known additive situations (shared mode, one type's route, no change) and one non-additive table written in the same file as the rules. Because the rules were written against these cases, the file is development material. It cannot serve as a held-out evaluation. A later confirmatory set would need new tables, fixed before looking at the scores. Recurrence is not tested. A finite menu of changes is not open-world adaptation. The protocol's narrower claim boundary applies: a fixed simulator can host a nonstationary lifetime, and that supports only the mechanisms it actually contains.

No evaluation distribution, sample size, or decision threshold is registered here. The equal-weight mean is a sensitivity display.

## 6. Next slice and kernel boundary

Do not build the job-service simulator. Do not add an API adapter. Do not tune case weights, the `1/5` drop gap, or the probe order to favor transfer.

Main confound: server and mode are visible parts of the action, so coordinate search is a fair baseline rather than a privileged parser.

Unresolved identifiability issue: the scalar on the exploited cell does not say which factor moved. One coordinate probe often does. After that probe, explicit assumption lists are not separately identified from the coordinate rule. Cross-type copying remains a hypothesis, and the trap punishes it.

Smallest experiment that could reject further research on this mechanism: keep the four cases, or a commander-registered superset that was not used to edit the rules, and require selective transfer to beat `per_type_stop` by a pre-registered margin after charging probes, including a stable control. If it does not, stop or simplify. Missing that margin is a successful negative result. These illustrations already point that way under equal weights; they do not replace a registered comparison.

If Codex later wants the desk inside the kernel, the implementation slice is an offline fixture and information-boundary check, still not a method pilot.

Kernel pieces that stay valid for that fixture:

- Events through `t-1` only, action before reward, no boundary signal, no reset at the change.
- Private log for the preferred server, pair effect, and regime. Public log for type, action, and scalar.
- Unchosen actions unrevealed. Sentinel private fields stay out of the public view.
- Schedule seed separate from innovation seed and agent seed. A stochastic successor can key innovations by `(innovation_seed, lifetime_id, t, action_id)`. The request type is schedule, not an innovation.
- Artifact freeze, hash, and deployment cutoff, if a procedure is stored later. The baseline rules do not need an artifact.
- Atomic run logs, nonempty-directory refusal, and rejection of checkpoint formats v1 and v2.
- Interaction budget as a tick count. Host compute still does not advance this clock.

Semantics that would have to change before such a fixture could be scored:

- `PublicView.public_current_observation` is fixed at `None` in the current runner. The desk needs the request type there, for every method. Existing bandit runs must keep `None`.
- `RunSpec` segments are Bernoulli probabilities keyed by arm. They cannot encode a route bonus plus a pair effect. Add a fixture. Do not reuse the dev bandit probabilities as desk utilities.
- Prefer integer tenths in `observed_outcome` so logs do not depend on binary floats.
- Resume allows only `RandomAgent`, `SlidingWindowUCB`, `ArtifactConfigAgent`, `ScriptedMeanUpdater`, and `NullUpdater`, as exact classes. A desk rule is not resumable until it is added explicitly. Do not checkpoint a test double.
- A schedule that is a function of `t` fits the current "world state is the committed index" checkpoint. A queue does not. Do not add a queue under the old checkpoint story.
- The primary sum can stay a sum, but the unit becomes the desk utility point. Report stable-world probe loss and whether unaffected types changed server. Do not report bandit regret, and do not treat the myopic per-tick gap as long-horizon regret if a queue is added later.
- The clock string can stay `interaction-indexed-v0` for this desk. A service-time clock, with arrivals during a long action, is a different convention. Freeze it before any comparison that uses it. This dossier does not freeze it, because the recommendation is not to build that environment next.

## Source of the numbers

`experiments/004-environment-feasibility/reference_witness.py` writes `witness.json`. It uses the standard library only and does not import `single_life`. The script raises if a checked total disagrees with the fractions in this note. Historical smokes, reviews, and `src/` were not modified.
