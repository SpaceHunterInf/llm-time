# Causal lifetime protocol and data contracts

Accepted offline implementation: `single-life-v0.2`, 29 September 2026; checkpoint format `single-life-checkpoint-v3`. Tasks 001–003 passed commander review within the trusted fixture scope. Scientific evaluation configurations are not frozen yet.

## Time and information

Use one-based action index t. At decision t, the public history contains completed outcomes only through t-1 plus any explicitly permitted current observation. The deployed action is selected before its outcome is revealed. If the hidden rule first changes for action 61, the choice at 61 cannot depend on outcomes 61 onward. A newly observed context can reveal current conditions only if the environment explicitly supplies it; the initial bandit fixture has no such signal.

Actor, memory updater, improver, candidate validator, and selector obey the same cutoff. No privileged metadata, schedule, true rewards/probabilities, optimal action, regime label, or unchosen counterfactual reward enters these components. Do not provide a list of future rule families or notify hidden boundaries. End-of-lifetime analysis is evaluator-only.

The evaluator may pre-generate a sealed future. The adaptive agent receives an allowlisted serialized public view, not the environment object or its metadata dictionary. Agent-produced code will require a restricted execution boundary before use; a Python naming convention or dataclass is not a sandbox. Task 001 tests accidental information-flow failures with trusted baseline code and does not claim adversarial isolation.

Private evaluator files must be outside the runtime agent's readable view. Repository access by the researcher/implementer is different from runtime access by the evaluated agent. Never put a sealed evaluator configuration into an LLM prompt, revision tool's working directory, or candidate scoring function.

## One life and replay

- No reset at hidden changes. Internal state, acquired artifacts, and external consequences persist.
- Historical observed records may be replayed. Unchosen bandit rewards remain unobserved even if available to the evaluator.
- A learned simulator may generate hypothetical futures. Tag these as model-generated; they are not environment observations or an oracle candidate score.
- Internal artifact rollback cannot undo external actions, consumed resources, or elapsed time.
- Researcher-only branches and independent replications can support analysis. Their feedback cannot flow into the original agent's earlier decisions.
- Resume restores one exact checkpoint, including world state, random states, artifacts, and last committed event. It does not create an extra trial or choose a favorable continuation. Fail closed on an incomplete/ambiguous checkpoint rather than repeating an external action.

## Clock and resource convention

The first fixture is interaction-indexed: one chosen action consumes one tick. Computation between ticks has a finite budget and is logged; simulated time does not advance during host computation. State this limitation wherever results are reported. It does not establish robustness to market/calendar latency.

A later service environment can charge revision/diagnostic actions in simulated time, advance exogenous arrivals while the agent is busy, and measure utility lost through missed work. Freeze that clock convention before comparing methods. Do not subtract dollars from rewards without a specified conversion; report reward and API cost separately in the initial fixture.

## Required records

These are semantic contracts. Task 001 should implement explicit Python types plus strict JSON parsing/serialization. Do not create a large schema framework. Use stable string IDs, JSON-compatible values, and reject non-finite numerical values. Extra fields must not silently pass into agent views.

| Record | Required content | Visibility |
|---|---|---|
| `PublicView` | protocol version, lifetime ID, decision index, permitted action IDs, public current observation, ordered observed events through t-1, accessible artifact references, remaining public resources | Agent and improver |
| `ObservedEvent` | event ID, action index, chosen action, observed outcome/reward, public observation, actual interaction cost | Public after the action completes |
| `ArtifactVersion` | artifact ID/version, kind, parent version, creation cutoff, supporting observed-event IDs, content/hash, deployment decision index | Runtime public content only |
| `AssumptionRecord` (later method) | procedure reference, predicted observable/condition, evidence IDs, dependent procedure IDs, last test cutoff, status, test outcome evidence | No hidden-truth status labels |
| `AttemptRecord` (API phase) | request ID, role, model/version, prompt/response artifact references, attempt index, timestamps, finish/parser status, observed usage or explicit unknown, price source/date if dollars estimated | Research accounting; only permitted response content returns to agent |
| `PrivateEvent` | public event ID, evaluator state/parameters needed for scoring, schedule position, seed/innovation references, evaluator-only diagnostics | Evaluator only |
| `RunManifest` | run/experiment IDs, status, protocol version, clock convention, agent/config IDs and hashes, code revision plus dirty-source snapshot/hash, dependency identity, initialization/seed identities, resource caps, artifact/log paths and hashes, completion/error reason | Researcher; never pass whole manifest to agent |

Maintain separate public and private logs. Evidence references must exist and be available by the artifact's creation cutoff. Deployment must occur after creation and before the action it influences. Record what was actually deployed, not just a proposed edit. Failed candidates and attempts remain in the research log.

Machine schemas may be added when an external consumer needs them. These records are deliberately smaller than a general experiment-tracking platform.

## Randomness and replication

Separate schedule/parameter generation, exogenous outcome innovations, and agent randomness. Same seed alone is not a pairing guarantee. For the bandit fixture use evaluator-owned innovations indexed by (lifetime, action index, action ID); selecting one action reveals only its result. Schedules that first diverge after a prefix must share exactly that prefix's outcomes under the same actions.

For endogenous environments, match exogenous arrivals/noise where appropriate while allowing different actions to produce different state paths. Do not describe instantaneous best-action gaps on a policy-dependent path as optimal long-horizon regret.

Independent lifetimes are the statistical replication unit. Initialize all adaptive state separately for each replicate. Researcher tuning uses development lifetimes; final evaluation needs held-out schedules and, for a stronger claim, withheld mechanisms/compositions. Report the prior coverage explicitly. Run-level paired differences are appropriate where exogenous pairing is defined; timesteps are not independent samples.

## Measurement

Primary eventual outcome: realized cumulative lifetime utility under a fixed interaction/clock budget, including consequences of probes, failures, and revisions. In the initial bandit fixture this is simply the sum of observed rewards. Also report stable-world performance, curves around evaluator-known changes, unnecessary revisions, persistence of useful unaffected procedures, actual resource usage, and recovery fractions with unrecovered cases retained/censored.

An oracle change marker or reset at the true boundary is diagnostic only. Recovery thresholds, comparison budgets, sample size, aggregation, and uncertainty calculation must be registered before confirmatory runs. They are not chosen to make a promising curve pass.

## Invariants for the first implementation

1. Future-perturbation test: change a sealed continuation while fixing the prefix and agent randomness; views, actions, revisions, and artifacts through the prefix remain identical, including the first action before new feedback.
2. Information test: inject sentinel private fields into evaluator state; none reaches serialized actor/improver inputs, artifact evidence, or public logs.
3. Evidence test: each completed observation enters statistics exactly once; future/nonexistent evidence references are rejected.
4. Persistence test: a hidden change neither resets agent/world state nor emits an unsolicited boundary signal.
5. Pairing test: changing future schedule generation cannot alter earlier reward innovations.
6. Resume test: uninterrupted and resumed runs produce the same semantic event/action/artifact stream with no duplicated interactions.
7. Budget test: invalid actions, exhausted budgets, and failures have explicit logged behavior and cannot earn free environment probes.

These tests establish protocol integrity, not method superiority or generalization.


## Review clarification — v0.2 repair target, 28 September 2026

The host must validate causal relationships, not only record types. An observed event must precede the current decision; accessible artifacts must have been created/deployed by the current decision and reference available evidence. Gaps due to invalid actions are permitted; duplicate or contradictory events are not.

Evidence cutoff and actual artifact creation/deployment are distinct. Old evidence may support a new artifact, but the updater cannot claim that a newly installed artifact influenced an earlier action. The host owns actual lifecycle time and the evidence catalog.

For the present fixture, one abstract compute unit is charged per attempted decision. A resumable completed-step checkpoint must reconcile that accounting with its committed prefix. Checksums alone do not establish semantic validity. Supported resume covers the supplied policies and deterministic updater; broader component state requires explicit support.

A fresh run must preserve pre-existing nonempty output directories. Initialize run identity before acting, persist completed interactions during execution, and retain a failed/aborted record for handled actor/updater/evaluator errors. Local synchronous crash handling does not imply distributed or external-service transaction recovery. Incompatible prior checkpoint formats must be rejected explicitly.

## Task 006 extension — specified v0.3, implementation pending

Authorized by commander after review 005, 29 September 2026. This extends the public observation and reward domain for a separate contextual development slice. Existing v0.2 runs/checkpoints remain unchanged; do not relabel or pool their scores with v0.3.

Before action t, the current request type is public. After action t, its observed event records that type, chosen action, and integer utility measured in tenths of a utility point. No future request list or private reward table is public. Negative integer utilities are valid; Bernoulli conversion is not. Time still advances once per executed action; API computation does not advance environmental time.

A validated JSON procedure is interpreted by trusted host code, with no arbitrary model-generated execution. Its exact DSL and counters are defined in task 006 and frozen in registration. At a scheduled call before action t, a proposed procedure uses only completed evidence through t-1, is validated and logged, and may deploy before t. The host owns cutoff/deployment/counters. Semantic identity is the procedure, not its explanation. A matching proposal keeps counters; a changed deployment resets per-type occurrence counters. Historical proposals do not influence an earlier action.

The next task must test the extended causal boundaries, execution semantics, and typed rewards before live use. Until reviewed, this is a specified contract, not an accepted implementation. Disable new-runner checkpoint/resume unless its complete state is supported; preserve existing v0.2 resume behavior. Independent lifetimes initialize separately and never receive another run's procedure or scores.
