# Task 001 review — revise before acceptance

Reviewed: 28 September 2026. Decision: **revise**. Scope: the submitted offline kernel, not the legacy engine or a proposed LLM method.

## Verified strengths

All 15 submitted tests passed on independent rerun. A fresh CLI smoke reproduced the saved public-log, private-log, and artifact hashes exactly. Saved files match their manifest, and the source hash matches the worker's recorded source identity. The supplied agents/updater pass the future-perturbation test; no hidden schedule leakage was found in the supplied smoke.

The independent package, indexed innovations, deduplicated baseline observations, artifact-use example, and ordinary resume tests are useful foundations. The worker correctly avoids claiming adaptation or RSI from these checks.

Evidence: [verification record](001-verification.json), [probe output](001-kernel-probes.json), and [reproducible probes](001-kernel-probes.py). Run from this project root:

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python reviews/001-kernel-probes.py
```

## Required corrections

### R1 — Validate causal relationships at the public boundary (P1)

Location: `src/single_life/contracts.py:298`, especially construction at lines 321–328.

The parser checks field types but does not check an observation's action index against the decision index. The probe supplies decision 2 with an outcome from action 60, and `PublicView.from_dict` accepts it. The artifact envelope likewise lacks checks against the view's current time and observed evidence.

This is a missing enforcement boundary, not evidence that the default runner generated a leaking view. An accidental indexing error or malformed restored view can pass the advertised strict interface. Validate temporal cutoff, ordered unique events, allowed actions, and artifact availability/evidence consistently when crossing into actor and updater inputs. Invalid-action gaps must remain legal.

### R2 — Bind artifact deployment to the actual host decision (P1)

Locations: `src/single_life/artifacts.py:90` (`deploy`) and `src/single_life/runner.py:164` (updater invocation; locate by symbol if line numbers change).

At real decision 4, an updater can create an artifact using observation 1 and call `store.deploy("v1", 2)`. The runner accepts it. The actor did not see it at decision 2, first sees it at decision 4, but the saved artifact claims deployment at 2. The CLI then reconstructs a historical log consistent with that false deployment time.

Evidence cutoff is not creation/deployment time. Let the host own actual lifecycle timestamps and the evidence catalog. Validate updates at the current decision instead of trusting a caller-provided timestamp/catalog. This is ordinary correctness enforcement for a faulty updater; adversarial execution isolation remains out of scope.

### R3 — Reject inconsistent checkpoint accounting (P2)

Location: `src/single_life/runner.py:389`.

The checkpoint checks only that remaining compute is nonnegative and no greater than the initial budget. Under this kernel's one-compute-unit-per-committed-decision rule, a checkpoint after four actions with initial budget five must have one unit left.

The probe changes that balance back to five and recomputes the checksum, as a buggy checkpoint writer could. Restore accepts it: the valid checkpoint stops at action 5, whereas the inconsistent one completes eight actions and reports only four units consumed. A checksum protects byte integrity, not semantic consistency.

Validate accounting against the committed prefix before any reveal. Extend consistency checks to horizon bounds, ordered/unique evidence identities, private/public event correspondence, restored baseline state, and artifact lifecycle times relative to the checkpoint. No general proof of arbitrary policy internals is requested.

### R4 — Preserve failures and unfinished run evidence (P1)

Locations: `src/single_life/cli.py:48–52`, `src/single_life/runner.py` actor/updater and observation callbacks.

The CLI only refuses an existing `manifest.json` and runs the whole lifetime before creating its output directory. Two concrete failures:

- An injected updater exception at decision 3 escapes after two completed interactions; there is no run directory, manifest, or public log.
- A directory containing an existing public log but no manifest is accepted, and the log is overwritten by a new run (exit status 0).

Preserve the initial run identity/config before interactions, record completed steps durably, and retain a structured failed/aborted record when actor/updater/evaluator handling fails. Refuse a nonempty run directory for a fresh run. Do not restart a partial lifetime as a fresh run at the same path. This repair needs bounded local-file failure handling, not a distributed transaction system or external-service recovery guarantee.

## Scope decisions and limitations

- Full public artifact content is acceptable; only public evidence may generate it. This is not itself a leak.
- Reserving one abstract compute unit before each decision is acceptable for this fixture. It is not measured wall-clock/token cost or a compute comparison between methods.
- Shared lifetime ID/innovation seed is acceptable for the paired fixture, not for independent statistical replications.
- Consuming a tick for invalid actions is acceptable. The current actor sees a history gap rather than an explicit failure record. Record that limitation; define actionable feedback before a method uses invalid-action recovery.
- The scripted artifact policy selecting only arm_a in the smoke is acceptable for this infrastructure task. Its separate controlled-prefix test establishes artifact use; neither establishes useful learning.
- Resume with the supplied deterministic updater is the supported case. Future stateful updaters must be serialized or explicitly rejected at checkpoint time; do not advertise general improver-state recovery.
- The worker reports accidentally syncing the parent environment and subsequently restoring its train group. This review cannot certify equivalence to the parent's previous environment; the repair task must remain within the independent subproject.

The probes intentionally construct malformed inputs/inconsistent state. They identify missing contract checks, not malicious access or observed future leakage in the recorded smoke. The supplied happy-path results remain reproducible.

## Next task

[Task 002 — kernel integrity repair](../tasks/002-kernel-integrity-commander.md). Keep task 001 unaccepted until these corrections and regression tests are reviewed. No API calls, training, service environment, or scientific comparison is queued.
