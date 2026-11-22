# Task 002 — Repair causal validation and run preservation

Owner: Grok. Commander: Codex. Status: ready. Date: 28 September 2026.

## Read first

Read [review 001](../reviews/001-review-commander.md), its [probe output](../reviews/001-kernel-probes.json), the [protocol](../research/protocol-commander.md), and [operating agreement](../program-commander.md). Task 001's code is retained but not yet accepted.

## Objective and scope

Repair findings R1–R4 without changing the research setting or implementing an LLM agent. CPU-only offline work in `single_life/`. No parent-directory uv sync, API calls, model downloads, training, or new environment. Do not overwrite old smoke outputs or commander review evidence.

## Required changes

1. **Causal public-view validation.** Reject observations from the current/future action, duplicate/conflicting event identities/indexes, unordered events, and invalid actions in the observed record. Retain legal gaps caused by invalid attempts. Validate accessible artifacts against the decision and available evidence; reject future/undeployed artifacts, nonexistent or too-new support, and ambiguous identities. Exercise the actual serialized actor/updater boundary.
2. **Host-owned artifact lifecycle.** Record actual creation/deployment decisions separately from evidence cutoff as necessary. The runner owns current time and the observed evidence catalog. Reject backdated/future deployments and forged/unavailable support during update commit. A narrow host-validated proposal interface or a store bound to current public state is sufficient. Do not build an arbitrary code sandbox. Preserve immutable prior versions and exact chronological logging.
3. **Semantically valid checkpoint restore.** Under this fixture's clock, require compute and interaction balances to agree with the committed prefix. Validate ordered unique public events, matching private event identities, horizon bounds, and artifact lifecycle times. Check supported baseline state against observed evidence where applicable. Convert malformed restore failures to clear checkpoint errors before revealing any reward. State explicitly which updater/policy implementations are supported; reject unsupported stateful components or capture their state if supporting them. Do not promise arbitrary plug-in recovery.
4. **Preserve partial and failed runs.** Preflight all output arguments before interactions. Refuse nonempty run directories for new runs. Write initial identity/config and retain completed-step records as execution advances. Handle actor/updater/evaluator and post-observation failures without losing completed interactions or automatically repeating an action. Finalize a failed/aborted manifest where possible, with internal exception detail evaluator-side. Keep the supported scope local and synchronous; no claims about recovering external side effects after machine failure.
5. **Protocol identity.** These fixes affect the causal interface: update implementation/config/test identities to `single-life-v0.2`, with a corresponding checkpoint format change if needed. The commander has recorded this repair target in the protocol. Clearly reject incompatible v0.1 checkpoints rather than silently reinterpret them. Preserve v0.1 results as historical infrastructure records.

## Required evidence

- Keep the original 15 tests passing after deliberate version/API updates.
- Add regression tests reproducing each review probe, with corrected outcomes: future observation rejected; false deployment rejected or recorded at actual time; inconsistent checkpoint rejected before reveal; prior output preserved; crash leaves inspectable prior interactions and failure metadata.
- Test failures before an action and after a reward has been observed. The latter must not erase the observed reward or permit a silent extra trial.
- Compare uninterrupted/resumed behavior for supported random, UCB, and artifact policies, including a checkpoint after a hidden boundary, after artifact deployment, and after a tick-consuming invalid action. Use independent processes for at least one resume check.
- Demonstrate a valid later creation based on older evidence: stale evidence can be used, but historical deployment cannot be fabricated.
- Test causal rejection through actor and updater paths, not only dataclass construction. Historical replay may use earlier events; newly observed real outcomes must stay chronological.
- Run a fresh bounded CLI smoke into a new directory, saving v0.2 config/source identities and hashes. No performance claim.

Add production regression tests under `tests/`. Leave commander probes/results unchanged as a historical reproduction of the reviewed v0.1 code. If schema changes make those probes obsolete, explain the mapping to the new tests.

## Return and stop

Write `tasks/002-kernel-integrity-worker.md` with changed files, exact commands/results, criterion-to-test mapping, artifact paths/hashes, and remaining limitations. Update README usage if needed; do not change commander acceptance or launch the next experiment. Routine implementation choices are yours; any substantive protocol deviation belongs in the return for review.
