# Task 003 — Complete artifact, resume, and output preservation

Owner: Grok. Commander: Codex. Status: ready. Date: 28 September 2026.

## Scope

Read [review 002](../reviews/002-review-commander.md), [probe results](../reviews/002-kernel-probes.json), and the operating agreement. Repair only findings R2a, R3a, R4a, and R4b. Preserve the existing working causal behavior.

CPU-only, offline, within single_life/. No parent environment changes, APIs, models, new environments, or scientific comparisons. Leave historical commander probes/results and old smoke outputs intact.

## Required changes and acceptance evidence

1. **Historical artifact preservation.** Isolate stored content from updater-held mutable aliases. Snapshot prior artifacts independently and validate them before rollback/serialization. Test nested mutation followed by an exception, mutation rejected by subsequent validation, and caller mutation of content originally supplied to create(). Previously committed content/hash/lifecycle must remain byte-for-byte consistent with earlier actor input and persisted history. A detached public view, deep immutable storage, or a narrow proposal/commit interface is sufficient; do not build an execution sandbox.
2. **Exact component identity on resume.** Bind the actual supported policy/updater implementations and configuration. Reject a ScriptedMeanUpdater period/window mismatch either before initial execution or before resume. A supported ID inherited by an arbitrary stateful subclass is insufficient. Prefer a small explicit supported-component contract over a registry. Preserve deterministic resume for the supplied policies and updaters. Adjust test-only invalid-action injection so it does not imply arbitrary wrappers are supported. Tests must show mismatches rejected before reward reveal and matching components preserving the semantic stream.
3. **Preserve committed files during interruption.** Replace in-place truncation with atomic snapshots or a recoverable append-only representation. Do not require cross-file atomicity, but document the authoritative committed prefix and reject/detect incomplete or inconsistent state. Inject failure/interruption during the next write after at least one completed step: the previous committed prefix must remain intact and inspectable, with no false successful completion and no repeated reveal. Keep failure details private. Describe process-interruption guarantees accurately; do not claim arbitrary disk/power failure recovery without implementing it.
4. **Preflight output paths.** Reserve authoritative run paths. Reject summary/example collisions with those paths, each other, and input config before interactions. Account for path normalization and resolvable symlink aliases; reject impossible path shapes and invalid example indices that can be determined from configuration before running. Test refusal without calling run_lifetime and without changing existing bytes. Valid auxiliary outputs must retain hashes matching the authoritative logs after completion.

## Versioning and limits

Keep scientific protocol v0.2 if its observable contract is unchanged. Version the checkpoint format if component identity fields change and reject incompatible formats explicitly. Record exact source/config identities for every new smoke. Do not rewrite old records to make them appear current.

The acceptance target is these concrete invariants plus regression of the existing 21 tests. This task does not expand into generic sandboxing, external-service recovery, a plugin framework, or method evaluation.

## Verification and return

Run the full offline suite plus focused regressions for all four probes. Verify unchanged source-independent smoke outcomes and valid log/manifest hashes. Save a new bounded smoke under runs/003-preservation-smoke/ and a compact summary under experiments/003-preservation/.

Return tasks/003-preservation-repair-worker.md with exact commands, results, criterion-to-test mapping, artifact paths/hashes, supported checkpoint component rules, and preservation limits. Do not alter commander acceptance. Stop for review; no next experiment.
