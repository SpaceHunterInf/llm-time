# Task 002 review — original reproductions repaired; acceptance pending

Reviewed: 28 September 2026. Decision: **revise**. Reviewed protocol: single-life-v0.2.

## Verified improvements

All 21 tests pass on independent rerun. A fresh CLI smoke reproduces all three saved data hashes, and source identity matches the worker's recorded source. Future observations are now rejected, deployment times are checked against the current decision, the compute-refill checkpoint is rejected, and the original nonempty-directory/updater-exception probes are covered.

Evidence: [verification](002-verification.json), [new probe output](002-kernel-probes.json), [probe script](002-kernel-probes.py). These findings concern incomplete implementations of task 002's immutability, exact resume, and preservation requirements. They do not show future leakage in the supplied smoke.

Reproduce from the project root:

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python reviews/002-kernel-probes.py
```

## Remaining findings

### R2a — Rollback retains mutations to historical artifact contents (P1)

Locations: `src/single_life/runner.py:175`, `src/single_life/artifacts.py:152`, `src/single_life/contracts.py:241`.

The snapshot is `list(state.store.versions)`: it copies the list, but retains the artifact objects and nested content dictionaries. A frozen dataclass does not make those dictionaries immutable. The prior/current comparison can compare the same mutated object against itself, and rollback reinstalls the same corrupted objects.

Probe: at decision 3, mutate a nested arm mean in the artifact deployed at decision 2 and raise an updater error. The runner aborts at committed index 2, but its returned historical artifact differs from what the actor saw at decision 2 and no longer matches its content hash. Thus the failed edit has rewritten the supposedly preserved history.

Required correction: preserve an independent validated snapshot or immutable representation; do not expose writable aliases to historical content. Validate restored content hashes and prove that rejected edits leave all previously committed artifact bytes unchanged. Include both an explicit exception and a mutation detected later by validation.

### R3a — Resume binds the updater name, not its configuration (P2)

Locations: `src/single_life/runner.py:337`, `src/single_life/runner.py:368`, `src/single_life/runner.py:470`.

The supported updater ID is the same for all ScriptedMeanUpdater configurations. Saving or restoring does not verify its actual period/window against the spec or checkpoint. Component support is also inferred from a declared/inherited ID rather than the actual implementation.

Probe: under a spec with period 2/window 4, run the supplied ScriptedMeanUpdater with period 3/window 1 and checkpoint after decision 4. Resuming with the default updater is accepted. The uninterrupted deployment sequence is [4, 7]; the resumed sequence is [4, 5, 7]. No checkpoint bytes were modified for this probe.

Required correction: either reject a runtime configuration inconsistent with the spec before running/checkpointing, or serialize the actual supported component type and configuration and require an exact match on restore. Unsupported stateful subclasses/wrappers must not be accepted merely because they inherit a supported name. Apply equivalent identity checks to supported policies. Test-only injections need an explicitly limited contract; a broader plugin system is unnecessary.

### R4a — Rewriting the log can destroy the committed prefix (P1)

Location: `src/single_life/cli.py:287` and adjacent persistence writes.

Every commit rewrites the whole log with `Path.write_text`, which truncates the existing file before replacement bytes are complete. Persisting after each step does not preserve previous steps if the next write is interrupted.

Probe: after event 1 has been saved, simulate interruption after the event-2 public-log write opens/truncates its file. The public log is empty, while the manifest still records one event and its earlier nonempty hash. The manifest's claim that a machine crash leaves flushed steps intact is too strong for this implementation.

Required correction: avoid in-place truncation of authoritative committed records. Atomic snapshot replacement or an append-only journal with validated complete records is enough. Detect incomplete/cross-file-inconsistent state and fail closed; do not silently replay. This is local process-interruption handling, not a request for distributed transactions or a guarantee against arbitrary storage failure.

### R4b — Auxiliary output can overwrite an authoritative log (P1)

Locations: `src/single_life/cli.py:165` (preflight), `src/single_life/cli.py:114` (summary write).

For a new run directory, `--summary <run-dir>/public_log.jsonl` passes preflight because the file does not yet exist. The run completes and the summary then replaces its public log. Exit status is zero; the saved manifest no longer matches the file hash.

Required correction: resolve output identities before interactions; reject collisions between summary/example paths, authoritative run files, and the input configuration, including equivalent relative paths and resolvable symlink aliases. Preflight deterministically invalid example indices as well. Retain the existing refusal to overwrite prior outputs.

## Review scope and outcome

The supplied v0.2 happy path is reproducible. The four original examples are fixed; the remaining findings exercise the same requirements under nested mutation, changed runtime configuration, interrupted writes, and colliding CLI arguments. None requires an LLM or a new environment.

The abstract compute clock, shared paired-fixture identity, public artifact payload, and invalid-action gap remain acceptable within the previously documented limits. Exact resume remains limited to explicitly supported components. No new research result is established.

Task 002 is not yet accepted. Proceed only with [task 003](../tasks/003-preservation-repair-commander.md). The review did not modify implementation code or historical evidence.
