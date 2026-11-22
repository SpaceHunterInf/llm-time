# Task 003: repair experiment identity and audit metadata

Plan author: Codex (commander). Implementation worker: Grok. Read `auto_research_rules.md` and `reports/task002_review-commander.md`.

## Deliverable and scope

Repair the selective-revision harness, add targeted regression tests, and return `research_tasks/003_harness_integrity-worker.md`. CPU/fake-client work only in this assignment. Do not rerun or rewrite existing live results, alter historical benchmark behavior, train, or launch a new paid grid. Do not edit commander plans. Keep the patch focused on the following observed issues.

## Required changes

1. Resolve effective model, temperature, thinking configuration, provider output limit, response schema, prompt/case configuration, retry policy, and relevant implementation hashes BEFORE computing scientific run identity. Store that effective configuration. Include its hash in keys and manifests. Resume into an incompatible directory must fail before constructing a client. Operational spending limits may increase explicitly without changing the scientific treatment, but cumulative usage must persist. Cover a CLI model override differing from the source YAML as well as thinking and temperature overrides.
2. Reproduce and fix the observed failure: one fake-client job at thinking=0, then same directory at thinking=8192 currently silently skips the job. Require rejection, not reuse or overwriting. Also test identical-setting resume makes no calls. Legacy manifests must be readable for analysis but must not silently resume without a verified effective identity.
3. Preserve exact request text and generation settings for every attempt, including repair prompts, with request hashes. Extend the provider result with resolved model version, finish reason, and usage-availability status where exposed. Mark unavailable metadata explicitly. Keep credentials and headers out of logs. Handle incomplete/non-success finishes explicitly rather than claiming every parseable object is a completed response. Do not require or expose private thinking text.
4. Replace the claimed hard character/token bound with a real pre-generation token count (cached by exact request hash) or a justified conservative bound. A local heuristic can remain in an offline estimate but must be labelled as such. Include tokenizer/counting overhead and the full retry allowance in whole-experiment planning. Do not introduce a tiny arbitrary spending cutoff. Document the actual provider output limit; currently it is the sum of configured output and thinking allowances, not an independent visible-text cap.
5. Log missing usage as unknown and retain its reservation; never convert absent usage metadata into zero cost. Ensure persistence/bookkeeping errors after a provider return are not retried as generation errors. Recover a saved completed response after a crash before calling the provider again. Test the response-saved / score-not-yet-saved interruption boundary, not just that a file exists.
6. Add `posterior_mse` (or `excess_expected_brier_latent`) for the existing squared-reference error. Add expected reward Brier with its variance term. Keep a documented legacy alias where compatibility is needed. Report maximum/mean absolute C error as well as signed drift, per-action denominators, and utility loss. Treat equal-utility actions as correct within a stated tolerance. Validate range errors separately from an internally inconsistent but otherwise valid probability/forecast pair; record the latter as a scientific consistency metric.

## Acceptance

All existing tests plus the new regression cases pass; old saved results can be analyzed unchanged; the exact same v1 responses still give 0 versus 7 decision errors and the same utilities. Show fake-client tests for the mismatch/resume, unknown usage, and saved-response recovery paths. Generate a fresh dry-run manifest in a new directory. No API calls required for this task; provider metadata/counting behavior can be tested using SDK-shaped fakes and checked against official provider documentation.

Return a concise worker note listing changed files, actual test results, effective-key rules, any remaining limitations, and a ready-to-use invocation for task 004's later live execution. Stop here; the next scientific task is specified separately.
