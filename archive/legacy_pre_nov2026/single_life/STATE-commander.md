# Current project state

Updated: 29 September 2026, after review 005. This is the only active queue for `single_life/`.

## Status

- Research target: adaptation of learned procedures during one uninterrupted lifetime, using only the observed prefix. Applicability testing and selective revision remain candidate mechanisms, unimplemented and unvalidated.
- Tasks 001–003: accepted trusted offline bandit fixture, protocol v0.2 / checkpoint v3.
- Task 004: accepted as arithmetic/design illustration only; supplied incumbent and boundary-conditioned policies are not valid online learning evidence.
- **Task 005: accepted as Gemini integration and narrow strategy-note diagnostic.** All 42 offline tests pass. Independent audit validates saved prompts, note timing, actions, rewards, source identity and accounting.
- **Gemini has run:** 196 successful generation requests (4 smoke + 192 pilot), no retries/failures, estimated cost **$0.6078675**. This is token-price accounting, not confirmed provider billing. No model calls during commander review.
- Stable realized reward: recent-only 26, note 34. Changing: recent-only 35, note 32; both changing trajectories have chosen-action expected-reward sum **34.5**. No evidence here for a general harmful-memory effect, procedure-level improvement, or RSI.
- Generic note-retention comparison closed. No replication or note-frequency sweep queued.
- **Current task: 006 — executable procedure acquisition and revision**, ready for Grok. Small contextual toy plus bounded live run in one assignment. Protocol v0.3 extension is specified, not yet implemented/accepted.
- API progress bar and cost estimates remain mandatory; task 006 separates environment ticks, model calls and provider attempts.

## Give Grok this task

Read [review 005](reviews/005-review-commander.md), then execute [006-procedure-acquisition-commander.md](tasks/006-procedure-acquisition-commander.md). Return `tasks/006-procedure-acquisition-worker.md`. The user invokes Grok; Codex has not launched it.

Scope: reuse task-004 utility tables without its incumbent or boundary-aware policies; let Gemini acquire a small executable JSON procedure; compare periodic revision with freezing after acquisition and competent online references. Eight 96-action lifetimes, 44 scheduled pilot generation calls, at most two smoke calls, at most 92 generation attempts including retries. No training or larger sweep. At task-005 rates the full generation reserve is $2.82624; verify rates and accounting before execution.

For a plain-language explanation, read [BRIEF](research/BRIEF-commander.md).

## Decision gates

1. Kernel/integration gates passed within the limits of reviews 003 and 005. Small carry-forward API cap/provenance corrections are included in 006; do not rerun 005.
2. Task 006 is a feasibility slice, not adoption of the desk as a paper benchmark. It must demonstrate acquisition from empty state and actual execution/revision. The simple local online learner is a serious alternative explanation.
3. If acquisition fails, report it without supplying a winning procedure. If local relearning suffices, close the toy as a fixture rather than adding complexity to protect the hypothesis.
4. If the slice works, commander decides whether a stronger procedure task and applicability-testing intervention are justified. No automatic paid follow-up or recursive improver work.
5. Final evaluation, mechanism generalization, substantial training, and paper claims remain unestablished.

## Latest review

[Review 005](reviews/005-review-commander.md), [independent verification](reviews/005-verification.json), and [re-runnable saved-data audit](reviews/005-audit-commander.py). Original worker reports and run artifacts are preserved. Hashes are evidence identities, not a provider invoice or a paper result.
