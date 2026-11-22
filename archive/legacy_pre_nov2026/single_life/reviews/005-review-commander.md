# Review 005 — accepted as an integration and note-retention diagnostic

29 September 2026. Commander: Codex.

## Decision

Accept the completed task and its saved trajectories. This establishes a working Gemini integration and a small, causal note-retention comparison. It does not establish harmful memory, a procedure-revision method, generalization, or RSI. Close this generic note comparison; do not launch the suggested note-freezing/frequency sweep automatically.

## Independent verification

Ran `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -q`: all 42 tests passed. Ran [the independent saved-data audit](005-audit-commander.py); [verification](005-verification.json) contains results and hashes. No generation requests were made during review.

The audit checked all 196 saved generation records: exact prompt instruction and allowlisted payload, own last-12 observed prefix, previous note only in the persistent condition, empty initial memory, parsed action against the action actually scored, and artifact cutoff/deployment. All 388 observed rewards across smoke, pilot, and offline references were recomputed from private probabilities and independently reconstructed indexed innovations. Paired pilot configurations differ only in agent identity. Source, config, and manifest log hashes match. The current source hash is `1e1772057b66e48a00ec1b54a411694cd7fc2fe694eec67926fdf560426d9265`.

There are 4 smoke requests and 192 pilot requests, all successful, with no retries, invalid actions, or unknown usage. Saved model versions are `gemini-3.5-flash`. Input tokens total 73,487; billed output including thinking totals 55,293, of which 44,502 are thinking. Estimated cost is **$0.6078675**, not an audited provider invoice. Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash), independently checked during review, lists standard input $1.50/M and output including thinking $9/M. These rates reproduce every saved attempt cost and the final ledger. Maximum observed input/output counts were 454/621, well below the declared 4096/2048 limits.

The code supplies a textual progress bar, attempt/retry counters, estimates, pending-call heartbeat, and atomic progress snapshot. Fake-client tests exercise these paths; the final live snapshot reconciles with the saved attempts. This review did not observe the original live terminal session.

## What the numbers mean

| World | Recent-only realized reward | Retained-note realized reward | Sum of chosen-action expected rewards, recent / note |
|---|---:|---:|---:|
| Stable | 26 | 34 | 29.1 / 37.5 |
| Changing | 35 | 32 | 34.5 / 34.5 |

Each row is one lifetime per condition, 48 actions. Expected reward here is the evaluator-only sum of the true probabilities of the actions actually chosen. It is an additional post-hoc diagnostic, not a replacement registered primary metric or an expectation over repeated complete agent trajectories.

The stable note agent chose A 46 times; recent-only chose A 24 times and explored B/C more. In the changing world, the observed three-reward difference does **not** establish an action-quality disadvantage: both chosen-action probability sums are 34.5. Action-dependent feedback can still affect later choices, so this is not a complete causal decomposition of the two trajectories.

Persistent notes changed text 46/47 times after the first response in the stable/changing runs. Inspection shows frequent recent-window summaries. Text edits are not executable procedure revisions. The model could choose an action inconsistent with its carried note. This implementation therefore gives no clean test of a long-lived learned procedure becoming invalid. It also does not isolate the value of remembering an earlier regime on recurrence.

UCB and random each ran offline; realized rewards were 29 and 28 in each world. A single UCB window is not an exhaustive strong-baseline comparison. No significance test, population effect, or new scientific contribution follows.

## Bounded implementation caveats

1. `estimate_tokens` uses characters/4. This is an estimate, not enforcement of a hard provider token limit; the advertised worst-case cost bound is conditional on that limit. All saved requests were comfortably within it, so this does not invalidate this run. Before the next live task, use provider token counting or a justified conservative bound and test rejection before generation.
2. The inherited `baseline_definition` field describes UCB even in LLM manifests. Other fields correctly identify the actual LLM policy. Remove the irrelevant field or make it null in new LLM manifests; preserve original evidence.
3. Persisted attempts contain exact prompt and visible response text plus normalized usage, not the full provider response envelope. Save raw nonsecret usage metadata/finish/model fields in future runs so token normalization itself can be independently reconstructed.
4. Failure handling and progress reserve paths were tested offline but were not exercised by this successful live run. Source hashes and local artifacts support a reproducibility audit, not independent confirmation from provider billing.

These are carry-forward requirements for task 006, not a reason to repeat task 005.

## Next decision

Authorize a small executable-procedure development slice using the route/mode tables already explored in task 004, now with real acquisition, continuous observation, and revision opportunities independent of hidden changes. Reuse the tables only; discard the supplied incumbent and boundary-conditioned policies. Include stable, local, shared, and non-additive cases plus a competent local online learner. This is a feasibility test, not adoption of a paper benchmark. Task 006 tests acquisition and revision before adding assumption records or recursive improvement.
