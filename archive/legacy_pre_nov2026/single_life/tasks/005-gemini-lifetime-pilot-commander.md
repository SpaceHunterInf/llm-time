# Task 005 — Gemini integration and strategy-note baseline

Owner: Grok. Commander: Codex. Status: accepted within integration/diagnostic scope; see review 005. Date: 29 September 2026.

## Purpose

Deliver the first Gemini trajectories in the new implementation, using the existing Bernoulli bandit fixture. Earlier repository experiments already studied hidden-change adaptation and carried state. This is a migration/integration baseline with a narrower note-retention ablation, not the first investigation of that phenomenon and not a new scientific contribution. Do not build the factored desk or a queue simulator.

Question: does carrying a self-written strategy note help or hinder behavior after hidden change, relative to the same model with recent observations only?

This is a small development diagnostic, not a powered comparison, paper result, or demonstration of RSI. Model weights and the note-update instruction remain fixed. A textual note is memory/policy guidance, not executable self-modification.

## What differs from the legacy work

Read the historical comparisons in [revision inertia](../../reports/revision_inertia_2026-09.md), [sequential-state task](../../research_tasks/004_sequential_revision-commander.md), and [handoff review](../../reports/task006_review-commander.md). Do not repeat their broad questions as if unanswered.

- The original bandit runner already chose actions and observed rewards chronologically inside each episode. Calling the new unit a lifetime does not create a new scientific setting.
- The inspected legacy bandit history renderer usually replayed actions/rewards/observations or engine statistics, not the emitted free-form heuristic. Its hypothesis-surprise variant did use a prior declared arm, so do not claim there was no feedback of any prior declaration.
- Legacy sequential tasks explicitly carried the model's previous numerical joint state, and a later condition used prior state plus an observation suffix. Their evidence tapes were fixed; the evaluated choices did not determine subsequent observations.
- Here the agent retains a free-form self-written note alongside the last 12 observations, and its action determines the next observed reward. The note-retention toggle is the narrow intervention. This combines previously explored ingredients rather than establishing a new self-improvement method.
- Public/private separation, indexed innovations, and better logs are measurement improvements. They are not novelty claims and do not prove historical LLM prompts leaked future information.

The old handoff review downgraded the broad carried-state-harm claim after its large penalty did not recur. Do not assume the new baseline will rediscover a strong harmful-memory effect. Its legitimate outputs are a working adapter, auditable live trajectories, and a baseline for later procedure-level experiments. No larger sweep follows automatically from a positive-looking reward gap.

## Live scope — part of this task

The user's existing authorization covers small Gemini development experiments. After the offline checks and cost preflight pass, run the bounded live work below without asking for the same permission again. Do not stop at a fake-client implementation and call this task complete. If provider access or the completion budget is blocked, return the concrete error and completed work.

Use the configured Gemini development provider. Resolve and record an actual supported model ID and generation settings; check availability and pricing against official Gemini sources at execution time. Do not invent a model ID from older reports. Credentials must never enter output artifacts or the conversation. No DeepSeek, training, downloads, or parent-environment sync.

## Registered diagnostic conditions

Use one fixed model/settings configuration throughout.

- **Recent-only:** current public action interface plus the last 12 global-time completed observation records; no retained strategy note.
- **Persistent-note:** the same observation window and action interface plus the last valid self-written strategy note, initially empty.

Both conditions return an action and a short proposed note using the same response schema and instruction. The recent-only condition's note is recorded for analysis but not carried into its next call. This isolates note retention as the intervention. It intentionally changes access to summarized older experience; do not claim both policies receive identical historical information. They have the same raw recent feedback and resource ceilings.

A single call chooses the next action and writes a proposed note from the evidence available before that action's outcome. Deploy its note for the next decision through host-owned artifact handling. No second improver call, self-reported success selection, or scoring on future rewards. Cap the retained note at a declared small size (initial ceiling 1,000 characters) and handle excess/invalid output by a fixed parser policy logged in advance.

## Worlds and size

Create two private development configurations, each 48 actions and three stable action IDs:

- **Stable:** probabilities [0.8, 0.2, 0.5] throughout.
- **Changing:** [0.8, 0.2, 0.5] at actions 1–16; [0.2, 0.8, 0.5] at 17–34; [0.8, 0.2, 0.5] at 35–48.

The boundaries, probability table, and recurrence are evaluator-only. They must never appear in prompts or control note updates. Update opportunities occur every decision in both worlds, including before the first change. Do not tune these settings after seeing the run.

Run both conditions once in each world: 4 independent initialized agent lifetimes, 192 scheduled action decisions total. Share indexed exogenous innovations for the paired comparisons; do not share acquired notes or provider responses across lifetimes. Initialize no winning strategy. The short acquisition period may be insufficient; report that result rather than supplying a good policy.

Run the existing sliding-window UCB and random references on the same configurations offline. Label them as references, not equally costly LLM methods.

## Integration and budgets

1. Before the pilot, an at-most-four-decision Gemini smoke in a separate lifetime may verify transport/schema only. Its outputs must not seed the pilot memories. Freeze prompts/configs before the pilot. Maximum scheduled live decisions for the whole task: 196.
2. At most two provider attempts per decision, including parser retries: maximum 392 requests over smoke and pilot. Never execute two environment actions to obtain one accepted response. A rejected action/failed decision must have explicit tick/accounting behavior consistent with the protocol.
3. Use bounded input/output and, where separately controlled, thinking-token limits. Initial ceilings: 4,096 input tokens and 1,024 generated output tokens per attempt; explicitly include any separately billed reasoning tokens in the estimate. If the provider cannot support this exact convention, record an equivalent bounded setting before running.
4. Calculate the full worst-case completion cost at verified current rates, including the retry reserve and thinking. Save the estimate and a sufficient run-wide cap before sending requests. Stay with a modest Gemini development configuration; a substantial scope/cost increase requires flagging before launch. Do not set an arbitrary low cap that predictably interrupts the registered batch.
5. Save every attempt's request/response, model identity, usage/unknown status, parser/finish result, and price source/date. Do not reuse legacy cost constants without verification. Keep reward, API cost, token usage, and elapsed time separate.
6. Prefer a small explicit provider adapter. Reviewed accounting patterns may be ported from the legacy bounded harness with provenance; no runtime imports from the parent project.
7. Record the actual LLM policy/memory condition in manifests, never label it as random or a scripted artifact policy merely to fit the old factory. Keep any new code/config identity explicit. Preserve old checkpoint behavior; either support the new policy state correctly or explicitly disable LLM checkpoint/resume for this diagnostic. Do not silently restart an interrupted lifetime or replace a failed replicate.

## Required API progress and cost display

This requirement comes directly from the user and applies to smoke and pilot runs.

- Before the first paid request, print and save the model/settings, planned decisions, maximum attempts including retries, token limits, verified price source/date, an expected-cost estimate or clearly stated estimate range, and a conservative full-completion upper bound including thinking and retry reserves. Show the actual configured spend cap and how the estimates were calculated. Unknown rates/usage must be labeled unknown, never priced as zero.
- Show a live terminal progress bar (tqdm or equivalent). Show completed decisions versus planned decisions, current world/condition and lifetime step, provider attempt/retry counts, failures, elapsed time, and estimated time remaining. Separate smoke and pilot counters while maintaining cumulative cost for the entire task.
- Also show estimated spend from reported token usage so far, unresolved/in-flight reserved cost, estimated remaining cost, projected final cost, and the configured cap/headroom. Distinguish a usage-based estimate from a provider billing statement. Compute projected remaining cost using observed usage after warmup, without replacing the conservative completion bound.
- A retry increments the request/cost counters, not the completed-decision bar. Only a committed or terminally failed decision advances that bar according to the registered failure policy. Slow calls show pending status; emit a heartbeat at least every 30 seconds without issuing a duplicate request.
- In non-interactive/logged execution, emit readable periodic progress lines rather than carriage-return-only output. Persist an atomic progress.json with the same counters, timestamps, cost estimates/reservations, and run status so the user can inspect a running task. Progress is monitoring data and must not enter agent prompts.
- Finish with a concise complete/failed/interrupted summary, actual decision/attempt/retry counts, observed token usage, estimated cost plus unknown/reserved usage, and output paths. Never display 100% success for an incomplete run.
- Verify the display and ledger using a fake sequence with success, retry, unknown usage, and failure. Check that retry costs accumulate once and that monitoring reveals no keys, credentials, prompts, or evaluator-private parameters. Add one example progress line or captured text output to the worker return.

## Minimum offline checks

Use a fake client to verify: allowlisted public prompts; memory starts empty; memory carried only in the persistent condition; bounded note contents and actual artifact use; no timing/regime labels; every retry accounted; no extra environment action; differing sealed futures leave prompts through the shared prefix identical; no parent imports or credential logging. Run the existing accepted suite plus targeted adapter tests. Avoid unrelated framework work.

The LLM sees no filesystem, evaluator object, source repository, simulator tool, or private logs. It receives only the serialized prompt. This is sufficient for this text-action diagnostic and does not implement generated-code execution.

## Analysis and deliverables

Register the effective model, prompts, private configurations, ceilings, pricing, failure rules, and reference settings before the pilot. Store outputs in a new run directory and compact results under experiments/005-gemini-lifetime/.

Report:
- exactly how many live Gemini requests succeeded/failed and their full observed/estimated cost;
- cumulative realized reward for each entire lifetime, with acquisition/stable/post-change windows as descriptive diagnostics only;
- plots or a compact timeline of chosen actions, observed rewards, and actual note changes;
- whether a useful strategy was acquired before the hidden change, whether the old rule persisted, and whether recurrence helped;
- matched-context and token-usage limitations, model stochasticity, and missing/failed runs;
- a few concrete prompt/note/action excerpts with evidence cutoffs, without evaluator annotations inside runtime messages.

One paired lifetime per world is not enough to infer a population effect or significance. Do not call a reward difference “harmful memory” without inspecting the trajectories and accounting for different action-dependent observed feedback. Do not tune prompts, seeds, windows, or schedules and report the selected result as this pilot.

## Return and stop

Write tasks/005-gemini-lifetime-pilot-worker.md. Lead with a plain-language account of what Gemini actually did, how this narrow baseline differs from legacy work, and what remains uncertain; then give files, commands, checks, costs, and limitations. Do not launch further replications or a new method. Return for commander review.

If the pilot shows only trivial estimation errors or no acquired strategy, recommend the cheapest discriminating follow-up rather than escalating to training. This task is intended to obtain the first real behavior, not to force a positive story.
