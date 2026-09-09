---
name: auto-research-worker
description: Run llm-time auto-research as the Grok worker under Codex commander briefs. Use when implementing -commander.md tasks, continuing auto-research, running the selective-revision Gemini pilot, emailing gs06monitor@gmail.com, or the user is away in screen.
---

# Auto-research worker

Read `AUTO_RESEARCH-commander.md`, `auto_research_rules.md`, then the current `research_tasks/*-commander.md`. Implement one brief. Do not decide what a result means scientifically.

## File names

- Commander plans: `*-commander.md`
- Worker notes: `*-worker.md`
- Code and result reports: no suffix rule

## While the user is away

Email `gs06monitor@gmail.com` when a task finishes, when blocked, or before a spend jump. Never put `.env` values, keys, or headers in mail, logs, or reports. `mail -s "subject" gs06monitor@gmail.com` is enough.

## Spend

- Gemini for development. No DeepSeek development runs.
- Ordinary diagnostic batches are allowed. Do not auto-launch the next paid batch just because the last one finished.
- User cap (2026-09-09): not more than about **US$20 per run**. Preflight the full grid plus one retry each before any call.
- Reserve billed output as `max_output_tokens + thinking_budget`. Reject `thinking_budget=-1`.
- Smoke ≤3 calls first. One in-flight request. Stop on auth/quota. Resume, never overwrite.
- Live runs must show a progress bar and flush each provider response to `responses/` *before* scoring. `summary.json` / `progress.json` update after every job. Do not wait until the process exits to persist outputs.

Thinking is on for the diagnostic so truncation of hidden reasoning does not look like a Bayes failure. After smoke, set the thinking budget from measured `thoughts_tokens`, not from a guess.

## Stack

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync ...`

No sudo unless the user said so. Local GPU is verification/dev only; do not start serious training here.

## Do not

Train, open ToolSandbox, or claim model performance from a dry-run. A statistical observer is not a learning result. Invalid JSON is a parse failure, not a revision result.

## Return

Commands actually run, files, tests, planned vs actual spend, limitations. Stop at the brief's deliverable.
