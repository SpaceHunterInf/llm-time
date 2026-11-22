# Where the research stands

Updated: 29 September 2026 after review 005. STATE is the authoritative queue.

## The idea

An agent lives through one changing world. It learns procedures from earlier experience, but those procedures may stop working. It must decide when to keep, test, or revise them using only what it has already observed. It cannot replay its life or train on its future.

We want to test whether learned conditions of applicability and selective revision are useful after accounting for investigation costs. We have not demonstrated that hypothesis. The model weights and revision algorithm are still fixed; recursive improvement of the improver is a later question.

## What actually happened

We built an auditable bandit runner, then ran Gemini 3.5 Flash for four 48-action lifetimes plus four smoke actions. This made **196 successful generation requests**, costing an estimated **$0.6078675** from saved token usage. Codex reran all 42 offline tests and independently checked the saved inputs, chosen actions, note handoffs, rewards, hashes and costs. No new model requests were needed for that review.

Both Gemini conditions saw their last 12 observations. One also retained its own previous strategy note.

| World | Recent observations only | Recent observations plus note |
|---|---:|---:|
| Stable | 26 rewards | 34 rewards |
| Hidden changes | 35 rewards | 32 rewards |

The stable note agent re-explored less. In the changing world, both agents' chosen actions sum to the same true expected reward, 34.5; the three-reward realized difference does not establish that memory hurt adaptation. There is only one lifetime per condition/world, so these are descriptive traces.

The notes were usually rewritten from recent observations. They did not constitute an executable procedure controlling future actions. We are closing this baseline instead of expanding it into another generic memory study.

## What differs from the stale project

The old bandit experiments already had chronological interaction and hidden changes. The old sequential experiments already studied carried numerical state, sometimes alongside a recent evidence suffix. A later review weakened their broad harmful-state claim. Calling an episode a lifetime does not create a new idea.

Task 005 combined explicit free-form note retention, a short history window, and action-dependent feedback in a repaired harness. That is a narrow integration/measurement exercise. It is not a new RSI contribution.

## What Grok should do next

Task 006 will let Gemini write a small executable routing procedure for three request types and four route/mode choices. The procedure runs between model calls. One condition can revise it every 12 actions; another keeps the procedure acquired early in life. They start without a winning rule. The world may change without warning, and exploratory actions consume real lifetime ticks.

A simple online learner will also start empty, notice an observed utility mismatch, and relearn the affected type. We must compare against it seriously. The stable, shared-change, local-change, and non-additive examples come from task 004, but its supplied winning rule and hidden-boundary-conditioned policies will not be reused.

This is eight short lifetimes and 44 scheduled pilot calls, plus at most two smoke calls. The full generation reserve at the task-005 token rates is about $2.83, with actual cost expected lower. Live progress and cost estimates are required. No training is involved.

The immediate question is whether the model can acquire and revise a procedure at all in this setup. It does not yet test our proposed applicability-record method. If a simple learner solves the toy, we should retain it as an engineering fixture and reconsider the paper task. A larger simulator is not automatically the answer.
