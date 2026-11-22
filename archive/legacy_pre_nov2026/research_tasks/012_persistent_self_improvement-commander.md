# Task 012: minimal persistent solver and improver loop

Author: Codex (commander), 25 September 2026. Worker: Grok.

Read `auto_research_rules.md` and [the current direction](../research-long-trajectory-self-improvement-commander.md). This supersedes task 011 as the next implementation. Do not implement the old fixed-library drift experiment first. Preserve all earlier artifacts and commander plans. Use uv, CPU only, at most four threads, no training/model downloads/subagents.

## Deliverable

One small runnable prototype with separately versioned solver code, improver code and experience, plus a reproducible worker report. Demonstrate that accepted code changes persist and influence future improvement requests. This is an integration task, not an efficacy study or evidence of novel RSI. Prefer reusing the existing Gemini client and robust attempt ledger; do not inherit the historical runner's incomplete accounting.

## 1. Explicit CPU task

Implement a small deterministic job-allocation problem with public job values, processing requirements, deadlines and worker capacities. Document one objective and feasibility semantics before experiments. Let `solve(instance)` return an assignment. Include three competent heuristic references and an independently checked exact solution for tiny fixtures. A simple enumeration or dynamic program is sufficient; avoid new solver dependencies. Record reference headroom on fixed development instances, including variable sizes and constraint combinations. Do not build latent regimes, change detectors or a fixed strategy selector.

Use separate deterministic seed namespaces for visible experience, candidate selection and later evaluation. Keep the evaluator/generator immutable. Freeze the generator configuration and seed manifest before model calls. Include obvious feasibility fixtures and a nontrivial case where the strong heuristic is suboptimal; label that illustrative case rather than using it as a population result. If broad development headroom is negligible, finish the offline report and omit the paid smoke.

## 2. Mutable artifacts and host boundary

The initial solver must be a competent heuristic. The initial improver is a small executable function that consumes the permitted archive, current artifact sources and requested edit type, then returns a structured proposal request: selected evidence IDs plus bounded instructions. The immutable host adds the task contract and artifact scope, calls the model and validates the returned edit. Permit algorithmic changes to evidence selection and proposal construction; do not reduce improvement to selecting one of a few hard-coded prompt IDs.

Accepted solver code runs on subsequent instances. Accepted improver code constructs subsequent solver-edit AND self-edit requests. Hash every version and record parent/descendant identities. Keep raw model responses, proposed files, rejection reasons and deployed versions. Experience truncation must be deterministic and visible in logs.

Generated code must not execute in the credential-bearing API process. Use a restricted, resource-limited evaluator with no network, secrets, repository write access or access to future evaluation data. A Python `exec` namespace or AST check alone is not a security boundary. Reuse an available isolation mechanism without sudo; if adequate isolation is unavailable, deliver scripted/offline execution and the concrete blocker rather than execute untrusted code with ambient access. The host retains all accounting and deployment authority. Candidate code cannot edit evaluator, manifest, model configuration or budgets.

## 3. Acceptance and a bounded smoke

For solver proposals, check interface/feasibility and compare candidate/incumbent on identical fresh selection instances. Deploy only a valid improvement under a predeclared deterministic rule; retain the incumbent on ties. Evaluate deployment on a separate instance block. Repeated selection against development data is allowed and logged; it is not held-out scientific evidence.

For improver proposals, validate the interface, then run old/candidate improvers from the same solver and history to obtain one solver proposal each. Score those descendants on the same fresh selection batch. Deploy the candidate improver only if its valid descendant beats the old improver's descendant; ties retain the old one. For this smoke, restore the common solver afterward so that any later behavioral difference comes from the selected improver. One paired descendant is noisy and is only an integration rule, not a validated research method.

First execute scripted offline cases that force an accepted solver edit, a rejected edit, an accepted improver edit and an invalid proposal. Verify that the next request is produced by the actual new improver. Then, if task headroom and isolation checks pass, run **one live Gemini smoke**:

1. Two sequential solver-edit requests, with CPU deployment blocks between them.
2. One improver-edit request.
3. Two descendant solver-edit requests, one from each improver at the common checkpoint.
4. Two continuation solver-edit requests using the selected improver.

That is seven logical requests. Allow at most two retries per request, for at most 21 billable attempts; failed attempts consume reserve. Skip invalid dependent branches and report them rather than inventing replacement experiments. Use eight instances per deployed block and sixteen per selection batch. These counts test wiring, not long-horizon research efficacy. Set explicit context and output-token limits after an offline prompt-size check; avoid truncating source code silently. Preserve unsuccessful live self-edits as results; the smoke need not produce an accepted meta-edit to be complete.

The user has authorized ordinary small API experiments and rejected an arbitrary $1 stop that interrupts a planned run. Verify the chosen existing development Gemini model's current official prices, preflight the entire seven-request workload plus retry reserve, save that estimate, and reserve enough to finish it. Do not switch to an expensive model or enlarge the workload. If the projected cost is materially beyond previous ordinary pilots (roughly single-digit to low-double-digit dollars), return the estimate before launch. This is a scope flag, not an arbitrary mid-run cap. A runtime overrun or provider anomaly should checkpoint safely; do not silently increase the preflight budget. DeepSeek is not for development.

## 4. Necessary verification and records

Test reward/feasibility against the independent tiny-case reference; inherited improver execution; candidate rejection retaining the incumbent; evaluator/budget immutability; isolation of future instances; charged descendant/retry calls; and checkpoint recovery without reissuing completed calls. Reuse existing tests where appropriate, and add only meaningful checks of the new behavior.

Keep paired descendant evaluations keyed by common starting solver, history hash and selection-instance IDs. Report every attempted edit and actual deployment utility, not just the best candidate. Record API and CPU resources separately; do not subtract dollars from task utility. A rejected improvement still costs resources. Capture actual model/version/config, token usage and pricing source/date; distinguish estimated from billed cost.

Provide commands for dry-run, smoke and resume. Do not launch a comparison grid after the smoke. A future matched-budget study needs frozen baselines and independent lineages; this task cannot establish transfer or recursive advantage.

## Worker return

Write `research_tasks/012_persistent_self_improvement-worker.md`, with links to code, fixed config, raw trace, tests, costs and an illustrative before/after solver and improver diff. Answer:

- Does changed improver code actually generate descendant requests, including a later self-edit request in the scripted test?
- Which edits were proposed, accepted and useful on the next deployment block?
- How much did the meta-edit and its validation cost relative to ordinary solver edits?
- Is there enough task headroom and implementation reliability to justify a small comparison?
- What remains blocked or untested, especially generalization and learning-to-improve transfer?

Return a concrete recommendation. Do not rename a successful plumbing test as RSI evidence, edit commander plans, or start the next paid experiment automatically.
