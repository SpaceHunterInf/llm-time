# Research operating agreement

Date: 28 September 2026. This is the operating procedure for the new project, not an instruction to launch an unbounded autonomous loop.

## Roles

- **User:** owns the direction, constraints, and substantive changes to scope/resources.
- **Codex, commander:** maintains the research question; checks literature and alternative explanations; specifies experiments, information boundaries, metrics, and acceptance criteria; reviews implementation and results; issues one next task.
- **Grok, implementer:** completes the bounded task; tests the actual paths; saves reproducible artifacts; reports failures and deviations. Grok can resolve ordinary implementation details, but cannot quietly change the research question, feedback available to the agent, evaluation distribution, or primary metric.

The user invokes Grok. No automated Grok launcher or multi-agent orchestration service is required. A failed experiment is an acceptable research outcome. The implementer is not tasked with obtaining a positive result.

## One iteration

1. Read STATE and the current numbered task. Check working-tree changes before editing.
2. Before a scientific run, fill an experiment registration using the template: hypothesis, comparison, data split, budget, metric, falsifier, and exact configuration. Infrastructure smoke tests need a task and run manifest, not an additional research essay.
3. Implement the smallest necessary change. Keep the evaluator fixed during the comparison. A bug fix that affects scoring or information access increments protocol identity and invalidates affected comparisons.
4. Run offline invariants and the bounded experiment authorized by the task. Record crashes, retries, aborted attempts, and negative outcomes. Do not silently replace failed seeds.
5. Return `tasks/NNN-<topic>-worker.md` with changed files, commands, artifacts, results, deviations, and unresolved issues. Do not mark commander acceptance yourself.
6. Codex reviews the diff and evidence, checks alternative explanations, records accepted/revise/rejected, updates STATE, and issues the next task.

Acceptance of working infrastructure does not establish the hypothesis. A successful API call does not establish adaptation. A reward increase after an edit does not by itself establish that the edit caused it.

## Two different loops

The **research loop** can compare independent lifetimes on development configurations. These runs inform the researchers' design decisions. Frozen evaluation configurations and independent agent initializations are needed for final claims; repeatedly tuning on test outcomes converts them into development data.

The **agent's lifetime** advances once. It can read its own historical records and revise internal artifacts. It cannot rewind environmental state, score candidates on hidden future outcomes, or inherit another evaluation replicate's experience. Offline researcher branches must never feed their later scores back into an earlier decision in the focal lifetime.

Do not transplant an autoresearch-style “run the whole benchmark, retain the winner, rewind” loop into the online agent. That answers a different question unless all such trials are real, charged interactions on the advancing world.

## Practical constraints

- Start with API-capable architecture and CPU diagnostics; no fine-tuning is queued. Later small-model training is optional if evidence justifies it. The local GPU is for development/verification; serious training arrangements remain with the user.
- Use independent uv dependencies. Pin actual dependencies when added and retain the lockfile. Avoid importing the legacy engine wholesale.
- A future paid task must specify model identity, maximum requests/tokens/retries, and an experiment-wide spend estimate/cap sufficient to complete its stated scope. Existing authorization for small API experiments is not an instruction to spend during this offline task.
- Save exact prompts, responses, parser failures, and per-attempt usage for API work. Unknown usage is unknown, not zero. Keep simulated reward, elapsed time, API dollars, and tokens in separate fields.
- Keep compact manifests, registrations, summaries, and worker reports versionable. Large run artifacts live in `runs/`; archive them with hashes before relying on them in a paper. Do not delete unsuccessful runs just because their code was superseded.
- Use one bounded task at a time. No automatic follow-on paid run, global Git reset, or generated manuscript before the evidence exists.

## API observability — user requirement, 29 September 2026

Every API experiment must include a live progress bar and visible cost estimation, not just an after-the-fact report. Before sending requests, show planned work, model/settings, expected cost or estimate range, a conservative completion bound including retries/thinking, the configured cap, and verified pricing provenance.

While running, show decision progress separately from provider attempts/retries, current condition/step, failures, elapsed time/ETA, usage-based estimated spend, in-flight/unknown-usage reserves, remaining-cost estimate, and projected final cost versus cap. Retries consume money without completing extra environment decisions. Unknown usage is not free. Keep smoke/pilot totals distinct and task-wide spend cumulative.

Use readable periodic status lines when no interactive terminal is available. Give pending calls a heartbeat at least every 30 seconds without duplicating requests. Persist atomic progress.json monitoring snapshots. Exclude monitoring data from agent inputs. End with a truthful completed/failed/interrupted summary and evidence paths. Reports must distinguish token-price estimates from checked provider billing.

Before interpreting a new API batch scientifically, state what earlier repository work already established and what the current intervention changes. A new directory or repaired harness is not a new research contribution.

## Record ownership

STATE is the current status; question is the scientific target; protocol is the validity contract; tasks are execution instructions. Decisions records changes to these choices. Related-work entries record verified overlap and remaining novelty questions. Do not copy a second active queue into other files.

Grok may propose a protocol change in its return. Codex reconciles the authoritative documents and updates the protocol version before the changed comparison. New user instructions can override the plan immediately; document the resulting change rather than using stale files to block the user.
