# Research question and claim boundaries

Date: 28 September 2026. Working direction; method and environment are revisable.

## Main question

**During one uninterrupted lifetime with hidden environmental changes, can an LLM agent improve cumulative utility by maintaining testable conditions for its learned procedures, spending interactions to investigate suspected failures, and revising the affected procedures?**

The motivation is that successful experience can create useful but conditional habits. A procedure acquired under one set of conditions may remain useful, become harmful, or become useful again. The agent must decide using the past and present; the world does not offer repeatable trials of its whole life.

This is not yet a claim of recursive self-improvement. First keep the model weights and revision algorithm fixed while allowing procedures and memories to change. Only after that works ask whether modifying the revision algorithm improves later adaptation beyond a fixed improver at matched resources.

## Candidate mechanism, not a settled method

A learned procedure may contain an executable/configurable rule, its supporting experience, a testable applicability condition, predictions about observable outcomes, and dependencies on other procedures. Example: batching jobs is useful while setup overhead dominates; latency rising specifically with batch size may invalidate that assumption while leaving routing knowledge intact.

On new evidence the agent may keep exploiting, collect a diagnostic observation, edit a local procedure, retire it, or revalidate an archived procedure. Diagnostics consume real interactions or a declared resource. Self-reported confidence alone is not a validity test. Language explanations need an observable behavioral consequence to support a mechanism claim.

The conditions may be wrong or unidentifiable. Test whether their extra structure earns its cost. Do not give only our method the true change dependencies or a hand-authored explanation of the hidden mechanism.

## Hypotheses and alternatives

| Hypothesis | Necessary comparison | Evidence against it |
|---|---|---|
| Acquired procedures sometimes hinder adaptation after hidden change | Accumulated experience versus recent evidence and operational reset/relearning, with stable controls | The effect disappears after matching context size or fixing the estimator |
| Testing applicability and revising locally improves lifetime utility | Same backbone, information, interaction and compute ceilings; matched generic probing and recency controls | Cheap forgetting/change detection or generic probing matches or beats the candidate |
| Local revision preserves still-useful knowledge | Partial changes and unaffected components; measure retention alongside recovery | Whole-agent reset performs equally well, or only the proposed method receives dependency hints |
| Archived procedures help on recurrence without obstructing unfamiliar change | Recurrence and withheld mechanism/composition conditions | Gains occur only for a small known menu of repeated regimes |

Do not require all hypotheses to succeed to report an informative result. A benchmark contribution needs a valid protocol and a reproducible gap that is not an artifact of weak baselines. A method contribution needs improvement beyond strong alternatives and evidence about why. A negative result can stop or redirect this project.

## Initial comparisons

For the eventual LLM pilot: (1) accumulating procedures/memory with generic revision; (2) recent evidence plus an evidence-triggered reset/relearning baseline; (3) candidate applicability testing and local revision. Share the base model, action interface, usable observations, total resource ceilings, and output opportunity. Extra revision/probing calls are charged to the same budget, not gifted to the candidate.

Before any method claim, include a competent task-specific online learner and a generic active-probing/memory-verification comparator motivated by GLOVE. Add ablations for applicability records, diagnostic action choice, and selective edits only after the full comparison warrants them. Report both total resource ceilings and actual usage; methods need not make identical choices about spending them.

## Environment choice

A small nonstationary bandit is the first **causality fixture**, because its hidden/public separation is easy to test. It is not the assumed paper environment. A 180-step, three-regime example is a unit-test illustration, not a required benchmark or public prior.

The preferred next candidate is a continuing resource-allocation/service task: jobs arrive, actions route or batch them, outcomes consume time/capacity, and hidden changes alter only some relationships. Learned procedures should compose and have measurable dependencies. Grok should not build that full environment in task 001.

Select a research environment only after checking: useful strategies can be acquired before change; chosen actions reveal limited feedback; partial change leaves something worth preserving; diagnostics have a cost; simple online baselines have measurable headroom; mechanisms and orderings can be withheld. If a recency-weighted estimator explains the whole task, it is a diagnostic result rather than evidence for LLM-specific research value.

## Scope and assumptions

Hidden changes must persist long enough, affect observable consequences, or have exploitable structure. We cannot promise adaptation to arbitrarily fast, unobservable, or adversarial changes. A fixed simulator can generate a nonstationary learner experience. Training on a fixed distribution of changing lifetimes is legitimate meta-learning when separated from evaluation, but it supports a narrower claim than adaptation to previously unrepresented mechanisms.

Single-life learning, memory verification, selective forgetting, and self-modifying policies all have precedents. The candidate contribution is their carefully tested combination around learned procedures and costly investigation under a causal lifetime protocol. Its novelty remains open; see related work.
