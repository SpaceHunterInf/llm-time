# Task 004 review — accept the calculation, limit its interpretation

Reviewed: 29 September 2026. Decision: **accept as an illustrative feasibility note; do not promote it to an autonomous online experiment or build the proposed desk yet**.

## What ran

A standard-library Python script evaluated hand-written rules on four deterministic reward tables. No LLM, Gemini request, GPU job, or new production fixture was used. I copied the script to a temporary directory and ran it there; every assertion passed and the generated witness JSON matched the supplied file byte for byte. All three worker-reported file hashes match. See [verification](004-verification.json).

The new single_life project has made no Gemini calls in tasks 001–004. Historical parent-repository reports document earlier Gemini experiments on different tasks; these are not evidence for the new lifetime protocol.

## What the arithmetic shows

In the shared-change example, copying a successful mode repair across request types yields response utility 7.3, versus 6.3 for per-type search. In the non-additive counterexample, that copy yields 7.4, versus 12.1 for per-type search. Under the arbitrarily equal case weights, per-type search yields 11.8 versus 10.875 for transfer.

These are exact illustrative totals, not estimated effects or population evidence. Their useful lesson is that transferring a local repair assumes a shared cause, and that assumption can be wrong. Nothing establishes that an LLM or explicit assumption records are needed.

## Corrections to interpretation

1. **The rules have privileged boundary information.** In reference_witness.py, CoordinateBase and OneTypeThenTransfer split history using CHANGE_AT; run_case activates adaptation at RESPONSE_AT. All reward observations used are in the past, so this is not future-reward leakage. But the exact change boundary is used to choose evidence windows and a fixed response phase is imposed. The dossier's claim that the fair rules use only current request type and observed scalars is incomplete: they also use researcher-provided timing structure. Treat this as a boundary-conditioned illustration, not hidden-boundary adaptation.
2. **The initial procedure is supplied.** INCUMBENT already selects the correct server/mode before the change. There is no strategy acquisition experiment. Calling ticks 1–12 an acquisition window should not be read as learning.
3. **The named baseline is not yet a demonstrated strong online baseline.** PerTypeStop is fixed-order, stop-at-first-improvement search using the known time split. It is neither a general recency estimator nor a calibrated change detector. Comparing two hand-written rules cannot settle whether a competent adaptive learner removes all headroom.
4. **The scope is too narrow to reject the larger direction.** A failed cross-type copy on one constructed trap rejects that particular extrapolation. It does not show that conditional procedure reuse, learned investigation, or nonstationary self-improvement is unpromising.

These limitations do not invalidate the arithmetic. Task 004 explicitly permitted hand-authored feasibility witnesses. Preserve its original outputs with this review as the authoritative classification; another documentation-repair round is unnecessary.

## Decision and next experiment

Do not build a queue simulator or the factored-desk fixture yet. The desk is still essentially a small contextual action table, and the added implementation would not answer whether real LLM behavior contains a useful phenomenon.

Instead run the first bounded Gemini diagnostic on the accepted bandit infrastructure. Compare an agent retaining its own short strategy note against the same model using the same recent observation window without retained notes. Include an unchanged world and a world with hidden changes, start each agent with no learned note, and let it act continuously from the first observation. No reset, announced boundary, hand-supplied winning policy, or oracle-timed evidence split.

This tests a necessary precursor: whether self-written historical strategy affects acquisition, persistence, and recovery. It does not test the full proposed method or recursive improvement. A tiny run can fail to acquire a useful strategy, reveal only numerical-estimation problems, or show no interesting memory effect; all are legitimate outcomes.

[Task 005](../tasks/005-gemini-lifetime-pilot-commander.md) combines the minimal adapter, offline checks, and the bounded live diagnostic. Do not issue another standalone infrastructure-only task before that live run unless a concrete blocker prevents it.
