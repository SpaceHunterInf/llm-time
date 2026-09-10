from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from llm_rl_engine.analysis.selective_revision import ReferenceModel, posterior
from llm_rl_engine.analysis.selective_revision_sequential import (
    DEFAULT_H,
    N_STATES,
    PROBE_AFTER,
    CONDITIONS,
    development_tapes,
    filter_prefix,
    generate_tape,
    joint_keys,
    observation_schedule,
    observe,
    predict,
    prior_joint,
    sequential_jobs,
    step_filter,
    transition_matrix,
    marginals,
    state_index,
)


def test_schedule_length_and_probes():
    sched = observation_schedule()
    assert len(sched) == 44
    assert PROBE_AFTER == (12, 20, 32, 44)
    assert sched[:12] == ["observed", "linked", "unrelated"] * 4
    assert sched[12:20] == ["observed"] * 8


def test_h0_identity():
    t = transition_matrix("shared", 0.0)
    for i in range(N_STATES):
        assert t[i][i] == pytest.approx(1.0)
    t2 = transition_matrix("stable", 0.3)
    for i in range(N_STATES):
        assert t2[i][i] == pytest.approx(1.0)


def test_h1_independent_redraw():
    t = transition_matrix("independent", 1.0)
    src = state_index(1, 1, 1)
    # u stays 1; o,l uniform
    for o, l in ((0, 0), (1, 0), (0, 1), (1, 1)):
        assert t[src][state_index(o, l, 1)] == pytest.approx(0.25)
        assert t[src][state_index(o, l, 0)] == pytest.approx(0.0)
    ts = transition_matrix("shared", 1.0)
    for dst in range(N_STATES):
        assert ts[src][dst] == pytest.approx(t[src][dst])


def test_one_step_matches_task001():
    observations = [0, 0, 0, 0, 0]
    for structure in ("shared", "independent"):
        joint = prior_joint(0.99)
        matrix = transition_matrix(structure, 0.1)
        joint = predict(joint, matrix)
        for y in observations:
            joint = observe(joint, "observed", y)
        p = marginals(joint)
        ref = posterior(
            ReferenceModel(structure=structure, hazard=0.1),
            {"A": observations},
        )
        assert p["observed"] == pytest.approx(ref["posterior_high"]["A"])
        assert p["linked"] == pytest.approx(ref["posterior_high"]["B"])
        assert p["unrelated"] == pytest.approx(ref["posterior_high"]["C"])


def test_shared_correlation_retained():
    joint = [0.0] * 8
    joint[state_index(1, 1, 1)] = 1.0
    pred = predict(joint, transition_matrix("shared", 0.2))
    p = marginals(pred)
    both = pred[state_index(1, 1, 1)] + pred[state_index(1, 1, 0)]
    assert both > p["observed"] * p["linked"] + 1e-9


def test_chunked_equals_replay():
    tape = generate_tape(0, "shared")
    full = filter_prefix(tape.prefix(32), structure="shared", hazard=DEFAULT_H)
    mid = filter_prefix(tape.prefix(20), structure="shared", hazard=DEFAULT_H)
    matrix = transition_matrix("shared", DEFAULT_H)
    for role, y in tape.prefix(32)[20:]:
        mid = step_filter(mid, matrix, role, y)
    assert full == pytest.approx(mid)


def _step_shared(p, o, l, u, role, y, h, low=0.2, high=0.8):
    out = []
    for e in (0, 1):
        pe = h if e else (1 - h)
        o_opts = (0, 1) if e else (o,)
        l_opts = (0, 1) if e else (l,)
        pr_e = pe * (0.25 if e else 1.0)
        for o1 in o_opts:
            for l1 in l_opts:
                bit = {"observed": o1, "linked": l1, "unrelated": u}[role]
                rate = high if bit else low
                out.append((p * pr_e * (rate if y else 1 - rate), o1, l1, u))
    return out


def _step_indep(p, o, l, u, role, y, h, low=0.2, high=0.8):
    out = []
    for eo, el in product((0, 1), (0, 1)):
        o_opts = (0, 1) if eo else (o,)
        l_opts = (0, 1) if el else (l,)
        pr = p
        pr *= (h if eo else 1 - h) * (0.5 if eo else 1.0)
        pr *= (h if el else 1 - h) * (0.5 if el else 1.0)
        for o1 in o_opts:
            for l1 in l_opts:
                bit = {"observed": o1, "linked": l1, "unrelated": u}[role]
                rate = high if bit else low
                out.append((pr * (rate if y else 1 - rate), o1, l1, u))
    return out


def _enumerate_two_step(structure: str, hazard: float, obs: list[tuple[str, int]], q=0.99):
    total = [0.0] * 8
    step = _step_shared if structure == "shared" else _step_indep
    for o0, l0, u0 in product((0, 1), (0, 1), (0, 1)):
        p0 = (q if o0 else 1 - q) * (q if l0 else 1 - q) * (q if u0 else 1 - q)
        for p1, o1, l1, u1 in step(p0, o0, l0, u0, obs[0][0], obs[0][1], hazard):
            for p2, o2, l2, u2 in step(p1, o1, l1, u1, obs[1][0], obs[1][1], hazard):
                total[state_index(o2, l2, u2)] += p2
    z = math.fsum(total)
    return [x / z for x in total]


def test_two_step_enumeration_shared():
    obs = [("observed", 0), ("linked", 1)]
    filt = filter_prefix(obs, structure="shared", hazard=0.2)
    brute = _enumerate_two_step("shared", 0.2, obs)
    assert filt == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_two_step_enumeration_independent():
    obs = [("observed", 1), ("unrelated", 0)]
    filt = filter_prefix(obs, structure="independent", hazard=0.2)
    brute = _enumerate_two_step("independent", 0.2, obs)
    assert filt == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_label_permutation_invariant_numbers():
    a = generate_tape(0, "shared", labels={"observed": "A", "linked": "B", "unrelated": "C"})
    b = generate_tape(0, "shared", labels={"observed": "X", "linked": "Y", "unrelated": "Z"})
    ja = filter_prefix(a.prefix(12), structure="shared", hazard=DEFAULT_H)
    jb = filter_prefix(b.prefix(12), structure="shared", hazard=DEFAULT_H)
    assert ja == pytest.approx(jb)
    assert a.outcomes == b.outcomes


def test_development_tapes_and_jobs():
    tapes = development_tapes()
    assert len(tapes) == 8
    assert len(sequential_jobs(tapes)) == 96
    ids = [t.tape_id for t in tapes]
    assert len(set(ids)) == 8
    shared0 = next(t for t in tapes if t.tape_id == "shared_s0")
    indep0 = next(t for t in tapes if t.tape_id == "independent_s0")
    assert shared0.labels == indep0.labels
    # paired tapes share emission/init streams but not necessarily outcomes
    assert shared0.schedule == indep0.schedule
    stables = [t for t in tapes if t.structure == "stable"]
    assert all(t.hazard == 0.0 for t in stables)


def test_public_prefix_has_no_hidden_fields():
    tape = generate_tape(0, "shared")
    for row in tape.public_prefix(44):
        assert "reset" not in row
        assert "latent" not in row
        assert set(row) <= {"step", "component", "role", "outcome"}


def test_each_action_appears_in_reference_grid():
    seen = set()
    for tape in development_tapes():
        from llm_rl_engine.analysis.selective_revision_sequential import reference_at

        for n in PROBE_AFTER:
            seen.add(reference_at(tape, n)["decision"]["optimal_action"])
    assert seen  # at least one action type; fixtures below cover all three


def test_decision_fixture_actions():
    from llm_rl_engine.analysis.selective_revision import decision_values

    assert decision_values(0.2, cost=0.04)["optimal_action"] == "choose_safe"
    assert decision_values(0.99, cost=0.04)["optimal_action"] == "choose_B"
    assert decision_values(0.58, cost=0.04)["optimal_action"] == "diagnose_B"


def test_prompts_omit_future_and_hidden():
    from llm_rl_engine.analysis.selective_revision_sequential import prompt_for_job, sequential_jobs

    job = next(j for j in sequential_jobs(development_tapes()[:1]) if j["n"] == 12)
    prompt = prompt_for_job(job, None)
    assert "t=13:" not in prompt
    assert "private" not in prompt
    assert "reset_observed" not in prompt


def test_fake_sequential_resume_no_duplicate_calls():
    from llm_rl_engine.analysis.selective_revision_sequential import (
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
        serialize_state,
        previous_checkpoint,
        parse_sequential_response,
        initial_state_payload,
    )

    tapes = development_tapes()[:1]
    jobs = sequential_jobs(tapes)

    class Fake:
        def __init__(self):
            self.calls = 0

        def generate(self, job) -> str:
            self.calls += 1
            ref = job["reference"]
            return json.dumps(
                {
                    "joint": ref["joint"],
                    "probability_high": ref["probability_high"],
                    "predictive_mean": ref["predictive_mean"],
                    "action": map_action_from_v1(ref["decision"]["optimal_action"]),
                }
            )

    fake = Fake()
    done = set()
    prev_self_live = {}

    def run(limit=None):
        n = 0
        for job in jobs:
            key = (job["tape_id"], job["n"], job["condition"])
            if key in done:
                continue
            prompt_for_job(job, prev_self_live.get(job["tape_id"]))
            text = fake.generate(job)
            parsed = parse_sequential_response(text)
            done.add(key)
            if job["condition"] == "self_state":
                prev_self_live[job["tape_id"]] = serialize_state(
                    parsed["joint"], f"after_obs_{job['n']}"
                )
            n += 1
            if limit is not None and n >= limit:
                break

    run(limit=4)
    assert fake.calls == 4
    run()
    assert fake.calls == len(jobs)
    run()
    assert fake.calls == len(jobs)


def test_sequential_live_runner_fake_resume(tmp_path):
    from llm_rl_engine.analysis.selective_revision_harness import LiveSettings, ReferenceAnswerClient
    from llm_rl_engine.analysis.selective_revision_sequential import (
        initial_state_payload,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential

    tapes = development_tapes()[:1]
    jobs = sequential_jobs(tapes)[:3]
    answers = {}
    for job in jobs:
        prev = initial_state_payload() if job["condition"] != "reconstruct" else None
        prompt = prompt_for_job(job, prev)
        ref = job["reference"]
        answers[prompt] = {
            "joint": ref["joint"],
            "probability_high": ref["probability_high"],
            "predictive_mean": ref["predictive_mean"],
            "action": map_action_from_v1(ref["decision"]["optimal_action"]),
        }
    client = ReferenceAnswerClient(answers)
    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=6,
        max_usd=1.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    first = run_sequential(
        tmp_path, settings, client, repo_root=REPO, smoke_limit=3, tapes=tapes
    )
    assert first["summary"]["n_attempted"] == 3
    client2 = ReferenceAnswerClient(answers)
    second = run_sequential(
        tmp_path, settings, client2, repo_root=REPO, smoke_limit=3, tapes=tapes
    )
    assert client2.calls == 0
    assert second["ledger"]["spent_usd"] == first["ledger"]["spent_usd"]
