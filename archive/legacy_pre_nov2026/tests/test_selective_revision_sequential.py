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
            from llm_rl_engine.analysis.selective_revision_sequential import joint_to_keyed

            return json.dumps(
                {
                    "joint": joint_to_keyed(ref["joint"]),
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
    from llm_rl_engine.analysis.selective_revision_sequential import joint_to_keyed

    answers = {}
    for job in jobs:
        prev = initial_state_payload() if job["condition"] != "reconstruct" else None
        prompt = prompt_for_job(job, prev)
        ref = job["reference"]
        answers[prompt] = {
            "joint": joint_to_keyed(ref["joint"]),
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


def test_v1_audit_matches_commander_counts():
    from llm_rl_engine.analysis.selective_revision_sequential import audit_v1_final_outputs

    path = REPO / "runs/selective_revision_sequential_v1"
    if not path.exists():
        pytest.skip("v1 live run not present")
    audit = audit_v1_final_outputs(path)
    assert audit["n_sum_violations"] == 15
    assert audit["n_marginal_disagreements"] == 13


def test_shared_and_independent_prompts_share_redraw_law():
    from llm_rl_engine.analysis.selective_revision_sequential import (
        SHARED_GENERATIVE,
        prompt_for_job,
        sequential_jobs,
    )

    jobs = sequential_jobs()
    shared = next(j for j in jobs if j["tape"].structure == "shared" and j["n"] == 12 and j["condition"] == "reconstruct")
    indep = next(j for j in jobs if j["tape"].structure == "independent" and j["n"] == 12 and j["condition"] == "reconstruct")
    ps, pi = prompt_for_job(shared, None), prompt_for_job(indep, None)
    assert SHARED_GENERATIVE in ps
    assert SHARED_GENERATIVE in pi
    assert "high with probability 0.5" in ps
    assert "high with probability 0.5" in pi
    assert "share one Bernoulli" in ps
    assert "independent Bernoulli" in pi
    assert "least-significant-bit" not in ps
    assert "index = o" not in ps


def test_keyed_joint_rejects_bad_sum():
    from llm_rl_engine.analysis.selective_revision_pilot import ValidationError
    from llm_rl_engine.analysis.selective_revision_sequential import parse_sequential_response

    bad = {"joint": {k: 0.0 for k in ["000", "001", "010", "011", "100", "101", "110", "111"]}, "action": "choose_safe"}
    bad["joint"]["000"] = 0.5
    with pytest.raises(ValidationError, match="sum"):
        parse_sequential_response(json.dumps(bad))
    good = {"joint": {k: 0.0 for k in ["000", "001", "010", "011", "100", "101", "110", "111"]}, "action": "choose_safe"}
    good["joint"]["111"] = 1.0
    parsed = parse_sequential_response(json.dumps(good))
    assert parsed["action"] == "choose_safe"


def test_changed_tape_same_id_is_rejected(tmp_path):
    from dataclasses import replace
    from llm_rl_engine.analysis.selective_revision_harness import IdentityError, LiveSettings, ReferenceAnswerClient
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential

    tapes = development_tapes()[:1]
    jobs = sequential_jobs(tapes)[:3]
    answers = {}
    for job in jobs:
        prompt = prompt_for_job(job, None)
        answers[prompt] = {
            "joint": joint_to_keyed(job["reference"]["joint"]),
            "action": map_action_from_v1(job["reference"]["decision"]["optimal_action"]),
        }
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
    run_sequential(tmp_path, settings, ReferenceAnswerClient(answers), repo_root=REPO, smoke_limit=3, tapes=tapes)
    flipped = list(tapes[0].outcomes)
    flipped[0] = 1 - flipped[0]
    mutated = [replace(tapes[0], outcomes=tuple(flipped))]
    with pytest.raises(IdentityError):
        run_sequential(
            tmp_path, settings, ReferenceAnswerClient(answers), repo_root=REPO, smoke_limit=3, tapes=mutated
        )


def test_schema_retry_then_valid(tmp_path):
    from llm_rl_engine.agents.llm import GenerationResult
    from llm_rl_engine.analysis.selective_revision_harness import LiveSettings
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential

    tapes = development_tapes()[:1]
    jobs = sequential_jobs(tapes)[:1]
    good = {
        "joint": joint_to_keyed(jobs[0]["reference"]["joint"]),
        "action": map_action_from_v1(jobs[0]["reference"]["decision"]["optimal_action"]),
    }

    class Flaky:
        def __init__(self):
            self.calls = 0

        def count_tokens(self, text: str) -> int:
            return 100

        def generate(self, prompt: str) -> GenerationResult:
            self.calls += 1
            if self.calls == 1:
                return GenerationResult(text="not json", input_tokens=10, output_tokens=5, finish_reason="STOP")
            return GenerationResult(
                text=json.dumps(good), input_tokens=10, output_tokens=20, finish_reason="STOP"
            )

    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=4,
        max_usd=1.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    result = run_sequential(tmp_path, settings, Flaky(), repo_root=REPO, smoke_limit=1, tapes=tapes)
    assert result["summary"]["n_valid"] == 1
    row = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert row["attempts_used"] == 2


def test_sequential_saved_response_recovered_without_regeneration(tmp_path):
    from llm_rl_engine.agents.llm import GenerationResult
    from llm_rl_engine.analysis.selective_revision_harness import LiveSettings, persist_provider_response
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
        sha256_text,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential, sequential_identity, _key_for

    tapes = development_tapes()[:1]
    job = sequential_jobs(tapes)[0]
    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=4,
        max_usd=1.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    identity = sequential_identity(settings, REPO, tapes)
    prompt = prompt_for_job(job, None)
    key = _key_for(job, identity, {}, {})
    payload = {
        "joint": joint_to_keyed(job["reference"]["joint"]),
        "action": map_action_from_v1(job["reference"]["decision"]["optimal_action"]),
    }
    persist_provider_response(
        tmp_path,
        case_id=job["tape_id"],
        treatment=f"n{job['n']}_{job['condition']}",
        attempt_n=1,
        attempt_id="pre",
        result=GenerationResult(
            text=json.dumps(payload),
            input_tokens=10,
            output_tokens=20,
            finish_reason="STOP",
            usage_status="available",
        ),
        latency_s=0.1,
        request_text=prompt,
        request_hash=sha256_text(prompt),
        generation_settings={"model": settings.model},
        job_key_value=key,
        scored=False,
    )

    class Boom:
        def count_tokens(self, text: str) -> int:
            return 100

        def generate(self, prompt: str):
            raise AssertionError("should not generate")

    result = run_sequential(tmp_path, settings, Boom(), repo_root=REPO, smoke_limit=1, tapes=tapes)
    assert result["summary"]["n_valid"] == 1
    row = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert row.get("recovered_from_saved_response") is True


def test_own_forecast_uses_model_utilities():
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        parse_sequential_response,
        prior_joint,
        score_sequential,
        serialize_state,
    )

    ref_joint = prior_joint()
    reference = {
        "joint": ref_joint,
        "probability_high": {"observed": 0.99, "linked": 0.99, "unrelated": 0.99},
        "decision": {
            "optimal_action": "choose_B",
            "action_values": {"choose_safe": 0.65, "choose_B": 0.794, "diagnose_B": 0.7585},
            "net_diagnostic_advantage": -0.0355,
        },
    }
    # Model believes linked is almost surely low but still chooses linked.
    keyed = {k: 0.0 for k in ["000", "001", "010", "011", "100", "101", "110", "111"]}
    keyed["000"] = 1.0
    parsed = parse_sequential_response(json.dumps({"joint": keyed, "action": "choose_linked"}))
    scored = score_sequential(parsed, reference, valid=True)
    assert scored["decision_error"] is False
    assert scored["own_forecast_decision_error"] is True
    assert scored["own_forecast_action"] == "choose_safe"
    assert scored["optimal_action"] == "choose_linked"


def test_boolean_joint_mass_rejected():
    from llm_rl_engine.analysis.selective_revision_pilot import ValidationError
    from llm_rl_engine.analysis.selective_revision_sequential import parse_sequential_response

    payload = {"joint": {k: 0.0 for k in ["000", "001", "010", "011", "100", "101", "110", "111"]}, "action": "choose_safe"}
    payload["joint"]["111"] = True
    with pytest.raises(ValidationError):
        parse_sequential_response(json.dumps(payload))
    payload["joint"]["111"] = "1"
    with pytest.raises(ValidationError):
        parse_sequential_response(json.dumps(payload))


def test_handoff_continuation_matches_reconstruct():
    from llm_rl_engine.analysis.selective_revision_sequential import (
        HANDOFF_N,
        HANDOFF_PREV,
        continue_filter,
        filter_prefix,
        handoff_tapes,
        reference_at,
    )

    for tape in handoff_tapes():
        full = filter_prefix(tape.prefix(HANDOFF_N), structure=tape.structure, hazard=tape.hazard)
        prev = reference_at(tape, HANDOFF_PREV)["joint"]
        cont = continue_filter(prev, tape, HANDOFF_PREV, HANDOFF_N)
        assert cont == pytest.approx(full, abs=1e-12)


def test_handoff_jobs_are_48_with_identical_repeats():
    from llm_rl_engine.analysis.selective_revision_sequential import handoff_jobs

    jobs = handoff_jobs()
    assert len(jobs) == 48
    by = {}
    for job in jobs:
        key = (job["tape_id"], job["interface"])
        by.setdefault(key, []).append(job["prompt"])
    assert len(by) == 24
    for prompts in by.values():
        assert len(prompts) == 2
        assert prompts[0] == prompts[1]


def test_v2_own_forecast_audit_matches_commander():
    from llm_rl_engine.analysis.selective_revision_sequential import audit_v2_own_forecast

    path = REPO / "runs/selective_revision_sequential_v2"
    if not path.exists():
        pytest.skip("v2 live run not present")
    audit = audit_v2_own_forecast(path)
    assert audit["by_condition"]["reconstruct"] == 2
    assert audit["by_condition"]["self_state"] == 10
    assert audit["by_condition"]["reference_state"] == 7
    assert audit["own_forecast_errors"] == 19
    assert audit["both_own_and_reference"] == 18


def test_invalid_first_response_still_gets_retry(tmp_path):
    from llm_rl_engine.agents.llm import GenerationResult
    from llm_rl_engine.analysis.selective_revision_harness import LiveSettings, persist_provider_response
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
        sha256_text,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import (
        _key_for,
        run_sequential,
        sequential_identity,
    )

    tapes = development_tapes()[:1]
    job = sequential_jobs(tapes)[0]
    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=4,
        max_usd=1.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    identity = sequential_identity(settings, REPO, tapes)
    prompt = prompt_for_job(job, None)
    key = _key_for(job, identity, {}, {})
    persist_provider_response(
        tmp_path,
        case_id=job["tape_id"],
        treatment=f"n{job['n']}_{job['condition']}",
        attempt_n=1,
        attempt_id="pre",
        result=GenerationResult(text="not json", input_tokens=10, output_tokens=5, finish_reason="STOP"),
        latency_s=0.1,
        request_text=prompt,
        request_hash=sha256_text(prompt),
        generation_settings={"model": settings.model},
        job_key_value=key,
        scored=False,
    )
    from llm_rl_engine.analysis.selective_revision_harness import atomic_write_json, append_jsonl

    atomic_write_json(
        tmp_path / "ledger.json",
        {"spent_usd": 0.01, "reserved_usd": 0.0, "attempts": 1, "completed_jobs": 0, "interrupted_reservations": 0},
    )
    append_jsonl(tmp_path / "attempts.jsonl", {"key": key, "status": "reserved"})
    good = {
        "joint": joint_to_keyed(job["reference"]["joint"]),
        "action": map_action_from_v1(job["reference"]["decision"]["optimal_action"]),
    }

    class Once:
        def __init__(self):
            self.calls = 0

        def count_tokens(self, text: str) -> int:
            return 100

        def generate(self, prompt: str) -> GenerationResult:
            self.calls += 1
            return GenerationResult(text=json.dumps(good), input_tokens=10, output_tokens=20, finish_reason="STOP")

    client = Once()
    result = run_sequential(tmp_path, settings, client, repo_root=REPO, smoke_limit=1, tapes=tapes)
    assert client.calls == 1
    assert result["summary"]["n_valid"] == 1


def test_valid_repair_saved_response_does_not_hash_against_base(tmp_path):
    from llm_rl_engine.agents.llm import GenerationResult
    from llm_rl_engine.analysis.selective_revision_harness import LiveSettings, persist_provider_response
    from llm_rl_engine.analysis.selective_revision_sequential import (
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
        sha256_text,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import (
        REPAIR_SUFFIX,
        _key_for,
        run_sequential,
        sequential_identity,
    )

    tapes = development_tapes()[:1]
    job = sequential_jobs(tapes)[0]
    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=4,
        max_usd=1.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    identity = sequential_identity(settings, REPO, tapes)
    prompt = prompt_for_job(job, None)
    key = _key_for(job, identity, {}, {})
    repair = prompt + REPAIR_SUFFIX.format(error="invalid JSON")
    good = {
        "joint": joint_to_keyed(job["reference"]["joint"]),
        "action": map_action_from_v1(job["reference"]["decision"]["optimal_action"]),
    }
    persist_provider_response(
        tmp_path,
        case_id=job["tape_id"],
        treatment=f"n{job['n']}_{job['condition']}",
        attempt_n=2,
        attempt_id="repair",
        result=GenerationResult(text=json.dumps(good), input_tokens=10, output_tokens=20, finish_reason="STOP"),
        latency_s=0.1,
        request_text=repair,
        request_hash=sha256_text(repair),
        generation_settings={"model": settings.model},
        job_key_value=key,
        scored=False,
    )

    class Boom:
        def count_tokens(self, text: str) -> int:
            return 100

        def generate(self, prompt: str):
            raise AssertionError("should recover repair without generate")

    result = run_sequential(tmp_path, settings, Boom(), repo_root=REPO, smoke_limit=1, tapes=tapes)
    assert result["summary"]["n_valid"] == 1


def test_changed_previous_state_rejects_later_checkpoint(tmp_path):
    from llm_rl_engine.analysis.selective_revision_harness import IdentityError, LiveSettings, ReferenceAnswerClient
    from llm_rl_engine.analysis.selective_revision_sequential import (
        initial_state_payload,
        joint_to_keyed,
        map_action_from_v1,
        prompt_for_job,
        sequential_jobs,
        serialize_state,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential

    tapes = development_tapes()[:1]
    jobs = sequential_jobs(tapes)[:5]
    answers = {}
    n12_self = next(j for j in jobs if j["condition"] == "self_state" and j["n"] == 12)
    for job in jobs:
        prev = None
        if job["condition"] == "self_state" and job["n"] == 12:
            prev = initial_state_payload()
        if job["condition"] == "self_state" and job["n"] == 20:
            prev = serialize_state(n12_self["reference"]["joint"], "after_obs_12")
        prompt = prompt_for_job(job, prev)
        answers[prompt] = {
            "joint": joint_to_keyed(job["reference"]["joint"]),
            "action": map_action_from_v1(job["reference"]["decision"]["optimal_action"]),
        }
    settings = LiveSettings(
        model="gemini-3.5-flash",
        max_calls=12,
        max_usd=2.0,
        max_input_tokens=4000,
        max_output_tokens=2048,
        thinking_budget=0,
        max_retries=1,
        temperature=0.0,
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    )
    run_sequential(
        tmp_path, settings, ReferenceAnswerClient(answers), repo_root=REPO, tapes=tapes, jobs=jobs
    )
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    for row in results:
        if row["condition"] == "self_state" and row["n"] == 12:
            row["parsed_joint"] = [1.0] + [0.0] * 7
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    with pytest.raises(IdentityError):
        run_sequential(
            tmp_path, settings, ReferenceAnswerClient(answers), repo_root=REPO, tapes=tapes, jobs=jobs
        )
