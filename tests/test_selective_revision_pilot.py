from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rl_engine.analysis.selective_revision import decision_values, posterior
from llm_rl_engine.analysis.selective_revision_harness import (
    BudgetError,
    LiveConfigError,
    PilotRun,
    ReferenceAnswerClient,
    live_settings_from_config,
    load_jsonl,
    load_pilot_config,
    preflight,
    reserved_cost_usd,
    run_jobs,
    snapshot_code,
)
from llm_rl_engine.analysis.selective_revision_pilot import (
    COMPONENTS,
    PilotSpec,
    ValidationError,
    build_jobs,
    build_public_cases,
    evaluator_record,
    parse_response,
    prompt_leaks_evaluator,
    render_prompt,
    score_response,
    sha256_text,
    shared_prompt_prefix,
    summarize_scores,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "llm_rl_engine" / "configs" / "pilot_selective_revision_v1.yaml"


def _perfect_payload(evaluator) -> dict:
    return {
        "probability_high": dict(evaluator.posterior_high),
        "predictive_mean": dict(evaluator.predictive_mean),
        "action": evaluator.decision["optimal_action"],
    }


def _client_for_jobs(jobs, mutate=None) -> ReferenceAnswerClient:
    answers = {}
    for case, treatment, evaluator in jobs:
        payload = _perfect_payload(evaluator)
        if mutate:
            payload = mutate(payload, case, treatment, evaluator)
        answers[render_prompt(case, treatment, evaluator)] = payload
    return ReferenceAnswerClient(answers)


def _settings(**overrides):
    config, _spec = load_pilot_config(CONFIG)
    values = {
        "model": "gemini-3.5-flash",
        "max_calls": 144,
        "max_usd": 4.0,
        "max_input_tokens": 2000,
        "max_output_tokens": 2048,
        "thinking_budget": 0,
        "max_retries": 1,
    }
    values.update(overrides)
    return live_settings_from_config(config, values)


def test_grid_size_and_neutral_ids():
    cases = build_public_cases()
    jobs = build_jobs()
    assert len(cases) == 24
    assert len(jobs) == 72
    assert [case.case_id for case in cases] == [f"case_{i:02d}" for i in range(1, 25)]
    assert all("event" not in case.case_id for case in cases)
    structures = {case.structure for case in cases}
    assert structures == {"shared", "independent", "stable"}
    assert all(case.hazard == 0.0 for case in cases if case.structure == "stable")
    assert all(case.hazard == 0.1 for case in cases if case.structure != "stable")


def test_raw_count_likelihood_equivalence():
    for case in build_public_cases():
        from_raw = posterior(case.model(), {"A": list(case.observations_a)})
        from_counts = posterior(
            case.model(),
            {"A": [1] * case.successes + [0] * case.failures},
        )
        assert from_raw["posterior_high"] == pytest.approx(from_counts["posterior_high"])
        raw = render_prompt(case, "raw", evaluator_record(case))
        counts = render_prompt(case, "counts", evaluator_record(case))
        prefix = shared_prompt_prefix(case)
        assert raw.startswith(prefix)
        assert counts.startswith(prefix)
        assert "Evidence on A after the possible reset, in order" in raw
        assert "Verified counts on A after the possible reset" in counts
        assert raw.split(prefix, 1)[0] == counts.split(prefix, 1)[0]


def test_supplied_predictions_determine_diagnostic_utility():
    for case in build_public_cases():
        record = evaluator_record(case)
        prompt = render_prompt(case, "reference", record)
        assert f"B={record.posterior_high['B']:.12g}" in prompt or "B=" in prompt
        recomputed = decision_values(
            record.posterior_high["B"],
            low=case.low,
            high=case.high,
            safe=case.safe,
            cost=case.diagnostic_cost,
        )
        assert recomputed["optimal_action"] == record.decision["optimal_action"]
        assert recomputed["action_values"]["diagnose_B"] == pytest.approx(
            record.decision["action_values"]["diagnose_B"]
        )


def test_ordinary_prompts_have_no_evaluator_fields():
    for case, treatment, record in build_jobs():
        prompt = render_prompt(case, treatment, record)
        assert prompt_leaks_evaluator(prompt, record) == []
        assert "optimal_action" not in prompt
        assert "reference_targets" not in prompt
        if treatment != "reference":
            assert "supplied_probability_high" not in prompt
            assert "trusted observation-limited reference" not in prompt
            mean_b = f"{record.predictive_mean['B']:.12f}"
            if mean_b not in prompt[:0] and abs(record.predictive_mean["B"] - (case.low + (case.high - case.low) * case.prior_high)) > 1e-9:
                assert mean_b not in prompt
        else:
            assert "not a recommended action" in prompt
            assert "B=" in prompt


def test_invalid_json_and_probabilities_rejected():
    with pytest.raises(ValidationError):
        parse_response("not json")
    with pytest.raises(ValidationError):
        parse_response('{"probability_high": {"A": 0.1, "B": 0.2, "C": 1.2}, "predictive_mean": {"A": 0.2, "B": 0.3, "C": 0.4}, "action": "choose_safe"}')
    with pytest.raises(ValidationError):
        parse_response('{"probability_high": {"A": true, "B": 0.2, "C": 0.3}, "predictive_mean": {"A": 0.2, "B": 0.3, "C": 0.4}, "action": "choose_safe"}')
    with pytest.raises(ValidationError):
        parse_response('{"probability_high": {"A": 0.1, "B": 0.2, "C": 0.3}, "predictive_mean": {"A": 0.2, "B": 0.3, "C": 0.4}, "action": "wait"}')
    parsed = parse_response(
        '{"probability_high": {"A": 0.1, "B": 0.2, "C": 0.3}, "predictive_mean": {"A": 0.2, "B": 0.3, "C": 0.4}, "action": "choose_safe"}'
    )
    assert parsed["action"] == "choose_safe"


def test_itt_fallback_is_safe_and_failures_are_counted():
    case = build_public_cases()[0]
    record = evaluator_record(case)
    scored = score_response(None, record, valid=False, failure_kind="parse_failure")
    assert scored["itt_fallback"] is True
    assert scored["chosen_action"] == "choose_safe"
    assert scored["decision_error"] is False
    summary = summarize_scores([scored, {**scored, "failure_kind": "api_failure"}])
    assert summary["n_attempted"] == 2
    assert summary["n_valid"] == 0
    assert summary["n_parse_failure"] == 1
    assert summary["n_api_failure"] == 1


def test_dry_run_is_deterministic_and_omits_credentials(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    first = PilotRun(tmp_path / "a", config, spec, repo_root=REPO)
    second = PilotRun(tmp_path / "b", config, spec, repo_root=REPO)
    report_a = first.write_dry_run()
    report_b = second.write_dry_run()
    assert report_a["manifest"]["n_planned_calls"] == 72
    assert report_a["manifest"]["n_planned_attempts_with_retry_reserve"] == 144
    assert report_a["manifest"]["config_hash"] == report_b["manifest"]["config_hash"]
    assert (tmp_path / "a" / "dryrun.json").read_text() == (tmp_path / "b" / "dryrun.json").read_text()
    prompts_a = sorted(p.read_text() for p in (tmp_path / "a" / "prompts").glob("*.txt"))
    prompts_b = sorted(p.read_text() for p in (tmp_path / "b" / "prompts").glob("*.txt"))
    assert prompts_a == prompts_b
    assert len(prompts_a) == 72
    evaluator = json.loads((tmp_path / "a" / "evaluator.json").read_text())
    for prompt in prompts_a:
        if "trusted observation-limited reference" not in prompt:
            for record in evaluator.values():
                assert record["decision"]["optimal_action"] not in [
                    line for line in prompt.splitlines() if "recommended" in line.lower()
                ]
            assert "optimal_action" not in prompt
            assert "reference_targets" not in prompt
    code = snapshot_code(REPO)
    assert ".env" not in code["dirty_file_hashes"] or code["dirty_file_hashes"].get(".env") == "omitted_credentials"


def test_dry_run_does_not_touch_env_or_client(tmp_path, monkeypatch):
    config, spec = load_pilot_config(CONFIG)

    def boom(*_args, **_kwargs):
        raise AssertionError("API client constructed during dry-run")

    monkeypatch.setattr(
        "llm_rl_engine.analysis.selective_revision_harness.BoundedGeminiClient", boom
    )
    PilotRun(tmp_path, config, spec, repo_root=REPO).write_dry_run()
    harness = (REPO / "llm_rl_engine/analysis/selective_revision_harness.py").read_text(
        encoding="utf-8"
    )
    prefix, _sep, _rest = harness.partition("class BoundedGeminiClient")
    assert "load_dotenv" not in prefix
    assert "GEMINI_API_KEY" not in prefix


def test_fake_client_end_to_end_matches_reference(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)
    client = _client_for_jobs(jobs)
    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    result = run_jobs(run, _settings(), client, resume=True)
    assert result["summary"]["n_attempted"] == 72
    assert result["summary"]["n_valid"] == 72
    assert result["summary"]["n_decision_error"] == 0
    assert result["summary"]["n_api_failure"] == 0
    assert result["summary"]["n_parse_failure"] == 0
    assert result["summary"]["itt_mean_decision_loss"] == pytest.approx(0.0)
    assert client.calls == 72
    for prompt in client.prompts:
        assert "optimal_action" not in prompt
        assert "reference_targets" not in prompt


def test_api_failure_remains_visible(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:1]

    class FailClient:
        def generate(self, prompt: str):
            raise RuntimeError("quota exceeded")

    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = jobs
    settings = _settings(max_calls=4, max_usd=1.0)
    result = run_jobs(run, settings, FailClient(), resume=True)
    row = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert row["failure_kind"] == "api_failure"
    assert row["valid"] is False
    assert row["stop_run"] is True
    attempts = load_jsonl(tmp_path / "attempts.jsonl")
    assert any(row.get("exception_category") == "quota" for row in attempts)
    assert result["summary"]["n_api_failure"] == 1


def test_each_response_is_saved_immediately(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:1]
    client = _client_for_jobs(jobs)
    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = jobs
    run_jobs(run, _settings(max_calls=4, max_usd=1.0), client)
    saved = list((tmp_path / "responses").glob("*.json"))
    assert saved, "provider response must be written before the run ends"
    payload = json.loads(saved[0].read_text())
    assert "raw_response" in payload
    assert payload["case_id"] == jobs[0][0].case_id
    last = json.loads((tmp_path / "last_response.json").read_text())
    assert last["raw_response"] == payload["raw_response"]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["n_results"] == 1
    assert (tmp_path / "summary.json").exists()


def test_retry_is_counted_once(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    case, treatment, evaluator = build_jobs(spec)[0]
    prompt = render_prompt(case, treatment, evaluator)
    payloads = ["not json", json.dumps(_perfect_payload(evaluator))]

    class Flaky:
        def __init__(self):
            self.calls = 0

        def generate(self, text: str):
            self.calls += 1
            body = payloads.pop(0)
            from llm_rl_engine.agents.llm import GenerationResult

            return GenerationResult(text=body, input_tokens=10, output_tokens=20, total_tokens=30)

    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = [(case, treatment, evaluator)]
    client = Flaky()
    result = run_jobs(run, _settings(max_calls=4, max_usd=1.0), client)
    assert client.calls == 2
    row = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert row["valid"] is True
    assert row["attempts_used"] == 2
    reserved = [row for row in load_jsonl(tmp_path / "attempts.jsonl") if row["status"] == "reserved"]
    assert len(reserved) == 2
    assert "not a recommended action is choose" not in prompt
    repair = load_jsonl(tmp_path / "attempts.jsonl")
    assert result["ledger"]["attempts"] == 2


def test_strict_precall_cap_rejects_underfunded_configs():
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)
    prompts = [render_prompt(case, treatment, rec) for case, treatment, rec in jobs]
    settings = _settings(max_calls=10, max_usd=4.0)
    with pytest.raises(BudgetError, match="max_calls"):
        preflight(72, settings, prompts)
    settings = _settings(max_calls=144, max_usd=0.01)
    with pytest.raises(BudgetError, match="max_usd"):
        preflight(72, settings, prompts)
    with pytest.raises(LiveConfigError):
        live_settings_from_config(config, {"model": "gemini-3.5-flash"})


def test_precall_cap_during_run(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:2]
    client = _client_for_jobs(jobs)
    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = jobs
    # One attempt costs the full reserved amount; max_usd covers exactly one attempt.
    settings = _settings(max_calls=10, max_usd=reserved_cost_usd(
        max_input_tokens=2000, max_output_tokens=2048, model="gemini-3.5-flash",
        pricing={"input_per_1m": 1.50, "output_per_1m": 9.00},
    ))
    with pytest.raises(BudgetError, match="pre-call cap|does not cover"):
        run_jobs(run, settings, client)


def test_interrupted_reservation_is_not_assumed_free(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:1]
    case, treatment, evaluator = jobs[0]
    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = jobs
    settings = _settings(max_calls=6, max_usd=1.0)
    reserved = settings.reserved_per_attempt
    from llm_rl_engine.analysis.selective_revision_harness import atomic_write_json, append_jsonl, job_key

    bundle_prompt = render_prompt(case, treatment, evaluator)
    from llm_rl_engine.analysis.selective_revision_pilot import prompt_bundle

    bundle = prompt_bundle(case, treatment, evaluator)
    key = job_key(case.case_id, treatment, settings.model, bundle["prompt_hash"], run.config_hash)
    atomic_write_json(
        tmp_path / "ledger.json",
        {
            "spent_usd": 0.0,
            "reserved_usd": reserved,
            "attempts": 1,
            "completed_jobs": 0,
            "interrupted_reservations": 0,
        },
    )
    append_jsonl(
        tmp_path / "attempts.jsonl",
        {"attempt_id": "old", "key": key, "status": "reserved", "reserved_usd": reserved},
    )
    atomic_write_json(
        tmp_path / "live_manifest.json",
        {"config_hash": run.config_hash, "model": settings.model},
    )
    client = _client_for_jobs(jobs)
    result = run_jobs(run, settings, client, resume=True)
    assert result["ledger"]["spent_usd"] >= reserved
    assert result["ledger"]["interrupted_reservations"] >= 1
    assert result["ledger"]["reserved_usd"] == 0.0
    assert client.calls == 1
    assert result["summary"]["n_valid"] == 1


def test_resume_is_idempotent_and_keeps_spend(tmp_path):
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:3]
    client = _client_for_jobs(jobs)
    run = PilotRun(tmp_path, config, spec, repo_root=REPO)
    run.jobs = jobs
    settings = _settings(max_calls=12, max_usd=1.0)
    first = run_jobs(run, settings, client, resume=True)
    spent = first["ledger"]["spent_usd"]
    assert client.calls == 3
    second_client = _client_for_jobs(jobs)
    second = run_jobs(run, settings, second_client, resume=True)
    assert second_client.calls == 0
    assert second["ledger"]["spent_usd"] == spent
    assert second["ledger"]["attempts"] == first["ledger"]["attempts"]
    assert len(load_jsonl(tmp_path / "results.jsonl")) == 3


def test_thinking_budget_is_reserved_as_billed_output():
    config, spec = load_pilot_config(CONFIG)
    jobs = build_jobs(spec)[:1]
    prompts = [render_prompt(*jobs[0])]
    off = _settings(thinking_budget=0, max_calls=4, max_usd=1.0)
    on = _settings(thinking_budget=128, max_calls=4, max_usd=1.0)
    assert on.billed_output_limit == on.max_output_tokens + 128
    assert on.reserved_per_attempt > off.reserved_per_attempt
    preflight(1, on, prompts)
    with pytest.raises(LiveConfigError):
        _settings(thinking_budget=-1)


def test_reference_treatment_is_enough_to_score_voi():
    case = next(c for c in build_public_cases() if c.structure == "shared" and c.successes == 0 and c.diagnostic_cost == 0.04)
    record = evaluator_record(case)
    assert record.decision["optimal_action"] == "diagnose_B"
    parsed = parse_response(json.dumps(_perfect_payload(record)))
    scored = score_response(parsed, record, valid=True, failure_kind=None)
    assert scored["decision_loss"] == pytest.approx(0.0)
    assert scored["c_drift"] == pytest.approx(0.0)
    for component in COMPONENTS:
        assert scored["probability_error"][component] == pytest.approx(0.0)
