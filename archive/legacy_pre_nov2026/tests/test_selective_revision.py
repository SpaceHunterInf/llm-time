from __future__ import annotations

import math
import pytest

from llm_rl_engine.analysis.selective_revision import ReferenceModel, decision_values, posterior


@pytest.mark.parametrize("successes", range(6))
def test_exact_marginal_and_selective_propagation(successes):
    observations = {"A": [1] * successes + [0] * (5 - successes)}
    shared = posterior(ReferenceModel(), observations)
    independent = posterior(ReferenceModel(structure="independent"), observations)
    p = .9 * .99 + .1 * .5
    high_likelihood = .8 ** successes * .2 ** (5 - successes)
    low_likelihood = .2 ** successes * .8 ** (5 - successes)
    expected = p * high_likelihood / (p * high_likelihood + (1 - p) * low_likelihood)
    for result in (shared, independent):
        assert sum(row["probability"] for row in result["joint"]) == pytest.approx(1)
        assert all(0 <= row["probability"] <= 1 for row in result["joint"])
        assert result["posterior_high"]["A"] == pytest.approx(expected)
        assert result["predictive_mean"]["C"] == pytest.approx(.794)
    assert independent["posterior_high"]["B"] == pytest.approx(p)
    if successes == 0:
        assert shared["predictive_mean"]["B"] == pytest.approx(.5484289975063971)
        assert shared["posterior_events"]["reset_A"] == pytest.approx(.8352755186857245)


@pytest.mark.parametrize("hazard", [0, 1])
def test_hazard_endpoints(hazard):
    obs = {"A": [0, 1, 0], "B": [1], "C": [0]}
    s = posterior(ReferenceModel(hazard=hazard), obs)
    i = posterior(ReferenceModel(structure="independent", hazard=hazard), obs)
    assert s["posterior_high"] == pytest.approx(i["posterior_high"])
    if hazard == 0:
        stable = posterior(ReferenceModel(structure="stable"), obs)
        assert s["posterior_high"] == pytest.approx(stable["posterior_high"])


def test_no_evidence_matched_marginals_and_dependent_joint():
    s = posterior(ReferenceModel(), {})
    i = posterior(ReferenceModel(structure="independent"), {})
    assert s["posterior_high"] == pytest.approx(i["posterior_high"])
    both = lambda result: sum(r["probability"] for r in result["joint"] if r["high"]["A"] and r["high"]["B"])
    assert both(s) > both(i)


def test_diagnostic_values_costs_and_ties():
    shared = posterior(ReferenceModel(), {"A": [0] * 5})
    independent = posterior(ReferenceModel(structure="independent"), {"A": [0] * 5})
    d = decision_values(shared["posterior_high"]["B"])
    assert d["optimal_action"] == "diagnose_B"
    assert d["action_values"]["diagnose_B"] == pytest.approx(.6971072493765993)
    assert decision_values(independent["posterior_high"]["B"])["optimal_action"] == "choose_B"
    assert decision_values(.5, cost=.2)["optimal_action"] == "choose_safe"
    assert decision_values(0, cost=0)["optimal_action"] == "choose_safe"
    assert decision_values(1, cost=0)["optimal_action"] == "choose_B"
    assert decision_values(0, cost=.04)["action_values"]["diagnose_B"] == pytest.approx(.61)
    assert decision_values(1, cost=.04)["action_values"]["diagnose_B"] == pytest.approx(.76)
    for p in [0, .1, .5, .9, 1]:
        assert decision_values(p)["gross_value_of_information"] >= 0


def test_long_batches_are_stable_and_order_is_irrelevant():
    result = posterior(ReferenceModel(), {"A": [0] * 10000})
    assert math.isfinite(result["log_evidence"])
    assert result["predictive_mean"]["A"] == pytest.approx(.2)
    a = posterior(ReferenceModel(), {"A": [1, 0, 1, 0]})
    b = posterior(ReferenceModel(), {"A": [0, 0, 1, 1]})
    assert a == b


@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("hazard", [0, 1])
def test_extreme_priors(q, hazard):
    result = posterior(ReferenceModel(prior_high=q, hazard=hazard), {"A": [0, 1]})
    assert sum(r["probability"] for r in result["joint"]) == pytest.approx(1)


@pytest.mark.parametrize("kwargs", [
    {"prior_high": -1}, {"hazard": 2}, {"high": float("nan")},
    {"low": float("inf")}, {"low": .8}, {"structure": "unknown"}, {"hazard": True},
])
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        ReferenceModel(**kwargs)


@pytest.mark.parametrize("obs", [{"D": [0]}, {"A": [2]}, {"A": [True]}, {"A": "01"}])
def test_invalid_observations(obs):
    with pytest.raises(ValueError):
        posterior(ReferenceModel(), obs)


def test_impossible_evidence():
    with pytest.raises(ValueError, match="zero likelihood"):
        posterior(ReferenceModel(low=0, high=1), {"A": [0, 1]})


@pytest.mark.parametrize("kwargs", [{"cost": -1}, {"cost": float("inf")}, {"safe": float("nan")}])
def test_invalid_decision(kwargs):
    with pytest.raises(ValueError):
        decision_values(.5, **kwargs)
