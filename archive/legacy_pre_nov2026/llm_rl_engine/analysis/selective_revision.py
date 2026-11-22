"""Exact, observation-limited reference for a single possible reset before a batch.

A/B share reset events or reset independently. Redrawn values are independent,
even for a shared event. C cannot reset. Fixed component labels are A, B, C.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from numbers import Real
from typing import Mapping, Sequence


COMPONENTS = ("A", "B", "C")
STRUCTURES = ("shared", "independent", "stable")


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _probability(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class ReferenceModel:
    structure: str = "shared"
    prior_high: float = 0.99
    hazard: float = 0.1
    low: float = 0.2
    high: float = 0.8

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}")
        for name in ("prior_high", "hazard", "low", "high"):
            _probability(name, getattr(self, name))
        if self.low >= self.high:
            raise ValueError("low must be less than high")


def posterior(
    model: ReferenceModel, observations: Mapping[str, Sequence[int]]
) -> dict:
    """Enumerate event/state pairs, then condition on permitted observations only.

    The prior summarizes earlier evidence. All supplied observations occur AFTER
    the sole possible reset. A stable model has effective hazard zero regardless
    of the configured hazard. Log weights avoid underflow on long batches.
    """
    if not isinstance(observations, Mapping) or set(observations) - set(COMPONENTS):
        raise ValueError("observations must map A/B/C to binary sequences")
    counts = {}
    for component in COMPONENTS:
        values = observations.get(component, ())
        if not isinstance(values, (list, tuple)) or any(
            type(value) is not int or value not in (0, 1) for value in values
        ):
            raise ValueError("observations must contain lists/tuples of integer 0 or 1")
        counts[component] = (sum(values), len(values) - sum(values))

    h = 0.0 if model.structure == "stable" else model.hazard
    if model.structure == "independent":
        events = [
            ((a, b), (h if a else 1 - h) * (h if b else 1 - h))
            for a, b in product((False, True), repeat=2)
        ]
    else:
        events = [((False, False), 1 - h), ((True, True), h)]

    weighted = []
    for (reset_a, reset_b), event_mass in events:
        if event_mass == 0:
            continue
        priors = (0.5 if reset_a else model.prior_high,
                  0.5 if reset_b else model.prior_high, model.prior_high)
        for bits in product((0, 1), repeat=3):
            log_weight = math.log(event_mass)
            possible = True
            for component, bit, prior in zip(COMPONENTS, bits, priors):
                mass = prior if bit else 1 - prior
                rate = model.high if bit else model.low
                successes, failures = counts[component]
                if mass == 0 or (successes and rate == 0) or (failures and rate == 1):
                    possible = False
                    break
                log_weight += math.log(mass)
                if successes:
                    log_weight += successes * math.log(rate)
                if failures:
                    log_weight += failures * math.log1p(-rate)
            if possible:
                weighted.append((bits, reset_a, reset_b, log_weight))
    if not weighted:
        raise ValueError("observations have zero likelihood under the stated model")

    peak = max(row[3] for row in weighted)
    denominator = math.fsum(math.exp(row[3] - peak) for row in weighted)
    joint = [
        {"high": dict(zip(COMPONENTS, bits)), "reset_A": a, "reset_B": b,
         "probability": math.exp(weight - peak) / denominator}
        for bits, a, b, weight in weighted
    ]
    high_prob = {
        component: math.fsum(row["probability"] for row in joint if row["high"][component])
        for component in COMPONENTS
    }
    return {
        "posterior_high": high_prob,
        "predictive_mean": {
            component: model.low + (model.high - model.low) * probability
            for component, probability in high_prob.items()
        },
        "posterior_events": {
            "reset_A": math.fsum(row["probability"] for row in joint if row["reset_A"]),
            "reset_B": math.fsum(row["probability"] for row in joint if row["reset_B"]),
            "reset_C": 0.0,
        },
        "log_evidence": peak + math.log(denominator),
        "joint": joint,
    }


def decision_values(
    probability_high: float, *, low: float = 0.2, high: float = 0.8,
    safe: float = 0.65, cost: float = 0.04,
) -> dict:
    """One B decision, optionally buying perfect information about B's RATE.

    Diagnostic cost is paid once. Ties within 1e-12 prefer safe, then B, then
    diagnosis. Post-diagnostic choice maximizes expected reward in each branch.
    """
    p = _probability("probability_high", probability_high)
    low, high = _probability("low", low), _probability("high", high)
    if low >= high:
        raise ValueError("low must be less than high")
    safe, cost = _finite("safe", safe), _finite("cost", cost)
    if cost < 0:
        raise ValueError("cost must be nonnegative")
    direct = p * high + (1 - p) * low
    informed = p * max(high, safe) + (1 - p) * max(low, safe)
    values = {"choose_safe": safe, "choose_B": direct, "diagnose_B": informed - cost}
    best = max(values.values())
    action = next(name for name, value in values.items() if best - value <= 1e-12)
    return {
        "action_values": values, "optimal_action": action,
        "gross_value_of_information": max(0.0, informed - max(direct, safe)),
        "net_diagnostic_advantage": informed - cost - max(direct, safe),
    }
