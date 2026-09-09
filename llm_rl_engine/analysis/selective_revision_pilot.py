"""Public cases, prompts, validation, and scoring for the selective-revision pilot.

Evaluator references are stored separately from ordinary prompts. Treatments
share one generative description and decision question; only the evidence block
changes. Diagnose commits to using a revealed rate optimally; this probe does
not test interpretation of diagnostic outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Iterable, Mapping

from llm_rl_engine.analysis.selective_revision import (
    COMPONENTS,
    STRUCTURES,
    ReferenceModel,
    decision_values,
    posterior,
)

ACTIONS = ("choose_safe", "choose_B", "diagnose_B")
TREATMENTS = ("raw", "counts", "reference")
DEFAULT_SUCCESSES = (0, 2, 4, 5)
DEFAULT_COSTS = (0.04, 0.20)
BATCH_LENGTH = 5
SCHEMA_VERSION = 1
HIDDEN_EVALUATOR_MARKERS = (
    "reference_targets",
    "optimal_action",
    "action_values",
    "gross_value_of_information",
    "net_diagnostic_advantage",
    "posterior_events",
    "log_evidence",
    "evaluator",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Conservative input-token estimate: 1 token per 2 characters plus overhead."""
    return 32 + (len(text) + 1) // 2


@dataclass(frozen=True)
class PilotSpec:
    prior_high: float = 0.99
    hazard: float = 0.1
    low: float = 0.2
    high: float = 0.8
    safe: float = 0.65
    batch_length: int = BATCH_LENGTH
    successes: tuple[int, ...] = DEFAULT_SUCCESSES
    costs: tuple[float, ...] = DEFAULT_COSTS
    structures: tuple[str, ...] = STRUCTURES
    treatments: tuple[str, ...] = TREATMENTS
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.batch_length <= 0:
            raise ValueError("batch_length must be positive")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        for structure in self.structures:
            if structure not in STRUCTURES:
                raise ValueError(f"unknown structure {structure}")
        for treatment in self.treatments:
            if treatment not in TREATMENTS:
                raise ValueError(f"unknown treatment {treatment}")
        for count in self.successes:
            if type(count) is not int or not 0 <= count <= self.batch_length:
                raise ValueError("successes must be integers in [0, batch_length]")
        for cost in self.costs:
            if isinstance(cost, bool) or not isinstance(cost, Real) or not math.isfinite(cost) or cost < 0:
                raise ValueError("costs must be finite and nonnegative")
        ReferenceModel(
            prior_high=self.prior_high, hazard=self.hazard, low=self.low, high=self.high
        )
        if not math.isfinite(self.safe):
            raise ValueError("safe must be finite")

    @property
    def prior_predictive_c(self) -> float:
        return self.low + (self.high - self.low) * self.prior_high

    def stated_hazard(self, structure: str) -> float:
        return 0.0 if structure == "stable" else self.hazard


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    structure: str
    prior_high: float
    hazard: float
    low: float
    high: float
    safe: float
    diagnostic_cost: float
    observations_a: tuple[int, ...]
    successes: int
    failures: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "structure": self.structure,
            "prior_high": self.prior_high,
            "hazard": self.hazard,
            "low": self.low,
            "high": self.high,
            "safe_utility": self.safe,
            "diagnostic_cost": self.diagnostic_cost,
            "observations": {
                "A": list(self.observations_a),
                "B": [],
                "C": [],
            },
            "observation_counts": {
                "A": {"successes": self.successes, "failures": self.failures},
                "B": {"successes": 0, "failures": 0},
                "C": {"successes": 0, "failures": 0},
            },
        }

    def observations(self) -> dict[str, list[int]]:
        return {"A": list(self.observations_a), "B": [], "C": []}

    def model(self) -> ReferenceModel:
        return ReferenceModel(
            structure=self.structure,
            prior_high=self.prior_high,
            hazard=self.hazard,
            low=self.low,
            high=self.high,
        )


@dataclass(frozen=True)
class EvaluatorRecord:
    case_id: str
    posterior_high: dict[str, float]
    predictive_mean: dict[str, float]
    posterior_events: dict[str, float]
    decision: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationError(ValueError):
    pass


def spec_from_mapping(raw: Mapping[str, Any] | None = None) -> PilotSpec:
    data = dict(raw or {})
    grid = dict(data.get("grid") or {})
    kwargs: dict[str, Any] = {}
    for key in ("prior_high", "hazard", "low", "high", "safe", "batch_length", "schema_version"):
        if key in grid:
            kwargs[key] = grid[key]
        elif key in data:
            kwargs[key] = data[key]
    if "safe_utility" in grid:
        kwargs["safe"] = grid["safe_utility"]
    if "successes" in grid:
        kwargs["successes"] = tuple(grid["successes"])
    if "costs" in grid:
        kwargs["costs"] = tuple(float(c) for c in grid["costs"])
    if "structures" in grid:
        kwargs["structures"] = tuple(grid["structures"])
    if "treatments" in grid:
        kwargs["treatments"] = tuple(grid["treatments"])
    return PilotSpec(**kwargs)


def build_public_cases(spec: PilotSpec | None = None) -> list[PublicCase]:
    spec = spec or PilotSpec()
    cases = []
    index = 1
    for structure in spec.structures:
        for successes in spec.successes:
            failures = spec.batch_length - successes
            observations = tuple([1] * successes + [0] * failures)
            for cost in spec.costs:
                cases.append(
                    PublicCase(
                        case_id=f"case_{index:02d}",
                        structure=structure,
                        prior_high=spec.prior_high,
                        hazard=spec.stated_hazard(structure),
                        low=spec.low,
                        high=spec.high,
                        safe=spec.safe,
                        diagnostic_cost=float(cost),
                        observations_a=observations,
                        successes=successes,
                        failures=failures,
                    )
                )
                index += 1
    return cases


def evaluator_record(case: PublicCase) -> EvaluatorRecord:
    result = posterior(case.model(), case.observations())
    decision = decision_values(
        result["posterior_high"]["B"],
        low=case.low,
        high=case.high,
        safe=case.safe,
        cost=case.diagnostic_cost,
    )
    return EvaluatorRecord(
        case_id=case.case_id,
        posterior_high=result["posterior_high"],
        predictive_mean=result["predictive_mean"],
        posterior_events=result["posterior_events"],
        decision=decision,
    )


def build_jobs(
    spec: PilotSpec | None = None,
) -> list[tuple[PublicCase, str, EvaluatorRecord]]:
    spec = spec or PilotSpec()
    jobs = []
    for case in build_public_cases(spec):
        record = evaluator_record(case)
        for treatment in spec.treatments:
            jobs.append((case, treatment, record))
    return jobs


def _reset_semantics(case: PublicCase) -> str:
    if case.structure == "shared":
        return (
            "A and B share one possible reset event. That event occurs with probability "
            f"{_fmt(case.hazard)}. If it occurs, A and B both reset; if it does not, neither "
            "resets. On reset, each affected component independently redraws its rate as high "
            "or low with probability 0.5. Sharing the event does not mean sharing the redrawn "
            "value. C cannot reset in this interval."
        )
    if case.structure == "independent":
        return (
            "A and B have independent reset events. Each occurs with probability "
            f"{_fmt(case.hazard)} and uses the same redraw rule: a reset redraws that "
            "component's rate as high or low with probability 0.5. C cannot reset in this "
            "interval."
        )
    return (
        "No component can reset in this interval. The hazard is 0. This is the stated "
        "generative model, not a hidden label about whether an event occurred under a "
        "positive hazard."
    )


def _fmt(value: float) -> str:
    text = f"{value:.12g}"
    return text


def _shared_body(case: PublicCase) -> str:
    return (
        "You are given a fully stated generative model and public evidence. Infer the "
        "requested probabilities and choose one action. Do not assume an event occurred "
        "unless the stated model says the hazard is 1.\n\n"
        "Components A, B, and C each have a binary latent reward rate: low "
        f"{_fmt(case.low)} or high {_fmt(case.high)}. Independently of the others, each "
        f"starts high with probability {_fmt(case.prior_high)}. That prior already "
        "summarizes earlier evidence; you are not given raw prehistory.\n\n"
        "There is at most one possible reset before the observation batch. No component "
        "changes during the batch.\n\n"
        f"{_reset_semantics(case)}\n\n"
        "You will see evidence about A only. There is no fresh evidence about B or C.\n\n"
        "Choose exactly one action:\n"
        f"- choose_B: take B's unknown expected reward, equal to {_fmt(case.low)} + "
        f"{_fmt(case.high - case.low)} * P(B is high).\n"
        f"- choose_safe: take a known alternative worth {_fmt(case.safe)}.\n"
        f"- diagnose_B: pay {_fmt(case.diagnostic_cost)} once to learn B's latent RATE "
        "(not a future reward), then take the higher of that rate and "
        f"{_fmt(case.safe)}. The cost is charged exactly once even if the safe option is "
        "chosen afterward. Choosing diagnose_B commits to using the revealed rate "
        "optimally; this probe tests valuation of information, not interpretation of "
        "diagnostic results.\n"
    )


def _evidence_block(case: PublicCase, treatment: str, record: EvaluatorRecord | None) -> str:
    if treatment == "raw":
        sequence = ", ".join(str(v) for v in case.observations_a)
        return (
            "Evidence on A after the possible reset, in order (1 = success, 0 = failure): "
            f"[{sequence}]. B and C were not observed."
        )
    counts = (
        "Verified counts on A after the possible reset: "
        f"{case.successes} successes and {case.failures} failures out of "
        f"{case.successes + case.failures} observations. There is no within-batch change, "
        "so these counts preserve the likelihood of the corresponding raw sequence. "
        "B and C were not observed."
    )
    if treatment == "counts":
        return counts
    if record is None:
        raise ValueError("reference treatment requires evaluator probabilities")
    supplied = record.posterior_high
    return (
        counts
        + " A trusted observation-limited reference, computed from the same public "
        "evidence and the stated model, reports these posterior probabilities that each "
        "rate is high: "
        f"A={_fmt(supplied['A'])}, B={_fmt(supplied['B'])}, C={_fmt(supplied['C'])}. "
        "These are predictions, not a recommended action."
    )


OUTPUT_SCHEMA = """Return exactly one JSON object and no other text:
{
  "probability_high": {
    "A": <posterior probability that A's latent rate is high>,
    "B": <posterior probability that B's latent rate is high>,
    "C": <posterior probability that C's latent rate is high>
  },
  "predictive_mean": {
    "A": <expected latent reward rate of A>,
    "B": <expected latent reward rate of B>,
    "C": <expected latent reward rate of C>
  },
  "action": "choose_B" or "choose_safe" or "diagnose_B"
}

probability_high.X is P(the latent rate of X equals the high rate | permitted evidence and the stated model). It is not a verbal confidence rating.
predictive_mean.X is the expected latent rate of X, equal to low + (high-low) * probability_high.X, and is also the expected Bernoulli reward if that rate is used.
action is the unique choice among the three options above.
Do not include a rationale field.
"""


def render_prompt(
    case: PublicCase, treatment: str, record: EvaluatorRecord | None = None
) -> str:
    if treatment not in TREATMENTS:
        raise ValueError(f"unknown treatment {treatment}")
    if treatment == "reference" and record is None:
        raise ValueError("reference treatment requires evaluator probabilities")
    return (
        _shared_body(case)
        + "\n"
        + _evidence_block(case, treatment, record)
        + "\n\n"
        + OUTPUT_SCHEMA
    ).strip() + "\n"


def prompt_bundle(
    case: PublicCase, treatment: str, record: EvaluatorRecord
) -> dict[str, Any]:
    prompt = render_prompt(case, treatment, record)
    public = case.public_dict()
    public["treatment"] = treatment
    if treatment == "reference":
        public["supplied_probability_high"] = dict(record.posterior_high)
    return {
        "case_id": case.case_id,
        "treatment": treatment,
        "prompt": prompt,
        "prompt_hash": sha256_text(prompt),
        "case_hash": sha256_text(canonical_json(case.public_dict())),
        "public_inputs": public,
        "token_estimate": estimate_tokens(prompt),
    }


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number")
    return float(value)


def _probability_field(name: str, value: Any) -> float:
    value = _finite_number(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be in [0, 1]")
    return value


def parse_response(text: str) -> dict[str, Any]:
    from llm_rl_engine.agents.llm import extract_json_object

    try:
        parsed = extract_json_object(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("response must be a JSON object")
    for key in ("probability_high", "predictive_mean", "action"):
        if key not in parsed:
            raise ValidationError(f"missing field {key}")
    high = parsed["probability_high"]
    means = parsed["predictive_mean"]
    if not isinstance(high, dict) or not isinstance(means, dict):
        raise ValidationError("probability_high and predictive_mean must be objects")
    out_high = {}
    out_mean = {}
    for component in COMPONENTS:
        if component not in high or component not in means:
            raise ValidationError(f"missing component {component}")
        out_high[component] = _probability_field(
            f"probability_high.{component}", high[component]
        )
        out_mean[component] = _finite_number(
            f"predictive_mean.{component}", means[component]
        )
    action = parsed["action"]
    if action not in ACTIONS:
        raise ValidationError("action must be choose_B, choose_safe, or diagnose_B")
    return {
        "probability_high": out_high,
        "predictive_mean": out_mean,
        "action": action,
    }


def action_utility(action: str, decision: Mapping[str, Any]) -> float:
    values = decision["action_values"]
    if action not in values:
        raise ValidationError(f"unknown action {action}")
    return float(values[action])


def score_response(
    parsed: Mapping[str, Any] | None,
    record: EvaluatorRecord,
    *,
    valid: bool,
    failure_kind: str | None,
    spec: PilotSpec | None = None,
) -> dict[str, Any]:
    spec = spec or PilotSpec()
    decision = record.decision
    optimal = decision["optimal_action"]
    optimal_utility = float(decision["action_values"][optimal])
    if valid and parsed is not None:
        action = parsed["action"]
        utility = action_utility(action, decision)
        fallback = False
        decision_error = action != optimal
    else:
        action = "choose_safe"
        utility = action_utility(action, decision)
        fallback = True
        decision_error = False
    loss = optimal_utility - utility
    prior_c = spec.prior_predictive_c
    if valid and parsed is not None:
        p_err = {
            component: parsed["probability_high"][component] - record.posterior_high[component]
            for component in COMPONENTS
        }
        mean_err = {
            component: parsed["predictive_mean"][component] - record.predictive_mean[component]
            for component in COMPONENTS
        }
        brier = sum(err * err for err in p_err.values()) / len(COMPONENTS)
        c_drift = parsed["predictive_mean"]["C"] - prior_c
        overpriced = action == "diagnose_B" and optimal != "diagnose_B"
    else:
        p_err = {component: None for component in COMPONENTS}
        mean_err = {component: None for component in COMPONENTS}
        brier = None
        c_drift = None
        overpriced = False
    return {
        "valid": valid,
        "failure_kind": failure_kind,
        "itt_fallback": fallback,
        "chosen_action": action,
        "optimal_action": optimal,
        "chosen_utility": utility,
        "optimal_utility": optimal_utility,
        "decision_loss": loss,
        "decision_error": decision_error,
        "probability_error": p_err,
        "predictive_mean_error": mean_err,
        "brier_probability_high": brier,
        "c_drift": c_drift,
        "linked_b_correct": valid and parsed is not None and parsed["action"] == optimal,
        "overpriced_diagnostic": overpriced,
    }


def prompt_leaks_evaluator(prompt: str, record: EvaluatorRecord) -> list[str]:
    leaks = [marker for marker in HIDDEN_EVALUATOR_MARKERS if marker in prompt]
    optimal = record.decision["optimal_action"]
    # The schema lists all three action names; leaking the *recommended* action
    # would be a sentence that names the optimum as the answer.
    if f"recommended action is {optimal}" in prompt or f"best action is {optimal}" in prompt:
        leaks.append("recommended_optimal_action")
    return leaks


def shared_prompt_prefix(case: PublicCase) -> str:
    return _shared_body(case)


def summarize_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    attempted = len(rows)
    valid = [row for row in rows if row.get("valid")]
    api_fail = sum(1 for row in rows if row.get("failure_kind") == "api_failure")
    parse_fail = sum(1 for row in rows if row.get("failure_kind") == "parse_failure")
    decision_err = sum(1 for row in valid if row.get("decision_error"))
    def _mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        if not present:
            return None
        return math.fsum(present) / len(present)

    return {
        "n_attempted": attempted,
        "n_valid": len(valid),
        "n_api_failure": api_fail,
        "n_parse_failure": parse_fail,
        "n_decision_error": decision_err,
        "itt_mean_utility": _mean([row.get("chosen_utility") for row in rows]),
        "itt_mean_decision_loss": _mean([row.get("decision_loss") for row in rows]),
        "valid_mean_decision_loss": _mean([row.get("decision_loss") for row in valid]),
        "valid_mean_brier": _mean([row.get("brier_probability_high") for row in valid]),
        "valid_mean_c_drift": _mean([row.get("c_drift") for row in valid]),
        "linked_b_accuracy_valid": (
            sum(1 for row in valid if row.get("linked_b_correct")) / len(valid) if valid else None
        ),
        "overpriced_diagnostic_count": sum(1 for row in valid if row.get("overpriced_diagnostic")),
        "note": (
            "Descriptive macro-averages over this chosen grid, not estimated natural-world "
            "return. Treatments and cost variants are not independent episodes."
        ),
    }
