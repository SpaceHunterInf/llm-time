"""Sequential three-bit filter and frozen tapes. Separate from the one-event observer.

Joint over (observed, linked, unrelated) high-bits. Shared resets couple the pair;
three marginals are not a sufficient recursive state. Unrelated never resets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
import math
from numbers import Real
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from llm_rl_engine.analysis.selective_revision import decision_values
from llm_rl_engine.analysis.selective_revision_pilot import (
    ACTIONS,
    UTILITY_TIE_TOL,
    canonical_json,
    sha256_text,
)

ROLES = ("observed", "linked", "unrelated")
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}
# state index = observed + 2*linked + 4*unrelated
N_STATES = 8
DEFAULT_Q = 0.99
DEFAULT_H = 0.03
DEFAULT_LOW = 0.2
DEFAULT_HIGH = 0.8
DEFAULT_SAFE = 0.65
DEFAULT_COST = 0.04
PROBE_AFTER = (12, 20, 32, 44)
CONDITIONS = ("reconstruct", "self_state", "reference_state")
SCHEMA_VERSION = 2
JOINT_KEY_ORDER = (
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
)
SUM_TOLERANCE = 1e-4
SHARED_GENERATIVE = (
    "Initially each component is high with probability 0.99, independently. "
    "A reset is different from initialization: whenever a component resets, its new "
    "state is high with probability 0.5 and low with probability 0.5, independently "
    "of its previous state. Components that reset together share only the event; "
    "their redrawn values are independent. The unrelated component never resets. "
    "Before every observed outcome, apply that time step's reset transition, then "
    "generate the Bernoulli outcome from the current rate (0.8 if high, 0.2 if low). "
    "The decision uses the latent state immediately after the final listed observation; "
    "there is no extra transition before diagnosis or reward."
)
STATE_USE = (
    "previous_state summarizes observations through its timestamp and is not independent "
    "evidence. You may reconstruct from the original prior and all listed observations, "
    "or start from previous_state and process only later observations. Do not apply "
    "already summarized observations twice. The state is an estimate, not additional data."
)


def state_bits(index: int) -> tuple[int, int, int]:
    return index & 1, (index >> 1) & 1, (index >> 2) & 1


def state_index(observed: int, linked: int, unrelated: int) -> int:
    return (observed & 1) + 2 * (linked & 1) + 4 * (unrelated & 1)


def joint_keys() -> list[str]:
    return ["".join(str(b) for b in state_bits(s)) for s in range(N_STATES)]


def prior_joint(q: float = DEFAULT_Q) -> list[float]:
    q = float(q)
    joint = []
    for s in range(N_STATES):
        o, l, u = state_bits(s)
        p = 1.0
        for bit in (o, l, u):
            p *= q if bit else (1.0 - q)
        joint.append(p)
    z = math.fsum(joint)
    return [x / z for x in joint]


def transition_matrix(structure: str, hazard: float) -> list[list[float]]:
    h = float(hazard)
    if not 0.0 <= h <= 1.0:
        raise ValueError("hazard must be in [0, 1]")
    if structure not in ("shared", "independent", "stable"):
        raise ValueError(f"unknown structure {structure}")
    if structure == "stable":
        h = 0.0
    t = [[0.0] * N_STATES for _ in range(N_STATES)]
    for src in range(N_STATES):
        o, l, u = state_bits(src)
        if h == 0.0:
            t[src][src] = 1.0
            continue
        if structure == "shared":
            t[src][src] += 1.0 - h
            for o2, l2 in product((0, 1), repeat=2):
                dst = state_index(o2, l2, u)
                t[src][dst] += h * 0.25
        else:
            for o2, l2 in product((0, 1), repeat=2):
                po = (1.0 - h) * (1.0 if o2 == o else 0.0) + h * 0.5
                pl = (1.0 - h) * (1.0 if l2 == l else 0.0) + h * 0.5
                t[src][state_index(o2, l2, u)] += po * pl
    for row in t:
        z = math.fsum(row)
        if z <= 0:
            raise ValueError("invalid transition row")
        for j, value in enumerate(row):
            row[j] = value / z
    return t


def predict(joint: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    _check_joint(joint)
    out = []
    for dst in range(N_STATES):
        out.append(math.fsum(joint[src] * matrix[src][dst] for src in range(N_STATES)))
    return _normalize(out)


def observe(
    joint: Sequence[float],
    role: str,
    outcome: int,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> list[float]:
    _check_joint(joint)
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    bit_i = ROLE_INDEX[role]
    like = []
    for s in range(N_STATES):
        bits = state_bits(s)
        rate = high if bits[bit_i] else low
        like.append(rate if outcome == 1 else (1.0 - rate))
    unnorm = [joint[i] * like[i] for i in range(N_STATES)]
    return _normalize(unnorm)


def step_filter(
    joint: Sequence[float],
    matrix: Sequence[Sequence[float]],
    role: str,
    outcome: int,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> list[float]:
    return observe(predict(joint, matrix), role, outcome, low=low, high=high)


def marginals(joint: Sequence[float]) -> dict[str, float]:
    _check_joint(joint)
    out = {role: 0.0 for role in ROLES}
    for s, p in enumerate(joint):
        bits = state_bits(s)
        for role, bit in zip(ROLES, bits):
            if bit:
                out[role] += p
    return out


def predictive_means(
    joint: Sequence[float], *, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH
) -> dict[str, float]:
    p = marginals(joint)
    return {role: low + (high - low) * p[role] for role in ROLES}


def joint_dict(joint: Sequence[float]) -> dict[str, float]:
    return {key: joint[i] for i, key in enumerate(joint_keys())}


def from_joint_dict(payload: Mapping[str, float]) -> list[float]:
    return [_probability(payload[key]) for key in joint_keys()]


def filter_prefix(
    observations: Sequence[tuple[str, int]],
    *,
    structure: str,
    hazard: float,
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> list[float]:
    joint = prior_joint(q)
    matrix = transition_matrix(structure, hazard)
    for role, outcome in observations:
        joint = step_filter(joint, matrix, role, outcome, low=low, high=high)
    return joint


def _check_joint(joint: Sequence[float]) -> None:
    if len(joint) != N_STATES:
        raise ValueError("joint must have 8 masses")
    if any(not math.isfinite(x) or x < 0 for x in joint):
        raise ValueError("joint masses must be finite and nonnegative")


def _normalize(masses: Sequence[float]) -> list[float]:
    z = math.fsum(masses)
    if z <= 0 or not math.isfinite(z):
        raise ValueError("zero likelihood")
    return [x / z for x in masses]


def _probability(value: float) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError("probability must be finite and nonnegative")
    return x


def observation_schedule() -> list[str]:
    cycle = ["observed", "linked", "unrelated"] * 4
    block = ["observed"] * 8 + ["linked"] * 2 + ["unrelated"] * 2
    return cycle + block + block + ["observed"] * 8


def probe_indices() -> tuple[int, ...]:
    return PROBE_AFTER


class StreamRng:
    def __init__(self, seed: int, stream: str) -> None:
        self._rng = random.Random(f"{seed}:{stream}")

    def uniform(self) -> float:
        return self._rng.random()


def _bit_from_uniform(u: float, q: float) -> int:
    return 1 if u < q else 0


LABEL_ROTATIONS = (
    {"observed": "A", "linked": "B", "unrelated": "C"},
    {"observed": "B", "linked": "C", "unrelated": "A"},
    {"observed": "C", "linked": "A", "unrelated": "B"},
)


@dataclass(frozen=True)
class SequentialTape:
    tape_id: str
    seed: int
    structure: str
    hazard: float
    labels: dict[str, str]
    schedule: tuple[str, ...]
    outcomes: tuple[int, ...]
    private_latents: tuple[tuple[int, int, int], ...]
    private_events: tuple[dict[str, bool], ...]
    public_observations: tuple[dict[str, Any], ...]

    def prefix(self, n: int) -> list[tuple[str, int]]:
        return list(zip(self.schedule[:n], self.outcomes[:n]))

    def public_prefix(self, n: int) -> list[dict[str, Any]]:
        return list(self.public_observations[:n])


def generate_tape(
    seed: int,
    structure: str,
    *,
    hazard: float = DEFAULT_H,
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    labels: Mapping[str, str] | None = None,
    tape_id: str | None = None,
) -> SequentialTape:
    if structure == "stable":
        hazard = 0.0
    labels = dict(labels or LABEL_ROTATIONS[seed % 3])
    schedule = tuple(observation_schedule())
    init = StreamRng(seed, "init")
    emit = StreamRng(seed, "emit")
    event_shared = StreamRng(seed, "event_shared")
    event_o = StreamRng(seed, "event_o")
    event_l = StreamRng(seed, "event_l")
    redraw_o = StreamRng(seed, "redraw_o")
    redraw_l = StreamRng(seed, "redraw_l")
    o = _bit_from_uniform(init.uniform(), q)
    l = _bit_from_uniform(init.uniform(), q)
    u = _bit_from_uniform(init.uniform(), q)
    outcomes = []
    latents = []
    events = []
    public = []
    for t, role in enumerate(schedule):
        ev = {"reset_observed": False, "reset_linked": False}
        if structure == "shared":
            if event_shared.uniform() < hazard:
                ev["reset_observed"] = True
                ev["reset_linked"] = True
                o = 1 if redraw_o.uniform() < 0.5 else 0
                l = 1 if redraw_l.uniform() < 0.5 else 0
            else:
                redraw_o.uniform()
                redraw_l.uniform()
        elif structure == "independent":
            if event_o.uniform() < hazard:
                ev["reset_observed"] = True
                o = 1 if redraw_o.uniform() < 0.5 else 0
            else:
                redraw_o.uniform()
            if event_l.uniform() < hazard:
                ev["reset_linked"] = True
                l = 1 if redraw_l.uniform() < 0.5 else 0
            else:
                redraw_l.uniform()
        else:
            event_shared.uniform()
            event_o.uniform()
            event_l.uniform()
            redraw_o.uniform()
            redraw_l.uniform()
        bit = {"observed": o, "linked": l, "unrelated": u}[role]
        rate = high if bit else low
        y = 1 if emit.uniform() < rate else 0
        outcomes.append(y)
        latents.append((o, l, u))
        events.append(ev)
        public.append(
            {
                "step": t + 1,
                "component": labels[role],
                "role": role,
                "outcome": y,
            }
        )
    return SequentialTape(
        tape_id=tape_id or f"{structure}_s{seed}",
        seed=seed,
        structure=structure,
        hazard=hazard,
        labels=labels,
        schedule=schedule,
        outcomes=tuple(outcomes),
        private_latents=tuple(latents),
        private_events=tuple(events),
        public_observations=tuple(public),
    )


def development_tapes() -> list[SequentialTape]:
    tapes = []
    for seed in (0, 1, 2):
        labels = LABEL_ROTATIONS[seed]
        for structure in ("shared", "independent"):
            tapes.append(
                generate_tape(
                    seed,
                    structure,
                    hazard=DEFAULT_H,
                    labels=labels,
                    tape_id=f"{structure}_s{seed}",
                )
            )
    tapes.append(
        generate_tape(
            10, "stable", hazard=0.0, labels=LABEL_ROTATIONS[0], tape_id="stable_s10"
        )
    )
    tapes.append(
        generate_tape(
            11, "stable", hazard=0.0, labels=LABEL_ROTATIONS[1], tape_id="stable_s11"
        )
    )
    return tapes


def reference_at(tape: SequentialTape, n: int) -> dict[str, Any]:
    joint = filter_prefix(
        tape.prefix(n), structure=tape.structure, hazard=tape.hazard
    )
    p_high = marginals(joint)
    means = predictive_means(joint)
    decision = decision_values(p_high["linked"], cost=DEFAULT_COST, safe=DEFAULT_SAFE)
    return {
        "n": n,
        "joint": joint,
        "joint_dict": joint_dict(joint),
        "probability_high": p_high,
        "predictive_mean": means,
        "decision": decision,
    }


def joint_to_keyed(joint: Sequence[float]) -> dict[str, float]:
    _check_joint(joint)
    return {key: float(joint[key_to_index(key)]) for key in JOINT_KEY_ORDER}


def key_to_index(key: str) -> int:
    if len(key) != 3 or any(c not in "01" for c in key):
        raise ValueError(f"invalid joint key {key}")
    o, l, u = (int(key[0]), int(key[1]), int(key[2]))
    return state_index(o, l, u)


def _strict_mass(name: str, value: Any) -> float:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise ValueError(f"{name} must be a JSON number, not {type(value).__name__}")
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return x


def keyed_to_joint(payload: Mapping[str, Any]) -> list[float]:
    missing = [k for k in JOINT_KEY_ORDER if k not in payload]
    if missing:
        raise ValueError(f"joint missing keys {missing}")
    extra = [k for k in payload if k not in JOINT_KEY_ORDER]
    if extra:
        raise ValueError(f"joint has extra keys {extra}")
    masses = [0.0] * N_STATES
    for key in JOINT_KEY_ORDER:
        masses[key_to_index(key)] = _strict_mass(f"joint.{key}", payload[key])
    return masses


def serialize_state(joint: Sequence[float], checkpoint_label: str) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint_label,
        "schema_version": SCHEMA_VERSION,
        "joint": joint_to_keyed(joint),
    }


def tape_fingerprint(tape: SequentialTape) -> dict[str, Any]:
    return {
        "tape_id": tape.tape_id,
        "seed": tape.seed,
        "structure": tape.structure,
        "hazard": tape.hazard,
        "labels": dict(tape.labels),
        "schedule": list(tape.schedule),
        "outcomes": list(tape.outcomes),
        "probes": list(PROBE_AFTER),
    }


def state_hash(state: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(state))


def build_observation_schedule_text(tape: SequentialTape, n: int) -> str:
    lines = []
    for row in tape.public_prefix(n):
        lines.append(
            f"t={row['step']}: component {row['component']} -> {row['outcome']}"
        )
    return "\n".join(lines)


def coupling_paragraph(tape: SequentialTape) -> str:
    labels = tape.labels
    if tape.structure == "stable" or tape.hazard == 0.0:
        return (
            "No component can reset in this interval. The hazard is 0. This is the stated "
            "generative model, not a hidden label about whether an event occurred under a "
            "positive hazard."
        )
    if tape.structure == "shared":
        return (
            f"Observed component {labels['observed']} and linked component {labels['linked']} "
            f"share one Bernoulli({tape.hazard}) reset event. They share only the event."
        )
    return (
        f"Observed component {labels['observed']} and linked component {labels['linked']} "
        f"have independent Bernoulli({tape.hazard}) reset events."
    )


def render_sequential_prompt(
    tape: SequentialTape,
    n: int,
    condition: str,
    previous_state: Mapping[str, Any] | None,
    *,
    invalid_state_marker: bool = False,
) -> str:
    labels = tape.labels
    if condition == "reconstruct":
        state_block = (
            "previous_state: empty. Reconstruct from the original prior and the listed "
            "observations."
        )
    elif previous_state is not None:
        marker = (
            " The previous self-state failed validation and was not replaced by the "
            "evaluator; this is the last valid timestamped state."
            if invalid_state_marker
            else ""
        )
        state_block = (
            f"previous_state checkpoint={previous_state.get('checkpoint')}:{marker}\n"
            + canonical_json(previous_state)
        )
    else:
        state_block = "previous_state: empty"
    schema = (
        '{"joint": {"000": p, "001": p, "010": p, "011": p, '
        '"100": p, "101": p, "110": p, "111": p}, '
        '"action": "choose_linked" or "choose_safe" or "diagnose_linked"}'
    )
    return (
        f"Named components: observed={labels['observed']}, linked={labels['linked']}, "
        f"unrelated={labels['unrelated']}. High rate is 0.8 and low rate is 0.2.\n\n"
        f"{SHARED_GENERATIVE}\n\n"
        f"{coupling_paragraph(tape)}\n\n"
        "Each joint key has three characters, left to right: observed, linked, unrelated. "
        "1 means the component is high and 0 means low. Example: 100 means observed high, "
        "linked low, unrelated low. Keys are names, not an ordered array.\n\n"
        f"{STATE_USE}\n\n"
        "History-plus-state is redundant external state, not a memory-compression test. "
        "The decision is a one-shot expected-utility question about the linked component. "
        "It does not change the tape and does not reveal a diagnostic outcome to later "
        "checkpoints.\n\n"
        f"{state_block}\n\n"
        f"Observations 1..{n} (no hidden events, no future rows):\n"
        f"{build_observation_schedule_text(tape, n)}\n\n"
        "Return exactly one JSON object and no other text:\n"
        f"{schema}\n"
        "joint must contain exactly those eight keys, each a probability in [0, 1], summing "
        "to 1. Do not report separate marginals or predictive means.\n"
        f"choose_linked takes the linked component's expected rate. choose_safe takes a known "
        f"alternative worth {DEFAULT_SAFE}. diagnose_linked: you pay a cost of {DEFAULT_COST} "
        "once to learn the linked latent RATE, then take the higher of that rate and "
        f"{DEFAULT_SAFE}. Do not include a rationale.\n"
    )


SEQUENTIAL_ACTIONS = ("choose_safe", "choose_linked", "diagnose_linked")


def map_action_from_v1(action: str) -> str:
    return {
        "choose_B": "choose_linked",
        "choose_safe": "choose_safe",
        "diagnose_B": "diagnose_linked",
    }[action]


def sequential_jobs(tapes: Sequence[SequentialTape] | None = None) -> list[dict[str, Any]]:
    tapes = list(tapes or development_tapes())
    by_id = {t.tape_id: t for t in tapes}
    jobs = []
    seen = set()
    # Smoke-first: one reconstruct n=12 from shared, independent, and stable.
    for tape_id in ("shared_s0", "independent_s0", "stable_s10"):
        tape = by_id.get(tape_id)
        if tape is None:
            continue
        job = {
            "tape_id": tape.tape_id,
            "n": 12,
            "condition": "reconstruct",
            "tape": tape,
            "reference": reference_at(tape, 12),
        }
        jobs.append(job)
        seen.add((tape.tape_id, 12, "reconstruct"))
    for tape in tapes:
        refs = {n: reference_at(tape, n) for n in PROBE_AFTER}
        for n in PROBE_AFTER:
            for condition in CONDITIONS:
                key = (tape.tape_id, n, condition)
                if key in seen:
                    continue
                jobs.append(
                    {
                        "tape_id": tape.tape_id,
                        "n": n,
                        "condition": condition,
                        "tape": tape,
                        "reference": refs[n],
                    }
                )
    return jobs


def previous_checkpoint(n: int) -> int | None:
    probes = list(PROBE_AFTER)
    i = probes.index(n)
    return None if i == 0 else probes[i - 1]


def parse_sequential_response(text: str) -> dict[str, Any]:
    from llm_rl_engine.agents.llm import extract_json_object
    from llm_rl_engine.analysis.selective_revision_pilot import ValidationError

    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        raise ValidationError("response must be a JSON object")
    raw_joint = parsed.get("joint")
    if not isinstance(raw_joint, dict):
        raise ValidationError("joint must be an object with eight keyed masses")
    try:
        masses = keyed_to_joint(raw_joint)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    raw_sum = math.fsum(masses)
    if abs(raw_sum - 1.0) > SUM_TOLERANCE:
        raise ValidationError(f"joint sum {raw_sum} is outside 1 ± {SUM_TOLERANCE}")
    if raw_sum <= 0:
        raise ValidationError("joint has zero mass")
    normalized = [x / raw_sum for x in masses]
    action = parsed.get("action")
    if action not in SEQUENTIAL_ACTIONS:
        raise ValidationError("invalid action")
    p_high = marginals(normalized)
    means = predictive_means(normalized)
    return {
        "joint": normalized,
        "joint_raw": masses,
        "joint_raw_sum": raw_sum,
        "joint_correction": abs(raw_sum - 1.0),
        "joint_keyed": joint_to_keyed(normalized),
        "probability_high": p_high,
        "predictive_mean": means,
        "action": action,
    }


def audit_v1_final_outputs(run_dir: str | Path) -> dict[str, Any]:
    """Offline diagnostics on retained v1 finals. Does not rewrite scores."""
    from llm_rl_engine.agents.llm import extract_json_object

    run_dir = Path(run_dir)
    results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    last = {}
    for line in (run_dir / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "attempted" and row.get("raw_response") and row.get("key"):
            last[row["key"]] = row
    sum_violations = []
    marginal_disagreements = []
    for result in results:
        attempt = last.get(result.get("key"))
        if not attempt:
            continue
        obj = extract_json_object(attempt["raw_response"])
        joint = obj.get("joint")
        if not isinstance(joint, list) or len(joint) != N_STATES:
            continue
        masses = [float(x) for x in joint]
        raw_sum = math.fsum(masses)
        if abs(raw_sum - 1.0) > SUM_TOLERANCE:
            sum_violations.append(
                {
                    "tape_id": result.get("tape_id"),
                    "n": result.get("n"),
                    "condition": result.get("condition"),
                    "raw_sum": raw_sum,
                }
            )
        if raw_sum > 0:
            norm = [x / raw_sum for x in masses]
            implied = marginals(norm)
            reported = obj.get("probability_high") or {}
            if reported:
                dmax = max(
                    abs(implied[role] - float(reported[role]))
                    for role in ROLES
                    if role in reported
                )
                if dmax > 0.001:
                    marginal_disagreements.append(
                        {
                            "tape_id": result.get("tape_id"),
                            "n": result.get("n"),
                            "condition": result.get("condition"),
                            "max_abs_disagreement": dmax,
                        }
                    )
    return {
        "n_final": len(results),
        "n_sum_violations": len(sum_violations),
        "n_marginal_disagreements": len(marginal_disagreements),
        "sum_violations": sum_violations,
        "marginal_disagreements": marginal_disagreements,
    }


def action_utilities_from_p(
    p: float,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    safe: float = DEFAULT_SAFE,
    cost: float = DEFAULT_COST,
) -> dict[str, float]:
    return {
        "choose_safe": float(safe),
        "choose_linked": float(low + (high - low) * p),
        "diagnose_linked": float(p * max(high, safe) + (1.0 - p) * max(low, safe) - cost),
    }


def own_forecast_action(p: float) -> str:
    values = action_utilities_from_p(p)
    best = max(values.values())
    for name, value in values.items():
        if best - value <= UTILITY_TIE_TOL:
            return name
    return "choose_safe"


def score_sequential(parsed: Mapping[str, Any] | None, reference: Mapping[str, Any], *, valid: bool) -> dict[str, Any]:
    decision = reference["decision"]
    optimal_v1 = decision["optimal_action"]
    optimal = map_action_from_v1(optimal_v1)
    values = {
        "choose_safe": decision["action_values"]["choose_safe"],
        "choose_linked": decision["action_values"]["choose_B"],
        "diagnose_linked": decision["action_values"]["diagnose_B"],
    }
    best = values[optimal]
    ref_high = reference["probability_high"]
    if valid and parsed is not None:
        action = parsed["action"]
        utility = values[action]
        loss = best - utility
        error = loss > UTILITY_TIE_TOL
        mse = math.fsum((a - b) ** 2 for a, b in zip(parsed["joint"], reference["joint"])) / N_STATES
        own_p = parsed["probability_high"]["linked"]
        own_u = action_utilities_from_p(own_p)
        own_action = own_forecast_action(own_p)
        own_error = (own_u[own_action] - own_u[action]) > UTILITY_TIE_TOL
        linked_abs = abs(parsed["probability_high"]["linked"] - ref_high["linked"])
        unrelated_abs = abs(parsed["probability_high"]["unrelated"] - ref_high["unrelated"])
        offline_best = own_action
        offline_ref_regret = values[optimal] - values[offline_best]
    else:
        action = "choose_safe"
        utility = values[action]
        loss = best - utility
        error = False
        mse = None
        own_action = None
        own_error = False
        linked_abs = None
        unrelated_abs = None
        offline_best = None
        offline_ref_regret = None
    return {
        "valid": valid,
        "chosen_action": action,
        "optimal_action": optimal,
        "decision_loss": loss,
        "decision_error": error,
        "joint_mse": mse,
        "own_forecast_action": own_action,
        "own_forecast_decision_error": own_error,
        "linked_abs_error": linked_abs,
        "unrelated_abs_error": unrelated_abs,
        "offline_action_from_joint": offline_best,
        "offline_action_reference_regret": offline_ref_regret,
        "joint_raw_sum": None if parsed is None else parsed.get("joint_raw_sum"),
        "joint_correction": None if parsed is None else parsed.get("joint_correction"),
        "diagnosis_has_positive_value": decision["net_diagnostic_advantage"] > 0,
        "itt_fallback": not valid,
    }


def audit_v2_own_forecast(run_dir: str | Path) -> dict[str, Any]:
    """Recompute own-forecast errors on retained v2 finals. Does not rewrite rows."""
    from llm_rl_engine.agents.llm import extract_json_object

    run_dir = Path(run_dir)
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    last = {}
    for line in (run_dir / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "attempted" and row.get("raw_response") and row.get("key"):
            last[row["key"]] = row
    by_cond = {c: 0 for c in CONDITIONS}
    n_valid = 0
    n_own = 0
    n_ref_err = 0
    n_both = 0
    for result in results:
        if not result.get("valid"):
            continue
        n_valid += 1
        attempt = last.get(result.get("key"))
        if not attempt:
            continue
        obj = extract_json_object(attempt["raw_response"])
        masses = keyed_to_joint(obj["joint"])
        raw_sum = math.fsum(masses)
        if raw_sum <= 0:
            continue
        norm = [x / raw_sum for x in masses]
        p = marginals(norm)["linked"]
        own_act = own_forecast_action(p)
        chosen = obj.get("action")
        own_u = action_utilities_from_p(p)
        own_err = (own_u[own_act] - own_u[chosen]) > UTILITY_TIE_TOL if chosen in own_u else True
        if own_err:
            n_own += 1
            by_cond[result.get("condition")] = by_cond.get(result.get("condition"), 0) + 1
        if result.get("decision_error"):
            n_ref_err += 1
            if own_err:
                n_both += 1
    return {
        "n_valid": n_valid,
        "own_forecast_errors": n_own,
        "by_condition": by_cond,
        "reference_errors": n_ref_err,
        "both_own_and_reference": n_both,
    }


def initial_state_payload() -> dict[str, Any]:
    return serialize_state(prior_joint(), "before_obs_1")


def handoff_tapes() -> list[SequentialTape]:
    tapes = []
    for seed in (3, 4):
        labels = LABEL_ROTATIONS[seed % 3]
        for structure in ("shared", "independent"):
            tapes.append(
                generate_tape(
                    seed,
                    structure,
                    labels=labels,
                    tape_id=f"{structure}_s{seed}",
                )
            )
    tapes.append(
        generate_tape(12, "stable", hazard=0.0, labels=LABEL_ROTATIONS[0], tape_id="stable_s12")
    )
    tapes.append(
        generate_tape(13, "stable", hazard=0.0, labels=LABEL_ROTATIONS[1], tape_id="stable_s13")
    )
    return tapes


def continue_filter(joint: Sequence[float], tape: SequentialTape, start: int, end: int) -> list[float]:
    matrix = transition_matrix(tape.structure, tape.hazard)
    out = list(joint)
    for role, y in tape.prefix(end)[start:]:
        out = step_filter(out, matrix, role, y)
    return out


HANDOFF_N = 32
HANDOFF_PREV = 20
HANDOFF_INTERFACES = ("R", "F", "S", "C")
HANDOFF_ORDER_SEED = 6006


def render_handoff_prompt(tape: SequentialTape, interface: str) -> str:
    labels = tape.labels
    ref32 = reference_at(tape, HANDOFF_N)
    ref20 = reference_at(tape, HANDOFF_PREV)
    header = (
        f"Named components: observed={labels['observed']}, linked={labels['linked']}, "
        f"unrelated={labels['unrelated']}. High rate is 0.8 and low rate is 0.2.\n\n"
        f"{SHARED_GENERATIVE}\n\n"
        f"{coupling_paragraph(tape)}\n\n"
        "Each joint key has three characters, left to right: observed, linked, unrelated. "
        "1 means the component is high and 0 means low. Example: 100 means observed high, "
        "linked low, unrelated low. Keys are names, not an ordered array.\n\n"
    )
    schema = (
        '{"joint": {"000": p, "001": p, "010": p, "011": p, '
        '"100": p, "101": p, "110": p, "111": p}, '
        '"action": "choose_linked" or "choose_safe" or "diagnose_linked"}'
    )
    footer = (
        "Return exactly one JSON object and no other text:\n"
        f"{schema}\n"
        "joint must contain exactly those eight keys, each a probability in [0, 1], summing "
        "to 1. Do not report separate marginals or predictive means.\n"
        f"choose_linked takes the linked component's expected rate. choose_safe takes a known "
        f"alternative worth {DEFAULT_SAFE}. diagnose_linked: you pay a cost of {DEFAULT_COST} "
        "once to learn the linked latent RATE, then take the higher of that rate and "
        f"{DEFAULT_SAFE}. Do not include a rationale.\n"
    )
    if interface == "R":
        body = (
            f"{STATE_USE}\n\n"
            "previous_state: empty. Reconstruct from the original prior and the listed observations.\n\n"
            f"Observations 1..{HANDOFF_N} (no hidden events, no future rows):\n"
            f"{build_observation_schedule_text(tape, HANDOFF_N)}\n\n"
        )
    elif interface == "F":
        prev = serialize_state(ref20["joint"], f"after_obs_{HANDOFF_PREV}")
        body = (
            f"{STATE_USE}\n\n"
            f"previous_state checkpoint={prev['checkpoint']}:\n{canonical_json(prev)}\n\n"
            f"Observations 1..{HANDOFF_N} (no hidden events, no future rows):\n"
            f"{build_observation_schedule_text(tape, HANDOFF_N)}\n\n"
        )
    elif interface == "S":
        prev = serialize_state(ref20["joint"], f"after_obs_{HANDOFF_PREV}")
        suffix = build_observation_schedule_text(tape, HANDOFF_N)
        # only rows 21..32: filter lines with t>=21
        lines = [
            f"t={row['step']}: component {row['component']} -> {row['outcome']}"
            for row in tape.public_prefix(HANDOFF_N)
            if row["step"] > HANDOFF_PREV
        ]
        body = (
            "The supplied previous_state is the exact posterior after observation "
            f"{HANDOFF_PREV}. Start from that distribution. There are exactly "
            f"{HANDOFF_N - HANDOFF_PREV} subsequent reset-then-observe steps. "
            "Do not restart from the 0.99 prior. Do not apply an extra transition after "
            "the last listed observation.\n\n"
            f"previous_state checkpoint={prev['checkpoint']}:\n{canonical_json(prev)}\n\n"
            f"Observations {HANDOFF_PREV + 1}..{HANDOFF_N} only:\n"
            + "\n".join(lines)
            + "\n\n"
        )
    else:
        cur = serialize_state(ref32["joint"], f"after_obs_{HANDOFF_N}")
        body = (
            "The supplied distribution already conditions on all observations through the "
            "decision time. Copy this joint unchanged. No further inference or transition "
            "is needed. This is posterior assistance, not a revealed simulator latent.\n\n"
            f"current_state checkpoint={cur['checkpoint']}:\n{canonical_json(cur)}\n\n"
        )
    return header + body + footer


def handoff_jobs() -> list[dict[str, Any]]:
    tapes = handoff_tapes()
    jobs = []
    for tape in tapes:
        ref32 = reference_at(tape, HANDOFF_N)
        ref20 = reference_at(tape, HANDOFF_PREV)
        for interface in HANDOFF_INTERFACES:
            prompt = render_handoff_prompt(tape, interface)
            for repeat in (0, 1):
                jobs.append(
                    {
                        "tape_id": tape.tape_id,
                        "n": HANDOFF_N,
                        "condition": interface,
                        "interface": interface,
                        "repeat": repeat,
                        "tape": tape,
                        "reference": ref32,
                        "reference_prev": ref20,
                        "prompt": prompt,
                    }
                )
    rng = random.Random(HANDOFF_ORDER_SEED)
    order = list(range(len(jobs)))
    rng.shuffle(order)
    return [jobs[i] for i in order]


def prompt_for_job(
    job: Mapping[str, Any],
    previous_self_state: Mapping[str, Any] | None,
    *,
    invalid_self: bool = False,
) -> str:
    tape: SequentialTape = job["tape"]
    n = int(job["n"])
    condition = job["condition"]
    if condition == "reconstruct":
        prev = None
    elif condition == "reference_state":
        prev_n = previous_checkpoint(n)
        joint = prior_joint() if prev_n is None else reference_at(tape, prev_n)["joint"]
        label = "before_obs_1" if prev_n is None else f"after_obs_{prev_n}"
        prev = serialize_state(joint, label)
    else:
        prev = previous_self_state or initial_state_payload()
    return render_sequential_prompt(
        tape, n, condition, prev, invalid_state_marker=invalid_self
    )
