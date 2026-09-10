"""Sequential three-bit filter and frozen tapes. Separate from the one-event observer.

Joint over (observed, linked, unrelated) high-bits. Shared resets couple the pair;
three marginals are not a sufficient recursive state. Unrelated never resets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
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


def serialize_state(joint: Sequence[float], checkpoint_label: str) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint_label,
        "joint_order": joint_keys(),
        "joint": [float(x) for x in joint],
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


def render_sequential_prompt(
    tape: SequentialTape,
    n: int,
    condition: str,
    previous_state: Mapping[str, Any] | None,
    *,
    invalid_state_marker: bool = False,
) -> str:
    labels = tape.labels
    hazard_text = (
        "No component can reset (hazard 0). This is the stated model, not a hidden "
        "label about whether events occurred under a positive hazard."
        if tape.structure == "stable" or tape.hazard == 0.0
        else (
            "Before every observation a reset may occur. "
            + (
                f"{labels['observed']} and {labels['linked']} share one Bernoulli({tape.hazard}) "
                "reset event; on reset each redraws independently as high/low with probability 0.5. "
                if tape.structure == "shared"
                else f"{labels['observed']} and {labels['linked']} have independent Bernoulli({tape.hazard}) "
                "resets with the same redraw rule. "
            )
            + f"{labels['unrelated']} cannot reset."
        )
    )
    state_block = "previous_state: empty"
    if condition == "reconstruct":
        state_block = (
            "previous_state: empty. Reconstruct from the observation prefix. "
            "The generative prior over the eight latent states is the independent product "
            f"with P(high)={DEFAULT_Q} for each bit before observation 1."
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
    return (
        "You infer a factored binary latent (high rate 0.8 or low rate 0.2) for three "
        f"named components. Observed={labels['observed']}, linked={labels['linked']}, "
        f"unrelated={labels['unrelated']}. Initial P(high)={DEFAULT_Q} independently.\n\n"
        f"{hazard_text}\n\n"
        "Joint state order is eight bits observed,linked,unrelated as 000..111 with "
        "observed as the least significant bit (index = o + 2*l + 4*u).\n\n"
        "History-plus-state is redundant external state, not a memory-compression test. "
        "The decision is a one-shot expected-utility question about the linked component. "
        "It does not change the tape and does not reveal a diagnostic outcome to later "
        "checkpoints.\n\n"
        f"{state_block}\n\n"
        f"Observations 1..{n} (no hidden events, no future rows):\n"
        f"{build_observation_schedule_text(tape, n)}\n\n"
        "Return exactly one JSON object:\n"
        '{"joint": [p000, p100, p010, p110, p001, p101, p011, p111], '
        '"probability_high": {"observed": p, "linked": p, "unrelated": p}, '
        '"predictive_mean": {"observed": m, "linked": m, "unrelated": m}, '
        '"action": "choose_linked" or "choose_safe" or "diagnose_linked"}\n'
        "joint must be eight nonnegative numbers in that index order, summing to 1. "
        "choose_linked takes the linked component's expected rate; choose_safe pays "
        f"{DEFAULT_SAFE}; diagnose_linked pays {DEFAULT_COST} once for the linked RATE "
        "then takes max(rate, safe). Do not include a rationale.\n"
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
    jobs = []
    for tape in tapes:
        refs = {n: reference_at(tape, n) for n in PROBE_AFTER}
        for n in PROBE_AFTER:
            for condition in CONDITIONS:
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
    from llm_rl_engine.analysis.selective_revision_pilot import ValidationError, _finite_number, _probability_field

    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        raise ValidationError("response must be a JSON object")
    joint = parsed.get("joint")
    if not isinstance(joint, list) or len(joint) != N_STATES:
        raise ValidationError("joint must be an array of 8 numbers")
    masses = [_finite_number(f"joint[{i}]", joint[i]) for i in range(N_STATES)]
    if any(x < 0 for x in masses):
        raise ValidationError("joint masses must be nonnegative")
    z = math.fsum(masses)
    if z <= 0:
        raise ValidationError("joint has zero mass")
    masses = [x / z for x in masses]
    high = parsed.get("probability_high")
    means = parsed.get("predictive_mean")
    if not isinstance(high, dict) or not isinstance(means, dict):
        raise ValidationError("probability_high and predictive_mean required")
    out_high = {role: _probability_field(f"probability_high.{role}", high[role]) for role in ROLES}
    out_mean = {role: _finite_number(f"predictive_mean.{role}", means[role]) for role in ROLES}
    action = parsed.get("action")
    if action not in SEQUENTIAL_ACTIONS:
        raise ValidationError("invalid action")
    return {"joint": masses, "probability_high": out_high, "predictive_mean": out_mean, "action": action}


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
    if valid and parsed is not None:
        action = parsed["action"]
        utility = values[action]
        loss = best - utility
        error = loss > UTILITY_TIE_TOL
        mse = math.fsum((a - b) ** 2 for a, b in zip(parsed["joint"], reference["joint"])) / N_STATES
    else:
        action = "choose_safe"
        utility = values[action]
        loss = best - utility
        error = False
        mse = None
    return {
        "valid": valid,
        "chosen_action": action,
        "optimal_action": optimal,
        "decision_loss": loss,
        "decision_error": error,
        "joint_mse": mse,
        "diagnosis_has_positive_value": decision["net_diagnostic_advantage"] > 0,
    }


def initial_state_payload() -> dict[str, Any]:
    return serialize_state(prior_joint(), "before_obs_1")


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
