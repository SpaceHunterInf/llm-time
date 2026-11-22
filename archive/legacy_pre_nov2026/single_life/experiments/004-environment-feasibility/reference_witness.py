"""Illustrative feasibility witnesses for task 004.

Hand-specified strategies and exact rational rewards. This is not a learner,
not a kernel fixture, and not evidence that a method adapts. Oracle and
wrong-factor strategies read hidden case tables; the other strategies read
only the request type and scalar rewards already observed.

Run from single_life/:
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync \\
    python experiments/004-environment-feasibility/reference_witness.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

TYPES = ("T1", "T2", "T3")
HORIZON = 36
CHANGE_AT = 13
RESPONSE_AT = 19
GAP = Fraction(1, 5)
PRE_PAIR = Fraction(3, 10)
ROUTE_MATCH = Fraction(1, 2)
INCUMBENT = {"T1": ("A", "pair"), "T2": ("B", "pair"), "T3": ("A", "pair")}
PREFERRED = {"T1": "A", "T2": "B", "T3": "A"}


def type_at(t: int) -> str:
    return TYPES[(t - 1) % 3]


def other_server(server: str) -> str:
    return "B" if server == "A" else "A"


def other_mode(mode: str) -> str:
    return "single" if mode == "pair" else "pair"


def action_id(action: tuple[str, str]) -> str:
    return f"{action[0]}|{action[1]}"


def frac(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def tenths(value: Fraction) -> int:
    scaled = value * 10
    if scaled.denominator != 1:
        raise AssertionError(f"not an integer tenth: {value}")
    return scaled.numerator


def additive_reward(
    t: int,
    typ: str,
    action: tuple[str, str],
    *,
    pair_after: Fraction,
    preferred_after: dict[str, str],
) -> Fraction:
    server, mode = action
    if t < CHANGE_AT:
        preferred = PREFERRED[typ]
        pair_value = PRE_PAIR
    else:
        preferred = preferred_after[typ]
        pair_value = pair_after
    route = ROUTE_MATCH if server == preferred else Fraction(0)
    batch = pair_value if mode == "pair" else Fraction(0)
    return route + batch


TRAP_POST = {
    ("T1", "A", "pair"): Fraction(1, 10),
    ("T1", "B", "pair"): Fraction(0),
    ("T1", "A", "single"): Fraction(1, 2),
    ("T1", "B", "single"): Fraction(0),
    ("T2", "A", "pair"): Fraction(4, 5),
    ("T2", "B", "pair"): Fraction(1, 10),
    ("T2", "A", "single"): Fraction(1, 5),
    ("T2", "B", "single"): Fraction(0),
    ("T3", "A", "pair"): Fraction(4, 5),
    ("T3", "B", "pair"): Fraction(3, 10),
    ("T3", "A", "single"): Fraction(1, 2),
    ("T3", "B", "single"): Fraction(0),
}


def trap_reward(t: int, typ: str, action: tuple[str, str]) -> Fraction:
    if t < CHANGE_AT:
        return additive_reward(
            t, typ, action, pair_after=PRE_PAIR, preferred_after=PREFERRED
        )
    return TRAP_POST[(typ, action[0], action[1])]


RewardFn = Callable[[int, str, tuple[str, str]], Fraction]
ActionFn = Callable[[int, str], tuple[str, str]]


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    reward: RewardFn
    oracle_action: ActionFn
    wrong_action: ActionFn


def make_cases() -> dict[str, Case]:
    shared_pref = dict(PREFERRED)
    local_pref = {"T1": "B", "T2": "B", "T3": "A"}

    def shared_reward(t: int, typ: str, action: tuple[str, str]) -> Fraction:
        return additive_reward(
            t, typ, action, pair_after=Fraction(-2, 5), preferred_after=shared_pref
        )

    def local_reward(t: int, typ: str, action: tuple[str, str]) -> Fraction:
        return additive_reward(
            t, typ, action, pair_after=PRE_PAIR, preferred_after=local_pref
        )

    def stable_reward(t: int, typ: str, action: tuple[str, str]) -> Fraction:
        return additive_reward(
            t, typ, action, pair_after=PRE_PAIR, preferred_after=shared_pref
        )

    def incumbent_single(_t: int, typ: str) -> tuple[str, str]:
        return (INCUMBENT[typ][0], "single")

    def flipped_server(_t: int, typ: str) -> tuple[str, str]:
        return (other_server(INCUMBENT[typ][0]), "pair")

    return {
        "shared_batch": Case(
            name="shared_batch",
            description=(
                "From t=13 the shared pair effect changes from +3/10 to -2/5. "
                "Preferred servers stay T1=A, T2=B, T3=A."
            ),
            reward=shared_reward,
            oracle_action=incumbent_single,
            wrong_action=flipped_server,
        ),
        "type_local_route": Case(
            name="type_local_route",
            description=(
                "From t=13 only T1's preferred server changes from A to B. "
                "The pair effect stays +3/10."
            ),
            reward=local_reward,
            oracle_action=lambda _t, typ: (
                ("B", "pair") if typ == "T1" else INCUMBENT[typ]
            ),
            wrong_action=incumbent_single,
        ),
        "stable": Case(
            name="stable",
            description="No hidden change. Pair effect stays +3/10 and routes stay put.",
            reward=stable_reward,
            oracle_action=lambda _t, typ: INCUMBENT[typ],
            wrong_action=incumbent_single,
        ),
        "nonadditive_trap": Case(
            name="nonadditive_trap",
            description=(
                "Researcher-built counterexample. From t=13, T1 is repaired by "
                "single on A, while T2 is repaired by switching to A and keeping "
                "pair. Transferring the T1 mode conclusion to T2 is harmful. "
                "T3 is unchanged."
            ),
            reward=trap_reward,
            oracle_action=lambda _t, typ: (
                ("A", "single") if typ == "T1" else ("A", "pair")
            ),
            wrong_action=incumbent_single,
        ),
    }


@dataclass
class Step:
    t: int
    request_type: str
    action: tuple[str, str]
    reward: Fraction
    strategy: str
    case: str

    def to_json(self) -> dict[str, object]:
        return {
            "t": self.t,
            "request_type": self.request_type,
            "action": action_id(self.action),
            "reward": frac(self.reward),
            "reward_tenths": tenths(self.reward),
            "runtime_observation": (
                f"request_type={self.request_type}, action={action_id(self.action)}, "
                f"reward={frac(self.reward)}"
            ),
        }


def window_sum(steps: list[Step], start: int, end: int) -> Fraction:
    return sum((step.reward for step in steps if start <= step.t <= end), Fraction(0))


class FeedbackStrategy:
    """Uses request type and past scalar rewards. No hidden case table."""

    def act(self, t: int, typ: str) -> tuple[str, str]:
        raise NotImplementedError

    def observe(self, t: int, typ: str, action: tuple[str, str], reward: Fraction) -> None:
        raise NotImplementedError


def _last_reward(history: list[Step], typ: str, start: int, end: int) -> Fraction:
    matches = [step.reward for step in history if step.request_type == typ and start <= step.t <= end]
    if not matches:
        raise AssertionError(f"missing history for {typ} in {start}..{end}")
    return matches[-1]


class Keep(FeedbackStrategy):
    def act(self, t: int, typ: str) -> tuple[str, str]:
        return INCUMBENT[typ]

    def observe(self, t: int, typ: str, action: tuple[str, str], reward: Fraction) -> None:
        return None


class UnnecessaryProbe(FeedbackStrategy):
    """Stable-control illustration: one mode probe, then revert. Feedback only."""

    def act(self, t: int, typ: str) -> tuple[str, str]:
        if t == RESPONSE_AT:
            return (INCUMBENT[typ][0], "single")
        return INCUMBENT[typ]

    def observe(self, t: int, typ: str, action: tuple[str, str], reward: Fraction) -> None:
        return None


class CoordinateBase(FeedbackStrategy):
    def __init__(self, history: list[Step], *, explore_all_types: bool) -> None:
        self.stale = {typ: _last_reward(history, typ, CHANGE_AT, RESPONSE_AT - 1) for typ in TYPES}
        self.pre = {typ: _last_reward(history, typ, 1, CHANGE_AT - 1) for typ in TYPES}
        self.dropped = [
            typ for typ in TYPES if self.stale[typ] <= self.pre[typ] - GAP
        ]
        targets = list(TYPES) if explore_all_types else list(self.dropped)
        self.pending = {typ: [] for typ in TYPES}
        for typ in targets:
            server, mode = INCUMBENT[typ]
            self.pending[typ] = [
                (other_server(server), mode),
                (server, other_mode(mode)),
                (other_server(server), other_mode(mode)),
            ]
        self.deployed = dict(INCUMBENT)
        self._probing: str | None = None

    def act(self, t: int, typ: str) -> tuple[str, str]:
        if self.pending[typ]:
            self._probing = typ
            return self.pending[typ][0]
        self._probing = None
        return self.deployed[typ]

    def observe(self, t: int, typ: str, action: tuple[str, str], reward: Fraction) -> None:
        if self._probing != typ:
            return
        self.pending[typ].pop(0)
        if reward > self.stale[typ]:
            self.deployed[typ] = action
            self.pending[typ] = []
        self._probing = None


class PerTypeStop(CoordinateBase):
    def __init__(self, history: list[Step]) -> None:
        super().__init__(history, explore_all_types=False)


class WholeReset(CoordinateBase):
    def __init__(self, history: list[Step]) -> None:
        super().__init__(history, explore_all_types=True)


class OneTypeThenTransfer(FeedbackStrategy):
    """Probe the first dropped type, route first. Transfer mode only after a gain.

    Other dropped types keep the incumbent action until that transfer fires.
    A server improvement is kept for the probed type only and is not copied.
    """

    def __init__(self, history: list[Step]) -> None:
        self.stale = {typ: _last_reward(history, typ, CHANGE_AT, RESPONSE_AT - 1) for typ in TYPES}
        self.pre = {typ: _last_reward(history, typ, 1, CHANGE_AT - 1) for typ in TYPES}
        self.dropped = [
            typ for typ in TYPES if self.stale[typ] <= self.pre[typ] - GAP
        ]
        self.exp_i = 0
        self.stage = 0
        self.deployed = dict(INCUMBENT)
        self.transferred: str | None = None
        self._probe: str | None = None

    def act(self, t: int, typ: str) -> tuple[str, str]:
        self._probe = None
        if self.transferred is not None and typ in self.dropped:
            return (INCUMBENT[typ][0], self.transferred)
        if self.exp_i >= len(self.dropped):
            return self.deployed[typ]
        exp = self.dropped[self.exp_i]
        if typ != exp:
            return self.deployed[typ]
        server, mode = INCUMBENT[exp]
        if self.stage == 0:
            action = (other_server(server), mode)
        elif self.stage == 1:
            action = (server, other_mode(mode))
        else:
            action = (other_server(server), other_mode(mode))
        self._probe = exp
        return action

    def observe(self, t: int, typ: str, action: tuple[str, str], reward: Fraction) -> None:
        if self._probe is None or typ != self._probe:
            return
        exp = self._probe
        improved = reward > self.stale[exp]
        if self.stage == 0 and improved:
            self.deployed[exp] = action
            self.exp_i += 1
            self.stage = 0
        elif self.stage == 0:
            self.stage = 1
        elif self.stage == 1 and improved:
            self.transferred = action[1]
            for dropped in self.dropped:
                self.deployed[dropped] = (INCUMBENT[dropped][0], self.transferred)
            self.exp_i = len(self.dropped)
        elif self.stage == 1:
            self.stage = 2
        elif self.stage == 2 and improved:
            self.deployed[exp] = action
            self.exp_i += 1
            self.stage = 0
        else:
            self.exp_i += 1
            self.stage = 0
        self._probe = None


def run_case(case: Case, strategy_name: str) -> list[Step]:
    steps: list[Step] = []
    learner: FeedbackStrategy | None = None
    for t in range(1, HORIZON + 1):
        typ = type_at(t)
        if t < RESPONSE_AT or strategy_name == "keep":
            action = INCUMBENT[typ]
        elif strategy_name == "oracle_repair":
            action = case.oracle_action(t, typ)
        elif strategy_name == "wrong_factor":
            action = case.wrong_action(t, typ)
        elif strategy_name == "unnecessary_probe":
            if learner is None:
                learner = UnnecessaryProbe()
            action = learner.act(t, typ)
        else:
            if learner is None:
                if strategy_name == "per_type_stop":
                    learner = PerTypeStop(steps)
                elif strategy_name == "whole_reset":
                    learner = WholeReset(steps)
                elif strategy_name == "one_type_then_transfer":
                    learner = OneTypeThenTransfer(steps)
                else:
                    raise AssertionError(strategy_name)
            action = learner.act(t, typ)
        reward = case.reward(t, typ, action)
        step = Step(t, typ, action, reward, strategy_name, case.name)
        steps.append(step)
        if learner is not None and t >= RESPONSE_AT:
            learner.observe(t, typ, action, reward)
    return steps


STRATEGIES = (
    "keep",
    "oracle_repair",
    "wrong_factor",
    "unnecessary_probe",
    "per_type_stop",
    "one_type_then_transfer",
    "whole_reset",
)

def summarize(steps: list[Step]) -> dict[str, object]:
    acquire = window_sum(steps, 1, CHANGE_AT - 1)
    stale = window_sum(steps, CHANGE_AT, RESPONSE_AT - 1)
    response = window_sum(steps, RESPONSE_AT, HORIZON)
    lifetime = acquire + stale + response
    return {
        "acquire_t1_12": frac(acquire),
        "stale_t13_18": frac(stale),
        "response_t19_36": frac(response),
        "lifetime_t1_36": frac(lifetime),
        "response_tenths": tenths(response),
        "lifetime_tenths": tenths(lifetime),
    }


def non_incumbent_server_ticks(steps: list[Step]) -> dict[str, int]:
    counts = {typ: 0 for typ in TYPES}
    for step in steps:
        if step.t < RESPONSE_AT:
            continue
        if step.action[0] != INCUMBENT[step.request_type][0]:
            counts[step.request_type] += 1
    return counts


def annotations_for_worked_trajectory() -> dict[int, str]:
    repaired = (
        "Researcher annotation: hidden pair effect is still -2/5 and the server "
        "still matches. Runtime observation remains the scalar only. Original "
        "servers stay in use."
    )
    notes = {
        1: (
            "Researcher annotation: the witness starts from an already chosen "
            "procedure (preferred server, pair), not from a learner discovering it. "
            "Hidden reward is 1/2 + 3/10. The runtime observation has no split."
        ),
        12: (
            "Researcher annotation: last tick before the hidden change. No boundary "
            "signal is emitted. The action at t=13 cannot use rewards from t=13 onward."
        ),
        13: (
            "Researcher annotation: same action as acquisition, hidden pair effect "
            "now -2/5, so the scalar falls from 4/5 to 1/10. The observation has "
            "request type, action, and scalar reward. It has no regime id and no "
            "factor split."
        ),
        18: (
            "Researcher annotation: every exploited cell now pays 1/10. That drop "
            "of 7/10 fits both a pair-effect change and a route-bonus change. "
            "The trajectory has not separated them."
        ),
        19: (
            "Researcher annotation: route-first probe. Hidden value is 0 + (-2/5). "
            "This tick is worse than both the stale action (1/10) and the untried "
            "single action (1/2). The agent is not told which fact failed."
        ),
        20: (
            "Researcher annotation: T2 is also dropped, but the rule waits for "
            "another T1 arrival before trying mode. T2 keeps the stale action."
        ),
        22: (
            "Researcher annotation: mode probe on the original server improves on "
            "the stale scalar, so the hand rule copies single onto every dropped "
            "type. That copy is a hypothesis. It is true in this case and false "
            "in nonadditive_trap."
        ),
    }
    for t in range(23, HORIZON + 1):
        notes[t] = repaired
    return notes


def confound_predictions() -> dict[str, object]:
    """Two explanations of the shared-case exploited cells, plus the truth.

    Both explanations match the observed incumbent reward 1/10. They disagree
    on actions the agent has not tried.
    """
    observed = Fraction(1, 10)
    batch_explanation = {
        "story": "Routes unchanged. Pair effect fell from 3/10 to -2/5.",
        "matches_exploited_cell": True,
        "predict_original_server_single": frac(ROUTE_MATCH),
        "predict_other_server_pair": frac(Fraction(-2, 5)),
    }
    route_explanation = {
        "story": "Pair effect stays 3/10. Match bonus fell by 7/10, from 1/2 to -1/5.",
        "matches_exploited_cell": True,
        "predict_original_server_single": frac(Fraction(-1, 5)),
        "predict_other_server_pair": frac(Fraction(3, 10)),
    }
    truth = {
        "story": "Hidden shared_batch parameters.",
        "original_server_single": frac(ROUTE_MATCH),
        "other_server_pair": frac(Fraction(-2, 5)),
        "exploited_cell": frac(observed),
    }
    if ROUTE_MATCH + Fraction(-2, 5) != observed:
        raise AssertionError("truth does not fit the exploited cell")
    if Fraction(-1, 5) + PRE_PAIR != observed:
        raise AssertionError("route explanation does not fit the exploited cell")
    return {
        "exploited_observation": frac(observed),
        "batch_explanation": batch_explanation,
        "route_explanation": route_explanation,
        "truth_hidden": truth,
    }


def equal_mixture(results: dict[str, dict[str, dict[str, object]]]) -> dict[str, str]:
    case_names = ("shared_batch", "type_local_route", "stable", "nonadditive_trap")
    mixture: dict[str, str] = {}
    for strategy in STRATEGIES:
        total = Fraction(0)
        for name in case_names:
            text = str(results[name][strategy]["response_t19_36"])
            num, den = text.split("/")
            total += Fraction(int(num), int(den))
        mixture[strategy] = frac(total / 4)
    return mixture


def preservation(runs: dict[str, dict[str, list[Step]]]) -> dict[str, dict[str, dict[str, int]]]:
    report: dict[str, dict[str, dict[str, int]]] = {}
    for case_name, strategies in runs.items():
        report[case_name] = {
            strategy: non_incumbent_server_ticks(steps)
            for strategy, steps in strategies.items()
        }
    return report


def assert_expected(results: dict[str, dict[str, dict[str, object]]]) -> None:
    expected = {
        "shared_batch": {
            "keep": "9/5",
            "oracle_repair": "9/1",
            "wrong_factor": "-36/5",
            "per_type_stop": "63/10",
            "one_type_then_transfer": "73/10",
            "whole_reset": "63/10",
        },
        "type_local_route": {
            "keep": "57/5",
            "oracle_repair": "72/5",
            "wrong_factor": "6/1",
            "per_type_stop": "72/5",
            "one_type_then_transfer": "72/5",
            "whole_reset": "56/5",
        },
        "stable": {
            "keep": "72/5",
            "oracle_repair": "72/5",
            "wrong_factor": "9/1",
            "unnecessary_probe": "141/10",
            "per_type_stop": "72/5",
            "one_type_then_transfer": "72/5",
            "whole_reset": "48/5",
        },
        "nonadditive_trap": {
            "keep": "6/1",
            "oracle_repair": "63/5",
            "wrong_factor": "6/1",
            "per_type_stop": "121/10",
            "one_type_then_transfer": "37/5",
        },
    }
    for case_name, strategies in expected.items():
        for strategy, response in strategies.items():
            got = results[case_name][strategy]["response_t19_36"]
            if got != response:
                raise AssertionError(f"{case_name} {strategy}: {got} != {response}")
        for strategy in STRATEGIES:
            if results[case_name][strategy]["acquire_t1_12"] != "48/5":
                raise AssertionError(f"acquisition drifted for {case_name} {strategy}")


def assert_mixture(mixture: dict[str, str]) -> None:
    if mixture["per_type_stop"] != "59/5":
        raise AssertionError(mixture["per_type_stop"])
    if mixture["one_type_then_transfer"] != "87/8":
        raise AssertionError(mixture["one_type_then_transfer"])
    if mixture["keep"] != "42/5":
        raise AssertionError(mixture["keep"])
    # 59/5 = 9. something wait: 59/5 = 11.8, 87/8 = 10.875. Per-type is higher.
    if Fraction(59, 5) <= Fraction(87, 8):
        raise AssertionError("expected per-type to beat transfer on the equal mixture")


def main() -> None:
    cases = make_cases()
    runs: dict[str, dict[str, list[Step]]] = {}
    results: dict[str, dict[str, dict[str, object]]] = {}
    for case in cases.values():
        runs[case.name] = {}
        results[case.name] = {}
        for strategy in STRATEGIES:
            steps = run_case(case, strategy)
            runs[case.name][strategy] = steps
            results[case.name][strategy] = summarize(steps)
            prefix = [(step.request_type, step.action, step.reward) for step in steps if step.t < RESPONSE_AT]
            keep_prefix = [
                (step.request_type, step.action, step.reward)
                for step in runs[case.name].get("keep", steps)
                if step.t < RESPONSE_AT
            ]
            if strategy != "keep" and "keep" in runs[case.name] and prefix != keep_prefix:
                raise AssertionError(f"response rule leaked into the prefix: {case.name} {strategy}")

    assert_expected(results)
    mixture = equal_mixture(results)
    assert_mixture(mixture)
    notes = annotations_for_worked_trajectory()
    trajectory = []
    for step in runs["shared_batch"]["one_type_then_transfer"]:
        row = step.to_json()
        row["researcher_annotation"] = notes.get(step.t, "")
        trajectory.append(row)

    servers_touched = preservation(runs)
    local_transfer = servers_touched["type_local_route"]["one_type_then_transfer"]
    if local_transfer != {"T1": 6, "T2": 0, "T3": 0}:
        raise AssertionError(local_transfer)
    local_per_type = servers_touched["type_local_route"]["per_type_stop"]
    if local_per_type != {"T1": 6, "T2": 0, "T3": 0}:
        raise AssertionError(local_per_type)
    local_reset = servers_touched["type_local_route"]["whole_reset"]
    if local_reset["T2"] != 2 or local_reset["T3"] != 2:
        raise AssertionError(local_reset)
    if servers_touched["stable"]["one_type_then_transfer"] != {"T1": 0, "T2": 0, "T3": 0}:
        raise AssertionError("stable transfer rule should not probe")
    if servers_touched["stable"]["whole_reset"] != {"T1": 2, "T2": 2, "T3": 2}:
        raise AssertionError(servers_touched["stable"]["whole_reset"])
    shared_transfer = servers_touched["shared_batch"]["one_type_then_transfer"]
    if shared_transfer != {"T1": 1, "T2": 0, "T3": 0}:
        raise AssertionError(shared_transfer)

    payload = {
        "label": (
            "Illustrative hand-specified witnesses. Not a learner result, "
            "not a population estimate, and not a method comparison."
        ),
        "clock": "interaction-indexed; one request and one action per tick",
        "reward_unit": "utility point; integer tenths are reward * 10",
        "horizon": HORIZON,
        "change_applies_at": CHANGE_AT,
        "response_starts_at": RESPONSE_AT,
        "gap": frac(GAP),
        "schedule": "T1, T2, T3, repeating; sealed from the learner except for the current request type",
        "cases": {
            case.name: {
                "description": case.description,
                "strategies": results[case.name],
            }
            for case in cases.values()
        },
        "equal_weight_mean_response": mixture,
        "non_incumbent_server_ticks_t19_36": servers_touched,
        "confound_at_t18_shared_batch": confound_predictions(),
        "worked_trajectory_shared_batch_one_type_then_transfer": trajectory,
    }
    out = Path(__file__).with_name("witness.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("response t=19..36")
    header = f"{'case':<20}" + "".join(f"{name[:16]:>16}" for name in STRATEGIES)
    print(header)
    for case_name in cases:
        row = f"{case_name:<20}"
        for strategy in STRATEGIES:
            row += f"{str(results[case_name][strategy]['response_t19_36']):>16}"
        print(row)
    print("equal-weight mean response")
    for strategy, value in mixture.items():
        print(f"  {strategy}: {value}")


if __name__ == "__main__":
    main()
