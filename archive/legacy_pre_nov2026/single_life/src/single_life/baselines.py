"""Development policies for the causal fixture. Not method evidence.

Sliding-window UCB
------------------
At decision t, let W be the window length. The window is global time, not a
per-arm pull count: the unique observed events whose action_index lies in
[t - W, t - 1]. Duplicate event ids in a supplied prefix are ignored.

Let N be the number of in-window observations of permitted arms and N_a the
number that chose arm a. The mean is the arithmetic mean of their outcomes.

Startup: if any permitted arm has N_a = 0, choose the lexicographically
smallest such arm. This includes arms that have slid out of the window.

Otherwise::

    score(a) = mean(a) + sqrt(2 * ln(N) / N_a)

ln is the natural logarithm. Ties keep the highest score, then the
lexicographically smallest action id. The agent RNG is not consulted.

Random policy
-------------
``rng.choice`` over the sorted permitted action ids. One draw per decision.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Iterable, Sequence

from single_life.contracts import ContractError, ObservedEvent, PublicView

UCB_DEFINITION = {
    "name": "sliding-window-ucb-v1",
    "equation": "score(a) = mean(a) + sqrt(2 * ln(N) / N_a)",
    "log": "natural",
    "window": "global time: action_index in [decision_index - window, decision_index - 1]",
    "startup": (
        "if any permitted arm has zero pulls inside the window, choose the "
        "lexicographically smallest such arm"
    ),
    "ties": "highest score, then lexicographically smallest action id",
    "agent_rng": "unused",
    "counting": "one count per event_id; a repeated full prefix does not add counts",
}


def dedupe_events(events: Iterable[ObservedEvent]) -> list[ObservedEvent]:
    by_id: dict[str, ObservedEvent] = {}
    order: list[str] = []
    for event in events:
        previous = by_id.get(event.event_id)
        if previous is None:
            by_id[event.event_id] = event
            order.append(event.event_id)
            continue
        if previous.to_dict() != event.to_dict():
            raise ContractError(f"conflicting observations for {event.event_id}")
    return [by_id[event_id] for event_id in order]


def window_events(
    events: Sequence[ObservedEvent], decision_index: int, window: int
) -> list[ObservedEvent]:
    if window < 1:
        raise ContractError("window must be >= 1")
    unique = dedupe_events(events)
    lower = decision_index - window
    upper = decision_index - 1
    return [event for event in unique if lower <= event.action_index <= upper]


def choose_ucb(
    events: Sequence[ObservedEvent],
    decision_index: int,
    window: int,
    permitted_action_ids: Sequence[str],
) -> str:
    permitted = list(permitted_action_ids)
    if not permitted:
        raise ContractError("no permitted actions")
    selected = window_events(events, decision_index, window)
    counts = {action_id: 0 for action_id in permitted}
    reward_sum = {action_id: 0.0 for action_id in permitted}
    for event in selected:
        if event.chosen_action not in counts:
            continue
        counts[event.chosen_action] += 1
        reward_sum[event.chosen_action] += float(event.observed_outcome)
    unpulled = sorted(action_id for action_id in permitted if counts[action_id] == 0)
    if unpulled:
        return unpulled[0]
    total = sum(counts.values())
    if total < 1:
        return sorted(permitted)[0]

    def score(action_id: str) -> float:
        mean = reward_sum[action_id] / counts[action_id]
        bonus = math.sqrt((2.0 * math.log(total)) / counts[action_id])
        return mean + bonus

    best = max(score(action_id) for action_id in permitted)
    tied = sorted(action_id for action_id in permitted if score(action_id) == best)
    return tied[0]


class RandomAgent:
    name = "random-v1"

    def select(self, view: PublicView, rng: random.Random) -> str:
        return rng.choice(tuple(sorted(view.permitted_action_ids)))

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, data: dict) -> None:
        if data != {}:
            raise ContractError("RandomAgent state must be empty")


class SlidingWindowUCB:
    name = "sliding-window-ucb-v1"

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ContractError("window must be >= 1")
        self.window = window
        self._by_id: dict[str, ObservedEvent] = {}
        self._order: list[ObservedEvent] = []

    def observe(self, events: Iterable[ObservedEvent]) -> None:
        for event in dedupe_events(events):
            previous = self._by_id.get(event.event_id)
            if previous is None:
                self._by_id[event.event_id] = event
                self._order.append(event)
                continue
            if previous.to_dict() != event.to_dict():
                raise ContractError(f"conflicting observations for {event.event_id}")

    @property
    def unique_count(self) -> int:
        return len(self._order)

    def counts_by_arm(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._order:
            counts[event.chosen_action] = counts.get(event.chosen_action, 0) + 1
        return counts

    def select(self, view: PublicView, rng: random.Random) -> str:
        del rng
        self.observe(view.observed_events)
        return choose_ucb(self._order, view.decision_index, self.window, view.permitted_action_ids)

    def state_dict(self) -> dict:
        return {"window": self.window, "events": [event.to_dict() for event in self._order]}

    def load_state_dict(self, data: dict) -> None:
        if not isinstance(data, dict) or set(data) != {"events", "window"}:
            raise ContractError("SlidingWindowUCB state is incomplete")
        if data["window"] != self.window:
            raise ContractError("SlidingWindowUCB window does not match checkpoint")
        events = data["events"]
        if not isinstance(events, list):
            raise ContractError("SlidingWindowUCB events must be a list")
        self._by_id = {}
        self._order = []
        self.observe(ObservedEvent.from_dict(item) for item in events)


class ArtifactConfigAgent:
    """Uses the newest deployed public artifact. Fallback is the smallest action id.

    This is a scripted interface check that the deployed content changes the
    action. It is not an improver.
    """

    name = "scripted-artifact-agent-v1"

    def select(self, view: PublicView, rng: random.Random) -> str:
        del rng
        permitted = list(view.permitted_action_ids)
        artifact = _latest_deployed(view)
        if artifact is None:
            return sorted(permitted)[0]
        means = artifact.content.get("arm_means")
        if not isinstance(means, Mapping) or isinstance(means, (str, bytes)):
            return sorted(permitted)[0]
        candidates = [action_id for action_id in permitted if action_id in means]
        if not candidates:
            return sorted(permitted)[0]
        best = max(float(means[action_id]) for action_id in candidates)
        tied = sorted(action_id for action_id in candidates if float(means[action_id]) == best)
        return tied[0]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, data: dict) -> None:
        if data != {}:
            raise ContractError("ArtifactConfigAgent state must be empty")


def _latest_deployed(view: PublicView):
    deployed = [
        artifact
        for artifact in view.accessible_artifacts
        if artifact.deployment_decision_index is not None
        and artifact.deployment_decision_index <= view.decision_index
        and artifact.creation_cutoff < view.decision_index
    ]
    if not deployed:
        return None
    return max(
        deployed,
        key=lambda artifact: (
            artifact.deployment_decision_index or 0,
            artifact.creation_cutoff,
            artifact.version,
        ),
    )


def make_policy(agent_name: str, ucb_window: int):
    if agent_name == "random":
        return RandomAgent()
    if agent_name == "ucb":
        return SlidingWindowUCB(window=ucb_window)
    if agent_name == "artifact":
        return ArtifactConfigAgent()
    if agent_name in {"llm-recent", "llm-note"}:
        raise ContractError(
            "llm policies are supplied by the pilot and are not checkpointed"
        )
    raise ContractError(f"unknown agent {agent_name}")
