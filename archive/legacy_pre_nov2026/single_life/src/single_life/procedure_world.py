"""Private route/mode tables for the task 006 development slice.

The numbers follow the task 004 additive and trap tables, with the switch at
action 43 and a horizon of 96. This module is evaluator-only. The prompt
builder does not import it. No incumbent, change detector, or boundary policy
from the task 004 witness is defined here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TYPES = ("T1", "T2", "T3")
ACTIONS = ("A|single", "A|pair", "B|single", "B|pair")
HORIZON = 96
CHANGE_AT = 43
SENTINEL = "SENTINEL_PRIVATE_do_not_leak 006"
ROUTE_TENTHS = 5
PAIR_PRE_TENTHS = 3
PAIR_SHARED_POST_TENTHS = -4
PREFERRED_PRE = {"T1": "A", "T2": "B", "T3": "A"}
PREFERRED_ROUTE_POST = {"T1": "B", "T2": "B", "T3": "A"}
WORLD_IDS = ("stable", "shared", "route", "trap")
TRAP_POST = {
    "T1": {"A|pair": 1, "B|pair": 0, "A|single": 5, "B|single": 0},
    "T2": {"A|pair": 8, "B|pair": 1, "A|single": 2, "B|single": 0},
    "T3": {"A|pair": 8, "B|pair": 3, "A|single": 5, "B|single": 0},
}


def additive_table(preferred: dict[str, str], pair_tenths: int) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for typ in TYPES:
        cells: dict[str, int] = {}
        for action in ACTIONS:
            server, mode = action.split("|")
            route = ROUTE_TENTHS if server == preferred[typ] else 0
            batch = pair_tenths if mode == "pair" else 0
            cells[action] = route + batch
        table[typ] = cells
    return table


def formula_worlds() -> dict[str, dict[str, Any]]:
    def pre() -> dict[str, dict[str, int]]:
        return additive_table(PREFERRED_PRE, PAIR_PRE_TENTHS)

    return {
        "stable": {
            "switches": False,
            "description": "Pair effect stays +3 tenths and preferred servers stay T1=A, T2=B, T3=A.",
            "pre": pre(),
            "post": additive_table(PREFERRED_PRE, PAIR_PRE_TENTHS),
        },
        "shared": {
            "switches": True,
            "description": "At action 43 the pair effect becomes -4 tenths. Preferred servers stay put.",
            "pre": pre(),
            "post": additive_table(PREFERRED_PRE, PAIR_SHARED_POST_TENTHS),
        },
        "route": {
            "switches": True,
            "description": "At action 43 only T1's preferred server becomes B. The pair effect stays +3 tenths.",
            "pre": pre(),
            "post": additive_table(PREFERRED_ROUTE_POST, PAIR_PRE_TENTHS),
        },
        "trap": {
            "switches": True,
            "description": (
                "At action 43 the post table is non-additive. T1 is best as A|single; "
                "T2 is best as A|pair; T3 stays A|pair."
            ),
            "pre": pre(),
            "post": {typ: dict(cells) for typ, cells in TRAP_POST.items()},
        },
    }


def config_document() -> dict[str, Any]:
    worlds = {}
    for name, spec in formula_worlds().items():
        worlds[name] = {
            "switches": spec["switches"],
            "description": spec["description"],
            "pre": spec["pre"],
            "post": spec["post"],
        }
    return {
        "sentinel": SENTINEL,
        "task": "006",
        "horizon": HORIZON,
        "change_at_action": CHANGE_AT,
        "utility_unit": "tenths of a utility point",
        "types": list(TYPES),
        "actions": list(ACTIONS),
        "type_cycle": "T1, T2, T3, repeating, one type before each action",
        "worlds": worlds,
        "note": (
            "Evaluator-only. Do not copy this file into a model prompt. "
            "change_at_action is the first action scored on the post table."
        ),
    }


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data != config_document():
        raise ValueError(f"{path} does not match the task 006 formula tables")
    return data


def utility_tenths(world: str, action_index: int, typ: str, action: str) -> int:
    spec = formula_worlds()[world]
    if action_index < 1:
        raise ValueError("action_index starts at 1")
    table = spec["pre"]
    if spec["switches"] and action_index >= CHANGE_AT:
        table = spec["post"]
    return int(table[typ][action])


def table_phase(world: str, action_index: int) -> str:
    spec = formula_worlds()[world]
    if spec["switches"] and action_index >= CHANGE_AT:
        return "post"
    return "pre"


def pre_table() -> dict[str, dict[str, int]]:
    return formula_worlds()["stable"]["pre"]


def uniform_mean(table: dict[str, dict[str, int]], typ: str) -> tuple[int, int]:
    total = sum(table[typ][action] for action in ACTIONS)
    return total, len(ACTIONS)


def optimum_cell(table: dict[str, dict[str, int]], typ: str) -> tuple[str, int]:
    best_action = ACTIONS[0]
    best_value = table[typ][best_action]
    for action in ACTIONS[1:]:
        value = table[typ][action]
        if value > best_value:
            best_action = action
            best_value = value
    return best_action, best_value


def evaluator_cycle_means(procedure: dict[str, list[str]]) -> dict[str, Any]:
    """Mean utility of one pass through each type list on the pre-change table.

    This is an after-the-run description. It must not be shown to the model or
    used to accept a deployment.
    """

    table = pre_table()
    per_type = {}
    for typ in TYPES:
        actions = list(procedure[typ])
        total = sum(table[typ][action] for action in actions)
        uniform_sum, uniform_count = uniform_mean(table, typ)
        best_action, best_value = optimum_cell(table, typ)
        per_type[typ] = {
            "listed_actions": actions,
            "cycle_sum_tenths": total,
            "cycle_count": len(actions),
            "cycle_mean_tenths": f"{total}/{len(actions)}",
            "uniform_mean_tenths": f"{uniform_sum}/{uniform_count}",
            "optimum_action": best_action,
            "optimum_tenths": best_value,
        }
    return {
        "label": "evaluator_only",
        "table": "pre-change",
        "per_type": per_type,
        "note": "Not supplied to the model and not used to choose a deployment.",
    }
