"""Observation interfaces: raw transcript vs engine-computed sufficient statistics.

Stats are tallies a human could compute from the same (arm, reward) history.
They must not include true p, best_arm, or regime labels.
"""
from __future__ import annotations

import json
from typing import Any

INTERFACE_MODES = ("raw", "stats", "stats_plus_recent")


def render_step_line(rec: dict[str, Any]) -> str:
    step = rec.get("step", "?")
    action = rec.get("action", {})
    reward = rec.get("reward", 0.0)
    obs = rec.get("observation", {}) or {}
    msg = obs.get("message", "")
    return f"Step {step}: action={json.dumps(action)}, reward={reward}, obs={msg}"


def render_history(history: list[dict[str, Any]], tail: int | None = None) -> str:
    if not history:
        return "(no prior steps)"
    rows = history[-tail:] if tail is not None and tail >= 0 else history
    if tail is not None and tail >= 0 and len(history) > tail:
        omitted = len(history) - tail
        prefix = f"(omitted {omitted} earlier steps)\n"
    else:
        prefix = ""
    return prefix + "\n".join(render_step_line(rec) for rec in rows)


def _arm_from_action(action: Any) -> int | None:
    if not isinstance(action, dict) or "arm" not in action:
        return None
    try:
        return int(action["arm"])
    except (TypeError, ValueError):
        return None


def render_bandit_stats(
    history: list[dict[str, Any]],
    n_arms: int,
    last_k: int = 5,
    recency_window: int = 10,
) -> str:
    """Per-arm lifetime mean, recency-window mean, and last-k realized rewards."""
    n_arms = max(int(n_arms), 1)
    last_k = max(int(last_k), 1)
    recency_window = max(int(recency_window), 1)
    counts = [0] * n_arms
    successes = [0.0] * n_arms
    recent: list[list[float]] = [[] for _ in range(n_arms)]
    for rec in history:
        arm = _arm_from_action(rec.get("action"))
        if arm is None or not (0 <= arm < n_arms):
            continue
        try:
            reward = float(rec.get("reward", 0.0))
        except (TypeError, ValueError):
            continue
        counts[arm] += 1
        successes[arm] += reward
        recent[arm].append(reward)

    w = recency_window
    lines = [
        "Per-arm tallies computed from your pulls so far (not hidden environment state).",
        f"mean_all = lifetime mean; mean_w{w} = mean of this arm's most recent {w} pulls.",
        f"{'arm':>4}  {'n':>5}  {'mean_all':>8}  {'n_w':>4}  {'mean_w'+str(w):>8}  last_{last_k}",
    ]
    for i in range(n_arms):
        n = counts[i]
        mean_all = (successes[i] / n) if n else float("nan")
        window = recent[i][-w:]
        n_w = len(window)
        mean_w = (sum(window) / n_w) if n_w else float("nan")
        tail = recent[i][-last_k:]
        tail_s = ",".join(str(int(x) if x in (0.0, 1.0) else round(x, 2)) for x in tail) or "-"
        mean_all_s = f"{mean_all:.3f}" if n else "     n/a"
        mean_w_s = f"{mean_w:.3f}" if n_w else "     n/a"
        lines.append(
            f"{i:4d}  {n:5d}  {mean_all_s:>8}  {n_w:4d}  {mean_w_s:>8}  {tail_s}"
        )
    if not history:
        lines.append("(no pulls yet)")
    return "\n".join(lines)


def render_interface_blocks(
    history: list[dict[str, Any]],
    n_arms: int,
    interface: str = "stats_plus_recent",
    history_tail: int = 10,
    stats_last_k: int = 5,
    recency_window: int = 10,
) -> tuple[str, str]:
    """Return (stats_block, history_block) for the prompt."""
    mode = (interface or "raw").lower().strip()
    if mode not in INTERFACE_MODES:
        raise ValueError(f"Unknown interface={interface!r}; expected {INTERFACE_MODES}")
    if mode == "raw":
        return "(not provided)", render_history(history)
    stats = render_bandit_stats(
        history, n_arms, last_k=stats_last_k, recency_window=recency_window
    )
    if mode == "stats":
        return stats, "(raw step list omitted; use the tallies above)"
    return stats, render_history(history, tail=history_tail)
