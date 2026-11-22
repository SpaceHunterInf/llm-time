"""Observation interfaces: raw transcript vs engine-computed sufficient statistics.

Stats are tallies a human could compute from the same (arm, reward) history.
They must not include true p, best_arm, or regime labels.
"""
from __future__ import annotations

import json
import math
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


def _reward_as_success(reward: Any) -> float | None:
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return None
    if value in (0.0, 1.0):
        return value
    return 1.0 if value > 0.5 else 0.0


def last_declared_best_arm(history: list[dict[str, Any]]) -> int | None:
    for rec in reversed(history):
        parsed = rec.get("parsed_response") or {}
        hyp = parsed.get("hypothesis") if isinstance(parsed, dict) else None
        if not isinstance(hyp, dict) or "best_arm" not in hyp:
            continue
        try:
            return int(hyp["best_arm"])
        except (TypeError, ValueError):
            continue
    return None


def binomial_cdf_le(successes: int, n: int, p: float) -> float:
    """P(X <= successes) for X ~ Binomial(n, p)."""
    if n <= 0:
        return float("nan")
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    s = max(0, min(int(successes), n))
    q = 1.0 - p
    total = 0.0
    for k in range(s + 1):
        total += math.comb(n, k) * (p**k) * (q ** (n - k))
    return min(1.0, max(0.0, total))


def _fmt_p(p: float) -> str:
    if p != p:  # NaN
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def render_hypothesis_check(
    history: list[dict[str, Any]],
    last_k: int = 5,
    n_arms: int | None = None,
) -> str:
    """How surprising the last-k pulls of the declared H arm are under its lifetime mean.

    Uses only (arm, reward) plus the agent's last declared best_arm. No hidden p.
    One-sided: P(this many successes or fewer | Bernoulli(lifetime mean)).
    """
    last_k = max(int(last_k), 1)
    arm = last_declared_best_arm(history)
    if arm is None:
        return "(no prior declared hypothesis)"
    if n_arms is not None and not (0 <= arm < int(n_arms)):
        return f"(last declared best_arm={arm} is outside the action space)"

    pulls: list[float] = []
    for rec in history:
        pulled = _arm_from_action(rec.get("action"))
        if pulled != arm:
            continue
        success = _reward_as_success(rec.get("reward"))
        if success is None:
            continue
        pulls.append(success)

    n_all = len(pulls)
    if n_all == 0:
        return f"last hypothesis: arm {arm}\n(no pulls of that arm yet)"

    p_hat = sum(pulls) / n_all
    recent = pulls[-last_k:]
    n_rec = len(recent)
    s_rec = int(round(sum(recent)))
    p_value = binomial_cdf_le(s_rec, n_rec, p_hat)
    tail = ",".join(str(int(x)) for x in recent)
    return "\n".join(
        [
            f"last hypothesis: arm {arm}",
            f"p_hat (lifetime mean of that arm): {p_hat:.3f}  (n={n_all})",
            f"last {n_rec} pulls of that arm: {tail}  ({s_rec}/{n_rec} successes)",
            (
                "P(this many successes or fewer | Bernoulli(p_hat)): "
                f"{_fmt_p(p_value)}"
            ),
            "A small value means those recent pulls are surprising under the lifetime mean.",
        ]
    )


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
