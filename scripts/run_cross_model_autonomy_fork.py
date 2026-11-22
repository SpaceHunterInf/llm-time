#!/usr/bin/env python3
"""Cross-Model Epistemic Autonomy vs. Controller Scaffolding Fork (`AlphaDesk-v0.3`).

Evaluates frozen models across distinct model families (`gemini-3.8-flash` vs. `gemma-4-31b-it`)
from an IDENTICAL acquired `W1` public trajectory & portfolio holdings `w_{t*-1}` across 3 arms:
  1. `scoped_cards_v3` (`ScopedCardBook [Baseline]`): Self-evolving structured scope cards (`reflect_every=3`).
  2. `factual_ledger_v3` (`Factual-Ledger [Ours — Autonomy Arm]`): Objective deterministic record of past
     public sessions & per-strategy probe/holding statistics (`0` LLM curator calls, `0` action recommendations).
  3. `sgeh_directive_v3` (`Directive-SGEH [Ours — Controller Arm]`): Exact superset = `Factual-Ledger` +
     `SGEH` explicit controller prescriptions (probe target, yield threshold, and recommended weights).

Tested across 5 shared-prefix post-fork worlds (`t_fork = 14`, `H_post = 16`, `N = 6` Latin-square seeds `501..506`):
  - `stationary` (`high_a` prefix -> `A` unchanged)
  - `abrupt_reversal` (`high_a` prefix -> `B` reversal, apparent loss `86.6 -> 18.9` under `orderly_normal`)
  - `freeze_matched_loss` (`high_a` prefix -> 2-day `acute_liquidity_freeze` at `19.0` matching `abrupt_reversal`
    apparent loss, then `14` sessions returning to `A`; Bellman DP optimal policy holds `current_holdings` with `0` churn!)
  - `opportunity_stationary` (`opp_a` prefix -> `A_opp` unchanged at `~61.3`)
  - `opportunity_rise` (`opp_a` prefix -> Type-II silent rise of 2 unprobed arms to `96 / 86` while `64 / 58` stay flat)

Strict Autonomy Enforcement:
  - ZERO silent near-compliant weight auto-repair (`0.65` weight breaches score `0.0` as true compliance violations).
  - Explicit tracking of `runner_interventions`, `compliance_violations`, `first_probe_was_untested`,
    `freeze_probes`, `freeze_turnover_pts`, and provider `model_version`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v3 import AlphaDeskV3ForkEnv, STRATEGIES_V3, l1_turnover_half  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.sgeh_v3 import (  # noqa: E402
    DirectiveSGEHV3Harness,
    FactualLedgerV3Harness,
    ScopedCardBookV3Harness,
    normalize_weights_v3,
)

DEFAULT_MODELS = ("gemini-3.8-flash", "gemma-4-31b-it")
DEFAULT_ARMS = ("scoped_cards_v3", "factual_ledger_v3", "sgeh_directive_v3")
DEFAULT_FORK_MODES = (
    "stationary",
    "abrupt_reversal",
    "freeze_matched_loss",
    "opportunity_stationary",
    "opportunity_rise",
)


def _prefix_family_for_mode(fork_mode: str) -> str:
    return "opp_a" if fork_mode in ("opportunity_stationary", "opportunity_rise") else "high_a"


def _build_acting_system_prompt(domain_prompt: str, tool_schemas: list[dict[str, Any]], harness_ctx: str) -> str:
    tools_desc = "\n".join(
        f"- `{t['name']}`: {t['description']} | args: {json.dumps(t['args'])}"
        for t in tool_schemas
    )
    return (
        f"{domain_prompt}\n"
        + (f"\n{harness_ctx}\n" if harness_ctx.strip() else "")
        + "\nAVAILABLE TOOLS (call ONE tool per turn as strict JSON):\n"
        + tools_desc
        + '\n\nRESPONSE FORMAT (strict JSON object ONLY): {"tool": "<name>", "args": {...}, "rationale": "<1-sentence reason>"}'
    )


def generate_shared_w1_acquisition(
    *,
    seed: int,
    prefix_family: str,
    t_fork: int = 14,
    h_post: int = 16,
) -> dict[str, Any]:
    """Generate a realistic, deterministic `W1` (`0 .. t_fork-1`) public trajectory shared across ALL models & arms.

    In `W1`, the acquisition policy inspects every brief, probes 3 calm strategies (`r0`, `r2`, `r1` — leaving
    3 calm strategies `r3, r4, r5` completely unprobed in calm sessions!) and 2 elevated-vol strategies (`r5, r4`),
    and converges to `0.55 r0 + 0.45 r1` in calm and `0.55 r5 + 0.45 r4` in elevated volatility.
    """
    base_mode = "opportunity_stationary" if prefix_family == "opp_a" else "stationary"
    env = AlphaDeskV3ForkEnv(fork_mode=base_mode, seed=seed, t_fork=t_fork, h_post=h_post)
    rmap = env._role_to_strat
    calm_probe_plan = [rmap["r0"], rmap["r2"], rmap["r1"]]
    hv_probe_plan = [rmap["r5"], rmap["r4"]]
    calm_idx = 0
    hv_idx = 0
    calm_discovered: dict[str, float] = {}
    hv_discovered: dict[str, float] = {}

    records: list[PublicEpisodeRecord] = []
    for ep in range(t_fork):
        spec = env.start_episode(ep)
        steps: list[dict[str, Any]] = []

        o1 = env.step_tool("inspect_desk_brief", {})
        brief = o1.observation
        steps.append(
            {
                "turn": 1,
                "tool": "inspect_desk_brief",
                "args": {},
                "rationale": "Inspect today's macro volatility, liquidity status, and current holdings.",
                "obs": brief,
            }
        )
        is_hv = brief.get("volatility_regime_hint") == "elevated_volatility"
        turn_no = 2
        if is_hv and hv_idx < len(hv_probe_plan):
            p_s = hv_probe_plan[hv_idx]
            hv_idx += 1
            op = env.step_tool("probe_strategy_execution", {"strategy": p_s})
            hv_discovered[p_s] = float(op.observation["live_net_yield_estimate"])
            steps.append(
                {
                    "turn": turn_no,
                    "tool": "probe_strategy_execution",
                    "args": {"strategy": p_s},
                    "rationale": f"Sample live yield of {p_s} in elevated_volatility.",
                    "obs": op.observation,
                }
            )
            turn_no += 1
        elif (not is_hv) and calm_idx < len(calm_probe_plan):
            p_s = calm_probe_plan[calm_idx]
            calm_idx += 1
            op = env.step_tool("probe_strategy_execution", {"strategy": p_s})
            calm_discovered[p_s] = float(op.observation["live_net_yield_estimate"])
            steps.append(
                {
                    "turn": turn_no,
                    "tool": "probe_strategy_execution",
                    "args": {"strategy": p_s},
                    "rationale": f"Sample live yield of {p_s} in calm_volatility.",
                    "obs": op.observation,
                }
            )
            turn_no += 1

        if is_hv:
            w_sub = normalize_weights_v3({}, rmap["r5"], rmap["r4"]) if len(hv_discovered) >= 2 else normalize_weights_v3({}, rmap["r5"], rmap["r0"])
        else:
            if len(calm_discovered) >= 3:
                w_sub = normalize_weights_v3({}, rmap["r0"], rmap["r1"])
            elif len(calm_discovered) == 2:
                w_sub = normalize_weights_v3({}, rmap["r0"], rmap["r2"])
            else:
                w_sub = normalize_weights_v3({}, rmap["r0"], rmap["r3"])

        ofin = env.step_tool("submit_allocation", {"weights": w_sub})
        steps.append(
            {
                "turn": turn_no,
                "tool": "submit_allocation",
                "args": {"weights": w_sub},
                "rationale": "Commit compliant 0.55 / 0.45 allocation based on observed regime yields.",
                "obs": ofin.observation,
            }
        )
        records.append(
            PublicEpisodeRecord(
                env_name=spec.env_name,
                episode_idx=ep,
                timestamp=spec.timestamp,
                task_prompt=spec.task_prompt,
                steps=steps,
                public_score=float(ofin.public_score or 0.0),
            )
        )

    return {
        "seed": seed,
        "prefix_family": prefix_family,
        "t_fork": t_fork,
        "fork_holdings": dict(env._holdings),
        "fork_nav": float(env._cumulative_nav),
        "role_to_strat": dict(rmap),
        "records": records,
    }


def build_frozen_arm_states_at_fork(
    acq: dict[str, Any],
) -> dict[str, Any]:
    """Construct deterministic, model-neutral initial harness states at `t_fork` from shared `W1` records.

    Uses a deterministic offline mock client (`$0.00`) to consolidate the shared `W1` observations so that
    `gemini-3.8-flash` and `gemma-4-31b-it` receive BIT-FOR-BIT IDENTICAL prompts at `t_fork` (`0` char diff)
    with zero model-favoritism in pre-fork card/bucket formation!
    """
    with tempfile.TemporaryDirectory() as tmp_d:
        dummy_ledger = CostLedger(results_root=Path(tmp_d), run_id="offline_w1_ckpt", phase="dev", max_usd=0.01)
        mock_client = MeteredLLMClient(model="mock", ledger=dummy_ledger, thinking_budget=0)

        rmap = acq["role_to_strat"]
        records: list[PublicEpisodeRecord] = acq["records"]

        # 1. Factual-Ledger (100% deterministic)
        factual_h = FactualLedgerV3Harness(recent_k=8)
        for r in records:
            factual_h.on_episode_complete(r, mock_client)

        # 2. Directive-SGEH (exact same Factual-Ledger + SGEH bucket state consolidated from W1)
        directive_h = DirectiveSGEHV3Harness(recent_k=8, enable_hazard=True)
        for r in records:
            directive_h.on_episode_complete(r, mock_client)

    # 3. ScopedCardBook (Baseline): populate exact acquired W1 calm + elevated-vol cards from W1 observations
    card_h = ScopedCardBookV3Harness(reflect_every=3)
    for r in records:
        card_h._history.append(r)
    calm_scores = [
        r.public_score
        for r in records
        if r.steps[0]["obs"].get("volatility_regime_hint") == "calm_volatility"
        and len(r.steps) == 2
    ]
    hv_scores = [
        r.public_score
        for r in records
        if r.steps[0]["obs"].get("volatility_regime_hint") == "elevated_volatility"
        and len(r.steps) == 2
    ]
    card_h.cards = [
        {
            "card_id": "C1",
            "kind": "calm_utility",
            "scope_condition": "volatility_regime_hint == 'calm_volatility' and liquidity_status == 'orderly_normal'",
            "target_strategy": rmap["r0"],
            "recommended_weights": normalize_weights_v3({}, rmap["r0"], rmap["r1"]),
            "uses": max(1, len(calm_scores)),
            "mean_score": round(statistics.mean(calm_scores), 1) if calm_scores else 82.0,
        },
        {
            "card_id": "C2",
            "kind": "invariant_or_hedge",
            "scope_condition": "volatility_regime_hint == 'elevated_volatility' and liquidity_status == 'orderly_normal'",
            "target_strategy": rmap["r5"],
            "recommended_weights": normalize_weights_v3({}, rmap["r5"], rmap["r4"]),
            "uses": max(1, len(hv_scores)),
            "mean_score": round(statistics.mean(hv_scores), 1) if hv_scores else 84.0,
        },
    ]

    return {
        "scoped_cards_v3": card_h,
        "factual_ledger_v3": factual_h,
        "sgeh_directive_v3": directive_h,
    }


def _mock_autonomy_actor(steps: list[dict[str, Any]], sys_prompt: str) -> dict[str, Any]:
    if not steps:
        return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Read desk brief."}
    obs0 = steps[0].get("obs", {})
    cur_h = obs0.get("current_holdings", {})
    already_probed = any(s["tool"] == "probe_strategy_execution" for s in steps)
    if not already_probed and "Call `probe_strategy_execution` on `" in sys_prompt:
        frag = sys_prompt.split("Call `probe_strategy_execution` on `", 1)[1]
        cand = frag.split("`", 1)[0].strip()
        if cand in STRATEGIES_V3:
            return {"tool": "probe_strategy_execution", "args": {"strategy": cand}, "rationale": f"Probe {cand}."}
    w = normalize_weights_v3(cur_h, "STRAT_S1", "STRAT_S2")
    return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Submit compliant weights."}


def run_autonomy_post_fork_branch(
    *,
    model_name: str,
    arm_name: str,
    fork_mode: str,
    seed: int,
    t_fork: int,
    h_post: int,
    acq: dict[str, Any],
    base_harness: Any,
    client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    env = AlphaDeskV3ForkEnv(
        fork_mode=fork_mode,
        seed=seed,
        t_fork=t_fork,
        h_post=h_post,
        initial_post_holdings=acq["fork_holdings"],
    )
    env.set_fork_state(acq["fork_holdings"], acq["fork_nav"])
    harness = copy.deepcopy(base_harness)

    rmap = acq["role_to_strat"]
    strat_to_role = {v: k for k, v in rmap.items()}
    # Strategies never probed in calm_volatility during W1 are roles r3, r4, r5 (and r2 was probed low ~9 calm sessions ago)
    untested_or_non_incumbent = {rmap["r2"], rmap["r3"], rmap["r4"], rmap["r5"]}
    truly_unprobed_in_calm = {rmap["r3"], rmap["r4"], rmap["r5"]}
    incumbent_pair = {rmap["r0"], rmap["r1"]}

    logs: list[dict[str, Any]] = []
    acting_calls = 0
    acting_usd = 0.0
    reflection_calls = 0
    reflection_usd = 0.0
    thinking_tokens = 0
    runner_interventions = 0
    compliance_violations = 0
    risk_tool_calls = 0

    for ep in range(t_fork, t_fork + h_post):
        spec = env.start_episode(ep)
        h_ctx = harness.build_context(spec)
        sys_prompt = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_ctx)

        steps: list[dict[str, Any]] = []
        done = False
        final_score = 0.0
        priv: dict[str, Any] = {}

        for turn in range(env.max_turns_per_episode):
            hist_str = (
                "\n".join(
                    f"Turn {idx + 1}: called `{s['tool']}`({json.dumps(s['args'])}) -> {json.dumps(s['obs'])}"
                    for idx, s in enumerate(steps)
                )
                or "(No tools called yet in this session.)"
            )
            rem = env.max_turns_per_episode - turn
            must_commit = " FINAL TURN: Call `submit_allocation` now!" if rem == 1 else ""
            usr_prompt = (
                f"{spec.task_prompt}\n\n"
                f"Session Tool History:\n{hist_str}\n\n"
                f"Turns remaining: {rem}.{must_commit}"
            )
            resp = client.generate_json(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                environment=spec.env_name,
                method=f"autonomy_{arm_name}",
                mock_responder=lambda s_p, _u_p, st=steps: _mock_autonomy_actor(st, s_p),
            )
            acting_calls += 1
            acting_usd += resp.usd
            thinking_tokens += resp.thinking_tokens

            parsed = resp.parsed_json or {}
            t_name = str(parsed.get("tool", "")).strip()
            t_args = parsed.get("args", {})
            if not isinstance(t_args, dict):
                t_args = {}

            intervened = False
            if not t_name or (rem == 1 and t_name != "submit_allocation"):
                intervened = True
                runner_interventions += 1
                if not steps:
                    t_name, t_args = "inspect_desk_brief", {}
                else:
                    # Passive no-trade hold on `env._holdings` so zero strategy preference is injected
                    t_name, t_args = "submit_allocation", {"weights": dict(env._holdings)}

            if t_name == "inspect_risk_and_turnover":
                risk_tool_calls += 1

            # STRICT ZERO AUTO-REPAIR: pass `t_args` directly to `env.step_tool` without clipping!
            outcome = env.step_tool(t_name, t_args)
            steps.append(
                {
                    "turn": turn + 1,
                    "tool": t_name,
                    "args": t_args,
                    "rationale": str(parsed.get("rationale", ""))[:220],
                    "runner_intervened": intervened,
                    "obs": outcome.observation,
                }
            )
            if outcome.done:
                done = True
                final_score = float(outcome.public_score or 0.0)
                priv = outcome.private_eval
                if priv.get("invariant_violation"):
                    compliance_violations += 1
                break

        if not done:
            runner_interventions += 1
            outcome = env.step_tool("submit_allocation", {"weights": dict(env._holdings)})
            final_score = float(outcome.public_score or 0.0)
            priv = outcome.private_eval
            if priv.get("invariant_violation"):
                compliance_violations += 1

        pub_rec = PublicEpisodeRecord(
            env_name=spec.env_name,
            episode_idx=ep,
            timestamp=spec.timestamp,
            task_prompt=spec.task_prompt,
            steps=steps,
            public_score=final_score,
        )
        h_meta = harness.on_episode_complete(pub_rec, client)
        reflection_calls += int(h_meta.get("reflection_calls", 0))
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))

        probed_strats = [
            str(s["obs"].get("strategy"))
            for s in steps
            if s["tool"] == "probe_strategy_execution" and isinstance(s.get("obs"), dict) and "strategy" in s["obs"]
        ]
        sub_step = next((s for s in reversed(steps) if s["tool"] == "submit_allocation"), None)
        sub_weights = sub_step["args"].get("weights", {}) if (sub_step and isinstance(sub_step.get("args"), dict)) else {}

        logs.append(
            {
                "episode": ep + 1,
                "regime": priv.get("regime", ""),
                "is_high_vol": bool(priv.get("is_high_vol", False)),
                "is_liquidity_freeze": bool(priv.get("is_liquidity_freeze", False)),
                "public_score": final_score,
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", 0.0)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "optimal_policy_match": bool(priv.get("optimal_policy_match", False)),
                "stale_policy_match": bool(priv.get("stale_policy_match", False)),
                "invariant_violation": bool(priv.get("invariant_violation", False)),
                "turnover_fee": float(priv.get("turnover_fee", 0.0)),
                "probes_used": int(priv.get("probes_used", 0)),
                "probed_strategies": probed_strats,
                "submitted_weights": sub_weights,
                "harness_meta": h_meta,
                "steps": steps,
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    dp_post_cum = sum(r["dp_oracle_step_net"] for r in logs)
    agent_post_cum = sum(r["expected_net_no_noise"] for r in logs)
    post_dp_utility_pct = round(100.0 * agent_post_cum / max(1.0, dp_post_cum), 2)

    calm_norm = [r for r in logs if not r["is_high_vol"] and not r["is_liquidity_freeze"]]
    calm_opt_pct = round(100.0 * sum(1 for r in calm_norm if r["optimal_policy_match"]) / max(1, len(calm_norm)), 2)
    stale_lock_pct = round(100.0 * sum(1 for r in calm_norm if r["stale_policy_match"]) / max(1, len(calm_norm)), 2)

    calm_probes: list[tuple[int, str]] = []
    for r in calm_norm:
        for ps in r["probed_strategies"]:
            calm_probes.append((r["episode"], ps))

    first_probe_ep = calm_probes[0][0] if calm_probes else None
    first_probe_strat = calm_probes[0][1] if calm_probes else None
    first_probe_role = strat_to_role.get(first_probe_strat, "") if first_probe_strat else ""
    first_probe_non_inc = bool(first_probe_strat in untested_or_non_incumbent) if first_probe_strat else False
    first_probe_truly_unprobed = bool(first_probe_strat in truly_unprobed_in_calm) if first_probe_strat else False
    redundant_inc_probes = sum(1 for _, ps in calm_probes if ps in incumbent_pair)
    non_inc_probes = sum(1 for _, ps in calm_probes if ps in untested_or_non_incumbent)

    freeze_rows = [r for r in logs if r["is_liquidity_freeze"]]
    freeze_probes = sum(r["probes_used"] for r in freeze_rows)
    freeze_turnover_pts = round(sum(r["turnover_fee"] for r in freeze_rows), 3)
    # Also check L1 drift away from fork_holdings immediately after freeze ends (ep = t_fork + 2)
    post_freeze_abandoned = False
    if fork_mode == "freeze_matched_loss" and len(logs) >= 3:
        w_after_freeze = logs[2].get("submitted_weights", {})
        if isinstance(w_after_freeze, dict) and w_after_freeze:
            post_freeze_abandoned = l1_turnover_half(w_after_freeze, acq["fork_holdings"]) > 0.25

    return {
        "model": model_name,
        "provider_model_versions": sorted(client.observed_model_versions) or [model_name],
        "arm": arm_name,
        "fork_mode": fork_mode,
        "seed": seed,
        "episodes_post": h_post,
        "post_dp_utility_pct": post_dp_utility_pct,
        "calm_optimal_pct": calm_opt_pct,
        "stale_lock_in_pct": stale_lock_pct,
        "probes_total": sum(r["probes_used"] for r in logs),
        "calm_non_incumbent_probes": non_inc_probes,
        "calm_redundant_incumbent_probes": redundant_inc_probes,
        "first_probe_ep": first_probe_ep,
        "first_probe_strategy": first_probe_strat,
        "first_probe_role": first_probe_role,
        "first_probe_non_incumbent": first_probe_non_inc,
        "first_probe_truly_unprobed": first_probe_truly_unprobed,
        "freeze_probes": freeze_probes,
        "freeze_turnover_pts": freeze_turnover_pts,
        "post_freeze_abandoned_incumbent": post_freeze_abandoned,
        "turnover_pts_total": round(sum(r["turnover_fee"] for r in logs), 2),
        "risk_tool_calls": risk_tool_calls,
        "compliance_violations": compliance_violations,
        "runner_interventions": runner_interventions,
        "acting_calls": acting_calls,
        "reflection_calls": reflection_calls,
        "thinking_tokens": thinking_tokens,
        "usd": round(acting_usd + reflection_usd, 6),
        "curve_utility": [round(100.0 * r["expected_net_no_noise"] / max(1.0, r["dp_oracle_step_net"]), 1) for r in logs],
        "logs": logs,
    }


def _bootstrap_ci(vals: list[float], seed: int = 20261121) -> dict[str, Any]:
    if not vals:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "per_seed": []}
    m = statistics.mean(vals)
    if len(vals) == 1:
        return {"mean": round(m, 2), "ci95": [round(m, 2), round(m, 2)], "per_seed": [round(v, 2) for v in vals]}
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(statistics.mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(4000))
    return {
        "mean": round(m, 2),
        "ci95": [round(boots[int(0.025 * 4000)], 2), round(boots[int(0.975 * 4000)], 2)],
        "per_seed": [round(v, 2) for v in vals],
    }


def summarize_autonomy_report(
    *,
    run_id: str,
    t_fork: int,
    h_post: int,
    records: list[dict[str, Any]],
    total_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    models = sorted({str(r["model"]) for r in records})
    modes = [m for m in DEFAULT_FORK_MODES if any(r["fork_mode"] == m for r in records)]
    seeds = sorted({int(r["seed"]) for r in records})

    by_model_mode: dict[str, dict[str, Any]] = {}
    paired_contrasts: dict[str, dict[str, Any]] = {}
    provider_versions: dict[str, list[str]] = {}

    for mdl in models:
        m_recs = [r for r in records if r["model"] == mdl]
        vers: set[str] = set()
        for r in m_recs:
            vers.update(r.get("provider_model_versions", []))
        provider_versions[mdl] = sorted(vers)
        by_model_mode[mdl] = {}
        paired_contrasts[mdl] = {}

        for fm in modes:
            sub_fm = [r for r in m_recs if r["fork_mode"] == fm]
            arm_map: dict[str, Any] = {}
            for arm in DEFAULT_ARMS:
                sub = [r for r in sub_fm if r["arm"] == arm]
                if not sub:
                    continue
                arm_map[arm] = {
                    "n_seeds": len(sub),
                    "post_dp_utility_pct": round(statistics.mean(r["post_dp_utility_pct"] for r in sub), 2),
                    "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in sub), 2),
                    "stale_lock_in_pct": round(statistics.mean(r["stale_lock_in_pct"] for r in sub), 2),
                    "mean_probes": round(statistics.mean(r["probes_total"] for r in sub), 2),
                    "mean_non_inc_probes": round(statistics.mean(r["calm_non_incumbent_probes"] for r in sub), 2),
                    "mean_redundant_inc_probes": round(statistics.mean(r["calm_redundant_incumbent_probes"] for r in sub), 2),
                    "first_probe_non_inc_rate_pct": round(100.0 * sum(1 for r in sub if r["first_probe_non_incumbent"]) / len(sub), 1),
                    "first_probe_truly_unprobed_rate_pct": round(100.0 * sum(1 for r in sub if r["first_probe_truly_unprobed"]) / len(sub), 1),
                    "mean_freeze_probes": round(statistics.mean(r["freeze_probes"] for r in sub), 2),
                    "mean_freeze_turnover_pts": round(statistics.mean(r["freeze_turnover_pts"] for r in sub), 2),
                    "post_freeze_abandon_rate_pct": round(100.0 * sum(1 for r in sub if r["post_freeze_abandoned_incumbent"]) / len(sub), 1),
                    "mean_turnover_pts": round(statistics.mean(r["turnover_pts_total"] for r in sub), 2),
                    "mean_risk_tool_calls": round(statistics.mean(r["risk_tool_calls"] for r in sub), 2),
                    "compliance_violations_sum": sum(int(r["compliance_violations"]) for r in sub),
                    "runner_interventions_sum": sum(int(r["runner_interventions"]) for r in sub),
                    "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in sub), 2),
                    "mean_usd": round(statistics.mean(r["usd"] for r in sub), 4),
                }
            by_model_mode[mdl][fm] = arm_map

            by_s_arm = {(int(r["seed"]), str(r["arm"])): r for r in sub_fm}
            fm_con: dict[str, Any] = {}
            for a_hi, a_lo, cname in (
                ("factual_ledger_v3", "scoped_cards_v3", "factual_minus_cardbook_rep_gain"),
                ("sgeh_directive_v3", "factual_ledger_v3", "directive_minus_factual_ctrl_gain"),
                ("sgeh_directive_v3", "scoped_cards_v3", "directive_minus_cardbook_total_gain"),
            ):
                diffs = [
                    by_s_arm[(sd, a_hi)]["post_dp_utility_pct"] - by_s_arm[(sd, a_lo)]["post_dp_utility_pct"]
                    for sd in seeds
                    if (sd, a_hi) in by_s_arm and (sd, a_lo) in by_s_arm
                ]
                if diffs:
                    fm_con[cname] = _bootstrap_ci(diffs)
            paired_contrasts[mdl][fm] = fm_con

        # Compute Stationary-Adjusted Difference-in-Differences (DiD) for each shift mode!
        did_map: dict[str, Any] = {}
        by_m_s_a = {(str(r["fork_mode"]), int(r["seed"]), str(r["arm"])): r for r in m_recs}
        for shift_m, stat_m in (
            ("abrupt_reversal", "stationary"),
            ("freeze_matched_loss", "stationary"),
            ("opportunity_rise", "opportunity_stationary"),
        ):
            for a_hi, a_lo, dname in (
                ("sgeh_directive_v3", "factual_ledger_v3", f"did_{shift_m}_directive_minus_factual"),
                ("factual_ledger_v3", "scoped_cards_v3", f"did_{shift_m}_factual_minus_cardbook"),
            ):
                d_vals = []
                for sd in seeds:
                    if all((mm, sd, aa) in by_m_s_a for mm in (shift_m, stat_m) for aa in (a_hi, a_lo)):
                        sh_gap = by_m_s_a[(shift_m, sd, a_hi)]["post_dp_utility_pct"] - by_m_s_a[(shift_m, sd, a_lo)]["post_dp_utility_pct"]
                        st_gap = by_m_s_a[(stat_m, sd, a_hi)]["post_dp_utility_pct"] - by_m_s_a[(stat_m, sd, a_lo)]["post_dp_utility_pct"]
                        d_vals.append(sh_gap - st_gap)
                if d_vals:
                    did_map[dname] = _bootstrap_ci(d_vals)
        paired_contrasts[mdl]["did_summary"] = did_map

    clean_records = [{k: v for k, v in r.items() if k != "logs"} for r in records]
    return {
        "experiment_type": "cross_model_autonomy_fork_v3",
        "run_id": run_id,
        "models": models,
        "provider_model_versions": provider_versions,
        "t_fork": t_fork,
        "h_post": h_post,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "total_run_usd": round(total_usd, 6),
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "by_model_mode": by_model_mode,
        "paired_contrasts": paired_contrasts,
        "records": clean_records,
    }


def render_autonomy_markdown(rep: dict[str, Any]) -> str:
    arm_labels = {
        "scoped_cards_v3": "ScopedCardBook (Baseline)",
        "factual_ledger_v3": "**Factual-Ledger (Ours — Evidence-Only Autonomy)**",
        "sgeh_directive_v3": "**Directive-SGEH (Ours — Prescriptive Controller)**",
    }
    lines = [
        f"# Cross-Model Epistemic Autonomy Fork (`AlphaDesk-v0.3`): `{rep['run_id']}`",
        "",
        f"- **Models & Provider Versions:** `{json.dumps(rep['provider_model_versions'])}`",
        f"- **Seeds:** `{rep['seeds']}` (`N={rep['n_seeds']}` Latin-square) | **Shared Fork:** `t*={rep['t_fork']}`, `H_post={rep['h_post']}` | **Total Spend:** `${rep['total_run_usd']:.4f}`",
        f"- **Observed Tokens:** `in={rep['observed_run_input_tokens']:,}`, `out={rep['observed_run_output_tokens']:,}`, `thinking={rep['observed_run_thinking_tokens']:,}`",
        "",
    ]
    for mdl in rep["models"]:
        ver_str = ", ".join(rep["provider_model_versions"].get(mdl, [mdl]))
        lines.append(f"## Model Family: `{mdl}` (provider `model_version`: `{ver_str}`)")
        lines.append("")
        mm = rep["by_model_mode"].get(mdl, {})
        pc = rep["paired_contrasts"].get(mdl, {})
        for fm, amap in mm.items():
            if not amap:
                continue
            lines.append(f"### Post-Fork World: `{fm}`")
            lines.append(
                "| Arm | Post-Fork DP Utility (%) | CalmOpt (%) | Probes (NonInc / Redund) | 1st Probe NonInc (%) | Freeze Prb / Turn | Viol / Interv | Ref Calls | USD |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for ak in DEFAULT_ARMS:
                if ak not in amap:
                    continue
                d = amap[ak]
                lines.append(
                    f"| {arm_labels.get(ak, ak)} | **{d['post_dp_utility_pct']:.1f}%** | {d['calm_optimal_pct']:.1f}% | "
                    f"{d['mean_probes']:.1f} (`{d['mean_non_inc_probes']:.1f}` / `{d['mean_redundant_inc_probes']:.1f}`) | "
                    f"{d['first_probe_non_inc_rate_pct']:.0f}% | "
                    f"`{d['mean_freeze_probes']:.1f}` / `{d['mean_freeze_turnover_pts']:.1f}` | "
                    f"`{d['compliance_violations_sum']}` / `{d['runner_interventions_sum']}` | "
                    f"{d['mean_reflection_calls']:.1f} | `${d['mean_usd']:.4f}` |"
                )
            lines.append("")
            fcon = pc.get(fm, {})
            for ck, cv in fcon.items():
                lines.append(
                    f"- **`{ck}`:** **`{cv['mean']:+.2f} pts`** `[{cv['ci95'][0]:+.2f}, {cv['ci95'][1]:+.2f}]` (per-seed: `{cv['per_seed']}`)"
                )
            lines.append("")

        did = pc.get("did_summary", {})
        if did:
            lines.append(f"### Stationary-Adjusted Difference-in-Differences (`{mdl}`)")
            for dk, dv in did.items():
                lines.append(
                    f"- **`{dk}`:** **`{dv['mean']:+.2f} pts`** `[{dv['ci95'][0]:+.2f}, {dv['ci95'][1]:+.2f}]` (per-seed: `{dv['per_seed']}`)"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def run_autonomy_batch(
    *,
    run_id: str,
    models: list[str],
    arms: list[str],
    fork_modes: list[str],
    seeds: list[int],
    t_fork: int,
    h_post: int,
    max_usd: float,
    workers: int,
    thinking_budget: int,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)
    total_eps = len(models) * len(arms) * len(fork_modes) * len(seeds) * h_post
    done_eps = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_eps
        with lock:
            done_eps += 1
            if done_eps % 4 == 0 or done_eps == total_eps:
                ledger.update_progress(
                    completed_units=min(done_eps, total_eps),
                    total_units=total_eps,
                    status="running",
                    current_label=lbl,
                    log_every=32,
                )

    clients = {
        mdl: MeteredLLMClient(
            model=mdl,
            ledger=ledger,
            dotenv_path=dotenv_path,
            thinking_budget=thinking_budget,
            temperature=0.0,
        )
        for mdl in models
    }

    # Precompute deterministic shared W1 trajectories & frozen arm states once per (seed, prefix_family)
    shared_cache: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for sd in seeds:
        for pf in ("high_a", "opp_a"):
            if any(_prefix_family_for_mode(fm) == pf for fm in fork_modes):
                acq = generate_shared_w1_acquisition(seed=sd, prefix_family=pf, t_fork=t_fork, h_post=h_post)
                arm_states = build_frozen_arm_states_at_fork(acq)
                shared_cache[(sd, pf)] = (acq, arm_states)

    traces_dir = ledger.results_root / "raw" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (mdl, arm, fm, sd)
        for mdl in models
        for sd in seeds
        for fm in fork_modes
        for arm in arms
    ]
    raw_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), workers))) as pool:
        futs = []
        for mdl, arm, fm, sd in jobs:
            acq, arm_states = shared_cache[(sd, _prefix_family_for_mode(fm))]
            futs.append(
                pool.submit(
                    run_autonomy_post_fork_branch,
                    model_name=mdl,
                    arm_name=arm,
                    fork_mode=fm,
                    seed=sd,
                    t_fork=t_fork,
                    h_post=h_post,
                    acq=acq,
                    base_harness=arm_states[arm],
                    client=clients[mdl],
                    on_ep_done=lambda m=mdl, a=arm, f=fm, s=sd: _tick(f"{m} {f} s{s} {a}"),
                )
            )
        for fut in as_completed(futs):
            res = fut.result()
            raw_results.append(res)
            mdl, arm, fm, sd = res["model"], res["arm"], res["fork_mode"], res["seed"]
            print(
                f"  -> [{mdl:16s} | {fm:22s} | s{sd} | {arm:18s}] "
                f"PostDP={res['post_dp_utility_pct']:5.1f}% (CalmOpt={res['calm_optimal_pct']:5.1f}%) | "
                f"Prb={res['probes_total']:2d} (NonInc={res['calm_non_incumbent_probes']}, Red={res['calm_redundant_incumbent_probes']}) "
                f"1st={res['first_probe_strategy'] or 'none':8s} | Viol={res['compliance_violations']} Intv={res['runner_interventions']} | "
                f"Cell=${res['usd']:.4f} Run=${ledger.run_usd:.4f}",
                flush=True,
            )
            cell_slug = f"alpha_desk_v3_fork_{fm}__{arm}__{mdl.replace('/', '_')}_s{sd}"
            with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as fh:
                for row in res["logs"]:
                    fh.write(json.dumps(row) + "\n")

            tot_c = res["acting_calls"] + res["reflection_calls"]
            atomic_write_json(
                cells_dir / f"{cell_slug}.json",
                {
                    "summary": {
                        "run_id": run_id,
                        "env_spec": f"alpha_desk_v3_fork:{fm}",
                        "harness_name": arm,
                        "model": mdl,
                        "seed": sd,
                        "episodes": h_post,
                        "mean_score": round(statistics.mean(r["public_score"] for r in res["logs"]), 2),
                        "mean_normalized_utility": res["post_dp_utility_pct"],
                        "pre_shift_utility": 100.0,
                        "post_shift_utility": res["post_dp_utility_pct"],
                        "stale_lock_in_rate": round(res["stale_lock_in_pct"] / 100.0, 4),
                        "invariant_violation_rate": round(res["compliance_violations"] / max(1, h_post), 4),
                        "total_acting_calls": res["acting_calls"],
                        "total_reflection_calls": res["reflection_calls"],
                        "rsi_tax_call_share": round(res["reflection_calls"] / max(1, tot_c), 4),
                        "total_usd": res["usd"],
                        "reflection_usd": 0.0,
                    },
                    "curve_utility": res["curve_utility"],
                    "curve_score": [r["public_score"] for r in res["logs"]],
                    "alarms": [r["episode"] for r in res["logs"] if r["harness_meta"].get("alarm_fired")],
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    raw_results.sort(key=lambda x: (x["model"], x["fork_mode"], x["seed"], x["arm"]))
    rep = summarize_autonomy_report(
        run_id=run_id,
        t_fork=t_fork,
        h_post=h_post,
        records=raw_results,
        total_usd=round(ledger.run_usd, 6),
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "autonomy_fork_report.json", rep)
    ledger.update_progress(
        completed_units=total_eps,
        total_units=total_eps,
        status="completed",
        current_label="completed autonomy fork batch",
    )
    return rep


def aggregate_autonomy_runs(results_root: Path, run_ids: list[str], out_run_id: str) -> Path:
    all_records: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    t_fork = 14
    h_post = 16
    for rid in run_ids:
        d = json.loads((results_root / "runs" / rid / "autonomy_fork_report.json").read_text(encoding="utf-8"))
        all_records.extend(d.get("records", []))
        tot_usd += float(d.get("total_run_usd", 0.0))
        in_t += int(d.get("observed_run_input_tokens", 0))
        out_t += int(d.get("observed_run_output_tokens", 0))
        th_t += int(d.get("observed_run_thinking_tokens", 0))
        t_fork = int(d.get("t_fork", t_fork))
        h_post = int(d.get("h_post", h_post))

    rep = summarize_autonomy_report(
        run_id=out_run_id,
        t_fork=t_fork,
        h_post=h_post,
        records=all_records,
        total_usd=round(tot_usd, 6),
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "autonomy_fork_report.json", rep)
    md = render_autonomy_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved merged autonomy fork report: {out_md}", flush=True)
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cross-Model Epistemic Autonomy Fork on AlphaDesk-v0.3")
    parser.add_argument("--run-id", default="autonomy_pilot", help="Run ID")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated models")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="Comma-separated harness arms")
    parser.add_argument("--modes", default=",".join(DEFAULT_FORK_MODES), help="Comma-separated fork modes")
    parser.add_argument("--seeds", default="501", help="Comma-separated seeds")
    parser.add_argument("--t-fork", type=int, default=14, help="Shared W1 acquisition sessions before fork")
    parser.add_argument("--h-post", type=int, default=16, help="Post-fork evaluation sessions")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget")
    parser.add_argument("--workers", type=int, default=5, help="Parallel threads")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 per-run USD cap")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated run IDs to merge")
    args = parser.parse_args()

    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        aggregate_autonomy_runs(ROOT / "results", rids, args.run_id)
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    try:
        rep = run_autonomy_batch(
            run_id=args.run_id,
            models=models,
            arms=arms,
            fork_modes=modes,
            seeds=seeds,
            t_fork=args.t_fork,
            h_post=args.h_post,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            results_root=ROOT / "results",
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET GUARDRAIL HALT] {exc}", flush=True)
        sys.exit(2)

    md = render_autonomy_markdown(rep)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved report: {out_md}", flush=True)


if __name__ == "__main__":
    main()
