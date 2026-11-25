#!/usr/bin/env python3
"""Post-Meeting Priorities 1 & 2 Runner:
  1. `--suite voi_fork`: Hidden-Ambiguity Temporary Dip (`hidden_transient_dip`) vs. Lasting Reversal (`abrupt_reversal`)
     + Stationary & Type-II Forks from **100% Live Unscripted `W1` Model Rollouts** (`0` `role_to_strat` scripting!).
  2. `--suite procedure_3w`: Self-Authored `Editable-Procedure` vs. Matched-Budget `Frozen-Procedure` vs. `No-Procedure`
     (`Factual-Ledger`) vs. `Directive-SGEH` across 3-Window (`W1 -> W2 -> W3`) Multi-Shift & Stationary Streams,
     saving **every verbatim procedure version in JSONL before future outcomes arrive**.
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
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v3 import (  # noqa: E402
    AlphaDeskV3ForkEnv,
    AlphaDeskV3ThreeWindowEnv,
    STRATEGIES_V3,
    l1_turnover_half,
)
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.procedure_harness import (  # noqa: E402
    SelfAuthoredProcedureHarness,
    distill_domain_agnostic_procedure,
)
from src.harnesses.sgeh_v3 import (  # noqa: E402
    DirectiveSGEHV3Harness,
    FactualLedgerV3Harness,
    ScopedCardBookV3Harness,
    normalize_weights_v3,
)

VOI_FORK_MODES = (
    "stationary",
    "hidden_transient_dip",
    "abrupt_reversal",
    "opportunity_stationary",
    "opportunity_rise",
)
PROC_3W_MODES = (
    "stationary_3w",
    "dip_then_reversal_3w",
    "double_reversal_3w",
    "opp_stationary_3w",
    "opp_double_rise_3w",
)


def _prefix_family(mode_name: str, suite: str) -> str:
    if suite == "voi_fork":
        return "opp_a" if mode_name in ("opportunity_stationary", "opportunity_rise") else "high_a"
    return "opp_3w" if mode_name in ("opp_stationary_3w", "opp_double_rise_3w") else "high_a"


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


def _snapshot_harness(harness: Any) -> dict[str, Any]:
    if hasattr(harness, "snapshot_state"):
        return harness.snapshot_state()
    if isinstance(harness, ScopedCardBookV3Harness):
        return {"harness_name": harness.name, "cards": copy.deepcopy(harness.cards)}
    if isinstance(harness, FactualLedgerV3Harness):
        return {"harness_name": harness.name, "history_len": len(harness.history)}
    if isinstance(harness, DirectiveSGEHV3Harness):
        return {
            "harness_name": harness.name,
            "history_len": len(harness.factual.history),
            "calm_weights": dict(harness.controller.buckets["calm_volatility"].active_weights),
        }
    return {"harness_name": getattr(harness, "name", "unknown")}


def _mock_actor(steps: list[dict[str, Any]], sys_prompt: str, ep_idx: int) -> dict[str, Any]:
    if not steps:
        return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Read desk brief."}
    obs0 = steps[0].get("obs", {})
    cur_h = obs0.get("current_holdings", {})
    already_probed = any(s["tool"] == "probe_strategy_execution" for s in steps)
    if not already_probed and ep_idx < 4:
        cand = STRATEGIES_V3[ep_idx % len(STRATEGIES_V3)]
        return {"tool": "probe_strategy_execution", "args": {"strategy": cand}, "rationale": f"Probe {cand}."}
    if not already_probed and "Call `probe_strategy_execution` on `" in sys_prompt:
        frag = sys_prompt.split("Call `probe_strategy_execution` on `", 1)[1]
        cand = frag.split("`", 1)[0].strip()
        if cand in STRATEGIES_V3:
            return {"tool": "probe_strategy_execution", "args": {"strategy": cand}, "rationale": f"Probe {cand}."}
    w = normalize_weights_v3(cur_h, "STRAT_S1", "STRAT_S2")
    return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Submit compliant weights."}


def _run_episode_loop(
    *,
    env: AlphaDeskV3ForkEnv,
    ep_range: range,
    harness: Any,
    client: MeteredLLMClient,
    method_tag: str,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    logs: list[dict[str, Any]] = []
    pub_records: list[PublicEpisodeRecord] = []
    acting_calls = 0
    acting_usd = 0.0
    reflection_calls = 0
    reflection_usd = 0.0
    thinking_tokens = 0
    runner_interventions = 0
    compliance_violations = 0
    risk_tool_calls = 0

    for ep in ep_range:
        spec = env.start_episode(ep)
        # Capture verbatim pre-episode harness context & procedure/rule state BEFORE any episode actions/outcomes!
        pre_state = _snapshot_harness(harness)
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
                method=method_tag,
                mock_responder=lambda s_p, _u_p, st=steps, e_i=ep: _mock_actor(st, s_p, e_i),
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
                    t_name, t_args = "submit_allocation", {"weights": dict(env._holdings)}

            if t_name == "inspect_risk_and_turnover":
                risk_tool_calls += 1

            outcome = env.step_tool(t_name, t_args)
            steps.append(
                {
                    "turn": turn + 1,
                    "tool": t_name,
                    "args": t_args,
                    "rationale": str(parsed.get("rationale", ""))[:240],
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
        pub_records.append(pub_rec)
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
                "window": int(priv.get("window", env.window_index(ep))),
                "regime": priv.get("regime", ""),
                "is_high_vol": bool(priv.get("is_high_vol", False)),
                "is_liquidity_freeze": bool(priv.get("is_liquidity_freeze", False)),
                "is_hidden_transient_dip": bool(priv.get("is_hidden_transient_dip", False)),
                "pre_episode_harness_state": pre_state,
                "pre_episode_harness_context": h_ctx,
                "public_score": final_score,
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", 0.0)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "optimal_policy_match": bool(priv.get("optimal_policy_match", False)),
                "dp_policy_match": bool(priv.get("dp_policy_match", False)),
                "stale_policy_match": bool(priv.get("stale_policy_match", False)),
                "invariant_violation": bool(priv.get("invariant_violation", False)),
                "turnover_fee": float(priv.get("turnover_fee", 0.0)),
                "probes_used": int(priv.get("probes_used", 0)),
                "probed_strategies": probed_strats,
                "submitted_weights": sub_weights,
                "post_episode_harness_meta": h_meta,
                "steps": steps,
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    return {
        "logs": logs,
        "pub_records": pub_records,
        "end_holdings": dict(env._holdings),
        "end_nav": float(env._cumulative_nav),
        "acting_calls": acting_calls,
        "acting_usd": acting_usd,
        "reflection_calls": reflection_calls,
        "reflection_usd": reflection_usd,
        "thinking_tokens": thinking_tokens,
        "runner_interventions": runner_interventions,
        "compliance_violations": compliance_violations,
        "risk_tool_calls": risk_tool_calls,
    }


def serialize_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": ckpt["suite"],
        "model": ckpt["model"],
        "seed": ckpt["seed"],
        "prefix_family": ckpt["prefix_family"],
        "t_fork": ckpt["t_fork"],
        "fork_holdings": ckpt["fork_holdings"],
        "fork_nav": ckpt["fork_nav"],
        "w1_dp_utility_pct": ckpt["w1_dp_utility_pct"],
        "w1_probes": ckpt["w1_probes"],
        "w1_usd": ckpt["w1_usd"],
        "acquired_procedure_v": ckpt.get("acquired_procedure_v", 0),
        "acquired_self_procedure": ckpt.get("acquired_self_procedure", ""),
        "acquired_procedure_history": ckpt.get("acquired_procedure_history", []),
        "acquired_cards": ckpt.get("acquired_cards", []),
        "records": [
            {
                "env_name": r.env_name,
                "episode_idx": r.episode_idx,
                "timestamp": r.timestamp,
                "task_prompt": r.task_prompt,
                "steps": r.steps,
                "public_score": r.public_score,
            }
            for r in ckpt["records"]
        ],
        "w1_logs": ckpt.get("w1_logs", []),
    }


def deserialize_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    recs = [
        PublicEpisodeRecord(
            env_name=str(d["env_name"]),
            episode_idx=int(d["episode_idx"]),
            timestamp=int(d["timestamp"]),
            task_prompt=str(d["task_prompt"]),
            steps=list(d["steps"]),
            public_score=float(d["public_score"]),
        )
        for d in raw["records"]
    ]
    return {
        **raw,
        "records": recs,
    }


def acquire_live_w1_checkpoint(
    *,
    suite: str,
    model_name: str,
    seed: int,
    prefix_family: str,
    t_fork: int,
    h_post: int,
    client: MeteredLLMClient,
    arms_needed: list[str],
    on_ep_done: Any = None,
) -> dict[str, Any]:
    """Acquire `W1` (`0 .. t_fork-1`) strictly via LIVE model rollouts using ONLY public observations (`0` scripting!)."""
    if suite == "voi_fork":
        base_mode = "opportunity_stationary" if prefix_family == "opp_a" else "stationary"
        env: AlphaDeskV3ForkEnv = AlphaDeskV3ForkEnv(fork_mode=base_mode, seed=seed, t_fork=t_fork, h_post=h_post)
    else:
        w2_l = h_post // 2
        w3_l = h_post - w2_l
        base_mode = "opp_stationary_3w" if prefix_family == "opp_3w" else "stationary_3w"
        env = AlphaDeskV3ThreeWindowEnv(stream_mode=base_mode, seed=seed, w1_len=t_fork, w2_len=w2_l, w3_len=w3_l)

    # For `procedure_3w`, the live W1 agent starts with `Editable-Procedure` (blank `""` at ep 0!) so `P_W1` is
    # genuinely co-evolved with its own live W1 actions! For `voi_fork`, live W1 runs with pure `FactualLedgerV3`.
    if suite == "procedure_3w":
        w1_harness: Any = SelfAuthoredProcedureHarness(mode="editable", review_cadence=4, recent_k=8)
    else:
        w1_harness = FactualLedgerV3Harness(recent_k=8)

    out = _run_episode_loop(
        env=env,
        ep_range=range(0, t_fork),
        harness=w1_harness,
        client=client,
        method_tag=f"live_w1_acq_{suite}",
        on_ep_done=on_ep_done,
    )
    w1_dp_cum = sum(r["dp_oracle_step_net"] for r in out["logs"])
    w1_ag_cum = sum(r["expected_net_no_noise"] for r in out["logs"])
    w1_util = round(100.0 * w1_ag_cum / max(1.0, w1_dp_cum), 2)

    acq_proc = ""
    acq_proc_v = 0
    acq_proc_hist: list[dict[str, Any]] = []
    if isinstance(w1_harness, SelfAuthoredProcedureHarness):
        acq_proc = w1_harness.self_procedure
        acq_proc_v = w1_harness.procedure_version
        acq_proc_hist = copy.deepcopy(w1_harness.procedure_history)
    elif any(a in arms_needed for a in ("editable_proc_v3", "frozen_proc_v3")):
        # Replay the live W1 public records through SelfAuthoredProcedureHarness so P_W1 is authored strictly from public W1!
        tmp_p = SelfAuthoredProcedureHarness(mode="editable", review_cadence=4, recent_k=8)
        for rec in out["pub_records"]:
            m = tmp_p.on_episode_complete(rec, client)
            out["reflection_calls"] += int(m.get("reflection_calls", 0))
            out["reflection_usd"] += float(m.get("reflection_usd", 0.0))
        acq_proc = tmp_p.self_procedure
        acq_proc_v = tmp_p.procedure_version
        acq_proc_hist = copy.deepcopy(tmp_p.procedure_history)

    acq_cards: list[dict[str, Any]] = []
    if "scoped_cards_v3" in arms_needed:
        tmp_c = ScopedCardBookV3Harness(reflect_every=4)
        for rec in out["pub_records"]:
            m = tmp_c.on_episode_complete(rec, client)
            out["reflection_calls"] += int(m.get("reflection_calls", 0))
            out["reflection_usd"] += float(m.get("reflection_usd", 0.0))
        acq_cards = copy.deepcopy(tmp_c.cards)

    return {
        "suite": suite,
        "model": model_name,
        "seed": seed,
        "prefix_family": prefix_family,
        "t_fork": t_fork,
        "fork_holdings": out["end_holdings"],
        "fork_nav": out["end_nav"],
        "w1_dp_utility_pct": w1_util,
        "w1_probes": sum(r["probes_used"] for r in out["logs"]),
        "w1_usd": round(out["acting_usd"] + out["reflection_usd"], 6),
        "acquired_procedure_v": acq_proc_v,
        "acquired_self_procedure": acq_proc,
        "acquired_procedure_history": acq_proc_hist,
        "acquired_cards": acq_cards,
        "records": out["pub_records"],
        "w1_logs": out["logs"],
    }


def instantiate_arm_from_live_checkpoint(
    arm_name: str,
    ckpt: dict[str, Any],
    client: MeteredLLMClient,
) -> Any:
    """Reconstruct exact pre-fork harness state strictly from the live-acquired `W1` public checkpoint."""
    records: list[PublicEpisodeRecord] = ckpt["records"]

    if arm_name == "factual_ledger_v3":
        h = FactualLedgerV3Harness(recent_k=8)
        for r in records:
            h.on_episode_complete(r, client)
        return h

    if arm_name in ("editable_proc_v3", "frozen_proc_v3"):
        mode = "editable" if arm_name == "editable_proc_v3" else "frozen"
        h_p = SelfAuthoredProcedureHarness(
            mode=mode,
            review_cadence=4,
            recent_k=8,
            initial_procedure=ckpt.get("acquired_self_procedure", ""),
        )
        h_p.procedure_version = int(ckpt.get("acquired_procedure_v", 1 if h_p.self_procedure else 0))
        h_p.procedure_history = copy.deepcopy(ckpt.get("acquired_procedure_history", []))
        for r in records:
            h_p.factual.on_episode_complete(r, client)
        return h_p

    if arm_name == "scoped_cards_v3":
        h_c = ScopedCardBookV3Harness(reflect_every=3)
        h_c._history = list(records)
        h_c.cards = copy.deepcopy(ckpt.get("acquired_cards", []))
        return h_c

    if arm_name == "sgeh_directive_v3":
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_d:
            dummy_ledger = CostLedger(results_root=Path(tmp_d), run_id="_tmp_offline", phase="dev", max_usd=0.01)
            mock_c = MeteredLLMClient(model="mock", ledger=dummy_ledger, thinking_budget=0)
            h_d = DirectiveSGEHV3Harness(recent_k=8, enable_hazard=True)
            for r in records:
                h_d.on_episode_complete(r, mock_c)
        return h_d

    raise ValueError(f"Unsupported arm '{arm_name}'.")


def evaluate_post_checkpoint_branch(
    *,
    suite: str,
    model_name: str,
    arm_name: str,
    world_mode: str,
    seed: int,
    t_fork: int,
    h_post: int,
    ckpt: dict[str, Any],
    base_harness: Any,
    client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    if suite == "voi_fork":
        env: AlphaDeskV3ForkEnv = AlphaDeskV3ForkEnv(
            fork_mode=world_mode,
            seed=seed,
            t_fork=t_fork,
            h_post=h_post,
            initial_post_holdings=ckpt["fork_holdings"],
        )
    else:
        w2_l = h_post // 2
        w3_l = h_post - w2_l
        env = AlphaDeskV3ThreeWindowEnv(
            stream_mode=world_mode,
            seed=seed,
            w1_len=t_fork,
            w2_len=w2_l,
            w3_len=w3_l,
            initial_post_holdings=ckpt["fork_holdings"],
        )
    env.set_fork_state(ckpt["fork_holdings"], ckpt["fork_nav"])
    harness = copy.deepcopy(base_harness)

    out = _run_episode_loop(
        env=env,
        ep_range=range(t_fork, t_fork + h_post),
        harness=harness,
        client=client,
        method_tag=f"{suite}_{arm_name}",
        on_ep_done=on_ep_done,
    )
    logs = out["logs"]

    dp_post_cum = sum(r["dp_oracle_step_net"] for r in logs)
    agent_post_cum = sum(r["expected_net_no_noise"] for r in logs)
    post_dp_utility_pct = round(100.0 * agent_post_cum / max(1.0, dp_post_cum), 2)

    # Window-split metrics for 3-window streams (`W2` vs `W3`)
    w2_logs = [r for r in logs if r["window"] == 1]
    w3_logs = [r for r in logs if r["window"] == 2]
    w2_util = (
        round(100.0 * sum(r["expected_net_no_noise"] for r in w2_logs) / max(1.0, sum(r["dp_oracle_step_net"] for r in w2_logs)), 2)
        if w2_logs
        else post_dp_utility_pct
    )
    w3_util = (
        round(100.0 * sum(r["expected_net_no_noise"] for r in w3_logs) / max(1.0, sum(r["dp_oracle_step_net"] for r in w3_logs)), 2)
        if w3_logs
        else post_dp_utility_pct
    )

    calm_norm = [
        r for r in logs
        if not r["is_high_vol"] and not r["is_liquidity_freeze"] and not r["is_hidden_transient_dip"]
    ]
    calm_opt_pct = round(100.0 * sum(1 for r in calm_norm if r["optimal_policy_match"]) / max(1, len(calm_norm)), 2)
    stale_lock_pct = round(100.0 * sum(1 for r in calm_norm if r["stale_policy_match"]) / max(1, len(calm_norm)), 2)

    dip_rows = [r for r in logs if r["is_hidden_transient_dip"] or r["is_liquidity_freeze"]]
    dip_probes = sum(r["probes_used"] for r in dip_rows)
    dip_turnover_pts = round(sum(r["turnover_fee"] for r in dip_rows), 3)

    # Check if incumbent holdings were abandoned immediately after a 2-session hidden dip (`logs[2]`, i.e. `ep = t_fork + 2`)
    post_dip_abandoned = False
    if world_mode in ("hidden_transient_dip", "dip_then_reversal_3w") and len(logs) >= 3:
        w_after_dip = logs[2].get("submitted_weights", {})
        if isinstance(w_after_dip, dict) and w_after_dip:
            post_dip_abandoned = l1_turnover_half(w_after_dip, ckpt["fork_holdings"]) > 0.25

    incumbent_held_at_fork = {s for s, w in ckpt["fork_holdings"].items() if float(w) >= 0.25}
    non_inc_probes = 0
    red_inc_probes = 0
    for r in logs:
        if not r["is_high_vol"]:
            for ps in r["probed_strategies"]:
                if ps in incumbent_held_at_fork:
                    red_inc_probes += 1
                else:
                    non_inc_probes += 1

    proc_history = getattr(harness, "procedure_history", []) if isinstance(harness, SelfAuthoredProcedureHarness) else []
    final_proc = getattr(harness, "self_procedure", "") if isinstance(harness, SelfAuthoredProcedureHarness) else ""
    final_proc_v = getattr(harness, "procedure_version", 0) if isinstance(harness, SelfAuthoredProcedureHarness) else 0

    return {
        "suite": suite,
        "model": model_name,
        "provider_model_versions": sorted(client.observed_model_versions) or [model_name],
        "arm": arm_name,
        "world_mode": world_mode,
        "seed": seed,
        "t_fork": t_fork,
        "episodes_post": h_post,
        "w1_live_dp_utility_pct": ckpt["w1_dp_utility_pct"],
        "post_dp_utility_pct": post_dp_utility_pct,
        "w2_dp_utility_pct": w2_util,
        "w3_dp_utility_pct": w3_util,
        "calm_optimal_pct": calm_opt_pct,
        "stale_lock_in_pct": stale_lock_pct,
        "probes_total": sum(r["probes_used"] for r in logs),
        "calm_non_incumbent_probes": non_inc_probes,
        "calm_redundant_incumbent_probes": red_inc_probes,
        "dip_probes": dip_probes,
        "dip_turnover_pts": dip_turnover_pts,
        "post_dip_abandoned_incumbent": post_dip_abandoned,
        "turnover_pts_total": round(sum(r["turnover_fee"] for r in logs), 2),
        "risk_tool_calls": out["risk_tool_calls"],
        "compliance_violations": out["compliance_violations"],
        "runner_interventions": out["runner_interventions"],
        "acting_calls": out["acting_calls"],
        "reflection_calls": out["reflection_calls"],
        "thinking_tokens": out["thinking_tokens"],
        "usd": round(out["acting_usd"] + out["reflection_usd"], 6),
        "initial_self_procedure": ckpt.get("acquired_self_procedure", ""),
        "final_self_procedure": final_proc,
        "final_procedure_version": final_proc_v,
        "procedure_history": proc_history,
        "curve_utility": [round(100.0 * r["expected_net_no_noise"] / max(1.0, r["dp_oracle_step_net"]), 1) for r in logs],
        "logs": logs,
    }


def _bootstrap_ci(vals: list[float], seed: int = 20261125) -> dict[str, Any]:
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


def summarize_suite_report(
    *,
    suite: str,
    run_id: str,
    t_fork: int,
    h_post: int,
    arms_order: list[str],
    modes_order: list[str],
    records: list[dict[str, Any]],
    total_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    models = sorted({str(r["model"]) for r in records})
    seeds = sorted({int(r["seed"]) for r in records})
    modes = [m for m in modes_order if any(r["world_mode"] == m for r in records)]

    by_model_mode: dict[str, dict[str, Any]] = {}
    paired_contrasts: dict[str, dict[str, Any]] = {}
    provider_versions: dict[str, list[str]] = {}

    contrast_specs = [
        ("editable_proc_v3", "frozen_proc_v3", "editable_minus_frozen_proc"),
        ("editable_proc_v3", "factual_ledger_v3", "editable_proc_minus_no_proc"),
        ("frozen_proc_v3", "factual_ledger_v3", "frozen_proc_minus_no_proc"),
        ("factual_ledger_v3", "scoped_cards_v3", "factual_minus_cardbook"),
        ("sgeh_directive_v3", "editable_proc_v3", "directive_minus_editable_proc"),
        ("sgeh_directive_v3", "factual_ledger_v3", "directive_minus_factual"),
    ]

    for mdl in models:
        m_recs = [r for r in records if r["model"] == mdl]
        vers: set[str] = set()
        for r in m_recs:
            vers.update(r.get("provider_model_versions", []))
        provider_versions[mdl] = sorted(vers)
        by_model_mode[mdl] = {}
        paired_contrasts[mdl] = {}

        for wm in modes:
            sub_wm = [r for r in m_recs if r["world_mode"] == wm]
            amap: dict[str, Any] = {}
            for arm in arms_order:
                sub = [r for r in sub_wm if r["arm"] == arm]
                if not sub:
                    continue
                amap[arm] = {
                    "n_seeds": len(sub),
                    "w1_live_dp_utility_pct": round(statistics.mean(r["w1_live_dp_utility_pct"] for r in sub), 2),
                    "post_dp_utility_pct": round(statistics.mean(r["post_dp_utility_pct"] for r in sub), 2),
                    "w2_dp_utility_pct": round(statistics.mean(r["w2_dp_utility_pct"] for r in sub), 2),
                    "w3_dp_utility_pct": round(statistics.mean(r["w3_dp_utility_pct"] for r in sub), 2),
                    "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in sub), 2),
                    "stale_lock_in_pct": round(statistics.mean(r["stale_lock_in_pct"] for r in sub), 2),
                    "mean_probes": round(statistics.mean(r["probes_total"] for r in sub), 2),
                    "mean_non_inc_probes": round(statistics.mean(r["calm_non_incumbent_probes"] for r in sub), 2),
                    "mean_redundant_inc_probes": round(statistics.mean(r["calm_redundant_incumbent_probes"] for r in sub), 2),
                    "mean_dip_probes": round(statistics.mean(r["dip_probes"] for r in sub), 2),
                    "mean_dip_turnover_pts": round(statistics.mean(r["dip_turnover_pts"] for r in sub), 2),
                    "post_dip_abandon_rate_pct": round(100.0 * sum(1 for r in sub if r["post_dip_abandoned_incumbent"]) / len(sub), 1),
                    "mean_turnover_pts": round(statistics.mean(r["turnover_pts_total"] for r in sub), 2),
                    "compliance_violations_sum": sum(int(r["compliance_violations"]) for r in sub),
                    "runner_interventions_sum": sum(int(r["runner_interventions"]) for r in sub),
                    "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in sub), 2),
                    "mean_final_proc_version": round(statistics.mean(r["final_procedure_version"] for r in sub), 2),
                    "mean_usd": round(statistics.mean(r["usd"] for r in sub), 4),
                }
            by_model_mode[mdl][wm] = amap

            by_sa = {(int(r["seed"]), str(r["arm"])): r for r in sub_wm}
            wm_con: dict[str, Any] = {}
            for a_hi, a_lo, cname in contrast_specs:
                diffs = [
                    by_sa[(sd, a_hi)]["post_dp_utility_pct"] - by_sa[(sd, a_lo)]["post_dp_utility_pct"]
                    for sd in seeds
                    if (sd, a_hi) in by_sa and (sd, a_lo) in by_sa
                ]
                if diffs:
                    wm_con[cname] = _bootstrap_ci(diffs)
            paired_contrasts[mdl][wm] = wm_con

        # Stationary-adjusted Difference-in-Differences (DiD)
        did_map: dict[str, Any] = {}
        by_msa = {(str(r["world_mode"]), int(r["seed"]), str(r["arm"])): r for r in m_recs}
        stat_pairs = (
            [
                ("hidden_transient_dip", "stationary"),
                ("abrupt_reversal", "stationary"),
                ("opportunity_rise", "opportunity_stationary"),
            ]
            if suite == "voi_fork"
            else [
                ("dip_then_reversal_3w", "stationary_3w"),
                ("double_reversal_3w", "stationary_3w"),
                ("opp_double_rise_3w", "opp_stationary_3w"),
            ]
        )
        for sh_m, st_m in stat_pairs:
            for a_hi, a_lo, cname in contrast_specs:
                d_vals = [
                    (by_msa[(sh_m, sd, a_hi)]["post_dp_utility_pct"] - by_msa[(sh_m, sd, a_lo)]["post_dp_utility_pct"])
                    - (by_msa[(st_m, sd, a_hi)]["post_dp_utility_pct"] - by_msa[(st_m, sd, a_lo)]["post_dp_utility_pct"])
                    for sd in seeds
                    if all((mm, sd, aa) in by_msa for mm in (sh_m, st_m) for aa in (a_hi, a_lo))
                ]
                if d_vals:
                    did_map[f"did_{sh_m}__{cname}"] = _bootstrap_ci(d_vals)
        paired_contrasts[mdl]["did_summary"] = did_map

    clean_records = [{k: v for k, v in r.items() if k != "logs"} for r in records]
    return {
        "experiment_type": f"post_meeting_{suite}",
        "suite": suite,
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


def render_suite_markdown(rep: dict[str, Any]) -> str:
    arm_labels = {
        "scoped_cards_v3": "ScopedCardBook (Baseline)",
        "factual_ledger_v3": "**No-Procedure / Factual-Ledger (Ours — Autonomy Baseline)**",
        "frozen_proc_v3": "Frozen-Procedure (Matched-Budget Checkpoint Control)",
        "editable_proc_v3": "**Editable-Procedure (Ours — Self-Revising Procedure)**",
        "sgeh_directive_v3": "**Directive-SGEH (Ours — Engineered Controller Reference)**",
    }
    title = (
        "Hidden-Ambiguity VOI Fork (`hidden_transient_dip` vs `abrupt_reversal`, Unscripted Live `W1`)"
        if rep["suite"] == "voi_fork"
        else "Self-Authored Editable vs. Frozen Exploration Procedure (`W1 -> W2 -> W3` Multi-Shift & Stationary)"
    )
    lines = [
        f"# {title}: `{rep['run_id']}`",
        "",
        f"- **Models & Provider Versions:** `{json.dumps(rep['provider_model_versions'])}`",
        f"- **Seeds:** `{rep['seeds']}` (`N={rep['n_seeds']}` Latin-square) | **Live `W1`:** `{rep['t_fork']}` eps | **Post-Fork:** `{rep['h_post']}` eps | **Total Spend:** `${rep['total_run_usd']:.4f}`",
        f"- **Observed Tokens:** `in={rep['observed_run_input_tokens']:,}`, `out={rep['observed_run_output_tokens']:,}`, `thinking={rep['observed_run_thinking_tokens']:,}`",
        "",
    ]
    for mdl in rep["models"]:
        ver_str = ", ".join(rep["provider_model_versions"].get(mdl, [mdl]))
        lines.append(f"## Model: `{mdl}` (`{ver_str}`)")
        lines.append("")
        mm = rep["by_model_mode"].get(mdl, {})
        pc = rep["paired_contrasts"].get(mdl, {})
        for wm, amap in mm.items():
            if not amap:
                continue
            lines.append(f"### Post-Checkpoint Stream: `{wm}`")
            if rep["suite"] == "procedure_3w":
                lines.append(
                    "| Arm | Post-Ckpt DP (%) | W2 / W3 DP (%) | CalmOpt (%) | Probes (NonInc / Red) | Dip Prb / Turn | Viol / Intv | Ref Calls | Proc v# | USD |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for ak, d in amap.items():
                    lines.append(
                        f"| {arm_labels.get(ak, ak)} | **{d['post_dp_utility_pct']:.1f}%** | "
                        f"`{d['w2_dp_utility_pct']:.1f}%` / `{d['w3_dp_utility_pct']:.1f}%` | {d['calm_optimal_pct']:.1f}% | "
                        f"{d['mean_probes']:.1f} (`{d['mean_non_inc_probes']:.1f}` / `{d['mean_redundant_inc_probes']:.1f}`) | "
                        f"`{d['mean_dip_probes']:.1f}` / `{d['mean_dip_turnover_pts']:.1f}` | "
                        f"`{d['compliance_violations_sum']}` / `{d['runner_interventions_sum']}` | "
                        f"{d['mean_reflection_calls']:.1f} | `v{d['mean_final_proc_version']:.1f}` | `${d['mean_usd']:.4f}` |"
                    )
            else:
                lines.append(
                    "| Arm | Live W1 DP (%) | Post-Fork DP (%) | CalmOpt (%) | Probes (NonInc / Red) | Dip Prb / Turn / Abandon% | Viol / Intv | Ref Calls | USD |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
                for ak, d in amap.items():
                    lines.append(
                        f"| {arm_labels.get(ak, ak)} | {d['w1_live_dp_utility_pct']:.1f}% | **{d['post_dp_utility_pct']:.1f}%** | "
                        f"{d['calm_optimal_pct']:.1f}% | "
                        f"{d['mean_probes']:.1f} (`{d['mean_non_inc_probes']:.1f}` / `{d['mean_redundant_inc_probes']:.1f}`) | "
                        f"`{d['mean_dip_probes']:.1f}` / `{d['mean_dip_turnover_pts']:.1f}` / `{d['post_dip_abandon_rate_pct']:.0f}%` | "
                        f"`{d['compliance_violations_sum']}` / `{d['runner_interventions_sum']}` | "
                        f"{d['mean_reflection_calls']:.1f} | `${d['mean_usd']:.4f}` |"
                    )
            lines.append("")
            for ck, cv in pc.get(wm, {}).items():
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


def run_suite_batch(
    *,
    suite: str,
    run_id: str,
    models: list[str],
    arms: list[str],
    world_modes: list[str],
    seeds: list[int],
    t_fork: int,
    h_post: int,
    max_usd: float,
    workers: int,
    thinking_budget: int,
    checkpoint_file: Path | None,
    distill_transfer_out: Path | None,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)

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

    # Determine required live W1 checkpoints per (model, seed, prefix_family)
    needed_families = sorted({_prefix_family(wm, suite) for wm in world_modes})
    loaded_ckpts: dict[tuple[str, int, str], dict[str, Any]] = {}
    if checkpoint_file and checkpoint_file.is_file():
        raw_map = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        for k_str, v_obj in raw_map.items():
            parts = k_str.split("|")
            if len(parts) == 3:
                loaded_ckpts[(parts[0], int(parts[1]), parts[2])] = deserialize_checkpoint(v_obj)

    missing_ckpt_keys = [
        (mdl, sd, pf)
        for mdl in models
        for sd in seeds
        for pf in needed_families
        if (mdl, sd, pf) not in loaded_ckpts
    ]

    w1_eps_total = len(missing_ckpt_keys) * t_fork
    post_eps_total = len(models) * len(arms) * len(world_modes) * len(seeds) * h_post
    total_units = max(1, w1_eps_total + post_eps_total)
    done_units = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_units
        with lock:
            done_units += 1
            if done_units % 4 == 0 or done_units == total_units:
                ledger.update_progress(
                    completed_units=min(done_units, total_units),
                    total_units=total_units,
                    status="running",
                    current_label=lbl,
                    log_every=28,
                )

    traces_dir = ledger.results_root / "raw" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Acquire any missing live W1 checkpoints strictly from public observations!
    if missing_ckpt_keys:
        print(f"[{run_id}] Acquiring {len(missing_ckpt_keys)} live unscripted W1 checkpoints (t_fork={t_fork})...", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, min(len(missing_ckpt_keys), workers))) as pool:
            futs = {
                pool.submit(
                    acquire_live_w1_checkpoint,
                    suite=suite,
                    model_name=mdl,
                    seed=sd,
                    prefix_family=pf,
                    t_fork=t_fork,
                    h_post=h_post,
                    client=clients[mdl],
                    arms_needed=arms,
                    on_ep_done=lambda m=mdl, s=sd, p=pf: _tick(f"W1_live {m} s{s} {p}"),
                ): (mdl, sd, pf)
                for (mdl, sd, pf) in missing_ckpt_keys
            }
            for fut in as_completed(futs):
                key = futs[fut]
                ck = fut.result()
                loaded_ckpts[key] = ck
                print(
                    f"  [W1-LIVE] {key[0]} | s{key[1]} | {key[2]} -> W1_DP={ck['w1_dp_utility_pct']:.1f}% "
                    f"Prb={ck['w1_probes']} Proc_v={ck['acquired_procedure_v']} (${ck['w1_usd']:.4f})",
                    flush=True,
                )

    # Persist all live W1 checkpoints to run_dir (and optional shared `checkpoint_file`) for 100% auditability!
    serialized_ckpts = {
        f"{m}|{s}|{p}": serialize_checkpoint(ck)
        for (m, s, p), ck in sorted(loaded_ckpts.items())
    }
    atomic_write_json(ledger.run_dir / "w1_live_checkpoints.json", serialized_ckpts)
    if checkpoint_file is not None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(checkpoint_file, serialized_ckpts)

    # Build arm base states per (model, seed, prefix_family, arm)
    arm_state_cache: dict[tuple[str, int, str, str], Any] = {}
    for (mdl, sd, pf), ck in loaded_ckpts.items():
        if mdl not in clients:
            continue
        for arm in arms:
            arm_state_cache[(mdl, sd, pf, arm)] = instantiate_arm_from_live_checkpoint(arm, ck, clients[mdl])

    # Phase 2: Run post-checkpoint branches in parallel
    jobs = [
        (mdl, arm, wm, sd)
        for mdl in models
        for sd in seeds
        for wm in world_modes
        for arm in arms
    ]
    raw_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), workers))) as pool:
        futs_b = []
        for mdl, arm, wm, sd in jobs:
            pf = _prefix_family(wm, suite)
            ck = loaded_ckpts[(mdl, sd, pf)]
            base_h = arm_state_cache[(mdl, sd, pf, arm)]
            futs_b.append(
                pool.submit(
                    evaluate_post_checkpoint_branch,
                    suite=suite,
                    model_name=mdl,
                    arm_name=arm,
                    world_mode=wm,
                    seed=sd,
                    t_fork=t_fork,
                    h_post=h_post,
                    ckpt=ck,
                    base_harness=base_h,
                    client=clients[mdl],
                    on_ep_done=lambda m=mdl, a=arm, w=wm, s=sd: _tick(f"{m} {w} s{s} {a}"),
                )
            )
        for fut in as_completed(futs_b):
            res = fut.result()
            raw_results.append(res)
            mdl, arm, wm, sd = res["model"], res["arm"], res["world_mode"], res["seed"]
            print(
                f"  -> [{mdl:16s} | {wm:22s} | s{sd} | {arm:18s}] "
                f"PostDP={res['post_dp_utility_pct']:5.1f}% (W2={res['w2_dp_utility_pct']:5.1f}%/W3={res['w3_dp_utility_pct']:5.1f}%) | "
                f"Prb={res['probes_total']:2d} DipPrb={res['dip_probes']} DipTurn={res['dip_turnover_pts']:.1f} | "
                f"ProcV={res['final_procedure_version']} | Cell=${res['usd']:.4f} Run=${ledger.run_usd:.4f}",
                flush=True,
            )
            slug = f"{suite}_{wm}__{arm}__{mdl.replace('/', '_')}_s{sd}"
            with open(traces_dir / f"{slug}.jsonl", "w", encoding="utf-8") as fh:
                for row in res["logs"]:
                    fh.write(json.dumps(row) + "\n")
            atomic_write_json(
                cells_dir / f"{slug}.json",
                {
                    **{k: v for k, v in res.items() if k != "logs"},
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    # Save verbatim self-authored procedure audit file
    proc_audit = [
        {
            "model": r["model"],
            "seed": r["seed"],
            "world_mode": r["world_mode"],
            "arm": r["arm"],
            "initial_self_procedure": r["initial_self_procedure"],
            "final_self_procedure": r["final_self_procedure"],
            "final_procedure_version": r["final_procedure_version"],
            "procedure_history": r["procedure_history"],
        }
        for r in sorted(raw_results, key=lambda x: (x["model"], x["world_mode"], x["seed"], x["arm"]))
        if r["arm"] in ("editable_proc_v3", "frozen_proc_v3")
    ]
    if proc_audit:
        atomic_write_json(ledger.run_dir / "procedure_versions_audit.json", proc_audit)

    # Optional: Distill and freeze domain-agnostic transfer procedure BEFORE ServeLearnBench
    if distill_transfer_out is not None and proc_audit:
        editable_rows = [
            r for r in raw_results
            if r["arm"] == "editable_proc_v3" and r["world_mode"] in ("dip_then_reversal_3w", "double_reversal_3w", "abrupt_reversal")
            and r["final_self_procedure"]
        ] or [r for r in raw_results if r["final_self_procedure"]]
        if editable_rows:
            src_procs = [r["final_self_procedure"] for r in editable_rows[:4]]
            sample_ledger = editable_rows[0]["logs"][-1]["pre_episode_harness_context"] if editable_rows[0]["logs"] else ""
            distilled = distill_domain_agnostic_procedure(
                client=clients[models[0]],
                source_procedures=src_procs,
                source_ledger_excerpt=sample_ledger,
            )
            distill_transfer_out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(distill_transfer_out, distilled)
            print(
                f"  [FROZEN TRANSFER PROCEDURE] Saved SHA256={distilled['sha256'][:16]}... -> {distill_transfer_out}",
                flush=True,
            )

    raw_results.sort(key=lambda r: (r["model"], r["world_mode"], r["seed"], r["arm"]))
    rep = summarize_suite_report(
        suite=suite,
        run_id=run_id,
        t_fork=t_fork,
        h_post=h_post,
        arms_order=arms,
        modes_order=world_modes,
        records=raw_results,
        total_usd=ledger.run_usd,
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "suite_report.json", rep)
    md = render_suite_markdown(rep)
    (ledger.run_dir / "summary.md").write_text(md, encoding="utf-8")
    ledger.update_progress(
        completed_units=total_units,
        total_units=total_units,
        status="completed",
        current_label=f"completed {run_id}",
    )
    return rep


def aggregate_suite_runs(run_ids: list[str], out_run_id: str, results_root: Path) -> Path:
    all_records: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    suite = "voi_fork"
    t_fork = 14
    h_post = 16
    arms_seen: list[str] = []
    modes_seen: list[str] = []

    for rid in run_ids:
        rep_p = results_root / "runs" / rid / "suite_report.json"
        d = json.loads(rep_p.read_text(encoding="utf-8"))
        suite = str(d.get("suite", suite))
        t_fork = int(d.get("t_fork", t_fork))
        h_post = int(d.get("h_post", h_post))
        tot_usd += float(d.get("total_run_usd", 0.0))
        in_t += int(d.get("observed_run_input_tokens", 0))
        out_t += int(d.get("observed_run_output_tokens", 0))
        th_t += int(d.get("observed_run_thinking_tokens", 0))
        for rec in d.get("records", []):
            all_records.append(rec)
            if rec["arm"] not in arms_seen:
                arms_seen.append(rec["arm"])
            if rec["world_mode"] not in modes_seen:
                modes_seen.append(rec["world_mode"])

    dedup = {(r["model"], r["arm"], r["world_mode"], int(r["seed"])): r for r in all_records}
    merged = sorted(dedup.values(), key=lambda r: (r["model"], r["world_mode"], int(r["seed"]), r["arm"]))
    std_arms = [a for a in ("scoped_cards_v3", "factual_ledger_v3", "frozen_proc_v3", "editable_proc_v3", "sgeh_directive_v3") if a in arms_seen]
    std_modes = [m for m in (list(VOI_FORK_MODES) + list(PROC_3W_MODES)) if m in modes_seen]

    rep = summarize_suite_report(
        suite=suite,
        run_id=out_run_id,
        t_fork=t_fork,
        h_post=h_post,
        arms_order=std_arms or arms_seen,
        modes_order=std_modes or modes_seen,
        records=merged,
        total_usd=tot_usd,
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "suite_report.json", rep)

    # Also merge procedure_versions_audit.json if present across shards
    merged_procs: list[dict[str, Any]] = []
    for rid in run_ids:
        pa = results_root / "runs" / rid / "procedure_versions_audit.json"
        if pa.is_file():
            merged_procs.extend(json.loads(pa.read_text(encoding="utf-8")))
    if merged_procs:
        atomic_write_json(out_dir / "procedure_versions_audit.json", merged_procs)

    md = render_suite_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Post-Meeting Hidden-Ambiguity VOI & Editable-Procedure Suites")
    parser.add_argument("--suite", choices=("voi_fork", "procedure_3w"), default="voi_fork")
    parser.add_argument("--run-id", default="voi_fork_dev", help="Experiment run ID")
    parser.add_argument("--models", default="gemini-3.8-flash", help="Comma-separated model IDs")
    parser.add_argument("--arms", default="", help="Comma-separated arms (defaults per suite)")
    parser.add_argument("--modes", default="", help="Comma-separated world modes (defaults per suite)")
    parser.add_argument("--seeds", default="601,602,603,604,605,606", help="Comma-separated seeds")
    parser.add_argument("--t-fork", type=int, default=0, help="W1 length (default 14 for voi_fork, 12 for procedure_3w)")
    parser.add_argument("--h-post", type=int, default=0, help="Post-fork length (default 16 for voi_fork, 28 for procedure_3w)")
    parser.add_argument("--workers", type=int, default=12, help="Parallel threads")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Thinking token budget")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Hard USD guardrail per run")
    parser.add_argument("--checkpoint-file", default="", help="Optional shared JSON path to cache/load live W1 checkpoints")
    parser.add_argument("--distill-transfer-out", default="", help="Optional JSON output path for frozen domain-agnostic transfer procedure")
    parser.add_argument("--results-root", default=str(ROOT / "results"), help="Results directory root")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated shard run IDs to merge")
    args = parser.parse_args()

    res_root = Path(args.results_root)
    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        out_md = aggregate_suite_runs(rids, args.run_id, res_root)
        print(out_md.read_text(encoding="utf-8"))
        print(f"Saved aggregated report: {out_md}", flush=True)
        return

    default_arms = (
        ["scoped_cards_v3", "factual_ledger_v3", "editable_proc_v3", "sgeh_directive_v3"]
        if args.suite == "voi_fork"
        else ["factual_ledger_v3", "frozen_proc_v3", "editable_proc_v3", "sgeh_directive_v3"]
    )
    default_modes = (
        ["stationary", "hidden_transient_dip", "abrupt_reversal"]
        if args.suite == "voi_fork"
        else list(PROC_3W_MODES)
    )

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()] or default_arms
    modes = [x.strip() for x in args.modes.split(",") if x.strip()] or default_modes
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    t_fork = args.t_fork if args.t_fork > 0 else (14 if args.suite == "voi_fork" else 12)
    h_post = args.h_post if args.h_post > 0 else (16 if args.suite == "voi_fork" else 28)
    ckpt_p = Path(args.checkpoint_file) if args.checkpoint_file else None
    distill_p = Path(args.distill_transfer_out) if args.distill_transfer_out else None

    try:
        rep = run_suite_batch(
            suite=args.suite,
            run_id=args.run_id,
            models=models,
            arms=arms,
            world_modes=modes,
            seeds=seeds,
            t_fork=t_fork,
            h_post=h_post,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            checkpoint_file=ckpt_p,
            distill_transfer_out=distill_p,
            results_root=res_root,
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET CIRCUIT BREAKER] {exc}", flush=True)
        sys.exit(2)

    print("\n" + render_suite_markdown(rep), flush=True)


if __name__ == "__main__":
    main()
