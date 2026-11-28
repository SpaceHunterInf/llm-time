#!/usr/bin/env python3
"""Single-Lifetime `H=60` Two-Level Reviewer Self-Improvement Runner & Paired `W4` Counterfactual Replay.

Executes the pre-registered experiment from `setup-debates/2026-11-26-decomposed-procedure-and-vbq-resolution.md`:
  - `AlphaDeskV3MetaHorizonEnv` (`H = 60`, private seed-jittered 4-window schedule across `Seeds 701..706`):
    * `drift_meta_60`: `W1` (`A`, calm acquisition) -> `W2` (`2`-day hidden calm dip -> `A`, temporary loss) ->
      `W3` (`B`, lasting change) -> `W4` (`A'`, partial recurrence later surprise testing both revision & reuse).
    * `stationary_meta_60`: Matched `60`-session stream remaining at `A` throughout.
  - Common checkpoint fork at `t_ckpt = w1_end` copying `(N_ckpt, R_ckpt, observations, holdings)` into 4 arms:
    1. `factual_ledger_v3` (`Facts-Only` — no `task_note`, no `review_rule`)
    2. `frozen_reviewer_v3` (`Frozen-Reviewer` — `task_note` rewritten every 4 sessions, `review_rule` frozen at `R_ckpt`)
    3. `editable_reviewer_v3` (`Editable-Reviewer [Ours]` — both `task_note` and reviewer-only `review_rule` editable)
    4. `sgeh_vbq_v3` (`Directive-SGEH-VBQ` — hand-written upper-reference controller)
  - Logs `R_pre_w4` right before `W4` begins and executes a paired counterfactual review + 4-session inter-review actor
    rollout at the first `W4` post-surprise review step comparing `R_ckpt` (old rule) vs `R_pre_w4` (new rule) on the
    exact same `(observations, pre_review_task_note)`.
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
    AlphaDeskV3MetaHorizonEnv,
    STRATEGIES_V3,
    l1_turnover_half,
)
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.procedure_harness import TwoLevelReviewerHarness  # noqa: E402
from src.harnesses.sgeh_v3 import (  # noqa: E402
    DirectiveSGEHV3Harness,
    FactualLedgerV3Harness,
    normalize_weights_v3,
)

META_MODES = ("drift_meta_60", "stationary_meta_60", "drift_dip_c_60")
DEFAULT_ARMS = ("factual_ledger_v3", "frozen_reviewer_v3", "editable_reviewer_v3", "sgeh_vbq_v3")


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
    if not already_probed and "guided_by=" in sys_prompt:
        tag = sys_prompt.split("guided_by=", 1)[1][:6]
        idx = int(tag, 16) % len(STRATEGIES_V3)
        if idx % 2 == 0:
            cand = STRATEGIES_V3[idx]
            return {"tool": "probe_strategy_execution", "args": {"strategy": cand}, "rationale": f"Probe {cand}."}
    w = normalize_weights_v3(cur_h, "STRAT_S1", "STRAT_S2")
    return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Submit compliant weights."}


def _execute_single_episode(
    *,
    env: AlphaDeskV3MetaHorizonEnv,
    ep: int,
    harness: Any,
    client: MeteredLLMClient,
    method_tag: str,
) -> dict[str, Any]:
    spec = env.start_episode(ep)
    pre_state = _snapshot_harness(harness)
    h_ctx = harness.build_context(spec)
    sys_prompt = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_ctx)

    steps: list[dict[str, Any]] = []
    done = False
    final_score = 0.0
    priv: dict[str, Any] = {}
    acting_calls = 0
    acting_usd = 0.0
    thinking_tokens = 0
    runner_interventions = 0
    compliance_violations = 0
    risk_tool_calls = 0

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
    probed_strats = [
        str(s["obs"].get("strategy"))
        for s in steps
        if s["tool"] == "probe_strategy_execution" and isinstance(s.get("obs"), dict) and "strategy" in s["obs"]
    ]
    sub_step = next((s for s in reversed(steps) if s["tool"] == "submit_allocation"), None)
    sub_weights = sub_step["args"].get("weights", {}) if (sub_step and isinstance(sub_step.get("args"), dict)) else {}

    return {
        "spec": spec,
        "pub_rec": pub_rec,
        "pre_state": pre_state,
        "h_ctx": h_ctx,
        "steps": steps,
        "final_score": final_score,
        "priv": priv,
        "probed_strats": probed_strats,
        "sub_weights": sub_weights,
        "acting_calls": acting_calls,
        "acting_usd": acting_usd,
        "thinking_tokens": thinking_tokens,
        "runner_interventions": runner_interventions,
        "compliance_violations": compliance_violations,
        "risk_tool_calls": risk_tool_calls,
    }


def _run_counterfactual_w4_replay(
    *,
    env_after_ep: AlphaDeskV3MetaHorizonEnv,
    harness_before_review: TwoLevelReviewerHarness,
    pub_rec: PublicEpisodeRecord,
    ep: int,
    old_rule: str,
    old_rule_version: int,
    new_rule: str,
    new_rule_version: int,
    client: MeteredLLMClient,
) -> dict[str, Any]:
    """Replay the first `W4` review under `R_ckpt` (old) vs `R_pre_w4` (new) + 4-ep inter-review actor rollout."""
    snap_h = copy.deepcopy(harness_before_review)
    snap_h.factual.on_episode_complete(pub_rec, client)
    pre_review_note = snap_h.task_note

    # Hold the displayed version label identical across both counterfactual branches (`snap_h.rule_version`)
    shared_disp_ver = int(new_rule_version)
    rev_old = snap_h.counterfactual_review(
        env_name=pub_rec.env_name,
        episode_idx=ep,
        override_review_rule=old_rule,
        override_rule_version=shared_disp_ver,
        client=client,
        method_suffix="cf_old_rule",
    )
    rev_new = snap_h.counterfactual_review(
        env_name=pub_rec.env_name,
        episode_idx=ep,
        override_review_rule=new_rule,
        override_rule_version=shared_disp_ver,
        client=client,
        method_suffix="cf_new_rule",
    )

    note_old = rev_old["proposed_task_note"] or pre_review_note
    note_new = rev_new["proposed_task_note"] or pre_review_note
    rollout_eps = list(range(ep + 1, min(env_after_ep.horizon, ep + 1 + snap_h.review_cadence)))

    def _roll_fixed_note(fixed_note: str, tag: str) -> dict[str, Any]:
        env_c = copy.deepcopy(env_after_ep)
        h_c = copy.deepcopy(snap_h)
        h_c.task_note = fixed_note
        h_c.note_version = snap_h.note_version + 1
        net_sum = 0.0
        dp_sum = 0.0
        probes_sum = 0
        turn_sum = 0.0
        usd_sum = 0.0
        step1_probed: list[str] = []
        step1_weights: dict[str, float] = {}
        for idx_r, r_ep in enumerate(rollout_eps):
            ep_out = _execute_single_episode(
                env=env_c,
                ep=r_ep,
                harness=h_c,
                client=client,
                method_tag=f"cf_w4_actor_{tag}",
            )
            h_c.factual.on_episode_complete(ep_out["pub_rec"], client)
            priv = ep_out["priv"]
            net_sum += float(priv.get("expected_net_no_noise", 0.0))
            dp_sum += float(priv.get("dp_oracle_step_net", 85.0))
            probes_sum += int(priv.get("probes_used", 0))
            turn_sum += float(priv.get("turnover_fee", 0.0))
            usd_sum += float(ep_out["acting_usd"])
            if idx_r == 0:
                step1_probed = list(ep_out["probed_strats"])
                step1_weights = dict(ep_out["sub_weights"])
        return {
            "step1_probed": step1_probed,
            "step1_weights": step1_weights,
            "block_expected_net": round(net_sum, 3),
            "block_dp_net": round(dp_sum, 3),
            "block_dp_pct": round(100.0 * net_sum / max(1.0, dp_sum), 2),
            "block_probes": probes_sum,
            "block_turnover_pts": round(turn_sum, 2),
            "actor_usd": usd_sum,
        }

    act_old = _roll_fixed_note(note_old, "old")
    act_new = _roll_fixed_note(note_new, "new")

    w_l1_diff = (
        round(l1_turnover_half(act_new["step1_weights"], act_old["step1_weights"]), 4)
        if act_new["step1_weights"] and act_old["step1_weights"]
        else 0.0
    )
    rule_changed = old_rule.strip() != new_rule.strip()
    note_changed = note_old.strip() != note_old_baseline(note_new)
    step1_action_changed = (act_new["step1_probed"] != act_old["step1_probed"]) or (w_l1_diff > 0.05)
    delta_net = round(act_new["block_expected_net"] - act_old["block_expected_net"], 2)
    delta_dp_pct = round(act_new["block_dp_pct"] - act_old["block_dp_pct"], 2)
    replay_usd = float(rev_old["usd"]) + float(rev_new["usd"]) + float(act_old["actor_usd"]) + float(act_new["actor_usd"])

    return {
        "replay_after_episode": ep + 1,
        "w4_start_session": env_after_ep.t_w4 + 1,
        "rollout_sessions": [e + 1 for e in rollout_eps],
        "old_review_rule_ckpt": old_rule,
        "new_review_rule_pre_w4": new_rule,
        "pre_review_task_note": pre_review_note,
        "task_note_under_old_rule": note_old,
        "task_note_under_new_rule": note_new,
        "rationale_old_rule": rev_old["edit_rationale"],
        "rationale_new_rule": rev_new["edit_rationale"],
        "rule_changed_before_w4": rule_changed,
        "note_revision_changed": note_changed,
        "step1_probed_old": act_old["step1_probed"],
        "step1_probed_new": act_new["step1_probed"],
        "step1_weights_l1_diff": w_l1_diff,
        "step1_action_changed": step1_action_changed,
        "old_block_dp_pct": act_old["block_dp_pct"],
        "new_block_dp_pct": act_new["block_dp_pct"],
        "old_block_probes": act_old["block_probes"],
        "new_block_probes": act_new["block_probes"],
        "delta_block_net_pts": delta_net,
        "delta_block_dp_pct": delta_dp_pct,
        "full_causal_chain_positive": bool(rule_changed and note_changed and step1_action_changed and delta_net > 0.0),
        "replay_usd": round(replay_usd, 6),
    }


def note_old_baseline(txt: str) -> str:
    return txt.strip()


def serialize_meta_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": ckpt["model"],
        "seed": ckpt["seed"],
        "horizon": ckpt["horizon"],
        "t_ckpt": ckpt["t_ckpt"],
        "window_ranges": ckpt["window_ranges"],
        "fork_holdings": ckpt["fork_holdings"],
        "fork_nav": ckpt["fork_nav"],
        "w1_dp_utility_pct": ckpt["w1_dp_utility_pct"],
        "w1_probes": ckpt["w1_probes"],
        "w1_usd": ckpt["w1_usd"],
        "acquired_note_version": ckpt["acquired_note_version"],
        "acquired_rule_version": ckpt["acquired_rule_version"],
        "acquired_task_note": ckpt["acquired_task_note"],
        "acquired_review_rule": ckpt["acquired_review_rule"],
        "acquired_review_history": ckpt["acquired_review_history"],
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


def deserialize_meta_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
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
        "window_ranges": [tuple(x) for x in raw["window_ranges"]],
        "records": recs,
    }


def acquire_live_w1_meta_checkpoint(
    *,
    model_name: str,
    seed: int,
    horizon: int,
    client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    """Acquire `W1` (`0 .. t_ckpt - 1`) live from blank `N_0="", R_0=""` using only public observations."""
    env = AlphaDeskV3MetaHorizonEnv(meta_mode="stationary_meta_60", seed=seed, horizon=horizon)
    t_ckpt = env.t_w2
    harness = TwoLevelReviewerHarness(mode="editable_rule", review_cadence=4, recent_k=8)

    logs: list[dict[str, Any]] = []
    pub_records: list[PublicEpisodeRecord] = []
    acting_usd = 0.0
    reflection_usd = 0.0

    for ep in range(0, t_ckpt):
        ep_out = _execute_single_episode(
            env=env,
            ep=ep,
            harness=harness,
            client=client,
            method_tag="live_w1_meta_acq",
        )
        acting_usd += float(ep_out["acting_usd"])
        pub_rec = ep_out["pub_rec"]
        pub_records.append(pub_rec)
        h_meta = harness.on_episode_complete(pub_rec, client)
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))
        priv = ep_out["priv"]
        logs.append(
            {
                "episode": ep + 1,
                "window": int(priv.get("window", env.window_index(ep))),
                "regime": priv.get("regime", ""),
                "pre_episode_harness_state": ep_out["pre_state"],
                "public_score": ep_out["final_score"],
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", 0.0)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "probes_used": int(priv.get("probes_used", 0)),
                "submitted_weights": ep_out["sub_weights"],
                "post_episode_harness_meta": h_meta,
                "steps": ep_out["steps"],
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    w1_dp = sum(r["dp_oracle_step_net"] for r in logs)
    w1_ag = sum(r["expected_net_no_noise"] for r in logs)
    w1_util = round(100.0 * w1_ag / max(1.0, w1_dp), 2)

    return {
        "model": model_name,
        "seed": seed,
        "horizon": horizon,
        "t_ckpt": t_ckpt,
        "window_ranges": list(env.window_ranges),
        "fork_holdings": dict(env._holdings),
        "fork_nav": float(env._cumulative_nav),
        "w1_dp_utility_pct": w1_util,
        "w1_probes": sum(r["probes_used"] for r in logs),
        "w1_usd": round(acting_usd + reflection_usd, 6),
        "acquired_note_version": harness.note_version,
        "acquired_rule_version": harness.rule_version,
        "acquired_task_note": harness.task_note,
        "acquired_review_rule": harness.review_rule,
        "acquired_review_history": copy.deepcopy(harness.review_history),
        "records": pub_records,
        "w1_logs": logs,
    }


def instantiate_meta_arm(arm_name: str, ckpt: dict[str, Any], client: MeteredLLMClient) -> Any:
    records: list[PublicEpisodeRecord] = ckpt["records"]
    if arm_name == "factual_ledger_v3":
        h = FactualLedgerV3Harness(recent_k=8)
        for r in records:
            h.on_episode_complete(r, client)
        return h

    if arm_name in ("editable_reviewer_v3", "frozen_reviewer_v3"):
        mode = "editable_rule" if arm_name == "editable_reviewer_v3" else "frozen_rule"
        h_r = TwoLevelReviewerHarness(
            mode=mode,
            review_cadence=4,
            recent_k=8,
            initial_task_note=ckpt["acquired_task_note"],
            initial_review_rule=ckpt["acquired_review_rule"],
        )
        h_r.note_version = int(ckpt["acquired_note_version"])
        h_r.rule_version = int(ckpt["acquired_rule_version"])
        h_r.review_history = copy.deepcopy(ckpt["acquired_review_history"])
        for r in records:
            h_r.factual.on_episode_complete(r, client)
        return h_r

    if arm_name == "sgeh_vbq_v3":
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_d:
            dummy_ledger = CostLedger(results_root=Path(tmp_d), run_id="_tmp_vbq", phase="dev", max_usd=0.01)
            mock_c = MeteredLLMClient(model="mock", ledger=dummy_ledger, thinking_budget=0)
            h_v = DirectiveSGEHV3Harness(recent_k=8, enable_hazard=True, verify_before_quarantine=True)
            for r in records:
                h_v.on_episode_complete(r, mock_c)
        return h_v

    raise ValueError(f"Unsupported meta-reviewer arm '{arm_name}'.")


def evaluate_meta_branch(
    *,
    model_name: str,
    arm_name: str,
    world_mode: str,
    seed: int,
    horizon: int,
    ckpt: dict[str, Any],
    base_harness: Any,
    client: MeteredLLMClient,
    enable_w4_replay: bool = True,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    env = AlphaDeskV3MetaHorizonEnv(
        meta_mode=world_mode,
        seed=seed,
        horizon=horizon,
        initial_post_holdings=ckpt["fork_holdings"],
    )
    env.set_fork_state(ckpt["fork_holdings"], ckpt["fork_nav"])
    harness = copy.deepcopy(base_harness)
    t_ckpt = env.t_w2

    logs: list[dict[str, Any]] = []
    acting_calls = 0
    acting_usd = 0.0
    reflection_calls = 0
    reflection_usd = 0.0
    thinking_tokens = 0
    runner_interventions = 0
    compliance_violations = 0
    risk_tool_calls = 0

    r_pre_w4 = getattr(harness, "review_rule", "")
    r_pre_w4_version = int(getattr(harness, "rule_version", 0))
    n_pre_w4 = getattr(harness, "task_note", "")
    w4_replay: dict[str, Any] | None = None

    for ep in range(t_ckpt, env.horizon):
        ep_out = _execute_single_episode(
            env=env,
            ep=ep,
            harness=harness,
            client=client,
            method_tag=f"meta60_{arm_name}",
        )
        acting_calls += int(ep_out["acting_calls"])
        acting_usd += float(ep_out["acting_usd"])
        thinking_tokens += int(ep_out["thinking_tokens"])
        runner_interventions += int(ep_out["runner_interventions"])
        compliance_violations += int(ep_out["compliance_violations"])
        risk_tool_calls += int(ep_out["risk_tool_calls"])

        pub_rec = ep_out["pub_rec"]

        # Paired counterfactual W4 review replay right BEFORE the first scheduled W4 review mutates `Editable-Reviewer`!
        if (
            enable_w4_replay
            and arm_name == "editable_reviewer_v3"
            and world_mode in ("drift_meta_60", "drift_dip_c_60")
            and isinstance(harness, TwoLevelReviewerHarness)
            and ep >= env.t_w4
            and w4_replay is None
            and (len(harness.factual.history) + 1) % harness.review_cadence == 0
            and (ep + 1) < env.horizon
        ):
            w4_replay = _run_counterfactual_w4_replay(
                env_after_ep=env,
                harness_before_review=harness,
                pub_rec=pub_rec,
                ep=ep,
                old_rule=ckpt["acquired_review_rule"],
                old_rule_version=int(ckpt["acquired_rule_version"]),
                new_rule=r_pre_w4,
                new_rule_version=r_pre_w4_version,
                client=client,
            )

        h_meta = harness.on_episode_complete(pub_rec, client)
        reflection_calls += int(h_meta.get("reflection_calls", 0))
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))

        # Snapshot `R_pre_w4` right after the final `W3` session (`t_w4 - 1`) completes (including any end-of-W3 review)
        if ep == env.t_w4 - 1 and isinstance(harness, TwoLevelReviewerHarness):
            r_pre_w4 = harness.review_rule
            r_pre_w4_version = harness.rule_version
            n_pre_w4 = harness.task_note

        priv = ep_out["priv"]
        logs.append(
            {
                "episode": ep + 1,
                "window": int(priv.get("window", env.window_index(ep))),
                "regime": priv.get("regime", ""),
                "is_high_vol": bool(priv.get("is_high_vol", False)),
                "is_hidden_transient_dip": bool(priv.get("is_hidden_transient_dip", False)),
                "pre_episode_harness_state": ep_out["pre_state"],
                "pre_episode_harness_context": ep_out["h_ctx"],
                "public_score": ep_out["final_score"],
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", 0.0)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "optimal_policy_match": bool(priv.get("optimal_policy_match", False)),
                "dp_policy_match": bool(priv.get("dp_policy_match", False)),
                "stale_policy_match": bool(priv.get("stale_policy_match", False)),
                "invariant_violation": bool(priv.get("invariant_violation", False)),
                "turnover_fee": float(priv.get("turnover_fee", 0.0)),
                "probes_used": int(priv.get("probes_used", 0)),
                "probed_strategies": ep_out["probed_strats"],
                "submitted_weights": ep_out["sub_weights"],
                "post_episode_harness_meta": h_meta,
                "steps": ep_out["steps"],
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    def _win_util(win_idx: int) -> tuple[float, float]:
        sub = [r for r in logs if r["window"] == win_idx]
        if not sub:
            return 0.0, 0.0
        ag = sum(r["expected_net_no_noise"] for r in sub)
        dp = sum(r["dp_oracle_step_net"] for r in sub)
        return round(100.0 * ag / max(1.0, dp), 2), round(ag, 2)

    post_ag = sum(r["expected_net_no_noise"] for r in logs)
    post_dp = sum(r["dp_oracle_step_net"] for r in logs)
    post_dp_pct = round(100.0 * post_ag / max(1.0, post_dp), 2)
    w2_pct, _ = _win_util(1)
    w3_pct, _ = _win_util(2)
    w4_pct, w4_net_pts = _win_util(3)

    calm_norm = [r for r in logs if not r["is_high_vol"] and not r["is_hidden_transient_dip"]]
    calm_opt_pct = round(100.0 * sum(1 for r in calm_norm if r["optimal_policy_match"]) / max(1, len(calm_norm)), 2)
    w4_calm = [r for r in calm_norm if r["window"] == 3]
    w4_calm_opt_pct = round(100.0 * sum(1 for r in w4_calm if r["optimal_policy_match"]) / max(1, len(w4_calm)), 2)

    dip_rows = [r for r in logs if r["is_hidden_transient_dip"]]
    dip_probes = sum(r["probes_used"] for r in dip_rows)
    dip_turn = round(sum(r["turnover_fee"] for r in dip_rows), 3)
    post_dip_abandoned = False
    if world_mode in ("drift_meta_60", "drift_dip_c_60") and len(logs) >= 3:
        w_after_dip = logs[2].get("submitted_weights", {})
        if isinstance(w_after_dip, dict) and w_after_dip:
            post_dip_abandoned = l1_turnover_half(w_after_dip, ckpt["fork_holdings"]) > 0.25

    w4_probes = sum(r["probes_used"] for r in logs if r["window"] == 3)
    rev_hist = getattr(harness, "review_history", []) if isinstance(harness, TwoLevelReviewerHarness) else []
    post_revs = [e for e in rev_hist if int(e.get("after_episode", 0)) > t_ckpt]
    note_edits_post = sum(1 for e in post_revs if e.get("note_applied"))
    rule_edits_post = sum(1 for e in post_revs if e.get("rule_applied"))

    return {
        "suite": "meta_reviewer_h60",
        "model": model_name,
        "provider_model_versions": sorted(client.observed_model_versions) or [model_name],
        "arm": arm_name,
        "world_mode": world_mode,
        "seed": seed,
        "horizon": horizon,
        "t_ckpt": t_ckpt,
        "window_ranges": list(env.window_ranges),
        "episodes_post": env.horizon - t_ckpt,
        "w1_live_dp_utility_pct": ckpt["w1_dp_utility_pct"],
        "post_dp_utility_pct": post_dp_pct,
        "w2_dp_utility_pct": w2_pct,
        "w3_dp_utility_pct": w3_pct,
        "w4_dp_utility_pct": w4_pct,
        "w4_expected_net_pts": w4_net_pts,
        "calm_optimal_pct": calm_opt_pct,
        "w4_calm_optimal_pct": w4_calm_opt_pct,
        "probes_total": sum(r["probes_used"] for r in logs),
        "w4_probes": w4_probes,
        "dip_probes": dip_probes,
        "dip_turnover_pts": dip_turn,
        "post_dip_abandoned_incumbent": post_dip_abandoned,
        "turnover_pts_total": round(sum(r["turnover_fee"] for r in logs), 2),
        "risk_tool_calls": risk_tool_calls,
        "compliance_violations": compliance_violations,
        "runner_interventions": runner_interventions,
        "acting_calls": acting_calls,
        "reflection_calls": reflection_calls,
        "note_edits_post": note_edits_post,
        "rule_edits_post": rule_edits_post,
        "final_note_version": int(getattr(harness, "note_version", 0)),
        "final_rule_version": int(getattr(harness, "rule_version", 0)),
        "ckpt_task_note": ckpt["acquired_task_note"],
        "ckpt_review_rule": ckpt["acquired_review_rule"],
        "pre_w4_task_note": n_pre_w4,
        "pre_w4_review_rule": r_pre_w4,
        "final_task_note": getattr(harness, "task_note", ""),
        "final_review_rule": getattr(harness, "review_rule", ""),
        "review_history": rev_hist,
        "counterfactual_w4_replay": w4_replay,
        "thinking_tokens": thinking_tokens,
        "usd": round(acting_usd + reflection_usd + (float(w4_replay["replay_usd"]) if w4_replay else 0.0), 6),
        "logs": logs,
    }


def _bootstrap_ci(vals: list[float], seed: int = 20261127) -> dict[str, Any]:
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


def summarize_meta_report(
    *,
    run_id: str,
    horizon: int,
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
    w4_replays_by_model: dict[str, list[dict[str, Any]]] = {}
    provider_versions: dict[str, list[str]] = {}

    contrast_specs = [
        ("editable_reviewer_v3", "frozen_reviewer_v3", "editable_minus_frozen_reviewer"),
        ("editable_reviewer_v3", "factual_ledger_v3", "editable_reviewer_minus_facts_only"),
        ("frozen_reviewer_v3", "factual_ledger_v3", "frozen_reviewer_minus_facts_only"),
        ("sgeh_vbq_v3", "editable_reviewer_v3", "vbq_minus_editable_reviewer"),
        ("sgeh_vbq_v3", "factual_ledger_v3", "vbq_minus_facts_only"),
    ]

    for mdl in models:
        m_recs = [r for r in records if r["model"] == mdl]
        vers: set[str] = set()
        for r in m_recs:
            vers.update(r.get("provider_model_versions", []))
        provider_versions[mdl] = sorted(vers)
        by_model_mode[mdl] = {}
        paired_contrasts[mdl] = {}
        w4_replays_by_model[mdl] = [
            {"seed": r["seed"], "world_mode": r["world_mode"], **r["counterfactual_w4_replay"]}
            for r in sorted(m_recs, key=lambda x: (x["world_mode"], int(x["seed"])))
            if r.get("counterfactual_w4_replay")
        ]

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
                    "w4_dp_utility_pct": round(statistics.mean(r["w4_dp_utility_pct"] for r in sub), 2),
                    "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in sub), 2),
                    "w4_calm_optimal_pct": round(statistics.mean(r["w4_calm_optimal_pct"] for r in sub), 2),
                    "mean_probes": round(statistics.mean(r["probes_total"] for r in sub), 2),
                    "mean_w4_probes": round(statistics.mean(r["w4_probes"] for r in sub), 2),
                    "mean_dip_probes": round(statistics.mean(r["dip_probes"] for r in sub), 2),
                    "post_dip_abandon_rate_pct": round(
                        100.0 * sum(1 for r in sub if r["post_dip_abandoned_incumbent"]) / len(sub), 1
                    ),
                    "mean_turnover_pts": round(statistics.mean(r["turnover_pts_total"] for r in sub), 2),
                    "compliance_violations_sum": sum(int(r["compliance_violations"]) for r in sub),
                    "runner_interventions_sum": sum(int(r["runner_interventions"]) for r in sub),
                    "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in sub), 2),
                    "mean_note_edits_post": round(statistics.mean(r["note_edits_post"] for r in sub), 2),
                    "mean_rule_edits_post": round(statistics.mean(r["rule_edits_post"] for r in sub), 2),
                    "mean_usd": round(statistics.mean(r["usd"] for r in sub), 4),
                }
            by_model_mode[mdl][wm] = amap

            by_sa = {(int(r["seed"]), str(r["arm"])): r for r in sub_wm}
            wm_con: dict[str, Any] = {}
            for a_hi, a_lo, cname in contrast_specs:
                diffs_w4 = [
                    by_sa[(sd, a_hi)]["w4_dp_utility_pct"] - by_sa[(sd, a_lo)]["w4_dp_utility_pct"]
                    for sd in seeds
                    if (sd, a_hi) in by_sa and (sd, a_lo) in by_sa
                ]
                diffs_post = [
                    by_sa[(sd, a_hi)]["post_dp_utility_pct"] - by_sa[(sd, a_lo)]["post_dp_utility_pct"]
                    for sd in seeds
                    if (sd, a_hi) in by_sa and (sd, a_lo) in by_sa
                ]
                if diffs_w4:
                    wm_con[f"w4__{cname}"] = _bootstrap_ci(diffs_w4)
                    wm_con[f"post__{cname}"] = _bootstrap_ci(diffs_post)
            paired_contrasts[mdl][wm] = wm_con

        # Stationary-adjusted DiD (`drift_meta_60` vs `stationary_meta_60`)
        did_map: dict[str, Any] = {}
        by_msa = {(str(r["world_mode"]), int(r["seed"]), str(r["arm"])): r for r in m_recs}
        for sh_m in ("drift_meta_60", "drift_dip_c_60"):
            st_m = "stationary_meta_60"
            for a_hi, a_lo, cname in contrast_specs:
                d_w4 = [
                    (by_msa[(sh_m, sd, a_hi)]["w4_dp_utility_pct"] - by_msa[(sh_m, sd, a_lo)]["w4_dp_utility_pct"])
                    - (by_msa[(st_m, sd, a_hi)]["w4_dp_utility_pct"] - by_msa[(st_m, sd, a_lo)]["w4_dp_utility_pct"])
                    for sd in seeds
                    if all((mm, sd, aa) in by_msa for mm in (sh_m, st_m) for aa in (a_hi, a_lo))
                ]
                d_post = [
                    (by_msa[(sh_m, sd, a_hi)]["post_dp_utility_pct"] - by_msa[(sh_m, sd, a_lo)]["post_dp_utility_pct"])
                    - (by_msa[(st_m, sd, a_hi)]["post_dp_utility_pct"] - by_msa[(st_m, sd, a_lo)]["post_dp_utility_pct"])
                    for sd in seeds
                    if all((mm, sd, aa) in by_msa for mm in (sh_m, st_m) for aa in (a_hi, a_lo))
                ]
                if d_w4:
                    did_map[f"did_w4_{sh_m}__{cname}"] = _bootstrap_ci(d_w4)
                    did_map[f"did_post_{sh_m}__{cname}"] = _bootstrap_ci(d_post)
        paired_contrasts[mdl]["did_summary"] = did_map

    clean_records = [{k: v for k, v in r.items() if k != "logs"} for r in records]
    return {
        "experiment_type": "meta_reviewer_h60",
        "suite": "meta_reviewer_h60",
        "run_id": run_id,
        "horizon": horizon,
        "models": models,
        "provider_model_versions": provider_versions,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "total_run_usd": round(total_usd, 6),
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "by_model_mode": by_model_mode,
        "paired_contrasts": paired_contrasts,
        "counterfactual_w4_replays": w4_replays_by_model,
        "records": clean_records,
    }


def render_meta_markdown(rep: dict[str, Any]) -> str:
    arm_labels = {
        "factual_ledger_v3": "Facts-Only (Autonomy Baseline — No Notes)",
        "frozen_reviewer_v3": "Frozen-Reviewer (Control — Task-Note Editable, Review-Rule Frozen)",
        "editable_reviewer_v3": "**Editable-Reviewer (Ours — Both Task-Note & Review-Rule Editable)**",
        "sgeh_vbq_v3": "Directive-SGEH-VBQ (Hand-Written Reference Controller)",
    }
    lines = [
        f"# Single-Lifetime `H={rep['horizon']}` Two-Level Reviewer Self-Improvement: `{rep['run_id']}`",
        "",
        f"- **Models & Provider Versions:** `{json.dumps(rep['provider_model_versions'])}`",
        f"- **Seeds:** `{rep['seeds']}` (`N={rep['n_seeds']}` private jittered schedules) | **Horizon:** `{rep['horizon']}` sessions | **Total Spend:** `${rep['total_run_usd']:.4f}`",
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
            lines.append(f"### Stream: `{wm}`")
            lines.append(
                "| Arm | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Viol / Intv | USD |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for ak, d in amap.items():
                lines.append(
                    f"| {arm_labels.get(ak, ak)} | **{d['w4_dp_utility_pct']:.1f}%** | {d['post_dp_utility_pct']:.1f}% | "
                    f"`{d['w2_dp_utility_pct']:.1f}%` / `{d['w3_dp_utility_pct']:.1f}%` | {d['w4_calm_optimal_pct']:.1f}% | "
                    f"{d['mean_probes']:.1f} (`{d['mean_w4_probes']:.1f}`) | "
                    f"`{d['mean_note_edits_post']:.1f}` / `{d['mean_rule_edits_post']:.1f}` | "
                    f"`{d['compliance_violations_sum']}` / `{d['runner_interventions_sum']}` | `${d['mean_usd']:.4f}` |"
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

        replays = rep.get("counterfactual_w4_replays", {}).get(mdl, [])
        if replays:
            lines.append(f"### Paired Counterfactual `W4` Review Replay (`R_ckpt` Old Rule vs `R_pre_W4` New Rule — `{mdl}`)")
            lines.append(
                "| Seed | W4 Start | Replay Ep | `ΔRule`? | `ΔTaskNote`? | Step+1 `ΔAction`? | Old 4-Ep DP (%) | New 4-Ep DP (%) | `ΔNet` (4-Ep) | Full Chain Positive? |"
            )
            lines.append("|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|:---:|")
            for rp in replays:
                lines.append(
                    f"| `{rp['seed']}` | `{rp['w4_start_session']}` | `{rp['replay_after_episode']}` | "
                    f"`{rp['rule_changed_before_w4']}` | `{rp['note_revision_changed']}` | `{rp['step1_action_changed']}` | "
                    f"{rp['old_block_dp_pct']:.1f}% | {rp['new_block_dp_pct']:.1f}% | "
                    f"**`{rp['delta_block_net_pts']:+.2f} pts`** (`{rp['delta_block_dp_pct']:+.2f}%`) | "
                    f"`{rp['full_causal_chain_positive']}` |"
                )
            deltas = [float(rp["delta_block_dp_pct"]) for rp in replays]
            ci_cf = _bootstrap_ci(deltas)
            lines.append("")
            lines.append(
                f"- **Mean Counterfactual 4-Ep `New Rule − Old Rule` DP Gain:** **`{ci_cf['mean']:+.2f} pts`** "
                f"`[{ci_cf['ci95'][0]:+.2f}, {ci_cf['ci95'][1]:+.2f}]` (per-seed: `{ci_cf['per_seed']}`)"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def run_meta_batch(
    *,
    run_id: str,
    models: list[str],
    arms: list[str],
    world_modes: list[str],
    seeds: list[int],
    horizon: int,
    max_usd: float,
    workers: int,
    thinking_budget: int,
    checkpoint_file: Path | None,
    acquire_checkpoints_only: bool,
    enable_w4_replay: bool,
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

    loaded_ckpts: dict[tuple[str, int], dict[str, Any]] = {}
    if checkpoint_file and checkpoint_file.is_file():
        raw_map = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        for k_str, v_obj in raw_map.items():
            parts = k_str.split("|")
            if len(parts) == 2:
                loaded_ckpts[(parts[0], int(parts[1]))] = deserialize_meta_checkpoint(v_obj)

    missing_keys = [(mdl, sd) for mdl in models for sd in seeds if (mdl, sd) not in loaded_ckpts]

    w1_eps_est = len(missing_keys) * 12
    post_eps_est = 0 if acquire_checkpoints_only else len(models) * len(arms) * len(world_modes) * len(seeds) * (horizon - 12)
    total_units = max(1, w1_eps_est + post_eps_est)
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

    if missing_keys:
        print(f"[{run_id}] Acquiring {len(missing_keys)} live unscripted W1 meta-checkpoints (H={horizon})...", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, min(len(missing_keys), workers))) as pool:
            futs = {
                pool.submit(
                    acquire_live_w1_meta_checkpoint,
                    model_name=mdl,
                    seed=sd,
                    horizon=horizon,
                    client=clients[mdl],
                    on_ep_done=lambda m=mdl, s=sd: _tick(f"W1_meta {m} s{s}"),
                ): (mdl, sd)
                for (mdl, sd) in missing_keys
            }
            for fut in as_completed(futs):
                key = futs[fut]
                ck = fut.result()
                loaded_ckpts[key] = ck
                print(
                    f"  [W1-META-CKPT] {key[0]} | s{key[1]} | t_ckpt={ck['t_ckpt']} wins={ck['window_ranges']} -> "
                    f"W1_DP={ck['w1_dp_utility_pct']:.1f}% Note_v={ck['acquired_note_version']} "
                    f"Rule_v={ck['acquired_rule_version']} (${ck['w1_usd']:.4f})",
                    flush=True,
                )

    serialized_ckpts = {
        f"{m}|{s}": serialize_meta_checkpoint(ck)
        for (m, s), ck in sorted(loaded_ckpts.items())
    }
    atomic_write_json(ledger.run_dir / "w1_live_checkpoints.json", serialized_ckpts)
    if checkpoint_file is not None and missing_keys:
        # Merge with any existing file entries before atomic write so parallel subset runs never clobber!
        merged_on_disk: dict[str, Any] = {}
        if checkpoint_file.is_file():
            try:
                merged_on_disk = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                merged_on_disk = {}
        merged_on_disk.update(serialized_ckpts)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(checkpoint_file, merged_on_disk)

    if acquire_checkpoints_only:
        ledger.update_progress(
            completed_units=total_units,
            total_units=total_units,
            status="completed",
            current_label=f"completed W1 checkpoints {run_id}",
        )
        return {"run_id": run_id, "checkpoints_acquired": len(missing_keys), "total_run_usd": round(ledger.run_usd, 6)}

    arm_cache: dict[tuple[str, int, str], Any] = {}
    for (mdl, sd), ck in loaded_ckpts.items():
        if mdl not in clients or sd not in seeds:
            continue
        for arm in arms:
            arm_cache[(mdl, sd, arm)] = instantiate_meta_arm(arm, ck, clients[mdl])

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
            ck = loaded_ckpts[(mdl, sd)]
            base_h = arm_cache[(mdl, sd, arm)]
            futs_b.append(
                pool.submit(
                    evaluate_meta_branch,
                    model_name=mdl,
                    arm_name=arm,
                    world_mode=wm,
                    seed=sd,
                    horizon=horizon,
                    ckpt=ck,
                    base_harness=base_h,
                    client=clients[mdl],
                    enable_w4_replay=enable_w4_replay,
                    on_ep_done=lambda m=mdl, a=arm, w=wm, s=sd: _tick(f"{m} {w} s{s} {a}"),
                )
            )
        for fut in as_completed(futs_b):
            res = fut.result()
            raw_results.append(res)
            mdl, arm, wm, sd = res["model"], res["arm"], res["world_mode"], res["seed"]
            cf_str = ""
            if res.get("counterfactual_w4_replay"):
                cf = res["counterfactual_w4_replay"]
                cf_str = f" | CF_dDP={cf['delta_block_dp_pct']:+.1f}%"
            print(
                f"  -> [{mdl:16s} | {wm:18s} | s{sd} | {arm:20s}] "
                f"W4_DP={res['w4_dp_utility_pct']:5.1f}% PostDP={res['post_dp_utility_pct']:5.1f}% "
                f"(W2={res['w2_dp_utility_pct']:5.1f}%/W3={res['w3_dp_utility_pct']:5.1f}%) | "
                f"Prb={res['probes_total']:2d} NoteEd={res['note_edits_post']} RuleEd={res['rule_edits_post']}"
                f"{cf_str} | Cell=${res['usd']:.4f} Run=${ledger.run_usd:.4f}",
                flush=True,
            )
            slug = f"meta60_{wm}__{arm}__{mdl.replace('/', '_')}_s{sd}"
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

    reviewer_audit = [
        {
            "model": r["model"],
            "seed": r["seed"],
            "world_mode": r["world_mode"],
            "arm": r["arm"],
            "window_ranges": r["window_ranges"],
            "ckpt_task_note": r["ckpt_task_note"],
            "ckpt_review_rule": r["ckpt_review_rule"],
            "pre_w4_task_note": r["pre_w4_task_note"],
            "pre_w4_review_rule": r["pre_w4_review_rule"],
            "final_task_note": r["final_task_note"],
            "final_review_rule": r["final_review_rule"],
            "note_edits_post": r["note_edits_post"],
            "rule_edits_post": r["rule_edits_post"],
            "counterfactual_w4_replay": r["counterfactual_w4_replay"],
            "review_history": r["review_history"],
        }
        for r in sorted(raw_results, key=lambda x: (x["model"], x["world_mode"], int(x["seed"]), x["arm"]))
        if r["arm"] in ("editable_reviewer_v3", "frozen_reviewer_v3")
    ]
    if reviewer_audit:
        atomic_write_json(ledger.run_dir / "reviewer_versions_audit.json", reviewer_audit)

    cf_list = [
        {"model": r["model"], "seed": r["seed"], "world_mode": r["world_mode"], **r["counterfactual_w4_replay"]}
        for r in sorted(raw_results, key=lambda x: (x["model"], x["world_mode"], int(x["seed"])))
        if r.get("counterfactual_w4_replay")
    ]
    if cf_list:
        atomic_write_json(ledger.run_dir / "counterfactual_w4_replay.json", cf_list)

    raw_results.sort(key=lambda r: (r["model"], r["world_mode"], int(r["seed"]), r["arm"]))
    rep = summarize_meta_report(
        run_id=run_id,
        horizon=horizon,
        arms_order=arms,
        modes_order=world_modes,
        records=raw_results,
        total_usd=ledger.run_usd,
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "suite_report.json", rep)
    md = render_meta_markdown(rep)
    (ledger.run_dir / "summary.md").write_text(md, encoding="utf-8")
    ledger.update_progress(
        completed_units=total_units,
        total_units=total_units,
        status="completed",
        current_label=f"completed {run_id}",
    )
    return rep


def aggregate_meta_runs(run_ids: list[str], out_run_id: str, results_root: Path) -> Path:
    all_records: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    horizon = 60
    arms_seen: list[str] = []
    modes_seen: list[str] = []

    for rid in run_ids:
        rep_p = results_root / "runs" / rid / "suite_report.json"
        if not rep_p.is_file():
            # Allow including W1-checkpoint-only run_ids so total spend reflects checkpoint acquisition too!
            prog_p = results_root / "runs" / rid / "progress.json"
            if prog_p.is_file():
                tot_usd += float(json.loads(prog_p.read_text(encoding="utf-8")).get("run_usd", 0.0))
            continue
        d = json.loads(rep_p.read_text(encoding="utf-8"))
        horizon = int(d.get("horizon", horizon))
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
    std_arms = [a for a in DEFAULT_ARMS if a in arms_seen] or arms_seen
    std_modes = [m for m in META_MODES if m in modes_seen] or modes_seen

    rep = summarize_meta_report(
        run_id=out_run_id,
        horizon=horizon,
        arms_order=std_arms,
        modes_order=std_modes,
        records=merged,
        total_usd=tot_usd,
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "suite_report.json", rep)

    merged_audit: list[dict[str, Any]] = []
    merged_cf: list[dict[str, Any]] = []
    for rid in run_ids:
        pa = results_root / "runs" / rid / "reviewer_versions_audit.json"
        if pa.is_file():
            merged_audit.extend(json.loads(pa.read_text(encoding="utf-8")))
        cfp = results_root / "runs" / rid / "counterfactual_w4_replay.json"
        if cfp.is_file():
            merged_cf.extend(json.loads(cfp.read_text(encoding="utf-8")))
    if merged_audit:
        atomic_write_json(out_dir / "reviewer_versions_audit.json", merged_audit)
    if merged_cf:
        atomic_write_json(out_dir / "counterfactual_w4_replay.json", merged_cf)

    md = render_meta_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Single-Lifetime H=60 Two-Level Reviewer Self-Improvement Suite")
    parser.add_argument("--run-id", default="meta60_dev", help="Experiment run ID")
    parser.add_argument("--models", default="gemini-3.8-flash", help="Comma-separated model IDs")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="Comma-separated arms")
    parser.add_argument("--modes", default="drift_meta_60,stationary_meta_60", help="Comma-separated meta modes")
    parser.add_argument("--seeds", default="701,702,703,704,705,706", help="Comma-separated seeds")
    parser.add_argument("--horizon", type=int, default=60, help="Single-lifetime session horizon")
    parser.add_argument("--workers", type=int, default=8, help="Parallel threads")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Thinking token budget")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Hard USD cap per shard")
    parser.add_argument("--checkpoint-file", default="", help="Shared JSON path for live W1 meta-checkpoints")
    parser.add_argument("--acquire-checkpoints-only", action="store_true", help="Acquire & save W1 checkpoints only")
    parser.add_argument("--skip-w4-replay", action="store_true", help="Disable inline counterfactual W4 replay")
    parser.add_argument("--results-root", default=str(ROOT / "results"), help="Results directory root")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated shard run IDs to merge")
    args = parser.parse_args()

    res_root = Path(args.results_root)
    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        out_md = aggregate_meta_runs(rids, args.run_id, res_root)
        print(out_md.read_text(encoding="utf-8"))
        print(f"Saved aggregated report: {out_md}", flush=True)
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    ckpt_p = Path(args.checkpoint_file) if args.checkpoint_file else None

    try:
        rep = run_meta_batch(
            run_id=args.run_id,
            models=models,
            arms=arms,
            world_modes=modes,
            seeds=seeds,
            horizon=args.horizon,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            checkpoint_file=ckpt_p,
            acquire_checkpoints_only=args.acquire_checkpoints_only,
            enable_w4_replay=not args.skip_w4_replay,
            results_root=res_root,
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET CIRCUIT BREAKER] {exc}", flush=True)
        sys.exit(2)

    if not args.acquire_checkpoints_only:
        print("\n" + render_meta_markdown(rep), flush=True)


if __name__ == "__main__":
    main()
