#!/usr/bin/env python3
"""Stage-1 Repeated-Trial Counterfactual Rule Replay & Mechanical Strategy De-Binding Runner.

Implements the pre-registered repeatability + strategy-de-binding diagnostic agreed with Astra/Codex
(`setup-debates/2026-11-27-h60-reviewer-self-improvement.md`, 2026-11-28):
  - Reconstructs exact saved environment state (`_holdings`, `_cumulative_nav` verified bit-for-bit against saved
    `public_score`) and `FactualLedgerV3Harness` public history at saved review checkpoints (`Flash` `W4` checkpoints
    `s704@48`, `s705@44`, `s701@44`, `s706@44`, plus `Hydrogen` `W3` first-read checkpoints `edit_s702@32` &
    `froz_s702@32`).
  - Holds displayed version labels, incoming `task_note`, public history, holdings, actor prompt, and 4-session tool
    budget **100% identical** across all compared rule arms, varying ONLY the `review_rule` text string across:
      1. `initial_verbatim` (`R_ckpt`)
      2. `revised_verbatim` (`R_rev`, e.g., `R_pre_W4` or `R_28`)
      3. `initial_anonymized` (`anonymize_rule_strategies(R_ckpt)`)
      4. `revised_anonymized` (`anonymize_rule_strategies(R_rev)`)
  - Draws `K` independent `(1 review + 4-session actor)` trials per arm, reporting mean net-of-fee DP return,
    within-rule rerun SD (`sigma_within`), note uniqueness, probe counts, and exact two-sided Monte Carlo permutation
    p-values comparing between-rule differences against within-rule rerun variation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_reviewer_h60 import (  # noqa: E402
    _execute_single_episode,
    deserialize_meta_checkpoint,
)
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v3 import AlphaDeskV3MetaHorizonEnv  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.procedure_harness import (  # noqa: E402
    TwoLevelReviewerHarness,
    anonymize_rule_strategies,
)

PRESET_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "flash_s704_w4_ep48": {
        "label": "Flash s704 W4 (ep 48 -> 49..52, previously +52.15%)",
        "category": "positive_flash",
        "model": "gemini-3.8-flash",
        "seed": 704,
        "after_episode": 48,
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_flash.json",
        "raw_trace_file": "results/raw/meta60_drift_s704/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s704.jsonl",
        "audit_file": "results/runs/meta60_drift_s704/reviewer_versions_audit.json",
    },
    "flash_s705_w4_ep44": {
        "label": "Flash s705 W4 (ep 44 -> 45..48, previously +28.73%)",
        "category": "positive_flash",
        "model": "gemini-3.8-flash",
        "seed": 705,
        "after_episode": 44,
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_flash.json",
        "raw_trace_file": "results/raw/meta60_drift_s705/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s705.jsonl",
        "audit_file": "results/runs/meta60_drift_s705/reviewer_versions_audit.json",
    },
    "flash_s701_w4_ep44": {
        "label": "Flash s701 W4 (ep 44 -> 45..48, previously +0.00% neutral)",
        "category": "neutral_flash",
        "model": "gemini-3.8-flash",
        "seed": 701,
        "after_episode": 44,
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_flash.json",
        "raw_trace_file": "results/raw/meta60_drift_s701/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s701.jsonl",
        "audit_file": "results/runs/meta60_drift_s701/reviewer_versions_audit.json",
    },
    "flash_s706_w4_ep44": {
        "label": "Flash s706 W4 (ep 44 -> 45..48, previously +0.00% neutral)",
        "category": "neutral_flash",
        "model": "gemini-3.8-flash",
        "seed": 706,
        "after_episode": 44,
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_flash.json",
        "raw_trace_file": "results/raw/meta60_drift_s706/meta60_drift_meta_60__editable_reviewer_v3__gemini-3.8-flash_s706.jsonl",
        "audit_file": "results/runs/meta60_drift_s706/reviewer_versions_audit.json",
    },
    "hydro_s702_edit_w3_ep32": {
        "label": "Hydrogen s702 Editable History W3 First-Read (ep 32 -> 33..36)",
        "category": "hydrogen_w3",
        "model": "gemini-4-hydrogen",
        "seed": 702,
        "after_episode": 32,
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_hydro.json",
        "raw_trace_file": "results/raw/meta60_hy_drift_edit_s702/meta60_drift_meta_60__editable_reviewer_v3__gemini-4-hydrogen_s702.jsonl",
        "audit_file": "results/runs/meta60_hy_drift_edit_s702/reviewer_versions_audit.json",
    },
    "hydro_s702_froz_w3_ep32": {
        "label": "Hydrogen s702 Frozen History W3 Stalled State (ep 32 -> 33..36, R_28 vs R_ckpt)",
        "category": "hydrogen_w3",
        "model": "gemini-4-hydrogen",
        "seed": 702,
        "after_episode": 32,
        "meta_mode": "drift_meta_60",
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_hydro.json",
        "raw_trace_file": "results/raw/meta60_hy_drift_froz_s702/meta60_drift_meta_60__frozen_reviewer_v3__gemini-4-hydrogen_s702.jsonl",
        "audit_file": "results/runs/meta60_hy_drift_edit_s702/reviewer_versions_audit.json",
    },
    "flash_s709_dipc_w4_ep44": {
        "label": "Flash s709 Alternate-Ending Dip->C W4 (ep 44 -> 45..48, helpful rule edit +23.49%)",
        "category": "stage2_dip_c_flash",
        "model": "gemini-3.8-flash",
        "seed": 709,
        "after_episode": 44,
        "meta_mode": "drift_dip_c_60",
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_s2_flash.json",
        "raw_trace_file": "results/raw/meta60_dipc_s709/meta60_drift_dip_c_60__editable_reviewer_v3__gemini-3.8-flash_s709.jsonl",
        "audit_file": "results/runs/meta60_dipc_s709/reviewer_versions_audit.json",
    },
    "flash_s711_dipc_w4_ep44": {
        "label": "Flash s711 Alternate-Ending Dip->C W4 (ep 44 -> 45..48, harmful rule erasure -30.87%)",
        "category": "stage2_dip_c_flash",
        "model": "gemini-3.8-flash",
        "seed": 711,
        "after_episode": 44,
        "meta_mode": "drift_dip_c_60",
        "w1_ckpt_file": "results/runs/meta_reviewer_h60_w1_ckpts_s2_flash.json",
        "raw_trace_file": "results/raw/meta60_dipc_s711/meta60_drift_dip_c_60__editable_reviewer_v3__gemini-3.8-flash_s711.jsonl",
        "audit_file": "results/runs/meta60_dipc_s711/reviewer_versions_audit.json",
    },
}

DEFAULT_ARMS = (
    "initial_verbatim",
    "revised_verbatim",
    "initial_anonymized",
    "revised_anonymized",
)


def reconstruct_saved_review_checkpoint(
    spec_cfg: dict[str, Any],
    workspace_root: Path = ROOT,
) -> dict[str, Any]:
    """Deterministically reconstruct exact `(env_after_ep, harness_before_review, rules)` from saved artifacts."""
    model = str(spec_cfg["model"])
    seed = int(spec_cfg["seed"])
    after_ep = int(spec_cfg["after_episode"])
    ep_idx = after_ep - 1

    w1_map = json.loads((workspace_root / spec_cfg["w1_ckpt_file"]).read_text(encoding="utf-8"))
    w1_ck = deserialize_meta_checkpoint(w1_map[f"{model}|{seed}"])

    trace_path = workspace_root / spec_cfg["raw_trace_file"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_by_ep = {int(r["episode"]): r for r in rows}
    if after_ep not in rows_by_ep:
        raise ValueError(f"Episode {after_ep} not found in {trace_path}")

    env = AlphaDeskV3MetaHorizonEnv(
        meta_mode=str(spec_cfg.get("meta_mode", "drift_meta_60")),
        seed=seed,
        horizon=int(w1_ck["horizon"]),
        initial_post_holdings=w1_ck["fork_holdings"],
    )
    env.set_fork_state(w1_ck["fork_holdings"], w1_ck["fork_nav"])

    # Step `env` through exact saved tool calls from `t_ckpt + 1 .. after_ep` and verify bit-identical scores!
    t_ckpt = int(w1_ck["t_ckpt"])
    post_pub_records: list[PublicEpisodeRecord] = []
    for cur_ep_1idx in range(t_ckpt + 1, after_ep + 1):
        r_row = rows_by_ep[cur_ep_1idx]
        e_spec = env.start_episode(cur_ep_1idx - 1)
        last_score = 0.0
        for st in r_row["steps"]:
            out = env.step_tool(str(st["tool"]), dict(st.get("args") or {}))
            if out.done:
                last_score = float(out.public_score or 0.0)
        if abs(last_score - float(r_row["public_score"])) > 1e-4:
            raise AssertionError(
                f"Score mismatch reconstructing s{seed} ep {cur_ep_1idx}: got {last_score} vs {r_row['public_score']}"
            )
        post_pub_records.append(
            PublicEpisodeRecord(
                env_name=e_spec.env_name,
                episode_idx=cur_ep_1idx - 1,
                timestamp=e_spec.timestamp,
                task_prompt=e_spec.task_prompt,
                steps=list(r_row["steps"]),
                public_score=float(r_row["public_score"]),
            )
        )

    target_row = rows_by_ep[after_ep]
    pre_state = target_row["pre_episode_harness_state"]
    incoming_note = str(pre_state.get("active_task_note", ""))
    incoming_note_v = int(pre_state.get("note_version", 1))
    displayed_rule_v = int(pre_state.get("rule_version", 1))

    # Look up `R_ckpt` and the active `R_rev` entering `after_ep` from the editable audit file
    aud_list = json.loads((workspace_root / spec_cfg["audit_file"]).read_text(encoding="utf-8"))
    edit_aud = next(a for a in aud_list if a.get("arm") == "editable_reviewer_v3" and int(a["seed"]) == seed)
    initial_rule = str(edit_aud["ckpt_review_rule"])

    # The revised rule entering `after_ep` is the `active_review_rule` from the most recent review BEFORE `after_ep`
    prior_revs = [e for e in edit_aud["review_history"] if int(e["after_episode"]) < after_ep]
    revised_rule = str(prior_revs[-1]["active_review_rule"]) if prior_revs else initial_rule

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_d:
        dummy_ledger = CostLedger(results_root=Path(tmp_d), run_id="_tmp_rec", phase="dev", max_usd=0.01)
        mock_c = MeteredLLMClient(model="mock", ledger=dummy_ledger, thinking_budget=0)
        h_snap = TwoLevelReviewerHarness(
            mode="editable_rule",
            review_cadence=4,
            recent_k=8,
            initial_task_note=incoming_note,
            initial_review_rule=revised_rule,
        )
        h_snap.note_version = incoming_note_v
        h_snap.rule_version = displayed_rule_v
        for rec in list(w1_ck["records"]) + post_pub_records:
            h_snap.factual.on_episode_complete(rec, mock_c)

    # Also compute the saved live 4-session DP% on this exact trajectory for reference
    rollout_sessions = list(range(after_ep + 1, min(env.horizon + 1, after_ep + 1 + h_snap.review_cadence)))
    live_sub = [rows_by_ep[s] for s in rollout_sessions if s in rows_by_ep]
    live_4ep_dp = (
        round(
            100.0
            * sum(float(x["expected_net_no_noise"]) for x in live_sub)
            / max(1.0, sum(float(x["dp_oracle_step_net"]) for x in live_sub)),
            2,
        )
        if live_sub
        else 0.0
    )

    rules_map = {
        "initial_verbatim": initial_rule,
        "revised_verbatim": revised_rule,
        "initial_anonymized": anonymize_rule_strategies(initial_rule),
        "revised_anonymized": anonymize_rule_strategies(revised_rule),
    }

    return {
        "model": model,
        "seed": seed,
        "after_episode": after_ep,
        "ep_idx": ep_idx,
        "rollout_sessions": rollout_sessions,
        "displayed_rule_version": displayed_rule_v,
        "incoming_task_note": incoming_note,
        "incoming_note_version": incoming_note_v,
        "live_4ep_dp_pct": live_4ep_dp,
        "env_after_ep": env,
        "harness_snap": h_snap,
        "rules_map": rules_map,
    }


def _run_single_trial(
    *,
    ckpt_bundle: dict[str, Any],
    arm_name: str,
    trial_idx: int,
    client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    snap_h: TwoLevelReviewerHarness = copy.deepcopy(ckpt_bundle["harness_snap"])
    env_c: AlphaDeskV3MetaHorizonEnv = copy.deepcopy(ckpt_bundle["env_after_ep"])
    rule_txt = ckpt_bundle["rules_map"][arm_name]
    disp_ver = int(ckpt_bundle["displayed_rule_version"])
    ep_idx = int(ckpt_bundle["ep_idx"])

    rev_out = snap_h.counterfactual_review(
        env_name="alpha_desk_v3",
        episode_idx=ep_idx,
        override_review_rule=rule_txt,
        override_rule_version=disp_ver,
        client=client,
        method_suffix=f"rep_{arm_name}_t{trial_idx}",
    )
    prop_note = rev_out["proposed_task_note"] or snap_h.task_note
    snap_h.task_note = prop_note
    snap_h.note_version = int(ckpt_bundle["incoming_note_version"]) + 1

    net_sum = 0.0
    dp_sum = 0.0
    probes_sum = 0
    turn_sum = 0.0
    actor_usd = 0.0
    step1_probed: list[str] = []
    step1_weights: dict[str, float] = {}

    for idx_r, sess_1idx in enumerate(ckpt_bundle["rollout_sessions"]):
        ep_out = _execute_single_episode(
            env=env_c,
            ep=sess_1idx - 1,
            harness=snap_h,
            client=client,
            method_tag=f"rep_actor_{arm_name}_t{trial_idx}",
        )
        snap_h.factual.on_episode_complete(ep_out["pub_rec"], client)
        priv = ep_out["priv"]
        net_sum += float(priv.get("expected_net_no_noise", 0.0))
        dp_sum += float(priv.get("dp_oracle_step_net", 85.0))
        probes_sum += int(priv.get("probes_used", 0))
        turn_sum += float(priv.get("turnover_fee", 0.0))
        actor_usd += float(ep_out["acting_usd"])
        if idx_r == 0:
            step1_probed = list(ep_out["probed_strats"])
            step1_weights = dict(ep_out["sub_weights"])
        if on_ep_done is not None:
            on_ep_done()

    return {
        "arm": arm_name,
        "trial": trial_idx,
        "proposed_task_note": prop_note,
        "edit_rationale": rev_out["edit_rationale"],
        "step1_probed": step1_probed,
        "step1_weights": step1_weights,
        "block_expected_net": round(net_sum, 3),
        "block_dp_net": round(dp_sum, 3),
        "block_dp_pct": round(100.0 * net_sum / max(1.0, dp_sum), 2),
        "block_probes": probes_sum,
        "block_turnover_pts": round(turn_sum, 2),
        "usd": round(float(rev_out["usd"]) + actor_usd, 6),
    }


def _two_sample_stats(vals_hi: list[float], vals_lo: list[float], seed: int = 20261128) -> dict[str, Any]:
    if not vals_hi or not vals_lo:
        return {"diff_mean": 0.0, "ci95": [0.0, 0.0], "pooled_sd": 0.0, "cohens_d": 0.0, "perm_p_two_sided": 1.0}
    m_hi = statistics.mean(vals_hi)
    m_lo = statistics.mean(vals_lo)
    obs_diff = m_hi - m_lo

    var_hi = statistics.variance(vals_hi) if len(vals_hi) > 1 else 0.0
    var_lo = statistics.variance(vals_lo) if len(vals_lo) > 1 else 0.0
    denom_df = max(1, len(vals_hi) + len(vals_lo) - 2)
    pooled_var = ((len(vals_hi) - 1) * var_hi + (len(vals_lo) - 1) * var_lo) / denom_df
    pooled_sd = math.sqrt(max(0.0, pooled_var))
    cohens_d = (obs_diff / pooled_sd) if pooled_sd > 1e-6 else (0.0 if abs(obs_diff) <= 1e-6 else 99.0)

    rng = random.Random(seed)
    n_hi, n_lo = len(vals_hi), len(vals_lo)
    boots = sorted(
        statistics.mean(vals_hi[rng.randrange(n_hi)] for _ in range(n_hi))
        - statistics.mean(vals_lo[rng.randrange(n_lo)] for _ in range(n_lo))
        for _ in range(5000)
    )
    ci95 = [round(boots[int(0.025 * 5000)], 2), round(boots[int(0.975 * 5000)], 2)]

    # Exact-style two-sided Monte Carlo permutation p-value (20,000 shuffles)
    combined = list(vals_hi) + list(vals_lo)
    extreme = 0
    n_perm = 20000
    if pooled_sd == 0.0 and abs(obs_diff) <= 1e-9:
        perm_p = 1.0
    else:
        for _ in range(n_perm):
            rng.shuffle(combined)
            d_perm = statistics.mean(combined[:n_hi]) - statistics.mean(combined[n_hi:])
            if abs(d_perm) >= abs(obs_diff) - 1e-9:
                extreme += 1
        perm_p = (extreme + 1) / (n_perm + 1)

    return {
        "diff_mean": round(obs_diff, 2),
        "ci95": ci95,
        "pooled_within_sd": round(pooled_sd, 2),
        "cohens_d": round(cohens_d, 2),
        "perm_p_two_sided": round(perm_p, 4),
    }


def summarize_checkpoint_trials(
    preset_key: str,
    spec_cfg: dict[str, Any],
    bundle: dict[str, Any],
    arms: list[str],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_summaries: dict[str, Any] = {}
    vals_by_arm: dict[str, list[float]] = {}
    for arm in arms:
        sub = sorted([t for t in trials if t["arm"] == arm], key=lambda x: int(x["trial"]))
        if not sub:
            continue
        dp_vals = [float(t["block_dp_pct"]) for t in sub]
        vals_by_arm[arm] = dp_vals
        uniq_notes = len({t["proposed_task_note"].strip() for t in sub})
        step1_probe_rate = 100.0 * sum(1 for t in sub if t["step1_probed"]) / len(sub)
        arm_summaries[arm] = {
            "k_trials": len(sub),
            "rule_text": bundle["rules_map"][arm],
            "mean_dp_pct": round(statistics.mean(dp_vals), 2),
            "sd_within_dp_pct": round(statistics.stdev(dp_vals), 2) if len(dp_vals) > 1 else 0.0,
            "min_dp_pct": round(min(dp_vals), 2),
            "max_dp_pct": round(max(dp_vals), 2),
            "per_trial_dp_pct": [round(v, 2) for v in dp_vals],
            "mean_net_pts": round(statistics.mean(float(t["block_expected_net"]) for t in sub), 2),
            "mean_probes": round(statistics.mean(float(t["block_probes"]) for t in sub), 2),
            "step1_probe_rate_pct": round(step1_probe_rate, 1),
            "unique_notes_count": uniq_notes,
            "mean_usd": round(statistics.mean(float(t["usd"]) for t in sub), 4),
        }

    contrasts: dict[str, Any] = {}
    contrast_pairs = [
        ("revised_verbatim", "initial_verbatim", "revised_minus_initial_verbatim"),
        ("revised_anonymized", "initial_anonymized", "revised_minus_initial_anonymized"),
        ("revised_anonymized", "initial_verbatim", "revised_anon_minus_initial_verbatim"),
        ("revised_verbatim", "revised_anonymized", "strategy_binding_effect_in_revised"),
        ("initial_anonymized", "initial_verbatim", "strategy_removal_effect_in_initial"),
    ]
    for a_hi, a_lo, cname in contrast_pairs:
        if a_hi in vals_by_arm and a_lo in vals_by_arm:
            contrasts[cname] = _two_sample_stats(vals_by_arm[a_hi], vals_by_arm[a_lo])

    return {
        "preset_key": preset_key,
        "label": spec_cfg["label"],
        "category": spec_cfg["category"],
        "model": bundle["model"],
        "seed": bundle["seed"],
        "after_episode": bundle["after_episode"],
        "rollout_sessions": bundle["rollout_sessions"],
        "displayed_rule_version_held_constant": bundle["displayed_rule_version"],
        "live_4ep_dp_pct": bundle["live_4ep_dp_pct"],
        "incoming_task_note": bundle["incoming_task_note"],
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "trials": trials,
    }


def render_repeated_replay_markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# Stage-1 Repeated-Trial Rule Replay & Strategy De-Binding: `{rep['run_id']}`",
        "",
        f"- **Trials per Rule Arm:** `K = {rep['k_trials']}` IID `(1 review + 4 actor sessions)` draws | **Total Spend:** `${rep['total_run_usd']:.4f}`",
        "- **Prompt Isolation:** Identical `displayed_rule_version`, `incoming_task_note`, `Factual-Ledger`, `holdings`, `NAV`, and 4-session tool budget across all arms on each checkpoint.",
        "",
    ]
    for ck in rep["checkpoints"]:
        lines.append(f"## {ck['label']} (`{ck['preset_key']}`, Live 4-Ep = `{ck['live_4ep_dp_pct']:.2f}%`, Constant Header `v{ck['displayed_rule_version_held_constant']}`)")
        lines.append("")
        lines.append("| Rule Arm | Mean 4-Ep DP (%) | Within-Rule SD (`σ`) | Min .. Max (%) | Mean Probes (Step+1 %) | Unique Notes / K | Trial Scores (%) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for arm_k, s in ck["arm_summaries"].items():
            lines.append(
                f"| `{arm_k}` | **{s['mean_dp_pct']:.2f}%** | `{s['sd_within_dp_pct']:.2f}` | "
                f"`{s['min_dp_pct']:.1f} .. {s['max_dp_pct']:.1f}` | "
                f"{s['mean_probes']:.2f} (`{s['step1_probe_rate_pct']:.0f}%`) | "
                f"`{s['unique_notes_count']}/{s['k_trials']}` | `{s['per_trial_dp_pct']}` |"
            )
        lines.append("")
        lines.append("**Exact Rule Texts Tested:**")
        for arm_k, s in ck["arm_summaries"].items():
            lines.append(f"- `{arm_k}`: *\"{s['rule_text']}\"*")
        lines.append("")
        lines.append("**Between-Rule Signal vs. Within-Rule Rerun Noise:**")
        for cname, c in ck["contrasts"].items():
            lines.append(
                f"- **`{cname}`:** **`{c['diff_mean']:+.2f} pts`** `[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]` | "
                f"`σ_within = {c['pooled_within_sd']:.2f}` | `d = {c['cohens_d']:+.2f}` | **`perm p = {c['perm_p_two_sided']:.4f}`**"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_repeated_replay_batch(
    *,
    run_id: str,
    preset_keys: list[str],
    arms: list[str],
    k_trials: int,
    workers: int,
    max_usd: float,
    use_mock: bool,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)

    needed_models = {"mock"} if use_mock else {str(PRESET_CHECKPOINTS[pk]["model"]) for pk in preset_keys}
    clients = {
        mdl: MeteredLLMClient(
            model=mdl,
            ledger=ledger,
            dotenv_path=None if use_mock else dotenv_path,
            thinking_budget=0,
            temperature=0.0,
        )
        for mdl in needed_models
    }

    bundles = {pk: reconstruct_saved_review_checkpoint(PRESET_CHECKPOINTS[pk], ROOT) for pk in preset_keys}
    total_eps = sum(len(bundles[pk]["rollout_sessions"]) * len(arms) * k_trials for pk in preset_keys)
    done_eps = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_eps
        with lock:
            done_eps += 1
            if done_eps % 4 == 0 or done_eps == total_eps:
                ledger.update_progress(
                    completed_units=min(done_eps, total_eps),
                    total_units=max(1, total_eps),
                    status="running",
                    current_label=lbl,
                    log_every=24,
                )

    ckpt_reports: list[dict[str, Any]] = []
    for pk in preset_keys:
        spec_cfg = PRESET_CHECKPOINTS[pk]
        bundle = bundles[pk]
        mdl_key = "mock" if use_mock else bundle["model"]
        client = clients[mdl_key]

        jobs = [(arm, t_i) for arm in arms for t_i in range(1, k_trials + 1)]
        trials_out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), workers))) as pool:
            futs = {
                pool.submit(
                    _run_single_trial,
                    ckpt_bundle=bundle,
                    arm_name=arm,
                    trial_idx=t_i,
                    client=client,
                    on_ep_done=lambda p=pk, a=arm, t=t_i: _tick(f"{p} {a} t{t}"),
                ): (arm, t_i)
                for (arm, t_i) in jobs
            }
            for fut in as_completed(futs):
                res = fut.result()
                trials_out.append(res)

        ck_rep = summarize_checkpoint_trials(pk, spec_cfg, bundle, arms, trials_out)
        ckpt_reports.append(ck_rep)
        for arm_k, s in ck_rep["arm_summaries"].items():
            print(
                f"  [{pk} | {arm_k:18s}] mean={s['mean_dp_pct']:5.2f}% (sd={s['sd_within_dp_pct']:4.2f}) | "
                f"probes={s['mean_probes']:.2f} | uniq_notes={s['unique_notes_count']}/{s['k_trials']}",
                flush=True,
            )

    rep = {
        "experiment_type": "repeated_rule_replay_stage1",
        "run_id": run_id,
        "k_trials": k_trials,
        "arms": arms,
        "preset_keys": preset_keys,
        "total_run_usd": round(ledger.run_usd, 6),
        "observed_run_input_tokens": ledger.run_input_tokens,
        "observed_run_output_tokens": ledger.run_output_tokens,
        "observed_run_thinking_tokens": ledger.run_thinking_tokens,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": [
            {k: v for k, v in ck.items() if k != "trials"}
            for ck in ckpt_reports
        ],
    }
    atomic_write_json(ledger.run_dir / "repeated_replay_report.json", rep)
    atomic_write_json(ledger.run_dir / "repeated_replay_trials_full.json", ckpt_reports)
    md = render_repeated_replay_markdown(rep)
    (ledger.run_dir / "summary.md").write_text(md, encoding="utf-8")
    ledger.update_progress(
        completed_units=max(1, total_eps),
        total_units=max(1, total_eps),
        status="completed",
        current_label=f"completed {run_id}",
    )
    return rep


def aggregate_repeated_runs(run_ids: list[str], out_run_id: str, results_root: Path) -> Path:
    all_ckpts: list[dict[str, Any]] = []
    all_full: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    k_trials = 8
    arms: list[str] = []

    for rid in run_ids:
        rdir = results_root / "runs" / rid
        d = json.loads((rdir / "repeated_replay_report.json").read_text(encoding="utf-8"))
        k_trials = int(d.get("k_trials", k_trials))
        tot_usd += float(d.get("total_run_usd", 0.0))
        in_t += int(d.get("observed_run_input_tokens", 0))
        out_t += int(d.get("observed_run_output_tokens", 0))
        th_t += int(d.get("observed_run_thinking_tokens", 0))
        for a in d.get("arms", []):
            if a not in arms:
                arms.append(a)
        all_ckpts.extend(d.get("checkpoints", []))
        fp = rdir / "repeated_replay_trials_full.json"
        if fp.is_file():
            all_full.extend(json.loads(fp.read_text(encoding="utf-8")))

    order = list(PRESET_CHECKPOINTS.keys())
    all_ckpts.sort(key=lambda c: order.index(c["preset_key"]) if c["preset_key"] in order else 99)
    all_full.sort(key=lambda c: order.index(c["preset_key"]) if c["preset_key"] in order else 99)

    rep = {
        "experiment_type": "repeated_rule_replay_stage1",
        "run_id": out_run_id,
        "k_trials": k_trials,
        "arms": arms,
        "preset_keys": [c["preset_key"] for c in all_ckpts],
        "total_run_usd": round(tot_usd, 6),
        "observed_run_input_tokens": in_t,
        "observed_run_output_tokens": out_t,
        "observed_run_thinking_tokens": th_t,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": all_ckpts,
    }
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "repeated_replay_report.json", rep)
    if all_full:
        atomic_write_json(out_dir / "repeated_replay_trials_full.json", all_full)
    md = render_repeated_replay_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Repeated-Trial Rule Replay & Strategy De-Binding")
    parser.add_argument("--run-id", default="rep_replay_dev", help="Run ID")
    parser.add_argument("--presets", default="flash_s704_w4_ep48", help="Comma-separated preset checkpoint keys")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="Comma-separated rule arms")
    parser.add_argument("--k-trials", type=int, default=8, help="IID trials per rule arm")
    parser.add_argument("--workers", type=int, default=8, help="Parallel threads")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Hard USD cap per shard")
    parser.add_argument("--mock", action="store_true", help="Run offline $0.00 mock mode")
    parser.add_argument("--results-root", default=str(ROOT / "results"), help="Results root")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated shard run IDs to aggregate")
    args = parser.parse_args()

    res_root = Path(args.results_root)
    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        out_md = aggregate_repeated_runs(rids, args.run_id, res_root)
        print(out_md.read_text(encoding="utf-8"))
        return

    pks = [x.strip() for x in args.presets.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    try:
        rep = run_repeated_replay_batch(
            run_id=args.run_id,
            preset_keys=pks,
            arms=arms,
            k_trials=args.k_trials,
            workers=args.workers,
            max_usd=args.max_usd,
            use_mock=args.mock,
            results_root=res_root,
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET CIRCUIT BREAKER] {exc}", flush=True)
        sys.exit(2)

    print("\n" + render_repeated_replay_markdown(rep), flush=True)


if __name__ == "__main__":
    main()
