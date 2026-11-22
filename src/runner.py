"""Sequential lifetime runner & metric aggregator for ParadigmBench."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.ledger import CostLedger, atomic_write_json
from src.core.llm_client import MeteredLLMClient
from src.envs import ParadigmEnv, make_env
from src.harnesses import EpistemicHarness, PublicEpisodeRecord, make_harness


@dataclass
class LifetimeSummary:
    run_id: str
    env_spec: str
    harness_name: str
    model: str
    seed: int
    episodes: int
    mean_score: float
    mean_normalized_utility: float
    pre_shift_utility: float
    post_shift_utility: float
    stale_lock_in_rate: float
    invariant_violation_rate: float
    total_acting_calls: int
    total_reflection_calls: int
    rsi_tax_call_share: float
    total_usd: float
    reflection_usd: float


def _build_acting_system_prompt(domain_prompt: str, tool_schemas: list[dict[str, Any]], harness_ctx: str) -> str:
    tools_desc = "\n".join(
        f"- `{t['name']}`: {t['description']} | args: {json.dumps(t['args'])}"
        for t in tool_schemas
    )
    return (
        f"{domain_prompt}\n\n"
        f"AVAILABLE TOOLS (call exactly ONE tool per turn as JSON):\n{tools_desc}\n"
        f"{harness_ctx}\n"
        "OUTPUT FORMAT (strict JSON object only):\n"
        "{\"tool\": \"<tool_name>\", \"args\": {...}, \"rationale\": \"<brief 1-sentence reason>\"}"
    )


def _scripted_mock_actor(env_spec: str, steps_so_far: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic valid tool-calling actor for `$0.00` Tier-0 offline tests."""
    domain = env_spec.split(":")[0]
    if domain == "alpha_desk_v2":
        if not steps_so_far:
            return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Read desk brief."}
        return {
            "tool": "submit_allocation",
            "args": {
                "weights": {
                    "STRAT_ALPHA": 0.65,
                    "STRAT_BETA": 0.35,
                    "STRAT_GAMMA": 0.0,
                    "STRAT_DELTA": 0.0,
                }
            },
            "rationale": "Compliant mock allocation.",
        }
    if domain == "alpha_desk":
        if not steps_so_far:
            return {"tool": "inspect_market_tape", "args": {}, "rationale": "Read macro tape."}
        return {
            "tool": "submit_allocation",
            "args": {
                "weights": {
                    "MOMENTUM_TECH": 0.50,
                    "MEAN_REV_FX": 0.25,
                    "CARRY_CREDIT": 0.15,
                    "DEFENSIVE_VOL": 0.10,
                }
            },
            "rationale": "Balanced compliant allocation.",
        }
    if domain == "life_companion":
        if not steps_so_far:
            return {"tool": "read_user_dossier", "args": {}, "rationale": "Check constraints."}
        if len(steps_so_far) == 1:
            return {"tool": "search_catalog", "args": {}, "rationale": "List options."}
        cands = steps_so_far[-1].get("obs", {}).get("candidates", [])
        safe = [c for c in cands if c.get("monthly_cost", 9999) <= 950 and "TRAP" not in c.get("option_id", "")]
        chosen = (safe[0] if safe else cands[0])["option_id"]
        return {"tool": "recommend_plan", "args": {"option_id": chosen}, "rationale": "Pick safe option."}
    if domain == "coalition_ops":
        if not steps_so_far:
            return {"tool": "inspect_dispatch_ledger", "args": {}, "rationale": "Inspect ledger."}
        obs = steps_so_far[-1].get("obs", {})
        partner = obs.get("active_counterparty", "VANGUARD_CORP")
        return {
            "tool": "execute_contract",
            "args": {"partner": partner, "posture": "escrow_hedged"},
            "rationale": "Safe escrow posture.",
        }
    return {"tool": "finish", "args": {}, "rationale": "Fallback"}


def run_lifetime(
    *,
    env_spec: str,
    harness_name: str,
    model: str,
    seed: int,
    horizon: int,
    ledger: CostLedger,
    dotenv_path: Path | None = None,
    thinking_budget: int | None = 256,
    progress_offset: int = 0,
    progress_total: int = 1,
) -> tuple[LifetimeSummary, list[dict[str, Any]]]:
    """Execute one uninterrupted sequential agent lifetime and record full metrics."""
    env: ParadigmEnv = make_env(env_spec, horizon=horizon, seed=seed)
    harness: EpistemicHarness = make_harness(harness_name)
    client = MeteredLLMClient(
        model=model,
        ledger=ledger,
        dotenv_path=dotenv_path,
        thinking_budget=thinking_budget,
        temperature=0.0,
    )

    episode_logs: list[dict[str, Any]] = []
    acting_calls = 0
    reflection_calls = 0
    acting_usd = 0.0
    reflection_usd = 0.0

    for ep in range(env.num_episodes()):
        spec = env.start_episode(ep)
        harness_ctx = harness.build_context(spec)
        sys_prompt = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, harness_ctx)

        steps: list[dict[str, Any]] = []
        done = False
        final_score = 0.0
        private_eval: dict[str, Any] = {}

        for turn in range(env.max_turns_per_episode):
            history_str = (
                "\n".join(
                    f"Turn {idx + 1}: called `{s['tool']}`({json.dumps(s['args'])}) -> {json.dumps(s['obs'])}"
                    for idx, s in enumerate(steps)
                )
                or "(No tools called yet in this episode.)"
            )
            rem = env.max_turns_per_episode - turn
            must_commit = (
                " FINAL TURN: You MUST call the terminal submission/execution tool now!"
                if rem == 1
                else ""
            )
            user_prompt = (
                f"{spec.task_prompt}\n\n"
                f"Current Episode Tool History:\n{history_str}\n\n"
                f"Turns remaining in this episode: {rem}.{must_commit}"
            )
            resp = client.generate_json(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                environment=spec.env_name,
                method=harness_name,
                mock_responder=lambda _s, _u, _st=list(steps): _scripted_mock_actor(env_spec, _st),
            )
            acting_calls += 1
            acting_usd += resp.usd

            parsed = resp.parsed_json or {}
            tool_name = str(parsed.get("tool", "")).strip()
            tool_args = parsed.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            outcome = env.step_tool(tool_name, tool_args)
            steps.append(
                {
                    "turn": turn + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "rationale": str(parsed.get("rationale", ""))[:240],
                    "obs": outcome.observation,
                }
            )
            if outcome.done:
                done = True
                final_score = float(outcome.public_score or 0.0)
                private_eval = outcome.private_eval
                break

        if not done:
            # Exhausted turn budget without calling terminal action -> score 0
            final_score = 0.0
            private_eval = {
                "regime": "timeout",
                "expected_score": 0.0,
                "oracle_expected_score": 90.0,
                "normalized_utility": 0.0,
                "stale_policy_match": False,
                "optimal_policy_match": False,
                "invariant_violation": True,
                "probes_used": 0,
            }

        pub_record = PublicEpisodeRecord(
            env_name=spec.env_name,  # Strictly mode-free!
            episode_idx=ep,
            timestamp=ep + 1,
            task_prompt=spec.task_prompt,
            steps=steps,
            public_score=final_score,
        )
        h_meta = harness.on_episode_complete(pub_record, client)
        ref_c = int(h_meta.get("reflection_calls", 0))
        ref_u = float(h_meta.get("reflection_usd", 0.0))
        reflection_calls += ref_c
        reflection_usd += ref_u

        episode_logs.append(
            {
                "episode": ep + 1,
                "public_score": final_score,
                "normalized_utility": float(private_eval.get("normalized_utility", 0.0)),
                "stale_policy_match": bool(private_eval.get("stale_policy_match", False)),
                "optimal_policy_match": bool(private_eval.get("optimal_policy_match", False)),
                "invariant_violation": bool(private_eval.get("invariant_violation", False)),
                "regime": str(private_eval.get("regime", "")),
                "probes_used": int(private_eval.get("probes_used", 0)),
                "harness_meta": h_meta,
                "steps": steps,
            }
        )

        ledger.update_progress(
            completed_units=progress_offset + ep + 1,
            total_units=progress_total,
            status="running",
            current_label=f"{env_spec} | {harness_name} | {model} | ep {ep + 1}/{env.num_episodes()}",
        )

    t1 = max(1, env.num_episodes() // 3)
    scores = [row["public_score"] for row in episode_logs]
    norm_u = [row["normalized_utility"] for row in episode_logs]
    pre_u = norm_u[:t1]
    post_u = norm_u[t1:] if len(norm_u) > t1 else norm_u
    post_rows = episode_logs[t1:] if len(episode_logs) > t1 else episode_logs

    total_calls = acting_calls + reflection_calls
    summary = LifetimeSummary(
        run_id=ledger.run_id,
        env_spec=env_spec,
        harness_name=harness_name,
        model=model,
        seed=seed,
        episodes=env.num_episodes(),
        mean_score=round(statistics.mean(scores), 2),
        mean_normalized_utility=round(100.0 * statistics.mean(norm_u), 2),
        pre_shift_utility=round(100.0 * statistics.mean(pre_u), 2),
        post_shift_utility=round(100.0 * statistics.mean(post_u), 2),
        stale_lock_in_rate=round(
            100.0 * sum(1 for r in post_rows if r["stale_policy_match"]) / max(1, len(post_rows)), 2
        ),
        invariant_violation_rate=round(
            100.0 * sum(1 for r in episode_logs if r["invariant_violation"]) / max(1, len(episode_logs)), 2
        ),
        total_acting_calls=acting_calls,
        total_reflection_calls=reflection_calls,
        rsi_tax_call_share=round(100.0 * reflection_calls / max(1, total_calls), 2),
        total_usd=round(acting_usd + reflection_usd, 6),
        reflection_usd=round(reflection_usd, 6),
    )

    cell_slug = f"{env_spec.replace(':', '_')}__{harness_name}__{model.replace('/', '_')}_s{seed}"
    traces_dir = ledger.results_root / "raw" / ledger.run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as handle:
        for row in episode_logs:
            handle.write(json.dumps(row) + "\n")

    # Save compact dashboard-friendly curve + cell summary under tracked `results/runs/<run_id>/`
    cell_record = {
        "summary": summary.__dict__,
        "curve_utility": [round(100.0 * r["normalized_utility"], 1) for r in episode_logs],
        "curve_score": [r["public_score"] for r in episode_logs],
        "alarms": [ep_row["episode"] for ep_row in episode_logs if ep_row["harness_meta"].get("alarm_fired")],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cells_dir / f"{cell_slug}.json", cell_record)

    return summary, episode_logs
