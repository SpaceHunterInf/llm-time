"""Act 3: Selective Epistemic Revision Training & 3-Fold Leave-One-Domain-Out (LODO) Harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.envs import make_env
from src.envs.alpha_desk_v2 import STRATEGIES_V2

LODO_FOLDS: dict[str, dict[str, Any]] = {
    "fold1_holdout_alphadesk": {
        "train_domains": ["life_companion", "coalition_ops"],
        "eval_domain": "alpha_desk_v2",
        "eval_specs": [
            "alpha_desk_v2:stationary",
            "alpha_desk_v2:abrupt_reversal",
            "alpha_desk_v2:gradual_decay",
            "alpha_desk_v2:opportunity_rise",
            "alpha_desk_v2:off_scope_shock",
        ],
    },
    "fold2_holdout_lifecompanion": {
        "train_domains": ["alpha_desk_v2", "coalition_ops"],
        "eval_domain": "life_companion",
        "eval_specs": ["life_companion:stationary", "life_companion:lifecycle"],
    },
    "fold3_holdout_coalitionops": {
        "train_domains": ["alpha_desk_v2", "life_companion"],
        "eval_domain": "coalition_ops",
        "eval_specs": ["coalition_ops:stationary", "coalition_ops:reversal"],
    },
}


@dataclass(frozen=True)
class Act3RewardBreakdown:
    """Decomposed trajectory reward for Arm B (`Outcome-GRPO`) vs Arm C (`Epistemic+Brier-GRPO`)."""

    arm: str
    normalized_utility: float
    brier_calibration_reward: float
    non_redundant_probe_bonus: float
    invariant_penalty: float
    total_reward: float


def compute_act3_grpo_reward(
    *,
    arm: str,
    normalized_utility: float,
    invariant_violation: bool,
    predicted_p_ref_high: float | None = None,
    actual_ref_event_high: int | None = None,
    probed_already_falsified_incumbent: bool = False,
    brier_weight: float = 0.25,
) -> Act3RewardBreakdown:
    """Compute strictly on-policy verifiable reward (zero teacher output distillation).

    Arms (`DECISIONS.md` D5):
      - `arm_b_outcome_grpo`: Pure realized net normalized utility minus hard invariant penalty.
      - `arm_c_epistemic_brier_grpo`: Net normalized utility + proper fixed-target Brier score
        `R_brier = 1 - (p_hat - y_ref)^2` minus redundant-incumbent-probe penalty.
    """
    if arm not in ("arm_b_outcome_grpo", "arm_c_epistemic_brier_grpo"):
        raise ValueError(f"Unsupported GRPO arm '{arm}'")

    u = max(0.0, min(1.0, float(normalized_utility)))
    inv_pen = 1.0 if invariant_violation else 0.0

    if arm == "arm_b_outcome_grpo":
        tot = round(u - inv_pen, 4)
        return Act3RewardBreakdown(
            arm=arm,
            normalized_utility=round(u, 4),
            brier_calibration_reward=0.0,
            non_redundant_probe_bonus=0.0,
            invariant_penalty=inv_pen,
            total_reward=tot,
        )

    # Arm C: Proper scoring rule on fixed reference event + anti-fixation probe shaping
    if predicted_p_ref_high is not None and actual_ref_event_high is not None:
        p_clamped = max(0.0, min(1.0, float(predicted_p_ref_high)))
        brier_r = 1.0 - (p_clamped - float(actual_ref_event_high)) ** 2
    else:
        brier_r = 0.75

    probe_adj = -0.10 if probed_already_falsified_incumbent else 0.0
    tot = round(u + brier_weight * brier_r + probe_adj - inv_pen, 4)
    return Act3RewardBreakdown(
        arm=arm,
        normalized_utility=round(u, 4),
        brier_calibration_reward=round(brier_r, 4),
        non_redundant_probe_bonus=probe_adj,
        invariant_penalty=inv_pen,
        total_reward=tot,
    )


def build_reference_sft_examples_for_fold(fold_name: str, n_seeds: int = 8) -> list[dict[str, Any]]:
    """Synthesize strictly held-in reference-consistent epistemic revision SFT warmup examples (`$0.00` Python).

    Crucial protocol rule (`DECISIONS.md` D5 & Round 4 ToS safeguard):
      1. Uses 100% synthetic Python-verifier-generated rollouts (ZERO Gemini API output distillation!).
      2. Strictly excludes `eval_domain` for `fold_name` so LODO transfer is zero-shot out-of-domain.
    """
    if fold_name not in LODO_FOLDS:
        raise ValueError(f"Unknown fold '{fold_name}'. Valid: {sorted(LODO_FOLDS)}")
    train_domains = LODO_FOLDS[fold_name]["train_domains"]
    sft_rows: list[dict[str, Any]] = []

    for dom in train_domains:
        for seed_offset in range(n_seeds):
            seed = 500 + seed_offset
            if dom == "alpha_desk_v2":
                env = make_env("alpha_desk_v2:abrupt_reversal", horizon=24, seed=seed)
                spec = env.start_episode(env.t_star + 1)
                obs = env.step_tool("inspect_desk_brief", {}).observation
                # Reference epistemic action after incumbent drop: probe untested candidate, never re-probe ALPHA
                target_completion = {
                    "epistemic_diagnosis": "calm_incumbent_shortfall_detected",
                    "predicted_p_incumbent_ge_75": 0.08,
                    "tool": "probe_strategy_execution",
                    "args": {"strategy": STRATEGIES_V2[(seed % 3) + 1]},
                    "rationale": "Incumbent calm portfolio dropped sharply under normal liquidity; probe an untested candidate instead of re-probing the falsified incumbent.",
                }
            elif dom == "life_companion":
                env = make_env("life_companion:lifecycle", horizon=18, seed=seed)
                spec = env.start_episode(8)
                obs = env.step_tool("read_user_dossier", {}).observation
                target_completion = {
                    "epistemic_diagnosis": "utility_drift_under_fixed_safety_invariants",
                    "predicted_p_incumbent_ge_75": 0.15,
                    "tool": "search_catalog",
                    "args": {},
                    "rationale": "Preserve hard nut-allergy and $950 budget invariants while listing candidate options to adapt to shifting life-stage utility.",
                }
            else:
                env = make_env("coalition_ops:reversal", horizon=18, seed=seed)
                spec = env.start_episode(8)
                obs = env.step_tool("inspect_dispatch_ledger", {}).observation
                active_partner = str(obs.get("active_counterparty", "VANGUARD_CORP"))
                target_completion = {
                    "epistemic_diagnosis": "counterparty_sla_inversion_suspected",
                    "predicted_p_incumbent_ge_75": 0.12,
                    "tool": "audit_counterparty",
                    "args": {"partner": active_partner},
                    "rationale": "Public dispatch score dropped on prior partner; run targeted compliance/incentive audit before committing contract posture.",
                }

            sft_rows.append(
                {
                    "fold": fold_name,
                    "domain": dom,
                    "held_out_eval_domain": LODO_FOLDS[fold_name]["eval_domain"],
                    "messages": [
                        {"role": "system", "content": spec.domain_prompt},
                        {
                            "role": "user",
                            "content": f"{spec.task_prompt}\nTurn 1 observation: {json.dumps(obs)}\nPrior incumbent calm score dropped from ~84 to ~30.",
                        },
                        {"role": "assistant", "content": json.dumps(target_completion)},
                    ],
                }
            )
    return sft_rows


def export_lodo_sft_datasets(out_dir: Path, n_seeds: int = 16) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for fold_name in LODO_FOLDS:
        rows = build_reference_sft_examples_for_fold(fold_name, n_seeds=n_seeds)
        target_f = out_dir / f"{fold_name}_sft_warmup.jsonl"
        with open(target_f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[fold_name] = len(rows)
    return counts
