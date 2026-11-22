"""AlphaDesk: Multi-turn quant allocation under gradual alpha decay & regime cycles."""

from __future__ import annotations

import math
import random
from typing import Any

from src.envs.base import ParadigmEnv, PublicTaskSpec, StepOutcome

STRATEGIES = ("MOMENTUM_TECH", "MEAN_REV_FX", "CARRY_CREDIT", "DEFENSIVE_VOL")
MAX_SINGLE_STRATEGY_WEIGHT = 0.65  # Invariant compliance constraint across all regimes


class AlphaDeskEnv(ParadigmEnv):
    """Quant portfolio desk where market tools & compliance limits stay fixed while factor alpha shifts.

    Drift modes:
      - `stationary`: Regime A holds for the entire lifetime (noisy observations).
      - `gradual_decay`: Momentum/Carry alpha smoothly crowds out and decays into Mean-Rev/Defensive.
      - `regime_cycle`: Abrupt shift A -> B at `t1`, then recurrence B -> A at `t2`.
    """

    env_name = "alpha_desk"
    max_turns_per_episode = 6

    def __init__(self, *, mode: str = "gradual_decay", horizon: int = 36, seed: int = 42) -> None:
        if mode not in ("stationary", "gradual_decay", "regime_cycle"):
            raise ValueError(f"Unsupported AlphaDesk mode: {mode}")
        self.mode = mode
        self.horizon = int(horizon)
        self.seed = int(seed)

        # Pre-generate independent RNG tapes per episode so pre-shift prefixes match across modes!
        self._episodes: list[dict[str, Any]] = []
        for ep in range(self.horizon):
            ep_rng = random.Random(self.seed * 10_007 + ep * 97 + 13)
            vol_index = round(ep_rng.uniform(12.0, 32.0), 1)
            dispersion = round(ep_rng.uniform(0.20, 0.85), 2)
            liquidity_bps = round(ep_rng.uniform(4.0, 18.0), 1)
            noise_vec = {s: ep_rng.gauss(0.0, 3.2) for s in STRATEGIES}
            self._episodes.append(
                {
                    "vol_index": vol_index,
                    "dispersion": dispersion,
                    "liquidity_bps": liquidity_bps,
                    "noise": noise_vec,
                }
            )

        self._cur_ep: int = 0
        self._inspected_tape: bool = False
        self._checked_crowding: set[str] = set()
        self._done: bool = True

    def num_episodes(self) -> int:
        return self.horizon

    def _blend_alpha(self, ep: int) -> tuple[float, str]:
        """Return `(lambda_B in [0, 1], regime_label)` where 0 = pure Regime A, 1 = pure Regime B."""
        t1 = self.horizon // 3
        t2 = (2 * self.horizon) // 3
        if self.mode == "stationary":
            return 0.0, "regime_A"
        if self.mode == "gradual_decay":
            if ep < t1:
                return 0.0, "regime_A"
            span = max(1, t2 - t1)
            lam = min(1.0, (ep - t1 + 1) / span)
            return lam, ("decay_transition" if lam < 1.0 else "regime_B")
        # regime_cycle: A -> B -> A recurrence
        if ep < t1:
            return 0.0, "regime_A"
        if ep < t2:
            return 1.0, "regime_B"
        return 0.0, "regime_A_recurrence"

    def _expected_strategy_payoffs(self, ep: int) -> tuple[dict[str, float], dict[str, float]]:
        ctx = self._episodes[ep]
        high_vol = ctx["vol_index"] >= 23.0
        high_disp = ctx["dispersion"] >= 0.50

        # Regime A: Trend/Carry market (Momentum & Carry rewarded; Defensive hedge when vol >= 23)
        payoff_a = {
            "MOMENTUM_TECH": 92.0 if not high_vol else 42.0,
            "CARRY_CREDIT": 82.0 if high_disp else 68.0,
            "MEAN_REV_FX": 28.0,
            "DEFENSIVE_VOL": 88.0 if high_vol else 32.0,
        }
        # Regime B: Crowded reversal / liquidity squeeze (Momentum & Carry invert; Mean-Rev & Defensive dominate)
        payoff_b = {
            "MOMENTUM_TECH": 14.0 if not high_vol else 8.0,
            "CARRY_CREDIT": 20.0,
            "MEAN_REV_FX": 94.0 if high_disp else 84.0,
            "DEFENSIVE_VOL": 90.0 if high_vol else 76.0,
        }

        lam, _ = self._blend_alpha(ep)
        active = {
            s: (1.0 - lam) * payoff_a[s] + lam * payoff_b[s]
            for s in STRATEGIES
        }
        return active, payoff_a

    @staticmethod
    def _optimal_weights(payoffs: dict[str, float]) -> dict[str, float]:
        """Exact LP optimum for sum(w)=1, 0 <= w_i <= MAX_SINGLE_STRATEGY_WEIGHT."""
        ranked = sorted(STRATEGIES, key=lambda s: payoffs[s], reverse=True)
        w = {s: 0.0 for s in STRATEGIES}
        w[ranked[0]] = MAX_SINGLE_STRATEGY_WEIGHT
        w[ranked[1]] = round(1.0 - MAX_SINGLE_STRATEGY_WEIGHT, 4)
        return w

    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        self._cur_ep = int(episode_idx)
        self._inspected_tape = False
        self._checked_crowding = set()
        self._done = False

        domain_prompt = (
            "You are the quantitative portfolio manager on AlphaDesk. "
            "In each trading session, inspect current market conditions and risk metrics using tools, "
            "then call `submit_allocation` with portfolio weights across the four strategies: "
            f"{', '.join(STRATEGIES)}.\n"
            "INVARIANT COMPLIANCE RULES (always enforced):\n"
            "1. Weights must be non-negative and sum to 1.0 (tolerance 0.02).\n"
            f"2. Single-strategy concentration cap: no strategy weight may exceed {MAX_SINGLE_STRATEGY_WEIGHT:.2f}.\n"
            "3. Violating compliance rules triggers an automatic compliance desk rejection (Score: 0/100).\n"
            "NOTE: Factor premia and crowding dynamics in live markets may evolve over time without announcement."
        )
        task_prompt = (
            f"Trading Session #{episode_idx + 1}: Inspect the live market tape and strategy diagnostics, "
            "then submit compliant capital weights to maximize session risk-adjusted return (0-100 scale)."
        )
        tool_schemas = [
            {
                "name": "inspect_market_tape",
                "description": "Read current session macro indicators (volatility index, dispersion, liquidity spread).",
                "args": {},
            },
            {
                "name": "check_strategy_diagnostics",
                "description": "Paid execution micro-probe (-12 pts session fee): measure live net alpha regime of one strategy.",
                "args": {"strategy": "One of MOMENTUM_TECH, MEAN_REV_FX, CARRY_CREDIT, DEFENSIVE_VOL"},
            },
            {
                "name": "submit_allocation",
                "description": "Terminal action: commit portfolio weights summing to 1.0 with each weight <= 0.65.",
                "args": {
                    "weights": "JSON object mapping strategy name to float weight in [0, 0.65], summing to 1.0"
                },
            },
        ]
        return PublicTaskSpec(
            env_name=self.env_name,
            episode_idx=episode_idx,
            timestamp=episode_idx + 1,
            domain_prompt=domain_prompt,
            task_prompt=task_prompt,
            tool_schemas=tool_schemas,
        )

    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        if self._done:
            raise RuntimeError("Episode already finished.")
        ctx = self._episodes[self._cur_ep]
        lam, regime_label = self._blend_alpha(self._cur_ep)

        if tool_name == "inspect_market_tape":
            self._inspected_tape = True
            return StepOutcome(
                observation={
                    "session": self._cur_ep + 1,
                    "vol_index": ctx["vol_index"],
                    "cross_asset_dispersion": ctx["dispersion"],
                    "liquidity_spread_bps": ctx["liquidity_bps"],
                    "compliance_cap_per_strategy": MAX_SINGLE_STRATEGY_WEIGHT,
                },
                done=False,
            )

        if tool_name == "check_strategy_diagnostics":
            strat = str(tool_args.get("strategy", "")).strip().upper()
            if strat not in STRATEGIES:
                return StepOutcome(
                    observation={"error": f"Unknown strategy '{strat}'. Valid: {list(STRATEGIES)}"},
                    done=False,
                )
            self._checked_crowding.add(strat)
            active_payoffs, _ = self._expected_strategy_payoffs(self._cur_ep)
            noisy_est = round(active_payoffs[strat] + ctx["noise"][strat], 1)
            return StepOutcome(
                observation={
                    "strategy": strat,
                    "diagnostic_fee_deducted": 12.0,
                    "probed_expected_yield_estimate": noisy_est,
                },
                done=False,
            )

        if tool_name == "submit_allocation":
            self._done = True
            raw_weights = tool_args.get("weights", {})
            if not isinstance(raw_weights, dict):
                return self._finalize_invalid(regime_label, reason="weights_not_dict")

            weights: dict[str, float] = {}
            for s in STRATEGIES:
                try:
                    val = float(raw_weights.get(s, 0.0))
                except (TypeError, ValueError):
                    return self._finalize_invalid(regime_label, reason="non_numeric_weight")
                if math.isnan(val) or math.isinf(val) or val < -1e-4:
                    return self._finalize_invalid(regime_label, reason="negative_or_nan_weight")
                weights[s] = max(0.0, val)

            total_w = sum(weights.values())
            if abs(total_w - 1.0) > 0.03 or any(w > MAX_SINGLE_STRATEGY_WEIGHT + 0.01 for w in weights.values()):
                return self._finalize_invalid(regime_label, reason="compliance_cap_or_sum_violation")

            # Normalize minor rounding within tolerance
            weights = {s: w / total_w for s, w in weights.items()}
            active_payoffs, regime_a_payoffs = self._expected_strategy_payoffs(self._cur_ep)
            opt_w = self._optimal_weights(active_payoffs)
            stale_w = self._optimal_weights(regime_a_payoffs)

            expected_score = sum(weights[s] * active_payoffs[s] for s in STRATEGIES)
            oracle_expected = sum(opt_w[s] * active_payoffs[s] for s in STRATEGIES)
            noise_term = sum(weights[s] * ctx["noise"][s] for s in STRATEGIES)

            # Penalty if agent never read the market tape + paid diagnostic probe fees (-12 pts each)
            probe_fees = 12.0 * len(self._checked_crowding)
            uninformed_penalty = (0.0 if self._inspected_tape else 18.0) + probe_fees
            realized = min(100.0, max(0.0, round(expected_score + noise_term - uninformed_penalty, 1)))

            l1_to_stale = sum(abs(weights[s] - stale_w[s]) for s in STRATEGIES)
            l1_to_opt = sum(abs(weights[s] - opt_w[s]) for s in STRATEGIES)
            stale_match = (lam >= 0.5) and (l1_to_stale < 0.35)

            return StepOutcome(
                observation={"status": "submitted", "score": realized},
                done=True,
                public_score=realized,
                private_eval={
                    "regime": regime_label,
                    "lambda_drift": round(lam, 3),
                    "expected_score": round(expected_score - uninformed_penalty, 2),
                    "oracle_expected_score": round(oracle_expected, 2),
                    "normalized_utility": round(
                        max(0.0, (expected_score - uninformed_penalty) / max(1.0, oracle_expected)), 4
                    ),
                    "stale_policy_match": stale_match,
                    "optimal_policy_match": l1_to_opt < 0.35,
                    "invariant_violation": False,
                    "probes_used": len(self._checked_crowding),
                },
            )

        return StepOutcome(
            observation={"error": f"Unknown tool '{tool_name}'."},
            done=False,
        )

    def _finalize_invalid(self, regime_label: str, reason: str) -> StepOutcome:
        active_payoffs, _ = self._expected_strategy_payoffs(self._cur_ep)
        opt_w = self._optimal_weights(active_payoffs)
        oracle_expected = sum(opt_w[s] * active_payoffs[s] for s in STRATEGIES)
        return StepOutcome(
            observation={"status": "rejected_by_compliance", "reason": reason, "score": 0.0},
            done=True,
            public_score=0.0,
            private_eval={
                "regime": regime_label,
                "expected_score": 0.0,
                "oracle_expected_score": round(oracle_expected, 2),
                "normalized_utility": 0.0,
                "stale_policy_match": False,
                "optimal_policy_match": False,
                "invariant_violation": True,
                "probes_used": len(self._checked_crowding),
            },
        )
