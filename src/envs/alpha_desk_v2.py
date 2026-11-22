"""AlphaDesk-v0.2: Semi-structured analyst briefs, anonymized strategies, and randomized shift t*."""

from __future__ import annotations

import math
import random
from typing import Any

from src.envs.base import ParadigmEnv, PublicTaskSpec, StepOutcome

STRATEGIES_V2 = ("STRAT_ALPHA", "STRAT_BETA", "STRAT_GAMMA", "STRAT_DELTA")
MAX_WEIGHT_V2 = 0.65
PROBE_FEE_V2 = 12.0


class AlphaDeskV2Env(ParadigmEnv):
    """Controlled v0.2 quant allocation benchmark supporting all 5 pre-registered regimes.

    Key upgrades over v0.1 (`src/envs/alpha_desk.py`):
      1. Anonymized strategy IDs (`STRAT_ALPHA..DELTA`) removing semantic name bias.
      2. Mode-free learner `env_name = 'alpha_desk_v2'` (zero mode string leakage).
      3. Semi-structured natural-language analyst & liquidity desk briefs (`inspect_desk_brief`).
      4. Seeded randomized hidden shift boundary `t_star in [12, 20]` over a 40-session stream.
      5. All 5 regimes share bit-identical pre-boundary (`t < t_star`) observations and RNG draws.
    """

    env_name = "alpha_desk_v2"
    max_turns_per_episode = 4

    def __init__(
        self,
        *,
        mode: str = "abrupt_reversal",
        horizon: int = 40,
        seed: int = 101,
        t_star_override: int | None = None,
        counterbalance: bool = False,
    ) -> None:
        valid_modes = (
            "stationary",
            "abrupt_reversal",
            "gradual_decay",
            "opportunity_rise",
            "opportunity_stationary",
            "off_scope_shock",
        )
        if mode not in valid_modes:
            raise ValueError(f"Unsupported AlphaDeskV2 mode '{mode}'. Valid: {valid_modes}")
        self.mode = mode
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.counterbalance = bool(counterbalance)

        if self.counterbalance:
            rot = self.seed % 4
            ordered = [STRATEGIES_V2[(rot + i) % 4] for i in range(4)]
            self._roles = {
                "inc_1": ordered[0],
                "inc_2": ordered[1],
                "unheld_alt": ordered[2],
                "hv_hedge": ordered[3],
            }
            self._hedge_overlaps_calm_winner = (self.seed % 2 == 0)
        else:
            self._roles = {
                "inc_1": "STRAT_ALPHA",
                "inc_2": "STRAT_BETA",
                "unheld_alt": "STRAT_GAMMA",
                "hv_hedge": "STRAT_DELTA",
            }
            self._hedge_overlaps_calm_winner = True

        boundary_rng = random.Random(self.seed * 31_337 + 17)
        self.t_star: int = (
            int(t_star_override) if t_star_override is not None else boundary_rng.randint(12, 16)
        )

        # Pre-generate independent per-episode RNG draws so pre-boundary prefixes match identically!
        self._episodes: list[dict[str, Any]] = []
        for ep in range(self.horizon):
            ep_rng = random.Random(self.seed * 100_003 + ep * 211 + 41)
            # ~25% elevated-volatility sessions where STRAT_DELTA is a permanent tail hedge
            is_high_vol = (ep % 4 == 2) or (ep_rng.random() < 0.08)
            vol_idx = round(ep_rng.uniform(24.5, 33.0) if is_high_vol else ep_rng.uniform(11.5, 20.5), 1)
            dispersion = round(ep_rng.uniform(0.28, 0.84), 2)
            normal_spread = round(ep_rng.uniform(4.2, 13.5), 1)
            noise_vec = {s: ep_rng.gauss(0.0, 2.4) for s in STRATEGIES_V2}
            note_variant = ep_rng.randint(0, 2)
            self._episodes.append(
                {
                    "is_high_vol": is_high_vol,
                    "vol_index": vol_idx,
                    "dispersion": dispersion,
                    "base_spread_bps": normal_spread,
                    "noise": noise_vec,
                    "note_variant": note_variant,
                }
            )

        self._cur_ep = 0
        self._read_brief = False
        self._probed: set[str] = set()
        self._done = True

    def num_episodes(self) -> int:
        return self.horizon

    def _is_liquidity_freeze(self, ep: int) -> bool:
        return self.mode == "off_scope_shock" and ep in (self.t_star, self.t_star + 1)

    def _desk_brief_payload(self, ep: int) -> dict[str, Any]:
        ctx = self._episodes[ep]
        freeze = self._is_liquidity_freeze(ep)
        spread = round(ctx["base_spread_bps"] + (31.0 if freeze else 0.0), 1)

        if freeze:
            commentary = (
                "ALERT — Acute market-wide liquidity freeze across primary dealers: orderbook depth "
                f"collapsed and bid-ask spreads widened to {spread} bps. All directional strategies face "
                "severe transient execution slippage today."
            )
            liq_state = "acute_liquidity_freeze"
        elif ctx["is_high_vol"]:
            notes = [
                f"Risk-off macro tape (vol index {ctx['vol_index']}); cross-asset volatility hedging demand is elevated while orderbooks remain orderly ({spread} bps).",
                f"Elevated implied volatility ({ctx['vol_index']}) on macro headline risk; funding and dealer depth remain normal ({spread} bps).",
                f"High-volatility session ({ctx['vol_index']}) with broad dispersion ({ctx['dispersion']}); protective convexity desks active, normal {spread} bps spread.",
            ]
            commentary = notes[ctx["note_variant"]]
            liq_state = "orderly_normal"
        else:
            notes = [
                f"Calm macro session (vol index {ctx['vol_index']}, dispersion {ctx['dispersion']}) with tight dealer spreads ({spread} bps).",
                f"Low-volatility tape ({ctx['vol_index']}) and steady institutional flow; execution spreads normal at {spread} bps.",
                f"Range-bound macro volatility ({ctx['vol_index']}), healthy liquidity depth ({spread} bps), dispersion at {ctx['dispersion']}.",
            ]
            commentary = notes[ctx["note_variant"]]
            liq_state = "orderly_normal"

        return {
            "session": ep + 1,
            "vol_index": ctx["vol_index"],
            "volatility_regime_hint": "elevated_volatility" if ctx["is_high_vol"] else "calm_volatility",
            "cross_sectional_dispersion": ctx["dispersion"],
            "bid_ask_spread_bps": spread,
            "liquidity_status": liq_state,
            "analyst_desk_brief": commentary,
            "compliance_limits": {"sum_weights": 1.0, "max_single_strategy_weight": MAX_WEIGHT_V2},
        }

    def private_role_map(self) -> dict[str, Any]:
        """Evaluator-only metadata for counterbalancing verification & fixed-target Brier scoring."""
        return {
            **self._roles,
            "hedge_overlaps_calm_winner": self._hedge_overlaps_calm_winner,
            "reference_incumbent_weights": {
                s: (0.65 if s == self._roles["inc_1"] else (0.35 if s == self._roles["inc_2"] else 0.0))
                for s in STRATEGIES_V2
            },
        }

    def _payoffs(self, ep: int) -> tuple[dict[str, float], dict[str, float], str]:
        """Return `(active_payoffs, pre_shift_payoffs, private_regime_label)`."""
        ctx = self._episodes[ep]
        high_vol = ctx["is_high_vol"]
        inc_1 = self._roles["inc_1"]
        inc_2 = self._roles["inc_2"]
        unheld = self._roles["unheld_alt"]
        hedge = self._roles["hv_hedge"]

        if self.counterbalance and not self._hedge_overlaps_calm_winner:
            high_vol_table = {inc_1: 22.0, inc_2: 74.0, unheld: 24.0, hedge: 92.0}
        else:
            high_vol_table = {inc_1: 22.0, inc_2: 28.0, unheld: 74.0, hedge: 92.0}

        if self.mode in ("opportunity_rise", "opportunity_stationary"):
            pre_calm = {inc_1: 72.0, inc_2: 66.0, unheld: 38.0, hedge: 24.0}
            post_calm = {inc_1: 72.0, inc_2: 66.0, unheld: 96.0, hedge: 24.0}
            pre_p = high_vol_table if high_vol else pre_calm
            if ep < self.t_star or self.mode == "opportunity_stationary":
                return pre_p, pre_p, ("pre_shift" if ep < self.t_star else "opportunity_stationary_post")
            return (high_vol_table if high_vol else post_calm), pre_p, "post_opportunity_rise"

        pre_calm = {
            inc_1: 92.0,
            inc_2: 78.0,
            unheld: 24.0,
            hedge: 18.0,
        }
        if not self.counterbalance:
            # Exact legacy mapping for counterbalance=False backward compatibility:
            rev_calm = {
                "STRAT_ALPHA": 16.0,
                "STRAT_BETA": 22.0,
                "STRAT_GAMMA": 94.0,
                "STRAT_DELTA": 78.0,
            }
        elif self._hedge_overlaps_calm_winner:
            rev_calm = {hedge: 94.0, unheld: 78.0, inc_2: 22.0, inc_1: 16.0}
        else:
            rev_calm = {unheld: 94.0, inc_2: 68.0, hedge: 22.0, inc_1: 16.0}
        pre_p = high_vol_table if high_vol else pre_calm

        if self._is_liquidity_freeze(ep):
            freeze_p = {s: 30.0 for s in STRATEGIES_V2}
            return freeze_p, pre_p, "off_scope_liquidity_freeze"

        if self.mode in ("stationary", "off_scope_shock") or ep < self.t_star:
            return pre_p, pre_p, ("pre_shift" if ep < self.t_star else "stationary_post")

        if self.mode == "abrupt_reversal":
            return (high_vol_table if high_vol else rev_calm), pre_p, "post_abrupt_reversal"

        # gradual_decay: smooth 10-step linear transition post-t_star in calm-vol sessions
        lam = min(1.0, (ep - self.t_star + 1) / 10.0)
        if high_vol:
            return high_vol_table, pre_p, "post_gradual_highvol"
        blended = {s: (1.0 - lam) * pre_calm[s] + lam * rev_calm[s] for s in STRATEGIES_V2}
        return blended, pre_p, f"post_gradual_lam_{lam:.2f}"

    @staticmethod
    def _optimal_weights(payoffs: dict[str, float]) -> dict[str, float]:
        ranked = sorted(STRATEGIES_V2, key=lambda s: payoffs[s], reverse=True)
        w = {s: 0.0 for s in STRATEGIES_V2}
        w[ranked[0]] = MAX_WEIGHT_V2
        w[ranked[1]] = round(1.0 - MAX_WEIGHT_V2, 4)
        return w

    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        self._cur_ep = int(episode_idx)
        self._read_brief = False
        self._probed = set()
        self._done = False

        domain_prompt = (
            "You are the Portfolio Manager on AlphaDesk-v2 managing capital across four anonymized quantitative "
            f"strategies: {', '.join(STRATEGIES_V2)}.\n"
            "WORKFLOW & PERMANENT COMPLIANCE RULES:\n"
            "1. Call `inspect_desk_brief` first to read the session macro commentary, volatility, and liquidity conditions.\n"
            f"2. Optionally call `probe_strategy_execution` (-{PROBE_FEE_V2:.0f} pts session fee; limit 1 probe per session) "
            "to measure the live single-strategy yield of one strategy before committing weights.\n"
            f"3. Call `submit_allocation` with non-negative weights summing to 1.0 and each weight <= {MAX_WEIGHT_V2:.2f} "
            "(compliance breach yields Score: 0.0/100).\n"
            "NOTE: Strategy alpha relationships in calm vs. volatile regimes must be learned from experience and may shift silently."
        )
        task_prompt = (
            f"Session #{episode_idx + 1}: Review today's desk brief and submit compliant portfolio weights "
            "to maximize realized session score (0-100)."
        )
        tool_schemas = [
            {
                "name": "inspect_desk_brief",
                "description": "Free: read today's semi-structured macro commentary, volatility index, dispersion, and liquidity spread.",
                "args": {},
            },
            {
                "name": "probe_strategy_execution",
                "description": f"Paid (-{PROBE_FEE_V2:.0f} pts session fee, max 1 probe per session): run live micro-execution check on one strategy.",
                "args": {"strategy": f"One of {', '.join(STRATEGIES_V2)}"},
            },
            {
                "name": "submit_allocation",
                "description": f"Terminal commit: map each strategy to weight in [0, {MAX_WEIGHT_V2:.2f}] summing to 1.0.",
                "args": {"weights": "JSON dict mapping strategy ID to float weight"},
            },
        ]
        return PublicTaskSpec(
            env_name=self.env_name,  # Strictly mode-free!
            episode_idx=episode_idx,
            timestamp=episode_idx + 1,
            domain_prompt=domain_prompt,
            task_prompt=task_prompt,
            tool_schemas=tool_schemas,
        )

    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        if self._done:
            raise RuntimeError("Episode already completed.")
        ctx = self._episodes[self._cur_ep]
        active_p, pre_p, regime_lbl = self._payoffs(self._cur_ep)

        if tool_name == "inspect_desk_brief":
            self._read_brief = True
            return StepOutcome(observation=self._desk_brief_payload(self._cur_ep), done=False)

        if tool_name == "probe_strategy_execution":
            if len(self._probed) >= 1:
                return StepOutcome(
                    observation={"error": "Limit 1 execution probe per session reached; call submit_allocation next."},
                    done=False,
                )
            strat = str(tool_args.get("strategy", "")).strip().upper()
            if strat not in STRATEGIES_V2:
                return StepOutcome(
                    observation={"error": f"Invalid strategy '{strat}'. Valid: {list(STRATEGIES_V2)}"},
                    done=False,
                )
            self._probed.add(strat)
            est = round(active_p[strat] + ctx["noise"][strat], 1)
            return StepOutcome(
                observation={
                    "strategy": strat,
                    "probe_fee_deducted": PROBE_FEE_V2,
                    "live_net_yield_estimate": est,
                },
                done=False,
            )

        if tool_name == "submit_allocation":
            self._done = True
            raw_w = tool_args.get("weights", {})
            if not isinstance(raw_w, dict):
                return self._invalid_outcome(active_p, regime_lbl, "weights_not_dict")

            weights: dict[str, float] = {}
            for s in STRATEGIES_V2:
                try:
                    v = float(raw_w.get(s, 0.0))
                except (TypeError, ValueError):
                    return self._invalid_outcome(active_p, regime_lbl, "non_numeric_weight")
                if math.isnan(v) or math.isinf(v) or v < -1e-4:
                    return self._invalid_outcome(active_p, regime_lbl, "negative_or_nan")
                weights[s] = max(0.0, v)

            tot = sum(weights.values())
            if abs(tot - 1.0) > 0.03 or any(w > MAX_WEIGHT_V2 + 0.01 for w in weights.values()):
                return self._invalid_outcome(active_p, regime_lbl, "compliance_cap_or_sum_breach")

            weights = {s: w / tot for s, w in weights.items()}
            opt_w = self._optimal_weights(active_p)
            stale_w = self._optimal_weights(pre_p)

            gross_exp = sum(weights[s] * active_p[s] for s in STRATEGIES_V2)
            oracle_exp = sum(opt_w[s] * active_p[s] for s in STRATEGIES_V2)
            noise_val = sum(weights[s] * ctx["noise"][s] for s in STRATEGIES_V2)
            penalties = (0.0 if self._read_brief else 18.0) + PROBE_FEE_V2 * len(self._probed)
            net_exp = max(0.0, gross_exp - penalties)
            realized = min(100.0, max(0.0, round(net_exp + noise_val, 1)))

            l1_stale = sum(abs(weights[s] - stale_w[s]) for s in STRATEGIES_V2)
            l1_opt = sum(abs(weights[s] - opt_w[s]) for s in STRATEGIES_V2)
            payoff_changed = opt_w != stale_w

            return StepOutcome(
                observation={"status": "submitted", "score": realized},
                done=True,
                public_score=realized,
                private_eval={
                    "regime": regime_lbl,
                    "t_star": self.t_star,
                    "is_high_vol": ctx["is_high_vol"],
                    "expected_score": round(net_exp, 2),
                    "oracle_expected_score": round(oracle_exp, 2),
                    "normalized_utility": round(net_exp / max(1.0, oracle_exp), 4),
                    "stale_policy_match": payoff_changed and (l1_stale < 0.35),
                    "optimal_policy_match": l1_opt < 0.35,
                    "invariant_violation": False,
                    "probes_used": len(self._probed),
                    "submitted_weights": {s: round(weights[s], 3) for s in STRATEGIES_V2},
                },
            )

        return StepOutcome(observation={"error": f"Unknown tool '{tool_name}'."}, done=False)

    def _invalid_outcome(
        self, active_p: dict[str, float], regime_lbl: str, reason: str
    ) -> StepOutcome:
        opt_w = self._optimal_weights(active_p)
        oracle_exp = sum(opt_w[s] * active_p[s] for s in STRATEGIES_V2)
        return StepOutcome(
            observation={"status": "rejected_by_compliance", "reason": reason, "score": 0.0},
            done=True,
            public_score=0.0,
            private_eval={
                "regime": regime_lbl,
                "t_star": self.t_star,
                "expected_score": 0.0,
                "oracle_expected_score": round(oracle_exp, 2),
                "normalized_utility": 0.0,
                "stale_policy_match": False,
                "optimal_policy_match": False,
                "invariant_violation": True,
                "probes_used": len(self._probed),
            },
        )
