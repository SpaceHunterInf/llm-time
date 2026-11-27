"""AlphaDesk-v0.3: Multi-Shift (`5` Shifts / `6` Windows, `H=84`), `6`-Strategy Stateful Portfolio Benchmark.

Upgrades `AlphaDesk-v0.2` (frozen strictly as Act 1's 1-shift causal unit test) to benchmark parity with
`ServeLearnBench L3` (5–7 shifts + non-monotonic recurrence) and `ReScopeBench` (persistent world state):
  1. `6` anonymized strategies (`STRAT_S1..STRAT_S6`), `1:1:1:1:1:1` Latin-square rotated per seed (`seed % 6`).
  2. Compliance cap `w_i <= 0.55` (`sum(w) = 1.0, w_i >= 0`), so optimal static portfolios place `0.55` on #1 +
     `0.45` on #2, leaving **4 unheld `0`-weight strategies** in every calm regime.
  3. Persistent cross-episode portfolio state (`current_holdings` w_{t-1} + `cumulative_nav`) with exact linear L1
     turnover slippage (`4.0 * 0.5 * ||w_t - w_{t-1}||_1`) and free `inspect_risk_and_turnover` planning tool.
  4. Two separate `5`-shift (`6`-window, `H = 84`) diagnostic tracks + matched `stationary` control:
     - `multi_abrupt_recurrence` (Track 1): Discrete reversals + mid-stream liquidity freeze + BOTH a Full
       Recurrence (`A` or `B`) and a Partial Recurrence (`A'` or `B'`: old #1 returns, old #2 broken), with
       window order randomized across seeds so session index is never a shortcut.
     - `mixed_decay_opportunity` (Track 2): Repeated Type-II silent opportunity rises + 8-step linear crowding
       decays + recurrence + late liquidity freeze & final shift.
  5. Exact backward-induction Bellman / Dynamic-Programming stateful oracle maximizing cumulative expected yield
     minus L1 turnover fees from the shared initial equal-weight portfolio `w_0`.
"""

from __future__ import annotations

import random
from typing import Any

from src.envs.base import ParadigmEnv, PublicTaskSpec, StepOutcome

STRATEGIES_V3 = ("STRAT_S1", "STRAT_S2", "STRAT_S3", "STRAT_S4", "STRAT_S5", "STRAT_S6")
MAX_WEIGHT_V3 = 0.55
PROBE_FEE_V3 = 10.0
SKIP_BRIEF_PENALTY_V3 = 18.0
TURNOVER_FULL_FLIP_PTS_V3 = 4.0


def l1_turnover_half(w_new: dict[str, float], w_prev: dict[str, float]) -> float:
    """Return `0.5 * sum_i |w_new[i] - w_prev[i]|` in `[0.0, 1.0]`."""
    return 0.5 * sum(abs(float(w_new.get(s, 0.0)) - float(w_prev.get(s, 0.0))) for s in STRATEGIES_V3)


def turnover_fee_pts(w_new: dict[str, float], w_prev: dict[str, float]) -> float:
    return round(TURNOVER_FULL_FLIP_PTS_V3 * l1_turnover_half(w_new, w_prev), 4)


class AlphaDeskV3Env(ParadigmEnv):
    """Multi-shift (`6`-window), `6`-strategy stateful portfolio desk with exact Bellman DP turnover oracle."""

    env_name = "alpha_desk_v3"
    max_turns_per_episode = 6

    def __init__(
        self,
        *,
        mode: str = "multi_abrupt_recurrence",
        horizon: int = 84,
        seed: int = 401,
    ) -> None:
        valid_modes = ("stationary", "multi_abrupt_recurrence", "mixed_decay_opportunity")
        if mode not in valid_modes:
            raise ValueError(f"Unsupported AlphaDeskV3 mode '{mode}'. Valid: {valid_modes}")
        if horizon < 24:
            raise ValueError(f"AlphaDeskV3 requires horizon >= 24 for 6 windows (got {horizon}).")

        self.mode = mode
        self.horizon = int(horizon)
        self.seed = int(seed)

        # 1:1:1:1:1:1 Latin-square rotation across all 6 strategies
        rot = self.seed % len(STRATEGIES_V3)
        ordered = [STRATEGIES_V3[(rot + i) % len(STRATEGIES_V3)] for i in range(len(STRATEGIES_V3))]
        self._role_to_strat = {f"r{i}": ordered[i] for i in range(6)}

        # Seed-randomized recurrence schedule on Track 1 so episode index cannot predict Full vs Partial recurrence
        self._partial_first = bool(self.seed % 2 == 1)
        self._swap_ab_rec = bool((self.seed // 2) % 2 == 1)

        # Partition `horizon` into 6 jittered windows (`5` shift boundaries) identically across modes for fixed seed
        win_rng = random.Random(self.seed * 41_999 + 23)
        base_len = self.horizon // 6
        lengths = [base_len] * 6
        rem = self.horizon - sum(lengths)
        for i in range(rem):
            lengths[i % 6] += 1
        # Apply zero-sum +-1 jitter across pairs so window boundaries vary unpredictably per seed
        for pair_i in (0, 2, 4):
            delta = win_rng.choice((-1, 0, 1))
            if lengths[pair_i] + delta >= 4 and lengths[pair_i + 1] - delta >= 4:
                lengths[pair_i] += delta
                lengths[pair_i + 1] -= delta

        self.window_lengths: list[int] = lengths
        self.shift_boundaries: list[int] = []
        acc = 0
        for w_len in lengths[:-1]:
            acc += w_len
            self.shift_boundaries.append(acc)
        # Window ranges: [(start, end_exclusive), ...]
        bounds = [0] + self.shift_boundaries + [self.horizon]
        self.window_ranges: list[tuple[int, int]] = [(bounds[i], bounds[i + 1]) for i in range(6)]

        # Pre-generate per-episode observable context & noise identically across modes!
        self._episodes: list[dict[str, Any]] = []
        for ep in range(self.horizon):
            ep_rng = random.Random(self.seed * 130_003 + ep * 313 + 53)
            is_high_vol = (ep % 5 == 3) or (ep_rng.random() < 0.04)
            vol_idx = round(ep_rng.uniform(25.2, 34.0) if is_high_vol else ep_rng.uniform(11.5, 20.5), 1)
            dispersion = round(ep_rng.uniform(0.25, 0.88), 2)
            base_spread = round(ep_rng.uniform(4.0, 13.0), 1)
            noise_vec = {s: ep_rng.gauss(0.0, 2.4) for s in STRATEGIES_V3}
            note_variant = ep_rng.randint(0, 2)
            self._episodes.append(
                {
                    "is_high_vol": is_high_vol,
                    "vol_index": vol_idx,
                    "dispersion": dispersion,
                    "base_spread_bps": base_spread,
                    "noise": noise_vec,
                    "note_variant": note_variant,
                }
            )

        self._initial_holdings: dict[str, float] = {
            s: (0.17 if i < 4 else 0.16) for i, s in enumerate(STRATEGIES_V3)
        }
        self._holdings: dict[str, float] = dict(self._initial_holdings)
        self._cumulative_nav: float = 100.0
        self._cur_ep: int = 0
        self._read_brief: bool = False
        self._probed: set[str] = set()
        self._done: bool = True

        # Pre-compute exact Bellman/DP net-of-turnover oracle trajectory
        self._dp_oracle = self._compute_bellman_oracle_trajectory()

    def num_episodes(self) -> int:
        return self.horizon

    def window_index(self, ep: int) -> int:
        for w_idx, (lo, hi) in enumerate(self.window_ranges):
            if lo <= ep < hi:
                return w_idx
        return 5

    def _is_liquidity_freeze(self, ep: int) -> bool:
        if self.mode == "multi_abrupt_recurrence":
            w1_lo, w1_hi = self.window_ranges[1]
            mid = w1_lo + max(2, (w1_hi - w1_lo) // 2)
            return ep in (mid, min(w1_hi - 1, mid + 1))
        if self.mode == "mixed_decay_opportunity":
            w5_lo, w5_hi = self.window_ranges[5]
            return ep in (w5_lo, min(w5_hi - 1, w5_lo + 1))
        return False

    @staticmethod
    def _role_vec(r0: float, r1: float, r2: float, r3: float, r4: float, r5: float) -> dict[str, float]:
        return {"r0": r0, "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5}

    def _calm_role_means_and_label(self, ep: int) -> tuple[dict[str, float], str]:
        reg_a = self._role_vec(92.0, 80.0, 22.0, 20.0, 24.0, 18.0)
        if self.mode == "stationary":
            return reg_a, "W1_stat_A"

        w_idx = self.window_index(ep)
        w_lo, _w_hi = self.window_ranges[w_idx]
        step_in_w = ep - w_lo

        if self.mode == "multi_abrupt_recurrence":
            reg_b = self._role_vec(18.0, 20.0, 93.0, 81.0, 24.0, 20.0)
            reg_c = self._role_vec(78.0, 18.0, 16.0, 20.0, 94.0, 22.0)
            reg_d = self._role_vec(18.0, 22.0, 18.0, 93.0, 16.0, 79.0)
            reg_a_prime = self._role_vec(92.0, 18.0, 81.0, 20.0, 24.0, 18.0)
            reg_b_prime = self._role_vec(18.0, 81.0, 93.0, 18.0, 24.0, 20.0)

            if w_idx == 0:
                return reg_a, "W1_acq_A"
            if w_idx == 1:
                return reg_b, "W2_rev_B"
            if w_idx == 2:
                return reg_c, "W3_rev_C"
            if w_idx == 4:
                return reg_d, "W5_rev_D"

            # Windows 3 (W4) and 5 (W6): randomized order of Full Recurrence vs Partial Recurrence!
            first_base, second_base = (("B", "A") if self._swap_ab_rec else ("A", "B"))
            if w_idx == 3:
                is_partial = self._partial_first
                target = first_base
                tag_prefix = "W4"
            else:
                is_partial = not self._partial_first
                target = second_base
                tag_prefix = "W6"

            if target == "A":
                return (reg_a_prime, f"{tag_prefix}_partial_rec_A_prime") if is_partial else (reg_a, f"{tag_prefix}_full_rec_A")
            return (reg_b_prime, f"{tag_prefix}_partial_rec_B_prime") if is_partial else (reg_b, f"{tag_prefix}_full_rec_B")

        # Track 2: `mixed_decay_opportunity`
        # Every Type-II window (W2 and W4) keeps the active incumbent pair 100% flat (zero downward surprise!)
        # while a fresh 2-strategy opportunity pair (96 / 86 -> 91.5 vs 61.3 incumbent = +30.2 pts/step) emerges silently!
        a_opp = self._role_vec(64.0, 58.0, 26.0, 22.0, 24.0, 22.0)
        b_opp = self._role_vec(64.0, 58.0, 96.0, 22.0, 24.0, 86.0)  # Shift 1 (Type-II): r0=64, r1=58 flat; (r2=96, r5=86) rise!
        c_mod = self._role_vec(20.0, 18.0, 18.0, 65.0, 59.0, 20.0)  # Shift 2 (Smooth Decay): settles at r3=65, r4=59
        d_opp = self._role_vec(86.0, 96.0, 18.0, 65.0, 59.0, 20.0)  # Shift 3 (Type-II): r3=65, r4=59 flat; (r1=96, r0=86) rise!
        e_fin = self._role_vec(20.0, 22.0, 94.0, 18.0, 82.0, 20.0)

        if w_idx == 0:
            return a_opp, "W1_acq_A_opp"
        if w_idx == 1:
            return b_opp, "W2_opp_rise_B"
        if w_idx == 2:
            lam = min(1.0, (step_in_w + 1) / 8.0)
            blended = {k: (1.0 - lam) * b_opp[k] + lam * c_mod[k] for k in b_opp}
            return blended, "W3_smooth_decay_C"
        if w_idx == 3:
            return d_opp, "W4_opp_rise_D"
        if w_idx == 4:
            lam = min(1.0, (step_in_w + 1) / 8.0)
            blended = {k: (1.0 - lam) * d_opp[k] + lam * reg_a[k] for k in d_opp}
            return blended, "W5_decay_rec_A"
        if step_in_w < 2:
            return reg_a, "W6_freeze_onset"
        return e_fin, "W6_post_freeze_E"

    def _expected_payoffs(self, ep: int) -> tuple[dict[str, float], dict[str, float], str]:
        """Return `(active_means_by_strat, w1_stale_means_by_strat, regime_label)`."""
        ctx = self._episodes[ep]
        r_map = self._role_to_strat

        w1_role, _ = self._calm_role_means_and_label(0)
        hv_role = self._role_vec(20.0, 22.0, 24.0, 26.0, 80.0, 92.0)

        if self._is_liquidity_freeze(ep):
            freeze_m = {s: 30.0 for s in STRATEGIES_V3}
            stale_m = {r_map[rk]: v for rk, v in (hv_role if ctx["is_high_vol"] else w1_role).items()}
            return freeze_m, stale_m, "acute_liquidity_freeze"

        if ctx["is_high_vol"]:
            hv_m = {r_map[rk]: v for rk, v in hv_role.items()}
            return hv_m, dict(hv_m), "elevated_vol_hedge"

        calm_role, tag = self._calm_role_means_and_label(ep)
        active_m = {r_map[rk]: v for rk, v in calm_role.items()}
        stale_m = {r_map[rk]: v for rk, v in w1_role.items()}
        return active_m, stale_m, tag

    @staticmethod
    def _static_optimal_weights(means: dict[str, float]) -> dict[str, float]:
        ranked = sorted(STRATEGIES_V3, key=lambda s: (means[s], -STRATEGIES_V3.index(s)), reverse=True)
        w = {s: 0.0 for s in STRATEGIES_V3}
        w[ranked[0]] = MAX_WEIGHT_V3
        w[ranked[1]] = round(1.0 - MAX_WEIGHT_V3, 4)
        return w

    def _compute_bellman_oracle_trajectory(self) -> dict[str, Any]:
        """Exact backward-induction Bellman DP maximizing expected yield minus L1 turnover from `w_0`."""
        candidates: list[dict[str, float]] = [dict(self._initial_holdings)]
        for i, s1 in enumerate(STRATEGIES_V3):
            for j, s2 in enumerate(STRATEGIES_V3):
                if i == j:
                    continue
                w = {s: 0.0 for s in STRATEGIES_V3}
                w[s1] = MAX_WEIGHT_V3
                w[s2] = round(1.0 - MAX_WEIGHT_V3, 4)
                candidates.append(w)

        n_c = len(candidates)
        trans_cost = [
            [turnover_fee_pts(candidates[nxt], candidates[prv]) for nxt in range(n_c)]
            for prv in range(n_c)
        ]
        exp_gross: list[list[float]] = []
        for ep in range(self.horizon):
            m, _, _ = self._expected_payoffs(ep)
            exp_gross.append([sum(c[s] * m[s] for s in STRATEGIES_V3) for c in candidates])

        # DP value V[ep][prev_idx] = max cumulative net score from `ep .. H-1` given `holdings_{ep-1} == candidates[prev_idx]`
        next_v = [0.0] * n_c
        policy_idx: list[list[int]] = [[0] * n_c for _ in range(self.horizon)]
        for ep in range(self.horizon - 1, -1, -1):
            cur_v = [0.0] * n_c
            eg = exp_gross[ep]
            for prv in range(n_c):
                best_val = -1e9
                best_a = 0
                tc_row = trans_cost[prv]
                for a in range(n_c):
                    val = eg[a] - tc_row[a] + next_v[a]
                    if val > best_val:
                        best_val = val
                        best_a = a
                cur_v[prv] = best_val
                policy_idx[ep][prv] = best_a
            next_v = cur_v

        # Roll forward from `prev_idx = 0` (`self._initial_holdings`)
        dp_weights: list[dict[str, float]] = []
        dp_step_net: list[float] = []
        prv = 0
        for ep in range(self.horizon):
            a = policy_idx[ep][prv]
            w_star = candidates[a]
            step_net = max(1.0, exp_gross[ep][a] - trans_cost[prv][a])
            dp_weights.append(w_star)
            dp_step_net.append(round(step_net, 4))
            prv = a

        return {
            "weights": dp_weights,
            "step_net": dp_step_net,
            "cumulative_net": round(sum(dp_step_net), 4),
        }

    def private_schedule_metadata(self) -> dict[str, Any]:
        return {
            "role_to_strat": dict(self._role_to_strat),
            "window_ranges": list(self.window_ranges),
            "shift_boundaries": list(self.shift_boundaries),
            "partial_first": self._partial_first,
            "swap_ab_rec": self._swap_ab_rec,
            "dp_cumulative_oracle_net": self._dp_oracle["cumulative_net"],
        }

    def _desk_brief_payload(self, ep: int) -> dict[str, Any]:
        ctx = self._episodes[ep]
        freeze = self._is_liquidity_freeze(ep)
        spread = round(ctx["base_spread_bps"] + (31.0 if freeze else 0.0), 1)
        if freeze:
            commentary = (
                "ALERT — Systemic dealer liquidity freeze: orderbook depth collapsed and bid-ask spreads "
                f"widened to {spread} bps. All directional and defensive strategies face acute slippage today."
            )
            liq_state = "acute_liquidity_freeze"
        elif ctx["is_high_vol"]:
            notes = [
                f"Risk-off macro session (vol index {ctx['vol_index']}); tail-hedging convexity demand elevated, orderly spreads ({spread} bps).",
                f"Elevated implied volatility ({ctx['vol_index']}) on macro shock risk; dealer funding normal ({spread} bps).",
                f"High-volatility tape ({ctx['vol_index']}, dispersion {ctx['dispersion']}); defensive hedges active, {spread} bps spread.",
            ]
            commentary = notes[ctx["note_variant"]]
            liq_state = "orderly_normal"
        else:
            notes = [
                f"Calm macro session (vol index {ctx['vol_index']}, dispersion {ctx['dispersion']}) with normal {spread} bps dealer spreads.",
                f"Low-volatility tape ({ctx['vol_index']}) and steady institutional flow; spreads orderly at {spread} bps.",
                f"Range-bound volatility ({ctx['vol_index']}), healthy depth ({spread} bps), dispersion {ctx['dispersion']}.",
            ]
            commentary = notes[ctx["note_variant"]]
            liq_state = "orderly_normal"

        return {
            "session": ep + 1,
            "current_holdings": {s: round(self._holdings[s], 4) for s in STRATEGIES_V3},
            "cumulative_nav": round(self._cumulative_nav, 2),
            "vol_index": ctx["vol_index"],
            "volatility_regime_hint": "elevated_volatility" if ctx["is_high_vol"] else "calm_volatility",
            "cross_sectional_dispersion": ctx["dispersion"],
            "bid_ask_spread_bps": spread,
            "liquidity_status": liq_state,
            "turnover_cost_per_full_flip": TURNOVER_FULL_FLIP_PTS_V3,
            "analyst_desk_brief": commentary,
            "compliance_limits": {
                "sum_weights": 1.0,
                "max_single_strategy_weight": MAX_WEIGHT_V3,
            },
        }

    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        self._cur_ep = int(episode_idx)
        if self._cur_ep == 0:
            self._holdings = dict(self._initial_holdings)
            self._cumulative_nav = 100.0
        self._read_brief = False
        self._probed = set()
        self._done = False

        domain_prompt = (
            "You are the Portfolio Manager on AlphaDesk-v3 managing stateful capital across six anonymized quantitative "
            f"strategies: {', '.join(STRATEGIES_V3)}.\n"
            "PERMANENT COMPLIANCE & EXECUTION RULES (strictly enforced every session):\n"
            "1. All portfolio weights must be non-negative (`w_i >= 0`) and sum to `1.0` (tolerance +/- 0.03).\n"
            f"2. Single-strategy concentration cap: NO strategy weight may exceed `{MAX_WEIGHT_V3}` (breach => Score: 0.0/100; "
            "optimal concentration places `0.55` on the top strategy and `0.45` on the runner-up).\n"
            f"3. Persistent Portfolio & Turnover Slippage: Yesterday's `current_holdings` carry into today; rebalancing pays "
            f"`{TURNOVER_FULL_FLIP_PTS_V3:.1f} * 0.5 * sum_i |w_i - prev_w_i|` points (0.0 if unchanged, 4.0 for a 100% flip).\n"
            "4. Call `inspect_desk_brief` at the start of every session. You may call `inspect_risk_and_turnover` (free) "
            "to verify compliance & exact turnover cost, and optionally call `probe_strategy_execution` (limit 1 paid probe "
            f"per session; deducts `{PROBE_FEE_V3:.0f}` pts) to inspect one strategy's live gross yield before `submit_allocation`."
        )
        task_prompt = (
            f"Session #{self._cur_ep + 1}: Inspect today's desk brief & current holdings, manage turnover/probing "
            "judiciously, and submit compliant portfolio weights."
        )
        tools = [
            {
                "name": "inspect_desk_brief",
                "description": "Read today's macro/liquidity brief, current persistent portfolio holdings, and compliance limits (free).",
                "args": {},
            },
            {
                "name": "inspect_risk_and_turnover",
                "description": "Free planning tool: check compliance (`sum=1, w_i <= 0.55`) and exact L1 turnover cost from `current_holdings`.",
                "args": {"candidate_weights": "{STRAT_S1..STRAT_S6: float}"},
            },
            {
                "name": "probe_strategy_execution",
                "description": f"Optional paid diagnostic (-{PROBE_FEE_V3:.0f} pts; max 1 per session): observe noisy live gross yield for one strategy.",
                "args": {"strategy": "|".join(STRATEGIES_V3)},
            },
            {
                "name": "submit_allocation",
                "description": "Commit today's compliant portfolio weights across all 6 strategies (updates persistent holdings & terminates session).",
                "args": {"weights": "{STRAT_S1..STRAT_S6: float summing to 1.0, each <= 0.55}"},
            },
        ]
        return PublicTaskSpec(
            env_name=self.env_name,
            episode_idx=self._cur_ep,
            timestamp=self._cur_ep + 1,
            domain_prompt=domain_prompt,
            task_prompt=task_prompt,
            tool_schemas=tools,
        )

    @staticmethod
    def _validate_weights(raw_weights: Any) -> tuple[bool, dict[str, float], str]:
        if not isinstance(raw_weights, dict):
            return False, {}, "weights must be a JSON object."
        parsed: dict[str, float] = {}
        for s in STRATEGIES_V3:
            try:
                val = float(raw_weights.get(s, 0.0))
            except (TypeError, ValueError):
                return False, {}, f"Non-numeric weight for {s}."
            if val < -1e-4:
                return False, {}, f"Negative weight ({val}) violates long-only rule."
            parsed[s] = max(0.0, val)
        tot = sum(parsed.values())
        if abs(tot - 1.0) > 0.035:
            return False, parsed, f"Weights sum to {tot:.3f} (must equal 1.0 +/- 0.03)."
        norm = {s: parsed[s] / tot for s in STRATEGIES_V3}
        if any(v > MAX_WEIGHT_V3 + 0.006 for v in norm.values()):
            return False, norm, f"Weight exceeds max cap {MAX_WEIGHT_V3}."
        return True, norm, ""

    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        if self._done:
            return StepOutcome({"error": "Session already submitted."}, done=True)

        ctx = self._episodes[self._cur_ep]
        if tool_name == "inspect_desk_brief":
            self._read_brief = True
            return StepOutcome(observation=self._desk_brief_payload(self._cur_ep), done=False)

        if tool_name == "inspect_risk_and_turnover":
            ok, norm_w, err = self._validate_weights(tool_args.get("candidate_weights", {}))
            if not ok:
                return StepOutcome(observation={"compliant": False, "reason": err}, done=False)
            l1_half = round(l1_turnover_half(norm_w, self._holdings), 4)
            fee = turnover_fee_pts(norm_w, self._holdings)
            return StepOutcome(
                observation={
                    "compliant": True,
                    "normalized_weights": {s: round(norm_w[s], 4) for s in STRATEGIES_V3},
                    "l1_turnover": l1_half,
                    "turnover_fee_pts": round(fee, 3),
                },
                done=False,
            )

        if tool_name == "probe_strategy_execution":
            strat = str(tool_args.get("strategy", "")).strip().upper()
            if strat not in STRATEGIES_V3:
                return StepOutcome({"error": f"Unknown strategy '{strat}'. Valid: {list(STRATEGIES_V3)}"}, done=False)
            if len(self._probed) >= 1:
                return StepOutcome({"error": "Probe limit (1 per session) already reached."}, done=False)
            self._probed.add(strat)
            means, _, _ = self._expected_payoffs(self._cur_ep)
            obs_yield = round(max(0.0, min(100.0, means[strat] + 0.65 * ctx["noise"][strat])), 1)
            return StepOutcome(
                observation={
                    "strategy": strat,
                    "probe_fee_deducted": PROBE_FEE_V3,
                    "live_net_yield_estimate": obs_yield,
                },
                done=False,
            )

        if tool_name == "submit_allocation":
            self._done = True
            active_m, stale_m, reg_tag = self._expected_payoffs(self._cur_ep)
            dp_w = self._dp_oracle["weights"][self._cur_ep]
            dp_step_net = self._dp_oracle["step_net"][self._cur_ep]
            static_opt_w = self._static_optimal_weights(active_m)
            stale_opt_w = self._static_optimal_weights(stale_m)

            ok, norm_w, err = self._validate_weights(tool_args.get("weights", {}))
            if not ok:
                return StepOutcome(
                    observation={"status": "compliance_violation", "reason": err, "score": 0.0},
                    done=True,
                    public_score=0.0,
                    private_eval={
                        "window": self.window_index(self._cur_ep),
                        "regime": reg_tag,
                        "invariant_violation": True,
                        "stale_policy_match": False,
                        "optimal_policy_match": False,
                        "normalized_utility": 0.0,
                        "dp_oracle_step_net": dp_step_net,
                        "expected_net_no_noise": 0.0,
                        "turnover_fee": 0.0,
                        "probes_used": len(self._probed),
                    },
                )

            turn_fee = turnover_fee_pts(norm_w, self._holdings)
            probe_fee = PROBE_FEE_V3 * len(self._probed) + (0.0 if self._read_brief else SKIP_BRIEF_PENALTY_V3)

            realized_gross = sum(norm_w[s] * (active_m[s] + ctx["noise"][s]) for s in STRATEGIES_V3)
            expected_gross = sum(norm_w[s] * active_m[s] for s in STRATEGIES_V3)

            realized_net = round(max(0.0, min(100.0, realized_gross - turn_fee - probe_fee)), 1)
            expected_net = max(0.0, expected_gross - turn_fee - probe_fee)

            # Update persistent portfolio holdings & NAV!
            self._holdings = dict(norm_w)
            self._cumulative_nav = round(self._cumulative_nav * (1.0 + (realized_net - 50.0) / 1000.0), 3)

            norm_util = min(1.0, max(0.0, expected_net / max(1.0, dp_step_net)))
            l1_to_opt = 0.5 * sum(abs(norm_w[s] - static_opt_w[s]) for s in STRATEGIES_V3)
            l1_to_stale = 0.5 * sum(abs(norm_w[s] - stale_opt_w[s]) for s in STRATEGIES_V3)
            is_shifted = self.window_index(self._cur_ep) >= 1 and self.mode != "stationary" and not ctx["is_high_vol"]

            return StepOutcome(
                observation={
                    "status": "submitted",
                    "gross_score": round(realized_gross, 1),
                    "turnover_fee_deducted": round(turn_fee, 2),
                    "probe_fee_deducted": round(probe_fee, 1),
                    "score": realized_net,
                    "updated_nav": self._cumulative_nav,
                },
                done=True,
                public_score=realized_net,
                private_eval={
                    "window": self.window_index(self._cur_ep),
                    "regime": reg_tag,
                    "is_high_vol": bool(ctx["is_high_vol"]),
                    "is_liquidity_freeze": self._is_liquidity_freeze(self._cur_ep),
                    "invariant_violation": False,
                    "stale_policy_match": bool(is_shifted and l1_to_stale <= 0.18 and l1_to_opt > 0.20),
                    "optimal_policy_match": bool(l1_to_opt <= 0.16),
                    "dp_policy_match": bool(0.5 * sum(abs(norm_w[s] - dp_w[s]) for s in STRATEGIES_V3) <= 0.16),
                    "normalized_utility": round(norm_util, 4),
                    "dp_oracle_step_net": dp_step_net,
                    "expected_net_no_noise": round(expected_net, 3),
                    "turnover_fee": round(turn_fee, 3),
                    "probes_used": len(self._probed),
                },
            )

        return StepOutcome({"error": f"Unknown tool '{tool_name}'."}, done=False)


class AlphaDeskV3ForkEnv(AlphaDeskV3Env):
    """Controlled 2-Window (`W1` Shared Prefix -> `W2` Fork) `AlphaDesk-v0.3` Environment for Autonomy & VOI Audits.

    Supports 6 post-fork worlds across two bit-identical `W1` prefix families (`t_fork = 14`, `H_post = 16`):
      - Family A (`high_a`: `r0=92, r1=80, r2=22, r3=20, r4=24, r5=18`):
        1. `stationary`: `W2` stays at `A` under `orderly_normal`.
        2. `abrupt_reversal`: `W2` shifts permanently to `B = (18, 20, 93, 81, 24, 20)` under `orderly_normal`
           (incumbent `0.55 r0 + 0.45 r1` suffers an immediate `86.6 -> 18.9` apparent loss; optimal response is to
           probe and reallocate).
        3. `hidden_transient_dip`: First 2 sessions of `W2` (`t_fork, t_fork+1`) enter an UNANNOUNCED broad calm factor
           dip (`r0=18, r1=20, r2..r5=19`) under `orderly_normal` (`vol_index` & `bid_ask_spread_bps` 100% identical to
           `stationary` and `abrupt_reversal`!), matching `abrupt_reversal`'s `18.9` apparent loss and `t_fork` realized
           gross/net/NAV bit-for-bit (`0.0` free bits of separation!). Only paying `-10.0` pts for
           `probe_strategy_execution` separates them (`desk_best_yield_estimate ~ 21` vs `~ 93`), followed by `14`
           sessions returning to `A` (Bellman DP optimal policy holds `current_holdings` with `0.0` turnover!).
        4. `freeze_matched_loss`: Observable `acute_liquidity_freeze` control (`bid_ask_spread_bps >= 40` on free brief).
      - Family B (`opp_a`: `r0=64, r1=58, r2=26, r3=22, r4=24, r5=22`):
        5. `opportunity_stationary`: `W2` stays at `A_opp` (`~61.3` calm gross).
        6. `opportunity_rise`: `W2` keeps `r0=64, r1=58` 100% flat (zero downward surprise!) while `r2=96, r5=86`
           (`91.5` gross, `+30.2 pts/session`) emerge silently among long-unprobed strategies.
    """

    VALID_FORK_MODES = (
        "stationary",
        "abrupt_reversal",
        "hidden_transient_dip",
        "freeze_matched_loss",
        "opportunity_stationary",
        "opportunity_rise",
    )

    def __init__(
        self,
        *,
        fork_mode: str = "abrupt_reversal",
        seed: int = 501,
        t_fork: int = 14,
        h_post: int = 16,
        initial_post_holdings: dict[str, float] | None = None,
    ) -> None:
        if fork_mode not in self.VALID_FORK_MODES:
            raise ValueError(f"Unsupported fork_mode '{fork_mode}'. Valid: {self.VALID_FORK_MODES}")
        self.fork_mode = fork_mode
        self.t_fork = int(t_fork)
        self.h_post = int(h_post)
        total_h = self.t_fork + self.h_post

        # Initialize base RNG & episodes deterministically via `stationary` mode, then override window ranges & payoffs
        super().__init__(mode="stationary", horizon=max(24, total_h), seed=seed)
        self.horizon = total_h
        self._episodes = self._episodes[:total_h]
        self.mode = fork_mode
        self.window_lengths = [self.t_fork, self.h_post]
        self.shift_boundaries = [self.t_fork]
        self.window_ranges = [(0, self.t_fork), (self.t_fork, total_h)]

        # Guarantee `t_fork - 1` (pre-fork holdings), `t_fork`, `t_fork + 1`, and `t_fork + 2` are calm_volatility
        # so fork holdings are always the calm incumbent pair and apparent loss + immediate post-dip rebound are clean!
        for boundary_ep in (self.t_fork - 1, self.t_fork, self.t_fork + 1, self.t_fork + 2):
            if 0 <= boundary_ep < total_h:
                self._episodes[boundary_ep]["is_high_vol"] = False
                self._episodes[boundary_ep]["vol_index"] = round(
                    13.5 + (self._episodes[boundary_ep]["vol_index"] % 5.0), 1
                )

        self._fork_holdings_override = dict(initial_post_holdings) if initial_post_holdings else None
        self._dp_oracle = self._compute_bellman_oracle_trajectory()
        self._post_dp_oracle = self._compute_post_fork_bellman_oracle(
            self._fork_holdings_override or self._dp_oracle["weights"][max(0, self.t_fork - 1)]
        )

    def window_index(self, ep: int) -> int:
        return 0 if ep < self.t_fork else 1

    def _is_liquidity_freeze(self, ep: int) -> bool:
        if self.fork_mode == "freeze_matched_loss":
            return ep in (self.t_fork, self.t_fork + 1)
        return False

    def _is_hidden_transient_dip(self, ep: int) -> bool:
        if self.fork_mode == "hidden_transient_dip":
            return ep in (self.t_fork, self.t_fork + 1)
        return False

    def _calm_role_means_and_label(self, ep: int) -> tuple[dict[str, float], str]:
        reg_a = self._role_vec(92.0, 80.0, 22.0, 20.0, 24.0, 18.0)
        reg_b = self._role_vec(18.0, 20.0, 93.0, 81.0, 24.0, 20.0)
        dip_role = self._role_vec(18.0, 20.0, 19.0, 19.0, 19.0, 19.0)
        a_opp = self._role_vec(64.0, 58.0, 26.0, 22.0, 24.0, 22.0)
        b_opp = self._role_vec(64.0, 58.0, 96.0, 22.0, 24.0, 86.0)

        is_opp_family = self.fork_mode in ("opportunity_stationary", "opportunity_rise")
        if ep < self.t_fork:
            return (a_opp, "W1_acq_A_opp") if is_opp_family else (reg_a, "W1_acq_A")

        if self.fork_mode == "stationary":
            return reg_a, "W2_stat_A"
        if self.fork_mode == "abrupt_reversal":
            return reg_b, "W2_rev_B"
        if self.fork_mode == "hidden_transient_dip":
            if self._is_hidden_transient_dip(ep):
                return dip_role, "hidden_calm_factor_dip_19"
            return reg_a, "W2_post_dip_A"
        if self.fork_mode == "freeze_matched_loss":
            return reg_a, "W2_post_freeze_A"
        if self.fork_mode == "opportunity_stationary":
            return a_opp, "W2_opp_stat_A"
        return b_opp, "W2_opp_rise_B"

    def _expected_payoffs(self, ep: int) -> tuple[dict[str, float], dict[str, float], str]:
        ctx = self._episodes[ep]
        r_map = self._role_to_strat
        w1_role, _ = self._calm_role_means_and_label(0)
        hv_role = self._role_vec(20.0, 22.0, 24.0, 26.0, 80.0, 92.0)

        if self._is_liquidity_freeze(ep):
            freeze_m = {s: 19.0 for s in STRATEGIES_V3}
            stale_m = {r_map[rk]: v for rk, v in w1_role.items()}
            return freeze_m, stale_m, "acute_liquidity_freeze_19"

        if ctx["is_high_vol"]:
            hv_m = {r_map[rk]: v for rk, v in hv_role.items()}
            return hv_m, dict(hv_m), "elevated_vol_hedge"

        calm_role, tag = self._calm_role_means_and_label(ep)
        active_m = {r_map[rk]: v for rk, v in calm_role.items()}
        stale_m = {r_map[rk]: v for rk, v in w1_role.items()}
        return active_m, stale_m, tag

    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        spec = super().start_episode(episode_idx)
        updated_tools = []
        for t in spec.tool_schemas:
            if t["name"] == "probe_strategy_execution":
                t = {
                    **t,
                    "description": (
                        f"Optional paid diagnostic (-{PROBE_FEE_V3:.0f} pts; max 1 per session): observe noisy live "
                        "gross yield for one strategy plus anonymized desk-wide mean and max strategy yields today."
                    ),
                }
            updated_tools.append(t)
        return PublicTaskSpec(
            env_name=spec.env_name,
            episode_idx=spec.episode_idx,
            timestamp=spec.timestamp,
            domain_prompt=spec.domain_prompt,
            task_prompt=spec.task_prompt,
            tool_schemas=updated_tools,
        )

    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        if tool_name == "probe_strategy_execution" and not self._done:
            strat = str(tool_args.get("strategy", "")).strip().upper()
            if strat not in STRATEGIES_V3:
                return StepOutcome({"error": f"Unknown strategy '{strat}'. Valid: {list(STRATEGIES_V3)}"}, done=False)
            if len(self._probed) >= 1:
                return StepOutcome({"error": "Probe limit (1 per session) already reached."}, done=False)
            self._probed.add(strat)
            ctx = self._episodes[self._cur_ep]
            means, _, _ = self._expected_payoffs(self._cur_ep)
            all_noisy = {
                s: round(max(0.0, min(100.0, means[s] + 0.65 * ctx["noise"][s])), 1)
                for s in STRATEGIES_V3
            }
            return StepOutcome(
                observation={
                    "strategy": strat,
                    "probe_fee_deducted": PROBE_FEE_V3,
                    "live_net_yield_estimate": all_noisy[strat],
                    "desk_mean_yield_estimate": round(sum(all_noisy.values()) / len(STRATEGIES_V3), 1),
                    "desk_best_yield_estimate": round(max(all_noisy.values()), 1),
                },
                done=False,
            )

        out = super().step_tool(tool_name, tool_args)
        if tool_name == "submit_allocation" and out.private_eval:
            out.private_eval["is_hidden_transient_dip"] = self._is_hidden_transient_dip(self._cur_ep)
        return out

    def _compute_post_fork_bellman_oracle(self, start_holdings: dict[str, float]) -> dict[str, Any]:
        """Exact backward-induction Bellman DP over `ep = t_fork .. H-1` starting from `start_holdings`."""
        candidates: list[dict[str, float]] = [dict(start_holdings)]
        for i, s1 in enumerate(STRATEGIES_V3):
            for j, s2 in enumerate(STRATEGIES_V3):
                if i == j:
                    continue
                w = {s: 0.0 for s in STRATEGIES_V3}
                w[s1] = MAX_WEIGHT_V3
                w[s2] = round(1.0 - MAX_WEIGHT_V3, 4)
                candidates.append(w)

        n_c = len(candidates)
        trans_cost = [
            [turnover_fee_pts(candidates[nxt], candidates[prv]) for nxt in range(n_c)]
            for prv in range(n_c)
        ]
        post_eps = list(range(self.t_fork, self.horizon))
        exp_gross = []
        for ep in post_eps:
            m, _, _ = self._expected_payoffs(ep)
            exp_gross.append([sum(c[s] * m[s] for s in STRATEGIES_V3) for c in candidates])

        next_v = [0.0] * n_c
        policy_idx = [[0] * n_c for _ in range(len(post_eps))]
        for idx in range(len(post_eps) - 1, -1, -1):
            cur_v = [0.0] * n_c
            eg = exp_gross[idx]
            for prv in range(n_c):
                best_val = -1e9
                best_a = 0
                tc_row = trans_cost[prv]
                for a in range(n_c):
                    val = eg[a] - tc_row[a] + next_v[a]
                    if val > best_val:
                        best_val = val
                        best_a = a
                cur_v[prv] = best_val
                policy_idx[idx][prv] = best_a
            next_v = cur_v

        dp_weights: dict[int, dict[str, float]] = {}
        dp_step_net: dict[int, float] = {}
        prv = 0
        for idx, ep in enumerate(post_eps):
            a = policy_idx[idx][prv]
            step_net = max(1.0, exp_gross[idx][a] - trans_cost[prv][a])
            dp_weights[ep] = candidates[a]
            dp_step_net[ep] = round(step_net, 4)
            prv = a
        return {
            "weights_by_ep": dp_weights,
            "step_net_by_ep": dp_step_net,
            "cumulative_net": round(sum(dp_step_net.values()), 4),
        }

    def set_fork_state(self, holdings: dict[str, float], cumulative_nav: float = 100.0) -> None:
        """Set exact shared portfolio holdings & NAV at `t_fork` and recompute post-fork Bellman DP reference."""
        self._holdings = {s: float(holdings.get(s, 0.0)) for s in STRATEGIES_V3}
        self._cumulative_nav = float(cumulative_nav)
        self._post_dp_oracle = self._compute_post_fork_bellman_oracle(self._holdings)
        for ep, w_s in self._post_dp_oracle["weights_by_ep"].items():
            self._dp_oracle["weights"][ep] = w_s
            self._dp_oracle["step_net"][ep] = self._post_dp_oracle["step_net_by_ep"][ep]


class AlphaDeskV3ThreeWindowEnv(AlphaDeskV3ForkEnv):
    """Controlled 3-Window (`W1 -> W2 -> W3`) Multi-Shift & Stationary Stream for Editable-Procedure Evaluation.

    Starts from a shared live-acquired `W1` (`0 .. w1_len-1`) checkpoint and runs `w2_len + w3_len` post-checkpoint
    episodes across two unannounced post-checkpoint windows (`W2` and `W3`):
      - Family A (`high_a`: `W1` has `A = (92, 80, 22, 20, 24, 18)`):
        1. `stationary_3w`: `W2 = A`, `W3 = A` (stationary VOI control: procedures should keep their points!).
        2. `dip_then_reversal_3w`: `W2` opens with a 2-session `hidden_transient_dip` (`18, 20, 19..19`) then returns
           to `A` for the remainder of `W2`; `W3` shifts permanently to `B = (18, 20, 93, 81, 24, 20)`.
        3. `double_reversal_3w`: `W2 = B = (18, 20, 93, 81, 24, 20)`, `W3 = C = (78, 18, 16, 20, 94, 22)`.
      - Family B (`opp_3w`: `W1` has `A_opp3 = (56, 52, 24, 20, 22, 20)`):
        4. `opp_stationary_3w`: `W2 = A_opp3`, `W3 = A_opp3` (stationary VOI control for Type-II family).
        5. `opp_double_rise_3w`: `W2 = B_opp3 = (56, 52, 78, 20, 22, 72)` (+21.1 pts/ep Type-II rise, 0 downward
           surprise), `W3 = D_opp3 = (56, 52, 78, 96, 90, 72)` (+18.0 pts/ep second Type-II rise, 0 downward surprise).
    """

    VALID_STREAM_MODES = (
        "stationary_3w",
        "dip_then_reversal_3w",
        "double_reversal_3w",
        "opp_stationary_3w",
        "opp_double_rise_3w",
    )

    def __init__(
        self,
        *,
        stream_mode: str = "dip_then_reversal_3w",
        seed: int = 601,
        w1_len: int = 12,
        w2_len: int = 14,
        w3_len: int = 14,
        initial_post_holdings: dict[str, float] | None = None,
    ) -> None:
        if stream_mode not in self.VALID_STREAM_MODES:
            raise ValueError(f"Unsupported stream_mode '{stream_mode}'. Valid: {self.VALID_STREAM_MODES}")
        self.stream_mode = stream_mode
        self.w1_len = int(w1_len)
        self.w2_len = int(w2_len)
        self.w3_len = int(w3_len)
        self.t_w3 = self.w1_len + self.w2_len

        super().__init__(
            fork_mode="stationary",
            seed=seed,
            t_fork=self.w1_len,
            h_post=self.w2_len + self.w3_len,
            initial_post_holdings=initial_post_holdings,
        )
        self.mode = stream_mode
        self.window_lengths = [self.w1_len, self.w2_len, self.w3_len]
        self.shift_boundaries = [self.w1_len, self.t_w3]
        self.window_ranges = [
            (0, self.w1_len),
            (self.w1_len, self.t_w3),
            (self.t_w3, self.horizon),
        ]

        # Ensure clean calm_volatility transitions around both W1->W2 and W2->W3 boundaries
        for boundary_ep in (
            self.w1_len - 1,
            self.w1_len,
            self.w1_len + 1,
            self.w1_len + 2,
            self.t_w3 - 1,
            self.t_w3,
            self.t_w3 + 1,
        ):
            if 0 <= boundary_ep < self.horizon:
                self._episodes[boundary_ep]["is_high_vol"] = False
                self._episodes[boundary_ep]["vol_index"] = round(
                    13.5 + (self._episodes[boundary_ep]["vol_index"] % 5.0), 1
                )

        self._dp_oracle = self._compute_bellman_oracle_trajectory()
        self._post_dp_oracle = self._compute_post_fork_bellman_oracle(
            self._fork_holdings_override or self._dp_oracle["weights"][max(0, self.t_fork - 1)]
        )
        for ep, w_s in self._post_dp_oracle["weights_by_ep"].items():
            self._dp_oracle["weights"][ep] = w_s
            self._dp_oracle["step_net"][ep] = self._post_dp_oracle["step_net_by_ep"][ep]

    def window_index(self, ep: int) -> int:
        if ep < self.w1_len:
            return 0
        if ep < self.t_w3:
            return 1
        return 2

    def _is_liquidity_freeze(self, ep: int) -> bool:
        return False

    def _is_hidden_transient_dip(self, ep: int) -> bool:
        if self.stream_mode == "dip_then_reversal_3w":
            return ep in (self.w1_len, self.w1_len + 1)
        return False

    def _calm_role_means_and_label(self, ep: int) -> tuple[dict[str, float], str]:
        reg_a = self._role_vec(92.0, 80.0, 22.0, 20.0, 24.0, 18.0)
        reg_b = self._role_vec(18.0, 20.0, 93.0, 81.0, 24.0, 20.0)
        reg_c = self._role_vec(78.0, 18.0, 16.0, 20.0, 94.0, 22.0)
        dip_role = self._role_vec(18.0, 20.0, 19.0, 19.0, 19.0, 19.0)

        a_opp3 = self._role_vec(56.0, 52.0, 24.0, 20.0, 22.0, 20.0)
        b_opp3 = self._role_vec(56.0, 52.0, 78.0, 20.0, 22.0, 72.0)
        d_opp3 = self._role_vec(56.0, 52.0, 78.0, 96.0, 90.0, 72.0)

        is_opp = self.stream_mode in ("opp_stationary_3w", "opp_double_rise_3w")
        w_idx = self.window_index(ep)
        if w_idx == 0:
            return (a_opp3, "W1_acq_A_opp3") if is_opp else (reg_a, "W1_acq_A")

        if self.stream_mode == "stationary_3w":
            return reg_a, f"W{w_idx + 1}_stat_A"
        if self.stream_mode == "dip_then_reversal_3w":
            if w_idx == 1:
                if self._is_hidden_transient_dip(ep):
                    return dip_role, "hidden_calm_factor_dip_19"
                return reg_a, "W2_post_dip_A"
            return reg_b, "W3_rev_B"
        if self.stream_mode == "double_reversal_3w":
            return (reg_b, "W2_rev_B") if w_idx == 1 else (reg_c, "W3_rev_C")
        if self.stream_mode == "opp_stationary_3w":
            return a_opp3, f"W{w_idx + 1}_opp_stat_A3"
        return (b_opp3, "W2_opp_rise_B3") if w_idx == 1 else (d_opp3, "W3_opp_rise_D3")


class AlphaDeskV3MetaHorizonEnv(AlphaDeskV3ForkEnv):
    """Single-Lifetime `H=60` 4-Window Private-Schedule Environment for Two-Level Reviewer Evaluation.

    Implements the pre-registered `H=60` protocol (`setup-debates/2026-11-26-decomposed-procedure-and-vbq-resolution.md`):
      - Seed-jittered private window boundaries (`w1_len = 12 +- 1`, `w2_len = 14 +- 2`, `w3_len = 16 +- 2`,
        `w4_len = H - (w1 + w2 + w3) >= 13` for `H = 60`), identical between `drift_meta_60` and `stationary_meta_60`
        for each seed so `W1` (`0 .. w1_len - 1`) and all observable context/noise vectors match bit-for-bit.
      - Modes:
        1. `stationary_meta_60`: Stays at `A = (92, 80, 22, 20, 24, 18)` across all 60 sessions.
        2. `drift_meta_60`:
           - `W1` (`Calm Acquisition`): `A = (92, 80, 22, 20, 24, 18)`.
           - `W2` (`Temporary Loss`): First 2 sessions (`t_w2, t_w2 + 1`) enter an unannounced `hidden_transient_dip`
             (`18, 20, 19..19` under `orderly_normal`, separable only via `-10 pt` `probe_strategy_execution`), then
             return to `A` for the remainder of `W2`.
           - `W3` (`Lasting Change`): Permanent shift to `B = (18, 20, 93, 81, 24, 20)` (penalizing sticky `W2` advice).
           - `W4` (`Later Surprise — Reuse + Revision`): Shifts to Partial Recurrence `A' = (92, 18, 81, 20, 24, 18)`,
             where `W3`'s runner-up (`r3=20`) breaks (`87.6 -> 53.6`), `W3`'s winner (`r2=81`) stays optimal runner-up,
             and `W1`'s winner (`r0=92`) returns.
        3. `drift_dip_c_60` (optional sensitivity stream): Same `W1..W3`, then `W4` opens with a 2-session calm dip on
           `B` before shifting to `C = (78, 18, 16, 20, 94, 22)`.
    """

    VALID_META_MODES = (
        "stationary_meta_60",
        "drift_meta_60",
        "drift_dip_c_60",
    )

    def __init__(
        self,
        *,
        meta_mode: str = "drift_meta_60",
        seed: int = 701,
        horizon: int = 60,
        initial_post_holdings: dict[str, float] | None = None,
    ) -> None:
        if meta_mode not in self.VALID_META_MODES:
            raise ValueError(f"Unsupported meta_mode '{meta_mode}'. Valid: {self.VALID_META_MODES}")
        if horizon < 32:
            raise ValueError(f"AlphaDeskV3MetaHorizonEnv requires horizon >= 32 (got {horizon}).")

        self.meta_mode = meta_mode
        win_rng = random.Random(int(seed) * 97_103 + 17)
        if horizon >= 52:
            w1_len = 12 + win_rng.choice((-1, 0, 1))
            w2_len = 14 + win_rng.choice((-2, -1, 0, 1, 2))
            w3_len = 16 + win_rng.choice((-2, -1, 0, 1, 2))
            w4_len = int(horizon) - (w1_len + w2_len + w3_len)
        else:
            q = int(horizon) // 4
            w1_len = q
            w2_len = q
            w3_len = q
            w4_len = int(horizon) - 3 * q

        self.w1_len = w1_len
        self.w2_len = w2_len
        self.w3_len = w3_len
        self.w4_len = w4_len
        self.t_w2 = w1_len
        self.t_w3 = w1_len + w2_len
        self.t_w4 = w1_len + w2_len + w3_len

        super().__init__(
            fork_mode="stationary",
            seed=seed,
            t_fork=self.w1_len,
            h_post=int(horizon) - self.w1_len,
            initial_post_holdings=initial_post_holdings,
        )
        self.mode = meta_mode
        self.window_lengths = [self.w1_len, self.w2_len, self.w3_len, self.w4_len]
        self.shift_boundaries = [self.t_w2, self.t_w3, self.t_w4]
        self.window_ranges = [
            (0, self.t_w2),
            (self.t_w2, self.t_w3),
            (self.t_w3, self.t_w4),
            (self.t_w4, self.horizon),
        ]

        # Enforce calm_volatility transitions around every private window boundary identically across meta_modes
        for b in (self.t_w2, self.t_w3, self.t_w4):
            for boundary_ep in (b - 1, b, b + 1, b + 2):
                if 0 <= boundary_ep < self.horizon:
                    self._episodes[boundary_ep]["is_high_vol"] = False
                    self._episodes[boundary_ep]["vol_index"] = round(
                        13.5 + (self._episodes[boundary_ep]["vol_index"] % 5.0), 1
                    )

        self._dp_oracle = self._compute_bellman_oracle_trajectory()
        self._post_dp_oracle = self._compute_post_fork_bellman_oracle(
            self._fork_holdings_override or self._dp_oracle["weights"][max(0, self.t_fork - 1)]
        )
        for ep, w_s in self._post_dp_oracle["weights_by_ep"].items():
            self._dp_oracle["weights"][ep] = w_s
            self._dp_oracle["step_net"][ep] = self._post_dp_oracle["step_net_by_ep"][ep]

    def window_index(self, ep: int) -> int:
        if ep < self.t_w2:
            return 0
        if ep < self.t_w3:
            return 1
        if ep < self.t_w4:
            return 2
        return 3

    def _is_liquidity_freeze(self, ep: int) -> bool:
        return False

    def _is_hidden_transient_dip(self, ep: int) -> bool:
        if self.meta_mode in ("drift_meta_60", "drift_dip_c_60") and ep in (self.t_w2, self.t_w2 + 1):
            return True
        if self.meta_mode == "drift_dip_c_60" and ep in (self.t_w4, self.t_w4 + 1):
            return True
        return False

    def _calm_role_means_and_label(self, ep: int) -> tuple[dict[str, float], str]:
        reg_a = self._role_vec(92.0, 80.0, 22.0, 20.0, 24.0, 18.0)
        reg_b = self._role_vec(18.0, 20.0, 93.0, 81.0, 24.0, 20.0)
        reg_a_prime = self._role_vec(92.0, 18.0, 81.0, 20.0, 24.0, 18.0)
        reg_c = self._role_vec(78.0, 18.0, 16.0, 20.0, 94.0, 22.0)
        dip_role = self._role_vec(18.0, 20.0, 19.0, 19.0, 19.0, 19.0)
        dip_b_role = self._role_vec(19.0, 19.0, 18.0, 20.0, 19.0, 19.0)

        w_idx = self.window_index(ep)
        if w_idx == 0:
            return reg_a, "W1_acq_A"
        if self.meta_mode == "stationary_meta_60":
            return reg_a, f"W{w_idx + 1}_stat_A"
        if w_idx == 1:
            if ep in (self.t_w2, self.t_w2 + 1):
                return dip_role, "W2_hidden_calm_dip_19"
            return reg_a, "W2_post_dip_A"
        if w_idx == 2:
            return reg_b, "W3_lasting_B"
        if self.meta_mode == "drift_dip_c_60":
            if ep in (self.t_w4, self.t_w4 + 1):
                return dip_b_role, "W4_hidden_calm_dip_19"
            return reg_c, "W4_surprise_C"
        return reg_a_prime, "W4_surprise_partial_A_prime"
