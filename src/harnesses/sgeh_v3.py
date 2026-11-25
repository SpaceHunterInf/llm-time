"""AlphaDesk-v0.3 Harnesses: `ScopedCardBookV3Harness` (Baseline) & `ScopedSGEHV3` (Ours + Shelf/Hazard Ablations).

Implements all three pre-registered `SGEH-v0.3` refinements agreed with PI & Codex (`2026-11-21`):
  1. Per-Strategy Shelf Estimates (`strategy_means: {s: mu_hat_s}`): Shelved calm regimes record individual
     probed/attributed strategy expectations rather than only an unattributed portfolio mean.
  2. `Shelf-Match` + Linear Runner-Up Residual Check (`A` vs. `A'` Partial Recurrence Protection):
     - On a post-shift alarm (`enable_shelf=True`), probes the top non-quarantined shelved regime's primary
       strategy `s1` first.
     - If `|y_probe(s1) - mu_shelf(s1)| <= 12` and `y_probe(s1) >= 80`, tentatively restores `0.55 s1 + 0.45 s2`
       and flags `pending_runnerup_verify = s2`.
     - On the next calm execution, checks implied runner-up yield `(y_gross - 0.55 * mu_hat(s1)) / 0.45`:
       under Full Recurrence (`A`), both legs hold (`true_restorations += 1`, 1 probe total!); under Partial
       Recurrence (`A'`, where `s2` decayed to `~18`), `s2` is immediately quarantined (`partial_recurrence_caught += 1`)
       while keeping `0.55` locked on verified `s1` and probing only for a replacement `0.45` runner-up!
  3. Zero Window-Spanning Cooldown: Resets `last_tested_ep[s] = cur_sc_ep` (`staleness = 0`) whenever `s` is
     probed or falsified, keeping standard `hazard_interval = 9` calm sessions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.alpha_desk_v3 import MAX_WEIGHT_V3, PROBE_FEE_V3, STRATEGIES_V3
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record
from src.harnesses.scoped_cards import extract_submitted_weights, sanitize_scope_text


def normalize_weights_v3(raw_w: Any, fallback_primary: str, fallback_secondary: str | None = None) -> dict[str, float]:
    """Project arbitrary LLM candidate weights onto the compliant `AlphaDesk-v0.3` simplex (`sum=1, 0 <= w_i <= 0.55`)."""
    if isinstance(raw_w, dict):
        parsed: dict[str, float] = {}
        for s in STRATEGIES_V3:
            try:
                v = max(0.0, min(MAX_WEIGHT_V3, float(raw_w.get(s, 0.0))))
            except (TypeError, ValueError):
                v = 0.0
            parsed[s] = v
        tot = sum(parsed.values())
        if tot > 0.12:
            norm = {s: parsed[s] / tot for s in STRATEGIES_V3}
            # Iteratively cap at MAX_WEIGHT_V3 and redistribute excess to next-highest strategies
            for _ in range(4):
                excess = 0.0
                uncapped = []
                for s in STRATEGIES_V3:
                    if norm[s] > MAX_WEIGHT_V3:
                        excess += norm[s] - MAX_WEIGHT_V3
                        norm[s] = MAX_WEIGHT_V3
                    else:
                        uncapped.append(s)
                if excess <= 1e-6 or not uncapped:
                    break
                uncapped.sort(key=lambda x: norm[x], reverse=True)
                add_each = excess / len(uncapped[:2])
                for u_s in uncapped[:2]:
                    norm[u_s] += add_each
            rounded = {s: round(norm[s], 2) for s in STRATEGIES_V3}
            diff = round(1.0 - sum(rounded.values()), 2)
            for cand in sorted(STRATEGIES_V3, key=lambda x: rounded[x], reverse=True):
                if 0.0 <= rounded[cand] + diff <= MAX_WEIGHT_V3:
                    rounded[cand] = round(rounded[cand] + diff, 2)
                    break
            return rounded

    p1 = fallback_primary if fallback_primary in STRATEGIES_V3 else STRATEGIES_V3[0]
    p2 = (
        fallback_secondary
        if (fallback_secondary in STRATEGIES_V3 and fallback_secondary != p1)
        else next(s for s in STRATEGIES_V3 if s != p1)
    )
    return {s: (0.55 if s == p1 else (0.45 if s == p2 else 0.0)) for s in STRATEGIES_V3}


def inspect_public_signals_v3(record: PublicEpisodeRecord) -> dict[str, Any]:
    """Extract strictly learner-public scope, liquidity status, probe results, turnover fee, and weights."""
    scope = "calm_volatility"
    off_scope_anomaly = False
    probed: dict[str, float] = {}
    probe_desk_stats: dict[str, dict[str, float]] = {}
    turnover_fee = 0.0
    gross_from_obs: float | None = None

    for st in record.steps:
        obs = st.get("obs", {})
        if not isinstance(obs, dict):
            continue
        if "volatility_regime_hint" in obs:
            scope = str(obs["volatility_regime_hint"])
        liq = str(obs.get("liquidity_status", "orderly_normal"))
        if liq not in ("orderly_normal", "", "normal"):
            off_scope_anomaly = True
        if st.get("tool") == "probe_strategy_execution" and "live_net_yield_estimate" in obs:
            strat = str(obs.get("strategy") or st.get("args", {}).get("strategy", "")).upper()
            if strat in STRATEGIES_V3:
                try:
                    probed[strat] = float(obs["live_net_yield_estimate"])
                    if "desk_best_yield_estimate" in obs or "desk_mean_yield_estimate" in obs:
                        probe_desk_stats[strat] = {
                            k: float(obs[k])
                            for k in ("desk_mean_yield_estimate", "desk_best_yield_estimate")
                            if k in obs
                        }
                except (TypeError, ValueError):
                    pass
        if st.get("tool") == "submit_allocation" and obs.get("status") == "submitted":
            try:
                turnover_fee = float(obs.get("turnover_fee_deducted", 0.0))
                if "gross_score" in obs:
                    gross_from_obs = float(obs["gross_score"])
            except (TypeError, ValueError):
                pass

    weights = extract_submitted_weights(record)
    if gross_from_obs is None:
        gross_est = min(100.0, float(record.public_score) + PROBE_FEE_V3 * len(probed) + turnover_fee)
    else:
        gross_est = gross_from_obs

    return {
        "scope": scope,
        "off_scope_anomaly": off_scope_anomaly,
        "probed": probed,
        "probe_desk_stats": probe_desk_stats,
        "weights": weights,
        "turnover_fee": turnover_fee,
        "gross_portfolio_score": gross_est,
    }


class ScopedCardBookV3Harness(EpistemicHarness):
    """6-strategy structured `ScopedCardBook (Baseline)` for `AlphaDesk-v0.3` (`reflect_every = 3`)."""

    name = "scoped_cards"

    def __init__(self, reflect_every: int = 3) -> None:
        self.reflect_every = max(1, int(reflect_every))
        self.cards: list[dict[str, Any]] = []
        self._history: list[PublicEpisodeRecord] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self.cards:
            return ""
        lines: list[str] = []
        for idx, c in enumerate(self.cards[:4], 1):
            u = int(c.get("uses", 0))
            m_val = c.get("mean_score")
            mean_txt = f"{float(m_val):.1f}" if m_val is not None else "null"
            w_ord = {s: round(float(c.get("recommended_weights", {}).get(s, 0.0)), 2) for s in STRATEGIES_V3}
            lines.append(
                f"- Card {c.get('card_id', f'C{idx}')} [{c.get('kind')} | Past sessions: n={u}, mean_portfolio_score={mean_txt}]: "
                f"IF {c.get('scope_condition')} -> recommended_weights = {json.dumps(w_ord)}"
            )
        return (
            "\nSCOPED HEURISTIC CARDBOOK (Baseline — structured scope cards):\n"
            + "\n".join(lines)
            + "\n"
        )

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self._history.append(record)
        sig = inspect_public_signals_v3(record)
        is_hv = sig["scope"] == "elevated_volatility"
        weights = sig["weights"]

        for c in self.cards:
            tgt = str(c.get("target_strategy") or "").upper()
            kind = str(c.get("kind", "calm_utility"))
            scope_matches = (kind == "invariant_or_hedge" and is_hv) or (kind == "calm_utility" and not is_hv)
            if scope_matches and weights.get(tgt, 0.0) >= 0.28 and not sig["off_scope_anomaly"]:
                old_u = int(c.get("uses", 0))
                old_m = c.get("mean_score")
                new_u = old_u + 1
                prev_sum = float(old_m) * old_u if (old_u > 0 and old_m is not None) else 0.0
                c["uses"] = new_u
                c["mean_score"] = round((prev_sum + float(record.public_score)) / new_u, 1)

        if len(self._history) % self.reflect_every != 0:
            return {"reflection_calls": 0, "reflection_usd": 0.0}

        recent_txt = "\n".join(summarize_public_record(r) for r in self._history[-self.reflect_every :])
        sys_p = (
            "You maintain a structured Scoped Heuristic Cardbook (max 3 cards) for AlphaDesk-v3 (6 strategies STRAT_S1..STRAT_S6).\n"
            "- `card_id`: 'C1'..'C3'\n"
            "- `kind`: `'invariant_or_hedge'` (for `elevated_volatility`) OR `'calm_utility'` (for `calm_volatility`)\n"
            "- `scope_condition`: short observable market condition ONLY\n"
            "- `target_strategy`: dominant strategy in `STRAT_S1..STRAT_S6`\n"
            "- `recommended_weights`: JSON dict mapping all 6 strategies to floats in [0, 0.55] summing to 1.0\n"
            "Respond ONLY with JSON: {\"cards\": [{...}, ...]}."
        )
        usr_p = (
            f"Current cards:\n{json.dumps(self.cards, indent=2)}\n\n"
            f"Recent {self.reflect_every} session traces:\n{recent_txt}\n\n"
            "Return updated `cards` JSON list."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method="scoped_cards_v3_curate",
            mock_responder=lambda _s, _u: {
                "cards": [
                    {
                        "card_id": "C1",
                        "kind": "calm_utility",
                        "scope_condition": "volatility_regime_hint == 'calm_volatility'",
                        "target_strategy": "STRAT_S1",
                        "recommended_weights": normalize_weights_v3({}, "STRAT_S1", "STRAT_S2"),
                    },
                    {
                        "card_id": "C2",
                        "kind": "invariant_or_hedge",
                        "scope_condition": "volatility_regime_hint == 'elevated_volatility'",
                        "target_strategy": "STRAT_S6",
                        "recommended_weights": normalize_weights_v3({}, "STRAT_S6", "STRAT_S5"),
                    },
                ]
            },
        )
        new_cards = (resp.parsed_json or {}).get("cards", [])
        if isinstance(new_cards, list) and new_cards:
            old_map = {(c.get("kind"), c.get("target_strategy")): c for c in self.cards}
            merged: list[dict[str, Any]] = []
            for idx, rc in enumerate(new_cards[:3], 1):
                if not isinstance(rc, dict):
                    continue
                kind = str(rc.get("kind", "calm_utility"))
                if kind not in ("invariant_or_hedge", "calm_utility"):
                    kind = "calm_utility"
                tgt = str(rc.get("target_strategy", "STRAT_S1")).strip().upper()
                if tgt not in STRATEGIES_V3:
                    tgt = "STRAT_S1"
                norm_w = normalize_weights_v3(rc.get("recommended_weights"), tgt)
                tgt = max(STRATEGIES_V3, key=lambda s: norm_w[s])
                prev = old_map.get((kind, tgt))
                merged.append(
                    {
                        "card_id": f"C{idx}",
                        "kind": kind,
                        "scope_condition": sanitize_scope_text(str(rc.get("scope_condition", kind))),
                        "target_strategy": tgt,
                        "recommended_weights": norm_w,
                        "uses": int(prev.get("uses", 1)) if prev else 1,
                        "mean_score": prev.get("mean_score", round(float(record.public_score), 1)) if prev else round(float(record.public_score), 1),
                    }
                )
            if merged:
                self.cards = merged
        return {"reflection_calls": 1, "reflection_usd": resp.usd}


@dataclass
class ScopeBucketV3:
    scope_name: str
    active_weights: dict[str, float] = field(default_factory=dict)
    primary_strategy: str | None = None
    secondary_strategy: str | None = None
    expected_mean: float = 82.0
    post_shift_n: int = 0
    cusum_stat: float = 0.0
    ewma_resid: float = 0.0
    probing_steps_remaining: int = 4
    quarantined_strategies: list[str] = field(default_factory=list)
    probe_estimates: dict[str, float] = field(default_factory=dict)
    last_tested_ep: dict[str, int] = field(default_factory=lambda: {s: -50 for s in STRATEGIES_V3})
    last_any_hazard_ep: int = 0
    shelf: list[dict[str, Any]] = field(default_factory=list)
    checked_shelf_ids: list[int] = field(default_factory=list)
    pending_runnerup_verify: dict[str, Any] | None = None
    alarms_fired: int = 0
    hazard_probes_triggered: int = 0
    shelf_probes_triggered: int = 0
    true_restorations: int = 0
    partial_recurrence_caught: int = 0
    recent_records: list[PublicEpisodeRecord] = field(default_factory=list)


class ScopedSGEHV3(EpistemicHarness):
    """`SGEH-v0.3 (Ours)` with clean flags for `enable_hazard` and `enable_shelf` on 6-strategy `AlphaDesk-v0.3`."""

    def __init__(
        self,
        *,
        enable_hazard: bool = True,
        enable_shelf: bool = True,
        drift_slack: float = 8.5,
        alarm_threshold: float = 18.0,
        ewma_alpha: float = 0.42,
        ewma_threshold: float = 12.5,
        hazard_interval: int = 7,
        hazard_desk_cadence: int = 4,
    ) -> None:
        self.enable_hazard = bool(enable_hazard)
        self.enable_shelf = bool(enable_shelf)
        if self.enable_hazard and self.enable_shelf:
            self.name = "sgeh_hazard_shelf"
        elif self.enable_hazard and not self.enable_shelf:
            self.name = "sgeh_hazard_noshelf"
        else:
            self.name = "sgeh_surprise_shelf"

        self.drift_slack = float(drift_slack)
        self.alarm_threshold = float(alarm_threshold)
        self.ewma_alpha = float(ewma_alpha)
        self.ewma_threshold = float(ewma_threshold)
        self.hazard_interval = max(5, int(hazard_interval))
        self.hazard_desk_cadence = max(4, int(hazard_desk_cadence))
        self.scope_ep_counts: dict[str, int] = {}
        self.buckets: dict[str, ScopeBucketV3] = {
            "calm_volatility": ScopeBucketV3(scope_name="calm_volatility", probing_steps_remaining=5),
            "elevated_volatility": ScopeBucketV3(scope_name="elevated_volatility", probing_steps_remaining=5),
        }

    def _get_bucket(self, scope: str) -> ScopeBucketV3:
        if scope not in self.buckets:
            self.buckets[scope] = ScopeBucketV3(scope_name=scope, probing_steps_remaining=4)
        return self.buckets[scope]

    def _candidate_shelf_entry(self, b: ScopeBucketV3) -> dict[str, Any] | None:
        if not self.enable_shelf or not b.shelf:
            return None
        # Prefer validated shelved regimes whose primary strategy is NOT currently quarantined and not yet checked this alarm
        for entry in reversed(b.shelf):
            rid = int(entry["regime_id"])
            p_s = str(entry["primary"])
            if rid not in b.checked_shelf_ids and p_s not in b.quarantined_strategies and p_s not in b.probe_estimates:
                return entry
        return None

    def _next_diagnostic_probe(self, b: ScopeBucketV3) -> str:
        shelf_cand = self._candidate_shelf_entry(b)
        if shelf_cand is not None:
            return str(shelf_cand["primary"])

        candidates = [
            s for s in STRATEGIES_V3
            if s not in b.quarantined_strategies and s not in b.probe_estimates
        ]
        if not candidates:
            candidates = [s for s in STRATEGIES_V3 if s not in b.quarantined_strategies] or list(STRATEGIES_V3)
        return min(candidates, key=lambda s: (b.last_tested_ep.get(s, -50), STRATEGIES_V3.index(s)))

    def _due_hazard_candidate(self, b: ScopeBucketV3) -> str | None:
        # Hazard check applies to calm regime exploration (elevated_volatility hedge is permanent invariant)
        if (
            not self.enable_hazard
            or b.scope_name != "calm_volatility"
            or b.probing_steps_remaining > 0
            or not b.active_weights
            or b.pending_runnerup_verify
        ):
            return None
        cur_n = self.scope_ep_counts.get(b.scope_name, 0)
        if cur_n - b.last_any_hazard_ep < self.hazard_desk_cadence:
            return None
        unheld = [s for s in STRATEGIES_V3 if float(b.active_weights.get(s, 0.0)) < 0.15]
        if not unheld:
            return None
        stalest = min(unheld, key=lambda s: (b.last_tested_ep.get(s, -50), STRATEGIES_V3.index(s)))
        staleness = cur_n - int(b.last_tested_ep.get(stalest, 0))
        if staleness >= self.hazard_interval:
            return stalest
        return None

    @staticmethod
    def _best_weights_from_estimates(b: ScopeBucketV3) -> dict[str, float] | None:
        if not b.probe_estimates:
            return None
        scored = {
            s: float(b.probe_estimates.get(s, 16.0 if s in b.quarantined_strategies else 48.0))
            for s in STRATEGIES_V3
        }
        ranked = sorted(STRATEGIES_V3, key=lambda s: (scored[s], -STRATEGIES_V3.index(s)), reverse=True)
        return {
            s: (0.55 if s == ranked[0] else (0.45 if s == ranked[1] else 0.0))
            for s in STRATEGIES_V3
        }

    def build_context(self, spec: PublicTaskSpec) -> str:
        lines: list[str] = [
            "SCOPE-CONDITIONED EPISTEMIC HARNESS v0.3 (match today's `inspect_desk_brief` observation to the active scope rule):",
            "- SYSTEMIC LIQUIDITY SAFEGUARD: If `liquidity_status != 'orderly_normal'` (e.g. `acute_liquidity_freeze`), "
            "do NOT probe (-10 pts) and do NOT churn `current_holdings` (hold current compliant weights to pay 0.0 turnover fee!).",
        ]
        for sc_name in ("calm_volatility", "elevated_volatility"):
            b = self._get_bucket(sc_name)
            if b.probing_steps_remaining > 0:
                nxt = self._next_diagnostic_probe(b)
                shelf_cand = self._candidate_shelf_entry(b)
                quar_txt = f" | Quarantined: {b.quarantined_strategies}" if b.quarantined_strategies else ""
                est_txt = f" | Probed yields: {json.dumps(b.probe_estimates)}" if b.probe_estimates else ""
                if shelf_cand is not None and nxt == shelf_cand["primary"]:
                    s2 = shelf_cand["secondary"]
                    mu1 = shelf_cand["strategy_means"].get(nxt, 90.0)
                    w_restore = json.dumps(shelf_cand["weights"])
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[SHELF-RECALL PROBE ({b.probing_steps_remaining} left){quar_txt}]: "
                        f"Call `probe_strategy_execution` on shelved primary `{nxt}` (historical yield ~{mu1:.0f}). "
                        f"If `{nxt}` live yield >= 78.0, immediately restore shelved pair {w_restore} (`0.55 {nxt} + 0.45 {s2}`); "
                        f"otherwise avoid `{nxt}` and diversify `0.55 / 0.45` across non-quarantined strategies."
                    )
                else:
                    hint_w = self._best_weights_from_estimates(b)
                    alloc_hint = (
                        f" After probing `{nxt}`, allocate 0.55 to the top strategy and 0.45 to the #2 strategy (prior: {json.dumps(hint_w)})."
                        if hint_w
                        else f" After probing `{nxt}`, allocate 0.55 to `{nxt}` if its yield >= 72, else place 0.55 / 0.45 on untested non-quarantined strategies."
                    )
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[DIAGNOSTIC PROBE MODE ({b.probing_steps_remaining} left){quar_txt}{est_txt}]: "
                        f"Call `probe_strategy_execution` on `{nxt}`.{alloc_hint}"
                    )
            else:
                haz = self._due_hazard_candidate(b)
                w_json = json.dumps({s: round(float(b.active_weights.get(s, 1.0 / 6.0)), 2) for s in STRATEGIES_V3})
                if haz is not None:
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[STALENESS HAZARD CHECK DUE | Active expected={b.expected_mean:.1f}, weights={w_json}]: "
                        f"Call `probe_strategy_execution` on long-untested `{haz}` once today. If `{haz}` live yield > {b.expected_mean + 8.0:.1f}, "
                        f"shift 0.55 to `{haz}` and 0.45 to `{b.primary_strategy}`; otherwise keep {w_json} (0 turnover!)."
                    )
                else:
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[EXPLOITATION | n={b.post_shift_n}, expected_gross={b.expected_mean:.1f}]: "
                        f"Submit `recommended_weights = {w_json}` directly without probing."
                    )
        return "\n" + "\n".join(lines) + "\n"

    def _consolidate_bucket(
        self, b: ScopeBucketV3, env_name: str, client: MeteredLLMClient
    ) -> tuple[int, float]:
        traces_txt = "\n".join(summarize_public_record(r) for r in b.recent_records[-4:])
        sys_p = (
            f"You consolidate the active 6-strategy portfolio hypothesis for scope `{b.scope_name}` on AlphaDesk-v3.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            "  \"primary_strategy\": \"STRAT_S1\",\n"
            "  \"secondary_strategy\": \"STRAT_S2\",\n"
            "  \"recommended_weights\": {\"STRAT_S1\": 0.55, \"STRAT_S2\": 0.45, \"STRAT_S3\": 0.0, \"STRAT_S4\": 0.0, \"STRAT_S5\": 0.0, \"STRAT_S6\": 0.0},\n"
            "  \"predicted_gross_score\": 86.5\n"
            "}"
        )
        usr_p = (
            f"Scope: {b.scope_name}\n"
            f"Quarantined (falsified) strategies: {json.dumps(b.quarantined_strategies)}\n"
            f"Verified single-strategy probe estimates: {json.dumps(b.probe_estimates)}\n"
            f"Recent scope traces:\n{traces_txt}\n\n"
            "Return compliant `recommended_weights` (`sum=1.0`, each `<= 0.55`, best pair `0.55 + 0.45`)."
        )

        def _mock_consolidator(_s: str, _u: str) -> dict[str, Any]:
            best_w = self._best_weights_from_estimates(b) or normalize_weights_v3({}, "STRAT_S1", "STRAT_S2")
            ranked = sorted(STRATEGIES_V3, key=lambda x: best_w[x], reverse=True)
            est_y = sum(best_w[x] * float(b.probe_estimates.get(x, 76.0)) for x in STRATEGIES_V3)
            return {
                "primary_strategy": ranked[0],
                "secondary_strategy": ranked[1],
                "recommended_weights": best_w,
                "predicted_gross_score": round(est_y, 1),
            }

        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=env_name,
            method=f"{self.name}_consolidate",
            mock_responder=_mock_consolidator,
        )
        parsed = resp.parsed_json or {}
        p1 = str(parsed.get("primary_strategy", "STRAT_S1")).strip().upper()
        p2 = str(parsed.get("secondary_strategy", "STRAT_S2")).strip().upper()
        norm_w = normalize_weights_v3(parsed.get("recommended_weights"), p1, p2)

        if len(b.probe_estimates) >= 2:
            emp_w = self._best_weights_from_estimates(b)
            if emp_w is not None:
                emp_exp = sum(emp_w[s] * float(b.probe_estimates.get(s, 45.0)) for s in STRATEGIES_V3)
                llm_exp = sum(norm_w[s] * float(b.probe_estimates.get(s, 45.0)) for s in STRATEGIES_V3)
                if emp_exp > llm_exp + 6.0:
                    norm_w = emp_w

        ranked_final = sorted(STRATEGIES_V3, key=lambda s: norm_w[s], reverse=True)
        b.active_weights = norm_w
        b.primary_strategy = ranked_final[0]
        b.secondary_strategy = ranked_final[1]

        if b.probe_estimates:
            implied_gross = sum(
                norm_w[s] * float(b.probe_estimates.get(s, 20.0 if s in b.quarantined_strategies else 62.0))
                for s in STRATEGIES_V3
            )
        else:
            implied_gross = 78.0
        try:
            pred = float(parsed.get("predicted_gross_score", implied_gross))
        except (TypeError, ValueError):
            pred = implied_gross
        b.expected_mean = round(min(95.0, max(52.0, 0.5 * pred + 0.5 * implied_gross)), 1)
        b.post_shift_n = max(1, len(b.recent_records))
        b.cusum_stat = 0.0
        b.ewma_resid = 0.0
        return 1, resp.usd

    def _archive_current_to_shelf(self, b: ScopeBucketV3) -> None:
        if not b.active_weights or not b.primary_strategy:
            return
        p1 = b.primary_strategy
        p2 = b.secondary_strategy or next(s for s in STRATEGIES_V3 if s != p1)
        s_means: dict[str, float] = {}
        for s in (p1, p2):
            if s in b.probe_estimates and float(b.probe_estimates[s]) >= 55.0:
                s_means[s] = round(float(b.probe_estimates[s]), 1)
            else:
                s_means[s] = round(b.expected_mean + (4.0 if s == p1 else -4.0), 1)

        # Deduplicate existing shelf entry with identical (p1, p2) pair by updating it
        for existing in b.shelf:
            if existing.get("primary") == p1 and existing.get("secondary") == p2:
                existing["strategy_means"] = s_means
                existing["expected_mean"] = b.expected_mean
                existing["validated_n"] = max(int(existing.get("validated_n", 1)), b.post_shift_n)
                return

        b.shelf.append(
            {
                "regime_id": len(b.shelf) + 1,
                "primary": p1,
                "secondary": p2,
                "weights": copy.deepcopy(b.active_weights),
                "strategy_means": s_means,
                "expected_mean": b.expected_mean,
                "validated_n": b.post_shift_n,
            }
        )

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        sig = inspect_public_signals_v3(record)
        if sig["off_scope_anomaly"]:
            return {
                "reflection_calls": 0,
                "reflection_usd": 0.0,
                "alarm_fired": False,
                "off_scope_ignored": True,
            }

        sc_name = sig["scope"] if sig["scope"] in ("calm_volatility", "elevated_volatility") else "calm_volatility"
        self.scope_ep_counts[sc_name] = self.scope_ep_counts.get(sc_name, 0) + 1
        cur_sc_ep = self.scope_ep_counts[sc_name]

        b = self._get_bucket(sc_name)
        b.recent_records.append(record)
        b.recent_records = b.recent_records[-6:]

        gross_y = float(sig["gross_portfolio_score"])

        # Record any fresh probe estimates and reset staleness (`staleness = 0`)
        shelf_cand = self._candidate_shelf_entry(b) if b.probing_steps_remaining > 0 else None
        for s, est in sig["probed"].items():
            b.probe_estimates[s] = est
            b.last_tested_ep[s] = cur_sc_ep
            if est >= 65.0 and s in b.quarantined_strategies:
                b.quarantined_strategies.remove(s)

        # Case 0: Verify runner-up leg on the calm step immediately following a tentative 1-probe Shelf-Match!
        if b.pending_runnerup_verify is not None and b.probing_steps_remaining == 0 and not sig["probed"]:
            ver = b.pending_runnerup_verify
            s1 = str(ver["primary"])
            s2 = str(ver["secondary"])
            mu1 = float(ver["mu1"])
            mu2_expected = float(ver["mu2"])
            if sig["weights"].get(s1, 0.0) >= 0.45 and sig["weights"].get(s2, 0.0) >= 0.35:
                implied_s2 = (gross_y - 0.55 * mu1) / 0.45
                b.pending_runnerup_verify = None
                if implied_s2 < mu2_expected - 22.0 or gross_y < b.expected_mean - 14.0:
                    # PARTIAL RECURRENCE (`A'`) CAUGHT! Keep `s1` verified, quarantine ONLY broken `s2`!
                    b.partial_recurrence_caught += 1
                    if s2 not in b.quarantined_strategies:
                        b.quarantined_strategies.append(s2)
                    b.probe_estimates[s2] = round(max(0.0, implied_s2), 1)
                    b.last_tested_ep[s2] = cur_sc_ep
                    b.probing_steps_remaining = 2  # Probe 2 candidates to find the new 0.45 co-winner!
                    return {
                        "reflection_calls": 0,
                        "reflection_usd": 0.0,
                        "alarm_fired": True,
                        "partial_recurrence_caught": s2,
                    }
                # FULL RECURRENCE (`A`) CONFIRMED with just 1 probe!
                b.true_restorations += 1
                b.probe_estimates[s2] = round(min(100.0, max(50.0, implied_s2)), 1)
                b.last_tested_ep[s2] = cur_sc_ep
                b.post_shift_n += 1
                return {
                    "reflection_calls": 0,
                    "reflection_usd": 0.0,
                    "alarm_fired": False,
                    "true_restoration_verified": True,
                }

        # Case 1: Acquisition or post-alarm diagnostic probing mode
        if b.probing_steps_remaining > 0:
            # Paid-probe cross-sectional check: if a paid probe after an alarm observes `desk_best_yield_estimate <= 32`,
            # no strategy rotated into leadership (broad calm factor dip) -> restore pre-alarm shelved pair with 0 churn!
            desk_bests = [
                float(ds["desk_best_yield_estimate"])
                for ds in sig.get("probe_desk_stats", {}).values()
                if "desk_best_yield_estimate" in ds
            ]
            if desk_bests and max(desk_bests) <= 32.0 and b.shelf:
                prior = b.shelf[-1]
                b.active_weights = copy.deepcopy(prior["weights"])
                b.primary_strategy = str(prior["primary"])
                b.secondary_strategy = str(prior["secondary"])
                b.expected_mean = float(prior.get("expected_mean", 84.0))
                b.quarantined_strategies = []
                b.probing_steps_remaining = 0
                b.cusum_stat = 0.0
                b.ewma_resid = 0.0
                b.post_shift_n = max(2, int(prior.get("validated_n", 2)))
                return {
                    "reflection_calls": 0,
                    "reflection_usd": 0.0,
                    "alarm_fired": False,
                    "paid_probe_confirmed_broad_dip": True,
                }

            # Check if a Shelf-Match probe just succeeded!
            if shelf_cand is not None and shelf_cand["primary"] in sig["probed"]:
                b.shelf_probes_triggered += 1
                rid = int(shelf_cand["regime_id"])
                b.checked_shelf_ids.append(rid)
                p1 = str(shelf_cand["primary"])
                p2 = str(shelf_cand["secondary"])
                y_p1 = float(sig["probed"][p1])
                mu1_shelf = float(shelf_cand["strategy_means"].get(p1, 90.0))
                mu2_shelf = float(shelf_cand["strategy_means"].get(p2, 80.0))
                if abs(y_p1 - mu1_shelf) <= 13.0 and y_p1 >= 78.0 and p2 not in b.quarantined_strategies:
                    w1_sub = float(sig["weights"].get(p1, 0.0))
                    w2_sub = float(sig["weights"].get(p2, 0.0))
                    if w1_sub >= 0.45 and w2_sub >= 0.35 and (w1_sub + w2_sub) >= 0.90:
                        implied_s2_now = (gross_y - w1_sub * y_p1) / max(0.20, w2_sub)
                        if implied_s2_now < mu2_shelf - 22.0:
                            # Immediate same-episode Partial Recurrence (`A'`) catch!
                            b.partial_recurrence_caught += 1
                            b.quarantined_strategies.append(p2)
                            b.probe_estimates[p2] = round(max(0.0, implied_s2_now), 1)
                            b.last_tested_ep[p2] = cur_sc_ep
                            b.probing_steps_remaining = 2
                            return {
                                "reflection_calls": 0,
                                "reflection_usd": 0.0,
                                "alarm_fired": True,
                                "partial_recurrence_caught": p2,
                            }
                        # Immediate same-episode Full Recurrence (`A`) verification!
                        b.true_restorations += 1
                        b.active_weights = copy.deepcopy(shelf_cand["weights"])
                        b.primary_strategy = p1
                        b.secondary_strategy = p2
                        b.probe_estimates[p2] = round(min(100.0, max(55.0, implied_s2_now)), 1)
                        b.last_tested_ep[p2] = cur_sc_ep
                        b.expected_mean = round(0.55 * y_p1 + 0.45 * b.probe_estimates[p2], 1)
                        b.post_shift_n = 2
                        b.cusum_stat = 0.0
                        b.ewma_resid = 0.0
                        b.probing_steps_remaining = 0
                        b.pending_runnerup_verify = None
                        return {
                            "reflection_calls": 0,
                            "reflection_usd": 0.0,
                            "alarm_fired": False,
                            "true_restoration_verified": True,
                        }

                    # Otherwise tentatively restore shelved `0.55 p1 + 0.45 p2` and verify `p2` on next calm step!
                    b.active_weights = copy.deepcopy(shelf_cand["weights"])
                    b.primary_strategy = p1
                    b.secondary_strategy = p2
                    b.expected_mean = round(0.55 * y_p1 + 0.45 * mu2_shelf, 1)
                    b.post_shift_n = 2
                    b.cusum_stat = 0.0
                    b.ewma_resid = 0.0
                    b.probing_steps_remaining = 0
                    b.pending_runnerup_verify = {
                        "primary": p1,
                        "secondary": p2,
                        "mu1": y_p1,
                        "mu2": mu2_shelf,
                    }
                    return {
                        "reflection_calls": 0,
                        "reflection_usd": 0.0,
                        "alarm_fired": False,
                        "shelf_tentative_restore": p1,
                    }

            b.probing_steps_remaining -= 1
            good_ests = [v for k, v in b.probe_estimates.items() if k not in b.quarantined_strategies and v >= 55.0]
            strong_pair_ready = len(good_ests) >= 2
            if b.probing_steps_remaining == 0 or strong_pair_ready:
                b.probing_steps_remaining = 0
                b.last_any_hazard_ep = cur_sc_ep
                ref_c, ref_u = self._consolidate_bucket(b, record.env_name, client)
                return {
                    "reflection_calls": ref_c,
                    "reflection_usd": ref_u,
                    "alarm_fired": False,
                    "consolidated_scope": sc_name,
                }
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}

        # Refresh `last_tested_ep` for currently held active strategies so hazard checks only target unheld arms
        for s, w in sig["weights"].items():
            if w >= 0.30:
                b.last_tested_ep[s] = cur_sc_ep

        # Case 2: Staleness Hazard Check on an unheld 0-weight arm
        if sig["probed"]:
            b.hazard_probes_triggered += 1
            b.last_any_hazard_ep = cur_sc_ep
            best_s, best_v = max(sig["probed"].items(), key=lambda kv: kv[1])
            cand_w = self._best_weights_from_estimates(b)
            port_gain = 0.0
            if cand_w is not None and b.active_weights:
                cand_exp = sum(cand_w[s] * float(b.probe_estimates.get(s, 50.0)) for s in STRATEGIES_V3)
                cur_exp = sum(b.active_weights.get(s, 0.0) * float(b.probe_estimates.get(s, b.expected_mean)) for s in STRATEGIES_V3)
                port_gain = cand_exp - cur_exp
            if (best_v > b.expected_mean + 8.0 and b.active_weights.get(best_s, 0.0) < 0.25) or port_gain > 7.5:
                self._archive_current_to_shelf(b)
                # Enter a 1-step follow-up probe to check if a second co-winner also rose alongside `best_s`!
                if b.primary_strategy and b.primary_strategy not in b.probe_estimates:
                    b.probe_estimates[b.primary_strategy] = b.expected_mean
                ref_c, ref_u = self._consolidate_bucket(b, record.env_name, client)
                return {
                    "reflection_calls": ref_c,
                    "reflection_usd": ref_u,
                    "alarm_fired": True,
                    "hazard_upgrade": best_s,
                }
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False, "hazard_checked": True}

        # Case 3: Exploitation — evaluate scope-conditioned Level-CUSUM + EWMA-Slope residual
        if b.primary_strategy and sig["weights"].get(b.primary_strategy, 0.0) < 0.30:
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}

        resid = b.expected_mean - gross_y
        shortfall = resid - self.drift_slack
        b.cusum_stat = max(0.0, b.cusum_stat + shortfall)
        b.ewma_resid = (1.0 - self.ewma_alpha) * b.ewma_resid + self.ewma_alpha * resid

        slope_alarm = (b.post_shift_n >= 2) and (b.ewma_resid >= self.ewma_threshold)
        if b.cusum_stat >= self.alarm_threshold or slope_alarm:
            self._archive_current_to_shelf(b)
            falsified = [s for s, w in b.active_weights.items() if w >= 0.30]
            b.quarantined_strategies = list(falsified)
            # Reset `last_tested_ep` on falsified arms (`staleness = 0`) without a window-blocking 16-step cooldown!
            for f_s in falsified:
                b.last_tested_ep[f_s] = cur_sc_ep
            b.probe_estimates = {s: round(gross_y, 1) for s in falsified}
            b.active_weights = {}
            b.primary_strategy = None
            b.secondary_strategy = None
            b.checked_shelf_ids = []
            b.pending_runnerup_verify = None
            b.post_shift_n = 0
            b.cusum_stat = 0.0
            b.ewma_resid = 0.0
            b.probing_steps_remaining = 4
            b.alarms_fired += 1
            b.recent_records = [record]
            return {
                "reflection_calls": 0,
                "reflection_usd": 0.0,
                "alarm_fired": True,
                "alarm_scope": sc_name,
            }

        b.post_shift_n += 1
        lr = 1.0 / max(2, min(8, b.post_shift_n))
        b.expected_mean = round((1.0 - lr) * b.expected_mean + lr * gross_y, 2)
        return {
            "reflection_calls": 0,
            "reflection_usd": 0.0,
            "alarm_fired": False,
            "cusum": round(b.cusum_stat, 2),
            "ewma_resid": round(b.ewma_resid, 2),
        }


def build_factual_ledger_text_v3(history: list[PublicEpisodeRecord], recent_k: int = 8) -> str:
    """Render a strictly objective, zero-prescription factual ledger from public `AlphaDesk-v0.3` observations.

    Contains ZERO action recommendations, ZERO yield thresholds, ZERO CUSUM alarms, ZERO quarantine labels,
    and ZERO liquidity-freeze instructions—only deterministic empirical facts observed so far.
    """
    if not history:
        return ""

    recent_lines: list[str] = []
    # Per-scope factual tracking under orderly_normal vs non-normal liquidity
    scope_counts: dict[str, int] = {"calm_volatility": 0, "elevated_volatility": 0}
    last_probe_val: dict[str, dict[str, float | None]] = {
        sc: {s: None for s in STRATEGIES_V3} for sc in ("calm_volatility", "elevated_volatility")
    }
    last_probe_sc_ep: dict[str, dict[str, int | None]] = {
        sc: {s: None for s in STRATEGIES_V3} for sc in ("calm_volatility", "elevated_volatility")
    }
    held_gross_hist: dict[str, dict[str, list[float]]] = {
        sc: {s: [] for s in STRATEGIES_V3} for sc in ("calm_volatility", "elevated_volatility")
    }

    parsed_rows: list[str] = []
    for rec in history:
        sig = inspect_public_signals_v3(rec)
        sc = sig["scope"] if sig["scope"] in ("calm_volatility", "elevated_volatility") else "calm_volatility"
        liq_status = "orderly_normal"
        spread_bps = None
        for st in rec.steps:
            obs = st.get("obs", {})
            if isinstance(obs, dict) and "liquidity_status" in obs:
                liq_status = str(obs.get("liquidity_status", "orderly_normal"))
                spread_bps = obs.get("bid_ask_spread_bps")

        if liq_status == "orderly_normal":
            scope_counts[sc] = scope_counts.get(sc, 0) + 1
            cur_idx = scope_counts[sc]
            for s, y_p in sig["probed"].items():
                last_probe_val[sc][s] = round(float(y_p), 1)
                last_probe_sc_ep[sc][s] = cur_idx
            for s, w_val in sig["weights"].items():
                if float(w_val) >= 0.25:
                    held_gross_hist[sc][s].append(round(float(sig["gross_portfolio_score"]), 1))

        desk_st = sig.get("probe_desk_stats", {})
        if sig["probed"]:
            prb_parts = []
            for k, v in sig["probed"].items():
                ds = desk_st.get(k, {})
                if ds:
                    dm = ds.get("desk_mean_yield_estimate")
                    db = ds.get("desk_best_yield_estimate")
                    extra = f"[desk_mean={dm:.1f},desk_max={db:.1f}]" if (dm is not None and db is not None) else ""
                    prb_parts.append(f"{k}:{v:.1f}{extra}")
                else:
                    prb_parts.append(f"{k}:{v:.1f}")
            prb_txt = ", ".join(prb_parts)
        else:
            prb_txt = "none"
        nonzero_w = {k: round(float(v), 2) for k, v in sig["weights"].items() if float(v) >= 0.05}
        sp_txt = f" ({spread_bps} bps)" if spread_bps is not None else ""
        parsed_rows.append(
            f"- Session #{rec.timestamp} | vol={sc} | liquidity={liq_status}{sp_txt} | "
            f"probed={prb_txt} | weights={json.dumps(nonzero_w)} | "
            f"gross={sig['gross_portfolio_score']:.1f} | net={rec.public_score:.1f}"
        )

    recent_lines.extend(parsed_rows[-recent_k:])

    lines = [
        "FACTUAL OBSERVATION LEDGER (objective record of past public sessions & per-strategy observations):",
        f"Recent {min(len(parsed_rows), recent_k)} Sessions (chronological):",
        *recent_lines,
    ]

    for sc in ("calm_volatility", "elevated_volatility"):
        tot_sc = scope_counts.get(sc, 0)
        lines.append(
            f"Observed Per-Strategy Record in `{sc}` + `orderly_normal` ({tot_sc} sessions observed):"
        )
        s_items: list[str] = []
        for s in STRATEGIES_V3:
            lp = last_probe_val[sc][s]
            lp_ep = last_probe_sc_ep[sc][s]
            if lp is not None and lp_ep is not None:
                ago = tot_sc - lp_ep
                p_part = f"last_probe={lp:.1f} ({ago} {sc.split('_')[0]} sessions ago)"
            else:
                p_part = f"last_probe=unprobed ({tot_sc} {sc.split('_')[0]} sessions elapsed)"
            gh = held_gross_hist[sc][s]
            if gh:
                mean_g = sum(gh) / len(gh)
                h_part = f"held_n={len(gh)}, last_port_gross={gh[-1]:.1f}, mean_port_gross={mean_g:.1f}"
            else:
                h_part = "held_n=0"
            s_items.append(f"  * {s}: {p_part} | {h_part}")
        lines.extend(s_items)

    return "\n" + "\n".join(lines) + "\n"


class FactualLedgerV3Harness(EpistemicHarness):
    """Evidence-Only Autonomy Arm (`factual_ledger_v3`): Pure deterministic factual history, 0 action prescriptions."""

    name = "factual_ledger_v3"

    def __init__(self, recent_k: int = 8) -> None:
        self.recent_k = int(recent_k)
        self.history: list[PublicEpisodeRecord] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        return build_factual_ledger_text_v3(self.history, recent_k=self.recent_k)

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self.history.append(record)
        return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}


class DirectiveSGEHV3Harness(EpistemicHarness):
    """Strict-Superset Controller Arm (`sgeh_directive_v3`): Identical `FactualLedgerV3` + `SGEH` action prescriptions."""

    name = "sgeh_directive_v3"

    def __init__(self, recent_k: int = 8, enable_hazard: bool = True) -> None:
        self.factual = FactualLedgerV3Harness(recent_k=recent_k)
        self.controller = ScopedSGEHV3(enable_hazard=enable_hazard, enable_shelf=False)

    def build_context(self, spec: PublicTaskSpec) -> str:
        fact_block = self.factual.build_context(spec)
        ctrl_block = self.controller.build_context(spec)
        return fact_block + ctrl_block

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self.factual.on_episode_complete(record, client)
        return self.controller.on_episode_complete(record, client)
