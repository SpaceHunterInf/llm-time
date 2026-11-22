"""Iteration 2 Epistemic Harnesses: `PeriodicProbeHarness`, `SGEH-SurpriseOnly`, and `SGEH+Hazard`."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.alpha_desk_v2 import STRATEGIES_V2
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record
from src.harnesses.scoped_cards import (
    ScopedCardBookHarness,
    extract_submitted_weights,
    normalize_card_weights,
)


def logistic_prob_ge(expected_val: float, threshold: float = 75.0, scale: float = 8.0) -> float:
    z = max(-12.0, min(12.0, (expected_val - threshold) / max(1.0, scale)))
    return round(1.0 / (1.0 + math.exp(-z)), 4)


def inspect_public_episode_signals(record: PublicEpisodeRecord) -> dict[str, Any]:
    """Extract strictly learner-public scope, anomaly flags, probe results, and weights from `record`."""
    scope = "default"
    off_scope_anomaly = False
    probed: dict[str, float] = {}

    for st in record.steps:
        obs = st.get("obs", {})
        if not isinstance(obs, dict):
            continue
        if "volatility_regime_hint" in obs:
            scope = str(obs["volatility_regime_hint"])
        elif "category" in obs and scope == "default":
            scope = str(obs["category"])
        liq = str(obs.get("liquidity_status", "orderly_normal"))
        if liq not in ("orderly_normal", "", "normal"):
            off_scope_anomaly = True
        if st.get("tool") == "probe_strategy_execution" and "live_net_yield_estimate" in obs:
            strat = str(obs.get("strategy") or st.get("args", {}).get("strategy", "")).upper()
            if strat in STRATEGIES_V2:
                try:
                    probed[strat] = float(obs["live_net_yield_estimate"])
                except (TypeError, ValueError):
                    pass

    weights = extract_submitted_weights(record)
    return {
        "scope": scope,
        "off_scope_anomaly": off_scope_anomaly,
        "probed": probed,
        "weights": weights,
    }


class PeriodicProbeHarness(EpistemicHarness):
    """Scheduled exploration baseline: `ScopedCardBookHarness` + mandatory untested probe every `probe_every` calm episodes."""

    name = "periodic_probe"

    def __init__(self, *, reflect_every: int = 3, probe_every: int = 4) -> None:
        self.probe_every = max(2, int(probe_every))
        self.inner = ScopedCardBookHarness(reflect_every=reflect_every)
        self.calm_count: int = 0
        self.last_probed_ep: dict[str, int] = {s: -100 for s in STRATEGIES_V2}
        self.latest_probe_est: dict[str, float] = {}
        self.ep_counter: int = 0

    def clone(self) -> "PeriodicProbeHarness":
        dup = PeriodicProbeHarness(reflect_every=self.inner.reflect_every, probe_every=self.probe_every)
        dup.inner = self.inner.clone()
        dup.calm_count = self.calm_count
        dup.last_probed_ep = dict(self.last_probed_ep)
        dup.latest_probe_est = dict(self.latest_probe_est)
        dup.ep_counter = self.ep_counter
        return dup

    def set_post_fork_schedule(self) -> None:
        self.inner.set_post_fork_schedule()

    def _select_stalest_strategy(self) -> str:
        return min(STRATEGIES_V2, key=lambda s: (self.last_probed_ep.get(s, -100), STRATEGIES_V2.index(s)))

    def build_context(self, spec: PublicTaskSpec) -> str:
        base_ctx = self.inner.build_context(spec)
        due_on_next_calm = ((self.calm_count + 1) % self.probe_every == 0)
        if due_on_next_calm:
            target_s = self._select_stalest_strategy()
            sched_note = (
                f"\nSCHEDULED PERIODIC PROBE: If today's session has `volatility_regime_hint == 'calm_volatility'` "
                f"and `liquidity_status == 'orderly_normal'`, call `probe_strategy_execution` on `{target_s}` "
                "before submitting compliant weights.\n"
            )
            return base_ctx + sched_note
        return base_ctx

    def on_episode_complete(
        self, record: PublicEpisodeRecord, client: MeteredLLMClient, *, force_curate: bool = False
    ) -> dict[str, Any]:
        self.ep_counter += 1
        sig = inspect_public_episode_signals(record)
        if sig["scope"] == "calm_volatility" and not sig["off_scope_anomaly"]:
            self.calm_count += 1
        for s, est in sig["probed"].items():
            self.last_probed_ep[s] = self.ep_counter
            self.latest_probe_est[s] = est
        return self.inner.on_episode_complete(record, client, force_curate=force_curate)

    def predict_calm_portfolio_prob_ge_75(self, w_ref: dict[str, float]) -> float:
        return self.inner.predict_calm_portfolio_prob_ge_75(w_ref)


@dataclass
class ScopeEpistemicBucket:
    scope_name: str
    active_weights: dict[str, float] = field(default_factory=dict)
    primary_strategy: str | None = None
    expected_mean: float = 80.0
    post_shift_n: int = 0
    cusum_stat: float = 0.0
    ewma_resid: float = 0.0
    probing_steps_remaining: int = 3
    quarantined_strategies: list[str] = field(default_factory=list)
    probe_estimates: dict[str, float] = field(default_factory=dict)
    last_tested_ep: dict[str, int] = field(default_factory=lambda: {s: -50 for s in STRATEGIES_V2})
    alarms_fired: int = 0
    hazard_probes_triggered: int = 0
    shelf: list[dict[str, Any]] = field(default_factory=list)
    recent_records: list[PublicEpisodeRecord] = field(default_factory=list)


class ScopedSGEHHardened(EpistemicHarness):
    """Scope-Conditioned Surprise-Gated Epistemic Harness with optional Staleness Hazard Probing (`Iteration 2`).

    Properties (strictly zero-leakage; uses only learner-public observations):
      1. Scope & Off-Scope Partitioning: Separates `calm_volatility` and `elevated_volatility` buckets; freezes
         CUSUM/counters during observable systemic anomalies (`liquidity_status != 'orderly_normal'`).
      2. Dual Level-CUSUM + EWMA-Slope Gate: 0 LLM reflection calls during in-control exploitation; catches both
         abrupt reversals (`1` step) and smooth `gradual_decay` slopes (`~3` steps).
      3. Non-Redundant Untested Probe Queue: On surprise alarm, quarantines the falsified incumbent strategy and
         queues untested strategies without replacement (eliminating redundant incumbent re-probing).
      4. Staleness Hazard Probing (`enable_hazard=True`): Schedules a single low-rate check on unheld strategies
         whose staleness exceeds `hazard_interval`, solving Type-II `opportunity_rise` at minimal stationary cost.
    """

    def __init__(
        self,
        *,
        enable_hazard: bool = True,
        drift_slack: float = 9.0,
        alarm_threshold: float = 20.0,
        ewma_alpha: float = 0.40,
        ewma_threshold: float = 13.5,
        hazard_interval: int = 8,
    ) -> None:
        self.enable_hazard = bool(enable_hazard)
        self.name = "sgeh_hazard" if self.enable_hazard else "sgeh_surprise"
        self.drift_slack = float(drift_slack)
        self.alarm_threshold = float(alarm_threshold)
        self.ewma_alpha = float(ewma_alpha)
        self.ewma_threshold = float(ewma_threshold)
        self.hazard_interval = max(4, int(hazard_interval))
        self.scope_ep_counts: dict[str, int] = {}
        self.buckets: dict[str, ScopeEpistemicBucket] = {
            "calm_volatility": ScopeEpistemicBucket(scope_name="calm_volatility", probing_steps_remaining=4),
            "elevated_volatility": ScopeEpistemicBucket(scope_name="elevated_volatility", probing_steps_remaining=3),
        }

    def clone(self) -> "ScopedSGEHHardened":
        dup = ScopedSGEHHardened(
            enable_hazard=self.enable_hazard,
            drift_slack=self.drift_slack,
            alarm_threshold=self.alarm_threshold,
            ewma_alpha=self.ewma_alpha,
            ewma_threshold=self.ewma_threshold,
            hazard_interval=self.hazard_interval,
        )
        dup.scope_ep_counts = copy.deepcopy(self.scope_ep_counts)
        dup.buckets = copy.deepcopy(self.buckets)
        return dup

    def set_post_fork_schedule(self) -> None:
        """Event-driven harness has no periodic modulo-3 clock, kept for interface symmetry."""
        return None

    def _get_bucket(self, scope: str) -> ScopeEpistemicBucket:
        if scope not in self.buckets:
            self.buckets[scope] = ScopeEpistemicBucket(scope_name=scope, probing_steps_remaining=3)
        return self.buckets[scope]

    def _next_diagnostic_probe(self, b: ScopeEpistemicBucket) -> str:
        candidates = [
            s for s in STRATEGIES_V2
            if s not in b.quarantined_strategies and s not in b.probe_estimates
        ]
        if not candidates:
            candidates = [s for s in STRATEGIES_V2 if s not in b.quarantined_strategies] or list(STRATEGIES_V2)
        return min(candidates, key=lambda s: (b.last_tested_ep.get(s, -50), STRATEGIES_V2.index(s)))

    def _due_hazard_candidate(self, b: ScopeEpistemicBucket) -> str | None:
        if not self.enable_hazard or b.probing_steps_remaining > 0 or not b.active_weights:
            return None
        cur_n = self.scope_ep_counts.get(b.scope_name, 0)
        unheld = [s for s in STRATEGIES_V2 if float(b.active_weights.get(s, 0.0)) < 0.15]
        if not unheld:
            return None
        stalest = min(unheld, key=lambda s: (b.last_tested_ep.get(s, -50), STRATEGIES_V2.index(s)))
        staleness = cur_n - int(b.last_tested_ep.get(stalest, 0))
        if staleness >= self.hazard_interval:
            return stalest
        return None

    @staticmethod
    def _best_known_weights_from_estimates(b: ScopeEpistemicBucket) -> dict[str, float] | None:
        if not b.probe_estimates:
            return None
        scored = {
            s: float(b.probe_estimates.get(s, 15.0 if s in b.quarantined_strategies else 50.0))
            for s in STRATEGIES_V2
        }
        ranked = sorted(STRATEGIES_V2, key=lambda s: scored[s], reverse=True)
        return {
            s: (0.65 if s == ranked[0] else (0.35 if s == ranked[1] else 0.0))
            for s in STRATEGIES_V2
        }

    def build_context(self, spec: PublicTaskSpec) -> str:
        lines: list[str] = [
            "SCOPE-CONDITIONED EPISTEMIC HARNESS (match today's `inspect_desk_brief` observation to the active scope rule):",
            "- SYSTEMIC ANOMALY SAFEGUARD: If `liquidity_status != 'orderly_normal'` (e.g. `acute_liquidity_freeze`), "
            "do NOT pay the -12 pt probe fee; submit compliant weights directly without altering calm-market beliefs.",
        ]
        for sc_name in ("calm_volatility", "elevated_volatility"):
            b = self._get_bucket(sc_name)
            if b.probing_steps_remaining > 0:
                nxt = self._next_diagnostic_probe(b)
                quar_txt = f" | Quarantined falsified: {b.quarantined_strategies}" if b.quarantined_strategies else ""
                est_txt = f" | Verified yields so far: {json.dumps(b.probe_estimates)}" if b.probe_estimates else ""
                hint_w = self._best_known_weights_from_estimates(b)
                alloc_hint = (
                    f" After probing `{nxt}`, allocate 0.65 to the highest-yielding strategy and 0.35 to the runner-up (current prior: {json.dumps(hint_w)})."
                    if hint_w
                    else f" After probing `{nxt}`, allocate 0.65 to `{nxt}` if its yield >= 70, else diversify across untested non-quarantined strategies."
                )
                lines.append(
                    f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                    f"[DIAGNOSTIC PROBE MODE ({b.probing_steps_remaining} left){quar_txt}{est_txt}]: "
                    f"Call `probe_strategy_execution` on untested `{nxt}` (do NOT re-probe {b.quarantined_strategies or 'already-probed arms'}).{alloc_hint}"
                )
            else:
                haz = self._due_hazard_candidate(b)
                w_json = json.dumps({s: round(float(b.active_weights.get(s, 0.25)), 2) for s in STRATEGIES_V2})
                if haz is not None:
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[STALENESS HAZARD CHECK DUE | Incumbent expected={b.expected_mean:.1f}, weights={w_json}]: "
                        f"Call `probe_strategy_execution` on long-untested `{haz}` once today. If `{haz}` live yield > {b.expected_mean + 8.0:.1f}, "
                        f"shift 0.65 weight to `{haz}` and 0.35 to `{b.primary_strategy}`; otherwise submit {w_json}."
                    )
                else:
                    lines.append(
                        f"- IF `volatility_regime_hint == '{sc_name}'` and `liquidity_status == 'orderly_normal'` -> "
                        f"[CALM EXPLOITATION | n={b.post_shift_n}, expected_score={b.expected_mean:.1f}]: "
                        f"Submit `recommended_weights = {w_json}` directly without paying probe fees."
                    )
        return "\n" + "\n".join(lines) + "\n"

    def _consolidate_bucket(
        self, b: ScopeEpistemicBucket, env_name: str, client: MeteredLLMClient
    ) -> tuple[int, float]:
        traces_txt = "\n".join(summarize_public_record(r) for r in b.recent_records[-4:])
        sys_p = (
            f"You consolidate the active portfolio weight hypothesis for scope `{b.scope_name}` on AlphaDesk-v2 "
            "using ONLY the provided post-shift traces and live probe estimates.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            "  \"primary_strategy\": \"STRAT_ALPHA|STRAT_BETA|STRAT_GAMMA|STRAT_DELTA\",\n"
            "  \"recommended_weights\": {\"STRAT_ALPHA\": 0.0, \"STRAT_BETA\": 0.0, \"STRAT_GAMMA\": 0.65, \"STRAT_DELTA\": 0.35},\n"
            "  \"predicted_portfolio_score\": 86.0\n"
            "}"
        )
        usr_p = (
            f"Scope: {b.scope_name}\n"
            f"Quarantined (falsified) strategies: {json.dumps(b.quarantined_strategies)}\n"
            f"Fresh live single-strategy probe estimates: {json.dumps(b.probe_estimates)}\n"
            f"Recent scope episode traces:\n{traces_txt}\n\n"
            "Return compliant `recommended_weights` (sum=1.0, each <= 0.65) maximizing expected score."
        )

        def _mock_consolidator(_s: str, _u: str) -> dict[str, Any]:
            best_w = self._best_known_weights_from_estimates(b) or {
                "STRAT_ALPHA": 0.65,
                "STRAT_BETA": 0.35,
                "STRAT_GAMMA": 0.0,
                "STRAT_DELTA": 0.0,
            }
            top_s = max(STRATEGIES_V2, key=lambda x: best_w[x])
            est_y = sum(best_w[x] * float(b.probe_estimates.get(x, 75.0)) for x in STRATEGIES_V2)
            return {
                "primary_strategy": top_s,
                "recommended_weights": best_w,
                "predicted_portfolio_score": round(est_y, 1),
            }

        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=env_name,
            method=f"{self.name}_consolidate",
            mock_responder=_mock_consolidator,
        )
        parsed = resp.parsed_json or {}
        tgt = str(parsed.get("primary_strategy", "STRAT_ALPHA")).strip().upper()
        if tgt not in STRATEGIES_V2:
            tgt = "STRAT_ALPHA"
        norm_w = normalize_card_weights(parsed.get("recommended_weights"), tgt)

        # Cross-check against direct empirical probe estimates if LLM picked a quarantined/dominated arm:
        if len(b.probe_estimates) >= 2:
            emp_w = self._best_known_weights_from_estimates(b)
            if emp_w is not None:
                emp_exp = sum(emp_w[s] * float(b.probe_estimates.get(s, 45.0)) for s in STRATEGIES_V2)
                llm_exp = sum(norm_w[s] * float(b.probe_estimates.get(s, 45.0)) for s in STRATEGIES_V2)
                if emp_exp > llm_exp + 8.0:
                    norm_w = emp_w

        b.active_weights = norm_w
        b.primary_strategy = max(STRATEGIES_V2, key=lambda s: norm_w[s])

        if b.probe_estimates:
            implied_gross = sum(
                norm_w[s] * float(b.probe_estimates.get(s, 20.0 if s in b.quarantined_strategies else 60.0))
                for s in STRATEGIES_V2
            )
        else:
            implied_gross = max((r.public_score for r in b.recent_records[-3:]), default=75.0)
        try:
            pred = float(parsed.get("predicted_portfolio_score", implied_gross))
        except (TypeError, ValueError):
            pred = implied_gross
        b.expected_mean = round(min(95.0, max(50.0, 0.5 * pred + 0.5 * implied_gross)), 1)
        b.post_shift_n = max(1, len(b.recent_records))
        b.cusum_stat = 0.0
        b.ewma_resid = 0.0
        return 1, resp.usd

    def on_episode_complete(
        self, record: PublicEpisodeRecord, client: MeteredLLMClient, *, force_curate: bool = False
    ) -> dict[str, Any]:
        sig = inspect_public_episode_signals(record)
        if sig["off_scope_anomaly"]:
            # Off-scope systemic shock (e.g. acute_liquidity_freeze): freeze CUSUM and preserve valid beliefs!
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

        for s, est in sig["probed"].items():
            b.probe_estimates[s] = est
            b.last_tested_ep[s] = cur_sc_ep
            if est >= 65.0 and s in b.quarantined_strategies:
                b.quarantined_strategies.remove(s)

        # Note:probe fees (-12 pts) depress `record.public_score` on probing turns; adjust realized portfolio score by +12 if 1 paid probe occurred
        probe_fee_adj = 12.0 * len(sig["probed"])
        est_gross_portfolio_y = min(100.0, float(record.public_score) + probe_fee_adj)

        # Case 1: Initial acquisition or post-alarm diagnostic probing window
        if b.probing_steps_remaining > 0:
            b.probing_steps_remaining -= 1
            non_quar_ests = [v for k, v in b.probe_estimates.items() if k not in b.quarantined_strategies]
            high_found = any(v >= 86.0 for v in non_quar_ests) and len(non_quar_ests) >= 2
            if b.probing_steps_remaining == 0 or high_found or force_curate:
                b.probing_steps_remaining = 0
                ref_c, ref_u = self._consolidate_bucket(b, record.env_name, client)
                if force_curate:
                    for ob in self.buckets.values():
                        if ob is not b and ob.probing_steps_remaining > 0 and ob.recent_records:
                            ob.probing_steps_remaining = 0
                            ec, eu = self._consolidate_bucket(ob, record.env_name, client)
                            ref_c += ec
                            ref_u += eu
                return {
                    "reflection_calls": ref_c,
                    "reflection_usd": ref_u,
                    "alarm_fired": False,
                    "consolidated_scope": sc_name,
                }
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}

        extra_c = 0
        extra_u = 0.0
        if force_curate:
            for ob in self.buckets.values():
                if ob.probing_steps_remaining > 0 and ob.recent_records:
                    ob.probing_steps_remaining = 0
                    ec, eu = self._consolidate_bucket(ob, record.env_name, client)
                    extra_c += ec
                    extra_u += eu

        # Mark primary held strategy as tested on non-probing exploitation turns
        if b.primary_strategy and sig["weights"].get(b.primary_strategy, 0.0) >= 0.30:
            b.last_tested_ep[b.primary_strategy] = cur_sc_ep

        # Check if a Staleness Hazard Probe just discovered a superior unheld strategy or runner-up upgrade!
        if sig["probed"]:
            b.hazard_probes_triggered += 1
            best_probed_s, best_probed_v = max(sig["probed"].items(), key=lambda kv: kv[1])
            cand_w = self._best_known_weights_from_estimates(b)
            portfolio_gain = 0.0
            if cand_w is not None and b.active_weights:
                cand_exp = sum(cand_w[s] * float(b.probe_estimates.get(s, 50.0)) for s in STRATEGIES_V2)
                cur_exp = sum(b.active_weights.get(s, 0.0) * float(b.probe_estimates.get(s, 50.0)) for s in STRATEGIES_V2)
                portfolio_gain = cand_exp - cur_exp
            if (best_probed_v > b.expected_mean + 10.0 and b.active_weights.get(best_probed_s, 0.0) < 0.30) or portfolio_gain > 7.0:
                ref_c, ref_u = self._consolidate_bucket(b, record.env_name, client)
                return {
                    "reflection_calls": ref_c,
                    "reflection_usd": ref_u,
                    "alarm_fired": True,
                    "hazard_upgrade": best_probed_s,
                }
            # Benign hazard check: do NOT penalize CUSUM for testing an unheld arm
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False, "hazard_checked": True}

        # Case 2: Calm exploitation — evaluate downward Level-CUSUM + EWMA-Slope residual on attributed executions
        if b.primary_strategy and sig["weights"].get(b.primary_strategy, 0.0) < 0.30:
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}

        resid = b.expected_mean - est_gross_portfolio_y
        shortfall = resid - self.drift_slack
        b.cusum_stat = max(0.0, b.cusum_stat + shortfall)
        b.ewma_resid = (1.0 - self.ewma_alpha) * b.ewma_resid + self.ewma_alpha * resid

        slope_alarm = (b.post_shift_n >= 2) and (b.ewma_resid >= self.ewma_threshold)
        if b.cusum_stat >= self.alarm_threshold or slope_alarm:
            b.shelf.append(
                {
                    "regime_id": len(b.shelf) + 1,
                    "weights": copy.deepcopy(b.active_weights),
                    "expected_mean": b.expected_mean,
                    "validated_n": b.post_shift_n,
                }
            )
            falsified = [s for s, w in b.active_weights.items() if w >= 0.30]
            b.quarantined_strategies = falsified
            b.probe_estimates = {s: est_gross_portfolio_y for s in falsified[:1]}
            b.active_weights = {}
            b.primary_strategy = None
            b.post_shift_n = 0
            b.cusum_stat = 0.0
            b.ewma_resid = 0.0
            # Probe up to 3 non-quarantined candidates (with early stop as soon as >=88 is found!)
            b.probing_steps_remaining = 3
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
        b.expected_mean = round((1.0 - lr) * b.expected_mean + lr * est_gross_portfolio_y, 2)
        if b.primary_strategy:
            b.probe_estimates[b.primary_strategy] = b.expected_mean
        return {
            "reflection_calls": extra_c,
            "reflection_usd": extra_u,
            "alarm_fired": False,
            "cusum": round(b.cusum_stat, 2),
            "ewma_resid": round(b.ewma_resid, 2),
        }

    def predict_calm_portfolio_prob_ge_75(self, w_ref: dict[str, float]) -> float:
        b = self._get_bucket("calm_volatility")
        est_y = 0.0
        for s, w in w_ref.items():
            if w <= 0.0:
                continue
            if s in b.quarantined_strategies:
                s_val = float(b.probe_estimates.get(s, 22.0))
            elif s in b.probe_estimates:
                s_val = float(b.probe_estimates[s])
            elif b.active_weights.get(s, 0.0) >= 0.30:
                s_val = b.expected_mean
            else:
                s_val = 50.0
            est_y += w * s_val
        return logistic_prob_ge(est_y, threshold=75.0, scale=8.0)
