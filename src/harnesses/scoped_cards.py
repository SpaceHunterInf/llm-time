"""ScopedCardBookHarness (Iteration 1b): Pure allocation-ranking cards with single-variable causal surgery."""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.alpha_desk_v2 import STRATEGIES_V2
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record


def extract_submitted_weights(record: PublicEpisodeRecord) -> dict[str, float]:
    for st in reversed(record.steps):
        if st.get("tool") == "submit_allocation":
            raw = st.get("args", {}).get("weights", {})
            if isinstance(raw, dict):
                out: dict[str, float] = {}
                for k, v in raw.items():
                    try:
                        out[str(k).upper()] = float(v)
                    except (TypeError, ValueError):
                        pass
                return out
    return {}


def detected_high_vol_in_trace(record: PublicEpisodeRecord) -> bool:
    for st in record.steps:
        obs = st.get("obs", {})
        if isinstance(obs, dict) and obs.get("volatility_regime_hint") == "elevated_volatility":
            return True
    return False


def sanitize_scope_text(raw_scope: str) -> str:
    """Ensure `scope_condition` describes ONLY observable market state (zero probing or tool directives)."""
    cleaned = re.sub(r"probe_strategy_execution", "market_check", raw_scope, flags=re.IGNORECASE)
    cleaned = re.sub(r"prob(?:e|ing|ed|es)", "check", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())[:180] or "volatility_regime_hint == 'calm_volatility'"


def normalize_card_weights(raw_w: Any, fallback_target: str) -> dict[str, float]:
    if isinstance(raw_w, dict):
        parsed: dict[str, float] = {}
        for s in STRATEGIES_V2:
            try:
                v = max(0.0, min(0.65, float(raw_w.get(s, 0.0))))
            except (TypeError, ValueError):
                v = 0.0
            parsed[s] = v
        tot = sum(parsed.values())
        if tot > 0.1:
            norm = {s: round(parsed[s] / tot, 2) for s in STRATEGIES_V2}
            for s in STRATEGIES_V2:
                if norm[s] > 0.65:
                    excess = round(norm[s] - 0.65, 2)
                    norm[s] = 0.65
                    other = next(x for x in STRATEGIES_V2 if x != s)
                    norm[other] = round(norm[other] + excess, 2)
            diff = round(1.0 - sum(norm.values()), 2)
            top_s = max(STRATEGIES_V2, key=lambda s: norm[s])
            if norm[top_s] + diff <= 0.65:
                norm[top_s] = round(norm[top_s] + diff, 2)
            else:
                sec_s = sorted(STRATEGIES_V2, key=lambda s: norm[s], reverse=True)[1]
                norm[sec_s] = round(norm[sec_s] + diff, 2)
            return norm

    tgt = fallback_target if fallback_target in STRATEGIES_V2 else "STRAT_ALPHA"
    others = [s for s in STRATEGIES_V2 if s != tgt]
    out = {tgt: 0.60, others[0]: 0.20, others[1]: 0.10, others[2]: 0.10}
    return {s: out[s] for s in STRATEGIES_V2}


def unranked_weights_dict() -> dict[str, None]:
    return {s: None for s in STRATEGIES_V2}


def render_allocation_payload(card: dict[str, Any]) -> str:
    if card.get("unranked", False):
        return f"recommended_weights = {json.dumps(unranked_weights_dict())}"
    if card.get("target_strategy") == "COMPLIANCE":
        return "enforce sum(weights) == 1.0 and max_single_strategy_weight <= 0.65"
    w = card.get("recommended_weights", {})
    ordered = {s: round(float(w.get(s, 0.0)), 2) for s in STRATEGIES_V2}
    return f"recommended_weights = {json.dumps(ordered)}"


def attributed_stats_over_records(
    records: list[PublicEpisodeRecord], kind: str, target_strategy: str
) -> tuple[int, float | None]:
    tgt = target_strategy.upper()
    scores: list[float] = []
    for r in records:
        w = extract_submitted_weights(r)
        is_hv = detected_high_vol_in_trace(r)
        scope_matches = (kind == "invariant_or_hedge" and is_hv) or (kind == "calm_utility" and not is_hv)
        if scope_matches and w.get(tgt, 0.0) >= 0.30:
            scores.append(float(r.public_score))
    if not scores:
        return 0, None
    return len(scores), round(sum(scores) / len(scores), 1)


class ScopedCardBookHarness(EpistemicHarness):
    """Single-variable-isolating Scoped Heuristic Cardbook (`Iteration 1b`).

    Invariants enforced across `Retained`, `Sham-Edited`, and `Utility-Excised`:
      1. Zero probing instructions inside any card or header (probing text lives solely in shared domain prompt).
      2. Identical `_history` retention and post-fork reflection schedule `(len(_history) - fork_offset) % 3 == 0`.
      3. `Excised` derives from `Sham-Edited` by replacing ONLY `calm_utility` cards' `recommended_weights` (`null`),
         `target_strategy` (`null`), and attributed `uses/mean_score` (`0 / null`) while preserving identical
         `card_id`, `kind`, `scope_condition`, and `invariant_or_hedge` cards in both actor and curator prompts.
    """

    name = "scoped_cards"

    def __init__(self, reflect_every: int = 3) -> None:
        self.reflect_every = max(1, int(reflect_every))
        self.cards: list[dict[str, Any]] = []
        self._history: list[PublicEpisodeRecord] = []
        self.fork_offset: int | None = None

    def clone(self) -> "ScopedCardBookHarness":
        dup = ScopedCardBookHarness(reflect_every=self.reflect_every)
        dup.cards = copy.deepcopy(self.cards)
        dup._history = list(self._history)
        dup.fork_offset = self.fork_offset
        return dup

    def set_post_fork_schedule(self) -> None:
        """Align post-boundary curation schedule identically across all branches (`t*+2, t*+5, ...`)."""
        self.fork_offset = len(self._history)

    def make_sham_edited_copy(
        self, client: MeteredLLMClient, env_name: str
    ) -> tuple["ScopedCardBookHarness", float]:
        """Branch 2 (`Sham-Edited`): Paraphrase `scope_condition` wording only; keep identical weights & stats."""
        dup = self.clone()
        if not dup.cards:
            return dup, 0.0

        sys_p = (
            "Paraphrase each card's `scope_condition` slightly using synonyms for operators/connectives while "
            "preserving exact single-quoted desk enum literals (e.g. 'calm_volatility', 'elevated_volatility', "
            "'orderly_normal') verbatim. Do NOT mention probing, tools, or strategy names. "
            "Respond ONLY with JSON: {\"paraphrased\": [{\"card_id\": \"...\", \"scope_condition\": \"...\"}, ...]}."
        )
        usr_p = json.dumps(
            [{"card_id": c["card_id"], "scope_condition": c["scope_condition"]} for c in dup.cards],
            indent=2,
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=env_name,
            method="scoped_cards_sham_edit",
            mock_responder=lambda _s, _u: {
                "paraphrased": [
                    {"card_id": c["card_id"], "scope_condition": f"session_state: {c['scope_condition']}"}
                    for c in dup.cards
                ]
            },
        )
        by_id = {
            str(item.get("card_id")): item
            for item in (resp.parsed_json or {}).get("paraphrased", [])
            if isinstance(item, dict)
        }
        for c in dup.cards:
            cid = str(c["card_id"])
            if cid in by_id:
                orig_s = str(c["scope_condition"])
                cand_s = sanitize_scope_text(str(by_id[cid].get("scope_condition") or orig_s))
                c["scope_condition"] = cand_s[: max(40, len(orig_s) + 30)]
        return dup, resp.usd

    def make_utility_excised_from_sham(self) -> "ScopedCardBookHarness":
        """Iteration 1b (`Utility-Excised`): Identical to `self` EXCEPT calm-card weights, uses, & score are null/0."""
        dup = self.clone()
        for c in dup.cards:
            if c.get("kind") == "calm_utility":
                c["unranked"] = True
                c["recommended_weights"] = unranked_weights_dict()
                c["target_strategy"] = None
                c["uses"] = 0
                c["mean_score"] = None
        return dup

    def make_weights_excised_score_shown(self) -> "ScopedCardBookHarness":
        """Iteration 1c (`Score-Shown`): Hide calm `recommended_weights`, KEEP exact `uses` AND `mean_score`."""
        dup = self.clone()
        for c in dup.cards:
            if c.get("kind") == "calm_utility":
                c["unranked"] = True
                c["recommended_weights"] = unranked_weights_dict()
                c["target_strategy"] = None
                # Preserve `c["uses"]` and `c["mean_score"]` verbatim!
        return dup

    def make_weights_excised_score_hidden(self) -> "ScopedCardBookHarness":
        """Iteration 1c (`Score-Hidden`): Hide calm `recommended_weights` AND `mean_score`, KEEP exact `uses`."""
        dup = self.clone()
        for c in dup.cards:
            if c.get("kind") == "calm_utility":
                c["unranked"] = True
                c["recommended_weights"] = unranked_weights_dict()
                c["target_strategy"] = None
                # Preserve `c["uses"]` verbatim; hide ONLY `mean_score`!
                c["mean_score"] = None
        return dup

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self.cards:
            return ""
        lines: list[str] = []
        for idx, c in enumerate(self.cards[:4], 1):
            u = int(c.get("uses", 0))
            m_val = c.get("mean_score")
            mean_txt = f"{float(m_val):.1f}" if m_val is not None else "null"
            payload_str = render_allocation_payload(c)
            lines.append(
                f"- Card {c.get('card_id', f'C{idx}')} [{c.get('kind')} | Past sessions: n={u}, mean_portfolio_score={mean_txt}]: "
                f"IF {c.get('scope_condition')} -> {payload_str}"
            )
        return (
            "\nSCOPED HEURISTIC CARDBOOK (reference allocation weights & realized session portfolio scores by scope; null = hidden/unranked):\n"
            + "\n".join(lines)
            + "\n"
        )

    def should_curate_now(self) -> bool:
        if self.fork_offset is None:
            return len(self._history) > 0 and (len(self._history) % self.reflect_every == 0)
        post_steps = len(self._history) - self.fork_offset
        return post_steps > 0 and (post_steps % self.reflect_every == 0)

    def curate_input_messages(self) -> tuple[str, str]:
        recent_txt = "\n".join(summarize_public_record(r) for r in self._history[-self.reflect_every :])
        public_cards_view = [
            {
                "card_id": c.get("card_id"),
                "kind": c.get("kind"),
                "scope_condition": c.get("scope_condition"),
                "target_strategy": None if c.get("unranked", False) else c.get("target_strategy"),
                "recommended_weights": (
                    unranked_weights_dict() if c.get("unranked", False) else c.get("recommended_weights")
                ),
                "uses": int(c.get("uses", 0)),
                "mean_portfolio_score": c.get("mean_score"),
            }
            for c in self.cards
        ]
        sys_p = (
            "You maintain a structured Scoped Heuristic Cardbook (max 3 cards) for AlphaDesk-v2.\n"
            "Each card represents an empirical portfolio weight hypothesis (structured weights only, zero prose rules):\n"
            "- `card_id`: 'C1'..'C3'\n"
            "- `kind`: `'invariant_or_hedge'` (for `elevated_volatility` sessions) OR `'calm_utility'` (for `calm_volatility` sessions)\n"
            "- `scope_condition`: short observable market condition ONLY (NEVER mention actions, tools, or probing)\n"
            "- `target_strategy`: dominant strategy (`STRAT_ALPHA`, `STRAT_BETA`, `STRAT_GAMMA`, or `STRAT_DELTA`)\n"
            "- `recommended_weights`: JSON dict mapping all 4 strategies to floats in [0, 0.65] summing to 1.0\n"
            "Update or replace cards based on recent session evidence. Respond ONLY with JSON: {\"cards\": [{...}, ...]}."
        )
        usr_p = (
            f"Current cards:\n{json.dumps(public_cards_view, indent=2)}\n\n"
            f"Recent {self.reflect_every} session traces:\n{recent_txt}\n\n"
            "Return updated `cards` JSON list."
        )
        return sys_p, usr_p

    def on_episode_complete(
        self, record: PublicEpisodeRecord, client: MeteredLLMClient, *, force_curate: bool = False
    ) -> dict[str, Any]:
        self._history.append(record)
        weights = extract_submitted_weights(record)
        is_hv = detected_high_vol_in_trace(record)

        for c in self.cards:
            if c.get("unranked", False):
                continue
            tgt = str(c.get("target_strategy") or "").upper()
            kind = str(c.get("kind", "calm_utility"))
            scope_matches = (kind == "invariant_or_hedge" and is_hv) or (kind == "calm_utility" and not is_hv)
            if scope_matches and weights.get(tgt, 0.0) >= 0.30:
                old_u = int(c.get("uses", 0))
                old_m = c.get("mean_score")
                new_u = old_u + 1
                prev_sum = float(old_m) * old_u if (old_u > 0 and old_m is not None) else 0.0
                c["uses"] = new_u
                c["mean_score"] = round((prev_sum + float(record.public_score)) / new_u, 1)

        if not force_curate and not self.should_curate_now():
            return {"reflection_calls": 0, "reflection_usd": 0.0}

        sys_p, usr_p = self.curate_input_messages()
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method="scoped_cards_curate",
            mock_responder=lambda _s, _u: {
                "cards": [
                    {
                        "card_id": "C1",
                        "kind": "calm_utility",
                        "scope_condition": "volatility_regime_hint == 'calm_volatility'",
                        "target_strategy": "STRAT_ALPHA",
                        "recommended_weights": {
                            "STRAT_ALPHA": 0.65,
                            "STRAT_BETA": 0.25,
                            "STRAT_GAMMA": 0.05,
                            "STRAT_DELTA": 0.05,
                        },
                    },
                    {
                        "card_id": "C2",
                        "kind": "invariant_or_hedge",
                        "scope_condition": "volatility_regime_hint == 'elevated_volatility'",
                        "target_strategy": "STRAT_DELTA",
                        "recommended_weights": {
                            "STRAT_ALPHA": 0.05,
                            "STRAT_BETA": 0.05,
                            "STRAT_GAMMA": 0.25,
                            "STRAT_DELTA": 0.65,
                        },
                    },
                ]
            },
        )
        new_cards = (resp.parsed_json or {}).get("cards", [])
        if isinstance(new_cards, list) and new_cards:
            old_by_target = {
                (c.get("kind"), c.get("target_strategy")): c
                for c in self.cards
                if not c.get("unranked", False) and c.get("target_strategy")
            }
            recent_slice = self._history[-self.reflect_every :]
            merged: list[dict[str, Any]] = []
            for idx, raw_c in enumerate(new_cards[:3], 1):
                if not isinstance(raw_c, dict):
                    continue
                kind = str(raw_c.get("kind", "calm_utility"))
                if kind not in ("invariant_or_hedge", "calm_utility"):
                    kind = "calm_utility"
                tgt = str(raw_c.get("target_strategy", "STRAT_ALPHA")).strip().upper()
                if tgt not in STRATEGIES_V2:
                    tgt = "STRAT_ALPHA"
                norm_w = normalize_card_weights(raw_c.get("recommended_weights"), tgt)
                tgt = max(STRATEGIES_V2, key=lambda s: norm_w[s])
                prev = old_by_target.get((kind, tgt))
                if prev is not None:
                    u_cnt = int(prev.get("uses", 0))
                    m_sc = prev.get("mean_score")
                else:
                    u_cnt, m_sc = attributed_stats_over_records(recent_slice, kind, tgt)
                merged.append(
                    {
                        "card_id": f"C{idx}",
                        "kind": kind,
                        "scope_condition": sanitize_scope_text(str(raw_c.get("scope_condition", kind))),
                        "target_strategy": tgt,
                        "recommended_weights": norm_w,
                        "uses": u_cnt,
                        "mean_score": m_sc,
                    }
                )
            if merged:
                self.cards = merged
        return {"reflection_calls": 1, "reflection_usd": resp.usd}

    def predict_calm_portfolio_prob_ge_75(self, w_ref: dict[str, float]) -> float:
        """Evaluate implied P(calm portfolio score >= 75 | weights = w_ref) from current calm card state."""
        calm_cards = [c for c in self.cards if c.get("kind") == "calm_utility" and not c.get("unranked", False)]
        if not calm_cards:
            return 0.5
        c = calm_cards[0]
        rec_w = c.get("recommended_weights") or {}
        m_sc = c.get("mean_score")
        card_mean = float(m_sc) if m_sc is not None else 75.0
        est_y = 0.0
        for s, w in w_ref.items():
            if w <= 0.0:
                continue
            # If card recommends `s` heavily, imply `card_mean`; if card avoids `s`, imply neutral/low 45.0
            s_w = float(rec_w.get(s, 0.0))
            s_val = card_mean if s_w >= 0.25 else (0.5 * card_mean + 22.5)
            est_y += w * s_val
        z = max(-12.0, min(12.0, (est_y - 75.0) / 8.0))
        return round(1.0 / (1.0 + math.exp(-z)), 4)
