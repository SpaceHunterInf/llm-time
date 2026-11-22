"""Baseline harnesses: Blind, SlidingWindow, SRN (Bounded Notes), and SkillBook (RSI Ratchet)."""

from __future__ import annotations

import json
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord


def summarize_public_record(rec: PublicEpisodeRecord) -> str:
    relevant_steps = [st for st in rec.steps if st.get("tool") != "read_docs"] or rec.steps
    compact_steps = []
    for st in relevant_steps[-5:]:
        compact_steps.append(
            f"{st.get('tool')}({json.dumps(st.get('args', {}))}) -> {json.dumps(st.get('obs', {}))[:220]}"
        )
    joined = " | ".join(compact_steps)
    prompt_snip = " ".join(str(rec.task_prompt).split())[:140]
    return f"[Ep {rec.timestamp}] Score={rec.public_score:.1f}/100 | Task: {prompt_snip} | Trace: {joined}"


class BlindHarness(EpistemicHarness):
    """Zero cross-episode memory baseline."""

    name = "blind"

    def build_context(self, spec: PublicTaskSpec) -> str:
        return ""

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        return {"reflection_calls": 0, "reflection_usd": 0.0}


class SlidingWindowHarness(EpistemicHarness):
    """Stateless recent-episode window baseline (no self-authored rules, 0 reflection calls)."""

    name = "window"

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = int(window_size)
        self._recent: list[str] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self._recent:
            return ""
        lines = "\n".join(self._recent[-self.window_size :])
        return f"\nRECENT EPISODE OUTCOMES (last {len(self._recent[-self.window_size:])}):\n{lines}\n"

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self._recent.append(summarize_public_record(record))
        return {"reflection_calls": 0, "reflection_usd": 0.0}


class SRNHarness(EpistemicHarness):
    """Simple Reflection with Bounded Notes (faithful to Zhang et al., 'The RSI Tax', 2611.13524)."""

    name = "srn"

    def __init__(self, max_chars: int = 1400, reflect_every: int = 2) -> None:
        self.max_chars = int(max_chars)
        self.reflect_every = max(1, int(reflect_every))
        self.note: str = ""
        self._buffer: list[str] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self.note:
            return ""
        return f"\nPERSISTENT STRATEGY NOTE (bounded):\n{self.note[: self.max_chars]}\n"

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self._buffer.append(summarize_public_record(record))
        if record.timestamp % self.reflect_every != 0:
            return {"reflection_calls": 0, "reflection_usd": 0.0}

        sys_p = (
            "You maintain a single bounded strategy note (max 1200 chars) for an online agent. "
            "Respond ONLY with JSON: {\"updated_note\": \"...\"}."
        )
        usr_p = (
            f"Current strategy note:\n{self.note or '(empty)'}\n\n"
            f"Recent episodes:\n" + "\n".join(self._buffer[-self.reflect_every :]) + "\n\n"
            "Update the concise note with hard constraints and best-performing decision heuristics."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method=f"{self.name}_reflect",
            mock_responder=lambda _s, _u: {"updated_note": f"Last score {record.public_score}"},
        )
        if resp.parsed_json and isinstance(resp.parsed_json.get("updated_note"), str):
            self.note = resp.parsed_json["updated_note"][: self.max_chars]
        return {"reflection_calls": 1, "reflection_usd": resp.usd}


class SkillBookHarness(EpistemicHarness):
    """ACE/SkillOpt-style rulebook with unclipped cumulative win/use counters (exposes RSI Ratchet)."""

    name = "skillbook"

    def __init__(self, reflect_every: int = 2) -> None:
        self.reflect_every = max(1, int(reflect_every))
        self.rules: list[dict[str, Any]] = []
        self._history: list[PublicEpisodeRecord] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self.rules:
            return ""
        ranked = sorted(
            self.rules,
            key=lambda r: (r.get("wins", 0) / max(1, r.get("uses", 1)), r.get("wins", 0)),
            reverse=True,
        )
        formatted = []
        for idx, r in enumerate(ranked[:5], 1):
            wins = int(r.get("wins", 0))
            uses = max(1, int(r.get("uses", 1)))
            formatted.append(
                f"{idx}. [Validated {wins}/{uses} wins ({100.0 * wins / uses:.0f}%)] {r.get('rule', '')}"
            )
        return (
            "\nVALIDATED SELF-EVOLVED PLAYBOOK (follow highest-validated rules strictly):\n"
            + "\n".join(formatted)
            + "\n"
        )

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self._history.append(record)
        won = 1 if record.public_score >= 70.0 else 0
        for r in self.rules:
            r["uses"] = int(r.get("uses", 0)) + 1
            r["wins"] = int(r.get("wins", 0)) + won

        if record.timestamp % self.reflect_every != 0:
            return {"reflection_calls": 0, "reflection_usd": 0.0}

        recent_txt = "\n".join(summarize_public_record(x) for x in self._history[-4:])
        sys_p = (
            "You are a SkillBook Curator for a self-evolving agent. "
            "Propose at most 2 concise decision rules from recent traces. "
            "Respond ONLY with JSON: {\"candidate_rules\": [\"rule 1\", \"rule 2\"]}."
        )
        usr_p = (
            f"Existing playbook:\n{json.dumps(self.rules, indent=2)}\n\n"
            f"Recent traces:\n{recent_txt}\n\n"
            "Return JSON with `candidate_rules`."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method=f"{self.name}_reflect",
            mock_responder=lambda _s, _u: {"candidate_rules": ["Follow prior high-scoring pattern."]},
        )
        cands = (resp.parsed_json or {}).get("candidate_rules", [])
        if isinstance(cands, list):
            existing_texts = {r.get("rule", "") for r in self.rules}
            for c in cands[:2]:
                if isinstance(c, str) and c.strip() and c.strip() not in existing_texts:
                    self.rules.append({"rule": c.strip()[:280], "wins": won, "uses": 1})
            # Retain top-scoring rules by cumulative wins (classic score ratchet!)
            self.rules = sorted(
                self.rules,
                key=lambda r: (r.get("wins", 0), r.get("wins", 0) / max(1, r.get("uses", 1))),
                reverse=True,
            )[:6]
        return {"reflection_calls": 1, "reflection_usd": resp.usd}
