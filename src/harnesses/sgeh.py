"""SGEH: Surprise-Gated Epistemic Harness (Act 2 Method)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record


@dataclass
class ShelvedRegime:
    regime_id: int
    utility_hypothesis: str
    mean_score: float
    episodes_validated: int


@dataclass
class EpistemicState:
    invariant_constraints: list[str] = field(default_factory=list)
    active_hypothesis: str = ""
    expected_mean: float = 80.0
    post_shift_n: int = 0
    cusum_stat: float = 0.0
    probing_steps_remaining: int = 3  # Start life with 3 initial exploration steps
    shift_alarms_triggered: int = 0
    shelf: list[ShelvedRegime] = field(default_factory=list)


class SurpriseGatedEpistemicHarness(EpistemicHarness):
    """Low-tax epistemic harness combining CUSUM surprise gating, scoped shelving, and counter reset.

    Key properties:
      1. Zero verification LLM calls while `cusum_stat < threshold` (kills the 75-94% RSI Tax).
      2. Separates `invariant_constraints` (never forgotten) from `active_hypothesis` (reward assumption).
      3. On surprise trigger (`cusum_stat >= threshold`):
         - Shelves the falsified `active_hypothesis` (for potential later recurrence).
         - Resets `post_shift_n = 0` and `cusum_stat = 0` (prevents large-n Resurrection!).
         - Enters 2-episode Active Diagnostic Probe mode before reconsolidating from post-shift data ONLY.
    """

    name = "sgeh"

    def __init__(
        self,
        *,
        drift_slack: float = 10.0,
        alarm_threshold: float = 24.0,
        post_alarm_probe_steps: int = 2,
        consolidate_window: int = 4,
    ) -> None:
        self.drift_slack = float(drift_slack)
        self.alarm_threshold = float(alarm_threshold)
        self.post_alarm_probe_steps = max(1, int(post_alarm_probe_steps))
        self.consolidate_window = max(3, int(consolidate_window))
        self.state = EpistemicState()
        self._post_shift_records: list[PublicEpisodeRecord] = []

    def build_context(self, spec: PublicTaskSpec) -> str:
        sections: list[str] = []
        if self.state.invariant_constraints:
            inv = "\n".join(f"- {c}" for c in self.state.invariant_constraints)
            sections.append(f"PERMANENT INVARIANT CONSTRAINTS (always obey):\n{inv}")

        if self.state.probing_steps_remaining > 0:
            shelf_hint = ""
            if self.state.shelf:
                archived = "; ".join(
                    f"[Regime #{s.regime_id}, prior mean {s.mean_score:.0f}]: {s.utility_hypothesis}"
                    for s in self.state.shelf[-3:]
                )
                shelf_hint = (
                    f"\nSHELVED PAST REGIMES (test if world reverted, do NOT assume blindly): {archived}"
                )
            recent_post = "\n".join(summarize_public_record(r) for r in self._post_shift_records[-3:])
            sections.append(
                "EPISTEMIC MODE: ACTIVE DIAGNOSTIC PROBING (Prior utility hypothesis suspended!)\n"
                "- Do NOT rely on pre-shift preferences, onboarding bios, or stale playbooks.\n"
                "- Invest in 1 diagnostic/inspection check or test an unexplored option to identify what currently drives high reward.\n"
                + (f"Post-shift observations so far:\n{recent_post}" if recent_post else "")
                + shelf_hint
            )
        elif self.state.active_hypothesis:
            sections.append(
                f"EPISTEMIC MODE: CALM EXPLOITATION (post-shift validated n={self.state.post_shift_n}, "
                f"expected score ~{self.state.expected_mean:.0f}/100):\n"
                f"{self.state.active_hypothesis}\n"
                "- Execute this validated hypothesis directly; avoid paying optional diagnostic audit/probe fees unless required by hard safety constraints."
            )

        return ("\n" + "\n\n".join(sections) + "\n") if sections else ""

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        y = float(record.public_score)
        self._post_shift_records.append(record)

        # Case 1: Currently in initial acquisition or post-alarm probing window
        if self.state.probing_steps_remaining > 0:
            self.state.probing_steps_remaining -= 1
            if self.state.probing_steps_remaining == 0:
                return self._consolidate_post_shift(record, client)
            return {
                "reflection_calls": 0,
                "reflection_usd": 0.0,
                "alarm_fired": False,
                "cusum": round(self.state.cusum_stat, 2),
            }

        # Case 2: Calm exploitation — run lightweight scalar Page-Hinkley / CUSUM gate (0 LLM tokens!)
        shortfall = self.state.expected_mean - y - self.drift_slack
        self.state.cusum_stat = max(0.0, self.state.cusum_stat + shortfall)

        if self.state.cusum_stat >= self.alarm_threshold:
            # SURPRISE ALARM! Quarantine & shelve active hypothesis, reset counters!
            if self.state.active_hypothesis:
                self.state.shelf.append(
                    ShelvedRegime(
                        regime_id=len(self.state.shelf) + 1,
                        utility_hypothesis=self.state.active_hypothesis,
                        mean_score=round(self.state.expected_mean, 1),
                        episodes_validated=self.state.post_shift_n,
                    )
                )
            self.state.active_hypothesis = ""
            self.state.post_shift_n = 0
            self.state.cusum_stat = 0.0
            self.state.probing_steps_remaining = self.post_alarm_probe_steps
            self.state.shift_alarms_triggered += 1
            # Keep only the triggering anomalous episode in post-shift evidence buffer!
            self._post_shift_records = [record]
            return {
                "reflection_calls": 0,
                "reflection_usd": 0.0,
                "alarm_fired": True,
                "cusum": 0.0,
            }

        # No alarm: update post-shift running mean in pure Python (0 LLM tokens!)
        self.state.post_shift_n += 1
        alpha = 1.0 / max(2, min(8, self.state.post_shift_n))
        self.state.expected_mean = (1.0 - alpha) * self.state.expected_mean + alpha * y
        return {
            "reflection_calls": 0,
            "reflection_usd": 0.0,
            "alarm_fired": False,
            "cusum": round(self.state.cusum_stat, 2),
        }

    def _consolidate_post_shift(
        self, record: PublicEpisodeRecord, client: MeteredLLMClient
    ) -> dict[str, Any]:
        traces_txt = "\n".join(summarize_public_record(r) for r in self._post_shift_records[-self.consolidate_window :])
        shelf_txt = json.dumps([s.__dict__ for s in self.state.shelf[-3:]], indent=2)
        sys_p = (
            "You are the Epistemic Hypothesis Consolidator. Separate permanent domain/safety constraints "
            "from the CURRENT active reward/utility hypothesis using ONLY the provided post-shift traces.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            "  \"invariant_constraints\": [\"hard constraint 1\", ...],\n"
            "  \"active_hypothesis\": \"concise falsifiable policy for maximizing score right now\",\n"
            "  \"predicted_mean_score\": 85.0\n"
            "}"
        )
        usr_p = (
            f"Known invariant constraints so far: {json.dumps(self.state.invariant_constraints)}\n"
            f"Shelved historical regimes (for reference if world reverted):\n{shelf_txt}\n\n"
            f"Fresh post-shift episode traces (ground truth for current regime):\n{traces_txt}\n\n"
            "Synthesize the updated JSON epistemic state."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method=f"{self.name}_consolidate",
            mock_responder=lambda _s, _u: {
                "invariant_constraints": self.state.invariant_constraints or ["Obey hard caps"],
                "active_hypothesis": "Follow recent post-shift high-reward actions.",
                "predicted_mean_score": max(65.0, record.public_score),
            },
        )
        parsed = resp.parsed_json or {}
        invs = parsed.get("invariant_constraints")
        if isinstance(invs, list) and invs:
            self.state.invariant_constraints = [str(x)[:200] for x in invs[:5]]
        hyp = parsed.get("active_hypothesis")
        if isinstance(hyp, str) and hyp.strip():
            self.state.active_hypothesis = hyp.strip()[:600]
        recent_scores = [r.public_score for r in self._post_shift_records[-3:]]
        empirical_best = max(recent_scores) if recent_scores else 75.0
        try:
            pred = float(parsed.get("predicted_mean_score", empirical_best))
        except (TypeError, ValueError):
            pred = empirical_best
        self.state.expected_mean = min(96.0, max(55.0, 0.5 * pred + 0.5 * empirical_best))
        self.state.post_shift_n = len(self._post_shift_records)
        self.state.cusum_stat = 0.0
        return {
            "reflection_calls": 1,
            "reflection_usd": resp.usd,
            "alarm_fired": False,
            "consolidated": True,
        }
