"""Self-Authored Editable vs. Frozen Exploration/Review Procedure Harnesses & Cross-Domain Transfer.

Implements the clean procedural autonomy & transfer experiments (`RESEARCH_STATE.md` §4.1 Priorities 2 & 3,
testing Sundaram et al. `arXiv:2611.23219` under non-stationary reward drift + matched stationary controls):

1. `SelfAuthoredProcedureHarness` (`editable_proc_v3` vs `frozen_proc_v3`):
   - Starts with `FactualLedgerV3Harness` (`0` prescriptions) + an **initially blank** `self_procedure: str = ""`
     (`0` researcher-written exploration rules, `0` yield thresholds, `0` hazard intervals, `0` liquidity hints).
   - Every `review_cadence` sessions, runs a neutral self-review call where the frozen LLM sees ONLY the public
     environment rules, tool fees, its current `self_procedure`, and its `Factual Observation Ledger`, and may write
     or revise `self_procedure` (`<= 140` words) for its future self.
   - Under `mode == "editable"` (`Editable-Procedure [Ours]`), edits overwrite `self_procedure`.
   - Under `mode == "frozen"` (`Frozen-Procedure [Matched-Budget Control]`), the **exact same review call schedule and
     token budget** execute (logging counterfactual proposed edits for auditability), while keeping active
     `self_procedure` locked at the checkpoint version `P_ckpt`.
   - Exposes `snapshot_state()` so runners log the exact verbatim `self_procedure` in JSONL **before** outcomes arrive.

2. `distill_domain_agnostic_procedure` + `SLBFactualLogHarness` + `SLBTransferredProcedureHarness`:
   - Distills a domain-agnostic epistemic checking/revision procedure (`<= 140` words, SHA-256 hashed & timestamped)
     from `AlphaDesk` experience **before** touching held-out `ServeLearnBench`.
   - Evaluates zero-shot cross-domain procedure transfer starting from empty task memory (`M_0 = empty`) with zero
     hand-coded Python streak/shelf triggers, logging 100% of generated rules/notes verbatim in JSONL traces.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.alpha_desk_v3 import MAX_WEIGHT_V3, PROBE_FEE_V3, STRATEGIES_V3, TURNOVER_FULL_FLIP_PTS_V3
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record
from src.harnesses.sgeh_slb import infer_public_slb_scope
from src.harnesses.sgeh_v3 import FactualLedgerV3Harness

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ACTION_BINDING_RE = re.compile(r"\bSTRAT_S[1-6]\b|\bS[1-6]\b", re.IGNORECASE)
_SESSION_BINDING_RE = re.compile(r"\b[Ss]essions?\s*#?\s*\d+(?:\s*[-–]\s*#?\s*\d+)?\b")


def sanitize_procedure_text(
    raw: Any,
    max_words: int = 140,
    max_chars: int = 960,
    strip_action_bindings: bool = False,
) -> str:
    """Strip ASCII control chars and enforce bounded word/character budgets on self-authored procedures."""
    txt = _CTRL_RE.sub(" ", str(raw or "")).strip()
    if strip_action_bindings:
        txt = _ACTION_BINDING_RE.sub("an observed strategy", txt)
        txt = _SESSION_BINDING_RE.sub("recent sessions", txt)
    words = txt.split()
    if len(words) > max_words:
        txt = " ".join(words[:max_words])
    return txt[:max_chars].strip()


class SelfAuthoredProcedureHarness(EpistemicHarness):
    """Factual-Ledger + Self-Authored `self_procedure` (`editable` vs `frozen`, free-form vs `action_agnostic_schema`)."""

    def __init__(
        self,
        *,
        mode: str = "editable",
        action_agnostic_schema: bool = False,
        review_cadence: int = 4,
        recent_k: int = 8,
        max_words: int = 140,
        initial_procedure: str = "",
    ) -> None:
        if mode not in ("editable", "frozen"):
            raise ValueError(f"Unsupported procedure mode '{mode}'. Expected 'editable' or 'frozen'.")
        self.mode = mode
        self.action_agnostic_schema = bool(action_agnostic_schema)
        if self.action_agnostic_schema:
            self.name = "decomposed_proc_v3" if mode == "editable" else "frozen_decomposed_proc_v3"
        else:
            self.name = "editable_proc_v3" if mode == "editable" else "frozen_proc_v3"
        self.review_cadence = max(1, int(review_cadence))
        self.max_words = max(40, int(max_words))
        self.factual = FactualLedgerV3Harness(recent_k=recent_k)
        self.self_procedure: str = sanitize_procedure_text(
            initial_procedure,
            max_words=self.max_words,
            strip_action_bindings=self.action_agnostic_schema,
        )
        self.procedure_version: int = 1 if self.self_procedure else 0
        self.procedure_history: list[dict[str, Any]] = []

    def set_mode(self, mode: str) -> None:
        if mode not in ("editable", "frozen"):
            raise ValueError(f"Unsupported procedure mode '{mode}'.")
        self.mode = mode
        if self.action_agnostic_schema:
            self.name = "decomposed_proc_v3" if mode == "editable" else "frozen_decomposed_proc_v3"
        else:
            self.name = "editable_proc_v3" if mode == "editable" else "frozen_proc_v3"

    def snapshot_state(self) -> dict[str, Any]:
        """Return verbatim active procedure state BEFORE an episode executes."""
        return {
            "harness_name": self.name,
            "mode": self.mode,
            "action_agnostic_schema": self.action_agnostic_schema,
            "procedure_version": self.procedure_version,
            "active_self_procedure": self.self_procedure,
            "n_reviews_logged": len(self.procedure_history),
        }

    def build_context(self, spec: PublicTaskSpec) -> str:
        fact_block = self.factual.build_context(spec)
        proc_body = (
            self.self_procedure
            if self.self_procedure.strip()
            else "(Blank — no procedure written yet; rely on public rules and factual observations.)"
        )
        proc_block = (
            f"\nSELF-AUTHORED OPERATING PROCEDURE (version v{self.procedure_version}, written by your past self):\n"
            f"{proc_body}\n"
        )
        return fact_block + proc_block

    def _run_procedure_review(self, env_name: str, episode_idx: int, client: MeteredLLMClient) -> dict[str, Any]:
        fact_txt = self.factual.build_context(
            PublicTaskSpec(
                env_name=env_name,
                episode_idx=episode_idx,
                timestamp=episode_idx + 1,
                domain_prompt="",
                task_prompt="",
                tool_schemas=[],
            )
        )
        schema_rule = (
            "\nSCHEMA SEPARATION RULE: Keep `self_procedure` strictly action-agnostic — do NOT write concrete "
            "strategy IDs (`STRAT_S1..STRAT_S6` or `S1..S6`) or specific session numbers into `self_procedure`. "
            "All strategy-specific facts already live in the `Factual Observation Ledger`; `self_procedure` must "
            "express only general decision principles for how to read the ledger and manage holding, rebalancing, "
            "and paid probing.\n"
            if self.action_agnostic_schema
            else ""
        )
        sys_p = (
            "You are reviewing your own decision-making on AlphaDesk-v3.\n"
            "Public environment mechanics:\n"
            "- Objective: maximize cumulative net session score across sessions.\n"
            f"- Compliance: non-negative weights across {', '.join(STRATEGIES_V3)} summing to 1.0, each <= {MAX_WEIGHT_V3}.\n"
            f"- Costs: rebalancing pays {TURNOVER_FULL_FLIP_PTS_V3:.1f} * 0.5 * L1_turnover pts; "
            f"`probe_strategy_execution` is optional (max 1/session, -{PROBE_FEE_V3:.1f} pts) and reports the probed "
            "strategy's live yield plus anonymized desk-wide mean and max strategy yields.\n"
            + schema_rule
            + f"\nWrite or revise `self_procedure` (<= {self.max_words} words) as an operating note to your future self "
            "for upcoming sessions (covering when to hold, rebalance, or pay -10 pts to probe, and when to keep your points).\n"
            'Respond ONLY with JSON: {"self_procedure": "...", "edit_rationale": "..."}.'
        )
        usr_p = (
            f"Current `self_procedure` (version v{self.procedure_version}):\n"
            f"{self.self_procedure or '(blank)'}\n\n"
            f"{fact_txt}\n"
            "Return JSON with `self_procedure` and `edit_rationale`."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=env_name,
            method=f"{self.name}_review",
            mock_responder=lambda _s, _u: {
                "self_procedure": (
                    f"Mock procedure (step {episode_idx + 1}): check brief, hold verified winners when calm, "
                    "and probe selectively when outcomes change."
                ),
                "edit_rationale": "Mock periodic procedure review.",
            },
        )
        parsed = resp.parsed_json or {}
        proposed = sanitize_procedure_text(
            parsed.get("self_procedure", ""),
            max_words=self.max_words,
            strip_action_bindings=self.action_agnostic_schema,
        )
        rationale = sanitize_procedure_text(parsed.get("edit_rationale", ""), max_words=60, max_chars=360)

        applied = False
        if self.mode == "editable" and proposed:
            if proposed != self.self_procedure:
                self.self_procedure = proposed
                self.procedure_version += 1
                applied = True
            elif self.procedure_version == 0:
                self.self_procedure = proposed
                self.procedure_version = 1
                applied = True

        entry = {
            "after_episode": episode_idx + 1,
            "mode": self.mode,
            "applied": applied,
            "active_procedure_version": self.procedure_version,
            "active_self_procedure": self.self_procedure,
            "proposed_self_procedure": proposed,
            "edit_rationale": rationale,
            "usd": round(resp.usd, 6),
        }
        self.procedure_history.append(entry)
        return {
            "reflection_calls": 1,
            "reflection_usd": resp.usd,
            "alarm_fired": applied,
            "procedure_review": entry,
        }

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self.factual.on_episode_complete(record, client)
        if len(self.factual.history) % self.review_cadence != 0:
            return {
                "reflection_calls": 0,
                "reflection_usd": 0.0,
                "alarm_fired": False,
                "active_procedure_version": self.procedure_version,
            }
        return self._run_procedure_review(record.env_name, record.episode_idx, client)


def distill_domain_agnostic_procedure(
    *,
    client: MeteredLLMClient,
    source_procedures: list[str],
    source_ledger_excerpt: str,
    max_words: int = 140,
) -> dict[str, Any]:
    """Distill a frozen, domain-agnostic epistemic checking/revision procedure BEFORE held-out SLB transfer."""
    sys_p = (
        "You are distilling a reusable, DOMAIN-AGNOSTIC epistemic adaptation procedure (<= 140 words) from an agent's "
        "experience in a non-stationary sequential decision environment.\n"
        "REQUIREMENTS:\n"
        "1. The procedure will be frozen and given verbatim to an agent starting with EMPTY task memory (`M_0 = empty`) "
        "in a completely different, unseen domain where documented baseline rules exist alongside unannounced policy "
        "shifts, scope-specific overrides, and potential reversals back to earlier regimes.\n"
        "2. Do NOT mention any finance, trading, strategy names (`STRAT_S1..S6`), portfolio weights, spreads, or "
        "volatility terms.\n"
        "3. Express general principles for organizing empirical observations by workflow/context scope, revising "
        "beliefs when recent outcomes contradict prior notes, preserving standard documented rules outside shifted "
        "scopes, and recognizing reversals to earlier regimes.\n"
        'Respond ONLY with JSON: {"domain_agnostic_procedure": "...", "rationale": "..."}.'
    )
    usr_p = (
        "Self-authored procedures developed during source non-stationary runs:\n"
        + json.dumps(source_procedures[-4:], indent=2)
        + "\n\nExcerpt of source factual observation log:\n"
        + str(source_ledger_excerpt)[:1800]
        + "\n\nReturn JSON with `domain_agnostic_procedure` and `rationale`."
    )
    resp = client.generate_json(
        system_prompt=sys_p,
        user_prompt=usr_p,
        environment="cross_domain_transfer_distill",
        method="distill_domain_agnostic_procedure",
        mock_responder=lambda _s, _u: {
            "domain_agnostic_procedure": (
                "Group empirical outcomes by task workflow scope. Obey standard documented rules unless recent "
                "outcomes in that exact scope fail unexpectedly. When repeated failures occur in a scope, supersede "
                "outdated overrides using only recent post-failure evidence, while keeping prior verified regimes "
                "available in case policies revert."
            ),
            "rationale": "Mock domain-agnostic distillation.",
        },
    )
    parsed = resp.parsed_json or {}
    proc_txt = sanitize_procedure_text(parsed.get("domain_agnostic_procedure", ""), max_words=max_words)
    rat_txt = sanitize_procedure_text(parsed.get("rationale", ""), max_words=60, max_chars=360)
    sha256_hex = hashlib.sha256(proc_txt.encode("utf-8")).hexdigest()
    return {
        "domain_agnostic_procedure": proc_txt,
        "rationale": rat_txt,
        "sha256": sha256_hex,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "usd": round(resp.usd, 6),
    }


class SLBFactualLogHarness(EpistemicHarness):
    """`M_0 = empty` Scoped Factual Serving Log for `ServeLearnBench` (`0` reflection calls, `0` synthesized rules)."""

    name = "factual_slb"

    def __init__(self, per_scope_k: int = 4) -> None:
        self.per_scope_k = max(2, int(per_scope_k))
        self.scope_logs: dict[str, list[str]] = {}

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "harness_name": self.name,
            "scope_counts": {k: len(v) for k, v in self.scope_logs.items()},
            "recent_by_scope": {k: list(v[-self.per_scope_k :]) for k, v in self.scope_logs.items()},
        }

    def build_context(self, spec: PublicTaskSpec) -> str:
        if not self.scope_logs:
            return ""
        req_scope = infer_public_slb_scope(spec.task_prompt) if spec.task_prompt else ""
        # When acting on a concrete task (`req_scope` known), inject ONLY that workflow scope's recent factual cases
        # so unrelated workflow logs neither bloat multi-turn acting tokens nor cause cross-scope confusion!
        if req_scope and req_scope in self.scope_logs:
            target_scopes = [req_scope]
        elif req_scope:
            return ""
        else:
            target_scopes = sorted(self.scope_logs.keys())[:4]

        blocks: list[str] = [
            "FACTUAL SERVING OUTCOME LOG (chronological public serving cases observed so far; `M_0 = empty`):"
        ]
        for sc in target_scopes:
            rows = self.scope_logs.get(sc, [])[-self.per_scope_k :]
            if not rows:
                continue
            blocks.append(f"Scope `{sc}` (last {len(rows)} observed cases):")
            for r in rows:
                blocks.append(f"  - {r[:240]}")
        return ("\n" + "\n".join(blocks) + "\n") if len(blocks) > 1 else ""

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        sc = infer_public_slb_scope(record.task_prompt, record.steps)
        self.scope_logs.setdefault(sc, []).append(summarize_public_record(record))
        self.scope_logs[sc] = self.scope_logs[sc][-12:]
        return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False, "scope": sc}


class SLBTransferredProcedureHarness(EpistemicHarness):
    """Zero-Shot Cross-Setup Transfer Arm (`transferred_proc_slb`) for Held-Out `ServeLearnBench`.

    Starts with empty task memory (`M_0 = empty`), `SLBFactualLogHarness`, and a **frozen domain-agnostic epistemic
    procedure** distilled on `AlphaDesk` BEFORE touching `ServeLearnBench` (zero hand-coded Python failure streaks,
    zero hand-coded `shelf` arrays—the LLM itself maintains a bounded empirical notebook guided ONLY by its frozen
    transferred procedure every `review_every` serving tasks!).
    """

    name = "transferred_proc_slb"

    def __init__(
        self,
        *,
        frozen_procedure: str = "",
        frozen_sha256: str = "",
        review_every: int = 4,
        max_notebook_chars: int = 1500,
    ) -> None:
        self.frozen_procedure = sanitize_procedure_text(frozen_procedure, max_words=150, max_chars=1050)
        self.frozen_sha256 = frozen_sha256 or hashlib.sha256(self.frozen_procedure.encode("utf-8")).hexdigest()
        self.review_every = max(2, int(review_every))
        self.max_notebook_chars = int(max_notebook_chars)
        self.factual = SLBFactualLogHarness(per_scope_k=3)
        self.empirical_notebook: str = ""
        self.notebook_version: int = 0
        self.notebook_history: list[dict[str, Any]] = []
        self._seen: int = 0

    def set_frozen_procedure(self, proc_text: str) -> None:
        self.frozen_procedure = sanitize_procedure_text(proc_text, max_words=150, max_chars=1050)
        self.frozen_sha256 = hashlib.sha256(self.frozen_procedure.encode("utf-8")).hexdigest()

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "harness_name": self.name,
            "frozen_procedure_sha256": self.frozen_sha256,
            "frozen_transferred_procedure": self.frozen_procedure,
            "notebook_version": self.notebook_version,
            "active_empirical_notebook": self.empirical_notebook,
        }

    def build_context(self, spec: PublicTaskSpec) -> str:
        parts: list[str] = []
        if self.frozen_procedure:
            parts.append(
                "TRANSFERRED EPISTEMIC ADAPTATION PROCEDURE (frozen prior to this benchmark):\n"
                + self.frozen_procedure
            )
        if self.empirical_notebook.strip():
            parts.append(
                f"AGENT-MAINTAINED EMPIRICAL NOTEBOOK (v{self.notebook_version}, M_0 started empty):\n"
                + self.empirical_notebook[:900]
            )
        else:
            fact_ctx = self.factual.build_context(spec)
            if fact_ctx.strip():
                parts.append(fact_ctx.strip())
        if not parts:
            return ""
        return "\n" + "\n\n".join(parts) + "\n"

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        self.factual.on_episode_complete(record, client)
        self._seen += 1
        if self._seen % self.review_every != 0:
            return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}

        fact_dump = self.factual.build_context(
            PublicTaskSpec(
                env_name=record.env_name,
                episode_idx=record.episode_idx,
                timestamp=record.timestamp,
                domain_prompt="",
                task_prompt="",
                tool_schemas=[],
            )
        )
        sys_p = (
            "You maintain an empirical policy notebook (<= 1400 chars) for an online serving agent starting from "
            "empty task memory (`M_0 = empty`).\n"
            "Follow the TRANSFERRED EPISTEMIC ADAPTATION PROCEDURE below to update your `empirical_notebook` from "
            "the chronological serving outcomes.\n\n"
            f"TRANSFERRED EPISTEMIC ADAPTATION PROCEDURE:\n{self.frozen_procedure or '(none)'}\n\n"
            'Respond ONLY with JSON: {"empirical_notebook": "..."}.'
        )
        usr_p = (
            f"Current `empirical_notebook` (v{self.notebook_version}):\n"
            f"{self.empirical_notebook or '(empty — M_0)'}\n\n"
            f"{fact_dump}\n"
            "Return JSON with updated `empirical_notebook`."
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=record.env_name,
            method="transferred_proc_slb_curate",
            mock_responder=lambda _s, _u: {
                "empirical_notebook": f"Updated scoped notes after {self._seen} serving episodes."
            },
        )
        nb = _CTRL_RE.sub(" ", str((resp.parsed_json or {}).get("empirical_notebook", ""))).strip()
        if nb:
            self.empirical_notebook = nb[: self.max_notebook_chars]
            self.notebook_version += 1
        snap = copy.deepcopy(self.snapshot_state())
        self.notebook_history.append({"after_serving_ep": self._seen, **snap})
        return {
            "reflection_calls": 1,
            "reflection_usd": resp.usd,
            "alarm_fired": False,
            "notebook_version": self.notebook_version,
            "active_empirical_notebook": self.empirical_notebook,
        }
