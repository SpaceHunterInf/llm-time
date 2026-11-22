"""Scoped Epistemic Harness (`ScopedSLBHarness`) for Multi-Window `ServeLearnBench L2/L3` (`banking_l3`, `retail_l3`).

Strict `ServeLearnBench` protocol compliance (`competitors/ServeLearnBench/docs/protocol.md`):
  - Zero access to `window`, `phase`, `task.z`, `task.actions`, or hidden policy notes.
  - Partitions experience by **public workflow scope** inferred strictly from task instructions & public tool calls
    (e.g., `card_txn`, `credit_limit`, `wire_transfer`, `batch_payment` in Banking; `return_exchange`,
    `cancel_modify`, `policy_inquiry` in Retail) so a policy shift in one workflow never corrupts unrelated workflows
    or `Fully-Specified` (`General`) document compliance!
  - Uses per-scope outcome surprise (`score == 0` streaks) + **shelved prior-regime rules (`shelf`)** to handle
    non-monotonic `L3` reversals (`A -> B -> C -> A`) with low reflection overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from src.core.llm_client import MeteredLLMClient
from src.envs.base import PublicTaskSpec
from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import summarize_public_record


def infer_public_slb_scope(task_prompt: str, steps: list[dict[str, Any]] | None = None) -> str:
    """Infer coarse workflow scope strictly from public task prompt & public tool names."""
    txt = str(task_prompt or "").lower()
    if steps:
        txt += " " + " ".join(str(s.get("tool", "")).lower() for s in steps)

    # Banking workflows
    if "decide_batch" in txt or "payment batch" in txt or "batch of" in txt:
        return "banking_batch"
    if "credit limit" in txt or "limit increase" in txt or "credit-limit" in txt:
        return "banking_limit"
    if "outbound transfer" in txt or "wire" in txt or "payee" in txt or "transfer request" in txt:
        return "banking_transfer"
    if "card transaction" in txt or "merchant" in txt or "pending card" in txt or "authorization" in txt:
        return "banking_card_txn"

    # Retail workflows
    if "exchange" in txt or "return" in txt or "refund" in txt:
        return "retail_return_exchange"
    if "cancel" in txt or "modify" in txt or "shipping address" in txt or "payment method" in txt:
        return "retail_cancel_modify"
    return "general_workflow"


@dataclass
class SLBScopeState:
    scope_name: str
    active_rule: str = ""
    shelf: list[str] = field(default_factory=list)
    recent_passed: list[str] = field(default_factory=list)
    recent_failed: list[str] = field(default_factory=list)
    total_seen: int = 0
    post_update_seen: int = 0
    consecutive_failures: int = 0
    recent_scores: list[float] = field(default_factory=list)
    alarms_fired: int = 0


class ScopedSLBHarness(EpistemicHarness):
    """Scope-partitioned, surprise-gated harness with `L3` regime shelf for `ServeLearnBench`."""

    name = "sgeh_slb"

    def __init__(self, *, min_cases_for_init: int = 4) -> None:
        self.min_cases_for_init = max(2, int(min_cases_for_init))
        self.scopes: dict[str, SLBScopeState] = {}

    def _get_scope(self, name: str) -> SLBScopeState:
        if name not in self.scopes:
            self.scopes[name] = SLBScopeState(scope_name=name)
        return self.scopes[name]

    def build_context(self, spec: PublicTaskSpec) -> str:
        active_rules = [
            f"- [{sc.scope_name}]: {sc.active_rule}"
            for sc in self.scopes.values()
            if sc.active_rule.strip()
        ]
        if not active_rules:
            return ""
        # If `spec.task_prompt` is non-empty, prioritize matching scope first while showing concise cross-scope overrides
        req_scope = infer_public_slb_scope(spec.task_prompt) if spec.task_prompt else ""
        ordered_rules = sorted(
            active_rules,
            key=lambda ln: (0 if (req_scope and f"[{req_scope}]" in ln) else 1, ln),
        )
        return (
            "\nEMPIRICAL SERVING POLICY REFINEMENTS (Scoped by workflow — ALWAYS call `read_docs` and obey standard "
            "documented rules on everything NOT explicitly overridden by a matching empirical observation below):\n"
            + "\n".join(ordered_rules[:6])
            + "\n"
        )

    def _synthesize_scope_rule(
        self, sc: SLBScopeState, env_name: str, client: MeteredLLMClient
    ) -> tuple[int, float]:
        sys_p = (
            "You infer concise empirical policy overrides for ONE specific customer/banking workflow scope from "
            "serving outcomes (`Score=100` vs `Score=0`).\n"
            "CRITICAL RULES:\n"
            "1. Standard documented rules in `read_docs` still hold unless contradicted by recent `Score=0` vs `Score=100` outcomes.\n"
            "2. Check `shelved_past_regime_rules` first: if recent `Score=100` & `Score=0` cases match an earlier shelved rule "
            "(policy reversal back to an earlier regime), restore or adapt that exact rule!\n"
            "3. State concrete numeric thresholds, category caps, or refusal/reason codes supported by the traces in <= 320 chars. "
            "Never guess unsupported restrictions that could break standard cases.\n"
            "Respond ONLY with JSON: {\"scope_rule\": \"...\"}."
        )
        usr_p = json.dumps(
            {
                "workflow_scope": sc.scope_name,
                "previous_active_rule": sc.active_rule or "(none yet)",
                "shelved_past_regime_rules": sc.shelf[-4:],
                "recent_PASSED_cases_score_100": sc.recent_passed[-5:],
                "recent_FAILED_cases_score_0": sc.recent_failed[-5:],
            },
            indent=2,
        )
        resp = client.generate_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            environment=env_name,
            method="sgeh_slb_consolidate",
            mock_responder=lambda _s, _u: {
                "scope_rule": f"Verified {sc.scope_name} thresholds from {len(sc.recent_passed)} passed / {len(sc.recent_failed)} failed cases."
            },
        )
        rule_txt = str((resp.parsed_json or {}).get("scope_rule", "")).strip()
        if rule_txt:
            sc.active_rule = rule_txt[:360]
        sc.post_update_seen = 0
        sc.consecutive_failures = 0
        return 1, resp.usd

    def on_episode_complete(self, record: PublicEpisodeRecord, client: MeteredLLMClient) -> dict[str, Any]:
        sc_name = infer_public_slb_scope(record.task_prompt, record.steps)
        sc = self._get_scope(sc_name)
        sc.total_seen += 1
        sc.post_update_seen += 1

        summary_line = summarize_public_record(record)
        passed = float(record.public_score) >= 99.0
        sc.recent_scores.append(100.0 if passed else 0.0)
        sc.recent_scores = sc.recent_scores[-6:]

        if passed:
            sc.recent_passed.append(summary_line)
            sc.recent_passed = sc.recent_passed[-6:]
            sc.consecutive_failures = 0
        else:
            sc.recent_failed.append(summary_line)
            sc.recent_failed = sc.recent_failed[-6:]
            sc.consecutive_failures += 1

        # Trigger 1: Initial rule acquisition for this scope once enough contrastive evidence arrives
        if not sc.active_rule and sc.total_seen >= self.min_cases_for_init and sc.recent_failed:
            c_cnt, c_usd = self._synthesize_scope_rule(sc, record.env_name, client)
            return {"reflection_calls": c_cnt, "reflection_usd": c_usd, "alarm_fired": False, "scope": sc_name}

        # Trigger 2: Scope-local surprise alarm (2 consecutive failures OR >=3 failures in last 4 scope cases)
        recent_4 = sc.recent_scores[-4:]
        fail_burst = len(recent_4) >= 3 and sum(1 for x in recent_4 if x < 50.0) >= 3
        if sc.active_rule and sc.post_update_seen >= 2 and (sc.consecutive_failures >= 2 or fail_burst):
            if sc.active_rule not in sc.shelf:
                sc.shelf.append(sc.active_rule)
            # Keep only post-shift evidence in the active contrastive buffers so stale pre-shift cases don't pollute synthesis
            sc.recent_passed = sc.recent_passed[-2:]
            sc.recent_failed = sc.recent_failed[-3:]
            sc.alarms_fired += 1
            c_cnt, c_usd = self._synthesize_scope_rule(sc, record.env_name, client)
            return {"reflection_calls": c_cnt, "reflection_usd": c_usd, "alarm_fired": True, "scope": sc_name}

        return {"reflection_calls": 0, "reflection_usd": 0.0, "alarm_fired": False}
