"""Tier-0 Offline ($0.00) Verification Suite for Post-Meeting Priorities 1–3:
  1. Hidden-ambiguity VOI (`hidden_transient_dip` vs `abrupt_reversal`): zero free-brief leak, exact `t_fork`
     realized score/NAV match, paid-probe separation (`desk_best_yield_estimate`), and Bellman DP optimality.
  2. `SelfAuthoredProcedureHarness`: initially blank `self_procedure`, matched-budget `Frozen-Procedure` vs
     `Editable-Procedure` causal isolation, and verbatim pre-episode trace snapshotting.
  3. Cross-domain `SLBTransferredProcedureHarness` & `ScopedSLBHarness` verbatim rule/notebook trace snapshots.
  4. Static security hygiene checks across newly introduced modules.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from src.core.ledger import CostLedger
from src.core.llm_client import MeteredLLMClient
from src.envs.alpha_desk_v3 import (
    AlphaDeskV3ForkEnv,
    AlphaDeskV3ThreeWindowEnv,
    STRATEGIES_V3,
    l1_turnover_half,
)
from src.envs.base import PublicTaskSpec
from src.harnesses import make_harness
from src.harnesses.base import PublicEpisodeRecord
from src.harnesses.procedure_harness import (
    SLBTransferredProcedureHarness,
    SelfAuthoredProcedureHarness,
    distill_domain_agnostic_procedure,
)
from src.harnesses.sgeh_v3 import normalize_weights_v3

ROOT = Path(__file__).resolve().parents[1]


def test_hidden_transient_dip_matches_free_signals_and_separates_only_via_paid_probe() -> None:
    """Verify 0.0 bits of free separation before probing and strong separation via paid `-10 pt` probe."""
    for seed in (601, 602, 603):
        env_dip = AlphaDeskV3ForkEnv(fork_mode="hidden_transient_dip", seed=seed, t_fork=14, h_post=16)
        env_rev = AlphaDeskV3ForkEnv(fork_mode="abrupt_reversal", seed=seed, t_fork=14, h_post=16)

        rmap = env_dip._role_to_strat
        inc_w = normalize_weights_v3({}, rmap["r0"], rmap["r1"])
        env_dip.set_fork_state(inc_w, 118.5)
        env_rev.set_fork_state(inc_w, 118.5)

        # Session `t_fork` (ep = 14): Free brief + unprobed incumbent submission must match 100% bit-for-bit!
        env_dip.start_episode(14)
        env_rev.start_episode(14)
        b_dip_14 = env_dip.step_tool("inspect_desk_brief", {}).observation
        b_rev_14 = env_rev.step_tool("inspect_desk_brief", {}).observation
        assert b_dip_14 == b_rev_14
        assert b_dip_14["liquidity_status"] == "orderly_normal"
        assert b_dip_14["volatility_regime_hint"] == "calm_volatility"

        sub_dip_14 = env_dip.step_tool("submit_allocation", {"weights": inc_w})
        sub_rev_14 = env_rev.step_tool("submit_allocation", {"weights": inc_w})
        assert sub_dip_14.observation == sub_rev_14.observation
        assert sub_dip_14.public_score == sub_rev_14.public_score

        # Session `t_fork + 1` (ep = 15): Free brief is STILL 100% identical!
        env_dip.start_episode(15)
        env_rev.start_episode(15)
        b_dip_15 = env_dip.step_tool("inspect_desk_brief", {}).observation
        b_rev_15 = env_rev.step_tool("inspect_desk_brief", {}).observation
        assert b_dip_15 == b_rev_15

        # Paid `-10 pt` probe on ANY strategy cleanly separates the two worlds via `desk_best_yield_estimate`!
        p_dip = env_dip.step_tool("probe_strategy_execution", {"strategy": rmap["r4"]}).observation
        p_rev = env_rev.step_tool("probe_strategy_execution", {"strategy": rmap["r4"]}).observation
        assert p_dip["probe_fee_deducted"] == 10.0
        assert p_dip["desk_best_yield_estimate"] < 30.0
        assert p_rev["desk_best_yield_estimate"] > 80.0

        # Bellman DP holds `inc_w` with 0 turnover on `hidden_transient_dip`, and reallocates on `abrupt_reversal`!
        assert l1_turnover_half(env_dip._post_dp_oracle["weights_by_ep"][14], inc_w) == 0.0
        assert l1_turnover_half(env_dip._post_dp_oracle["weights_by_ep"][15], inc_w) == 0.0
        assert l1_turnover_half(env_dip._post_dp_oracle["weights_by_ep"][16], inc_w) == 0.0
        assert l1_turnover_half(env_rev._post_dp_oracle["weights_by_ep"][14], inc_w) > 0.80


def test_editable_vs_frozen_procedure_harness_matched_budget_and_snapshots() -> None:
    """Verify blank init, matched review calls on Frozen-Procedure without mutation, and verbatim snapshots."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = CostLedger(results_root=Path(tmp), run_id="test_proc", phase="dev", max_usd=0.05)
        mock_client = MeteredLLMClient(model="mock", ledger=ledger, thinking_budget=0)

        edit_h = SelfAuthoredProcedureHarness(mode="editable", review_cadence=2)
        assert edit_h.self_procedure == ""
        assert edit_h.procedure_version == 0

        for ep in range(2):
            rec = PublicEpisodeRecord(
                env_name="alpha_desk_v3",
                episode_idx=ep,
                timestamp=ep + 1,
                task_prompt="Session",
                steps=[{"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}}],
                public_score=82.0,
            )
            m = edit_h.on_episode_complete(rec, mock_client)

        assert m["reflection_calls"] == 1
        assert edit_h.procedure_version == 1
        ckpt_proc = edit_h.self_procedure
        assert "session 2" in ckpt_proc

        # Fork into Frozen-Procedure at checkpoint!
        froz_h = SelfAuthoredProcedureHarness(mode="frozen", review_cadence=2, initial_procedure=ckpt_proc)
        for ep in range(2, 4):
            snap = froz_h.snapshot_state()
            assert snap["active_self_procedure"] == ckpt_proc
            rec = PublicEpisodeRecord(
                env_name="alpha_desk_v3",
                episode_idx=ep,
                timestamp=ep + 1,
                task_prompt="Session",
                steps=[],
                public_score=20.0,
            )
            m_f = froz_h.on_episode_complete(rec, mock_client)
            m_e = edit_h.on_episode_complete(rec, mock_client)

        # Matched review calls, but `froz_h` stays locked at `ckpt_proc` while `edit_h` advances to `v2`!
        assert m_f["reflection_calls"] == m_e["reflection_calls"] == 1
        assert froz_h.self_procedure == ckpt_proc
        assert edit_h.self_procedure != ckpt_proc
        assert edit_h.procedure_version == 2


def test_three_window_streams_and_cross_domain_transfer_harnesses() -> None:
    """Verify 3-window environment modes and transferred SLB procedure harness snapshots."""
    for sm in AlphaDeskV3ThreeWindowEnv.VALID_STREAM_MODES:
        env = AlphaDeskV3ThreeWindowEnv(stream_mode=sm, seed=601, w1_len=12, w2_len=14, w3_len=14)
        assert env.num_episodes() == 40
        assert env._post_dp_oracle["cumulative_net"] > 500.0

    with tempfile.TemporaryDirectory() as tmp:
        ledger = CostLedger(results_root=Path(tmp), run_id="test_distill", phase="dev", max_usd=0.05)
        mock_client = MeteredLLMClient(model="mock", ledger=ledger, thinking_budget=0)

        dist = distill_domain_agnostic_procedure(
            client=mock_client,
            source_procedures=["Check scope, hold when stable, probe selectively on drop."],
            source_ledger_excerpt="Session #1 calm gross=86.6",
        )
        assert len(dist["sha256"]) == 64
        assert "STRAT_S1" not in dist["domain_agnostic_procedure"]

        slb_h = SLBTransferredProcedureHarness(
            frozen_procedure=dist["domain_agnostic_procedure"],
            frozen_sha256=dist["sha256"],
            review_every=2,
        )
        for i in range(2):
            rec = PublicEpisodeRecord(
                env_name="slb_retail",
                episode_idx=i,
                timestamp=i + 1,
                task_prompt="Return order #101",
                steps=[{"turn": 1, "tool": "return_order", "args": {}, "obs": "ok"}],
                public_score=100.0 if i == 0 else 0.0,
            )
            slb_h.on_episode_complete(rec, mock_client)

        snap = slb_h.snapshot_state()
        assert snap["notebook_version"] == 1
        assert snap["frozen_transferred_procedure"] == dist["domain_agnostic_procedure"]
        ctx = slb_h.build_context(
            PublicTaskSpec("slb_retail", 2, 3, "", "Return order #102", [])
        )
        assert "TRANSFERRED EPISTEMIC ADAPTATION PROCEDURE" in ctx
        assert "AGENT-MAINTAINED EMPIRICAL NOTEBOOK" in ctx

    for reg_name in ("editable_proc_v3", "frozen_proc_v3", "factual_slb", "transferred_proc_slb"):
        h = make_harness(reg_name)
        assert h is not None


def test_new_modules_security_hygiene() -> None:
    """Ensure newly added Python modules contain zero hardcoded secrets, `0.0.0.0` bindings, or shell=True."""
    import re

    checked = [
        ROOT / "src" / "harnesses" / "procedure_harness.py",
        ROOT / "scripts" / "run_editable_procedure_experiment.py",
    ]
    for path in checked:
        txt = path.read_text(encoding="utf-8")
        assert "0.0.0.0" not in txt
        assert "shell=True" not in txt
        assert not re.search(r"AIza[0-9A-Za-z\-_]{20,}", txt)
        assert not re.search(r"\bsk-[0-9A-Za-z]{16,}", txt)
