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
        assert "(step 2)" in ckpt_proc

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

    for reg_name in (
        "editable_proc_v3",
        "frozen_proc_v3",
        "decomposed_proc_v3",
        "frozen_decomposed_proc_v3",
        "sgeh_vbq_v3",
        "factual_slb",
        "transferred_proc_slb",
    ):
        h = make_harness(reg_name)
        assert h is not None


def test_decomposed_procedure_scrubbing_and_vbq_incumbent_verification() -> None:
    """Verify action-agnostic scrubbing strips STRAT_S1..S6 and VBQ verifies incumbent before quarantining."""
    from src.harnesses.procedure_harness import sanitize_procedure_text
    from src.harnesses.sgeh_v3 import DirectiveSGEHV3Harness

    scrubbed = sanitize_procedure_text(
        "In Session #16, hold STRAT_S2 and S3 at 0.55/0.45.",
        strip_action_bindings=True,
    )
    assert "STRAT_S2" not in scrubbed
    assert " S3 " not in f" {scrubbed} "
    assert "Session #16" not in scrubbed

    with tempfile.TemporaryDirectory() as tmp:
        ledger = CostLedger(results_root=Path(tmp), run_id="test_vbq", phase="dev", max_usd=0.05)
        mock_c = MeteredLLMClient(model="mock", ledger=ledger, thinking_budget=0)
        vbq = DirectiveSGEHV3Harness(verify_before_quarantine=True)
        w_inc = normalize_weights_v3({}, "STRAT_S1", "STRAT_S2")

        # Establish calm W1 baseline on STRAT_S1 + STRAT_S2
        for ep in range(5):
            rec = PublicEpisodeRecord(
                env_name="alpha_desk_v3",
                episode_idx=ep,
                timestamp=ep + 1,
                task_prompt="Session",
                steps=[
                    {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}},
                    {"turn": 2, "tool": "probe_strategy_execution", "args": {"strategy": "STRAT_S1" if ep == 0 else "STRAT_S2"}, "obs": {"strategy": "STRAT_S1" if ep == 0 else "STRAT_S2", "live_net_yield_estimate": 91.0 if ep == 0 else 81.0}},
                    {"turn": 3, "tool": "submit_allocation", "args": {"weights": w_inc}, "obs": {"status": "submitted", "gross_score": 86.5, "turnover_fee_deducted": 0.0}},
                ] if ep < 2 else [
                    {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}},
                    {"turn": 2, "tool": "submit_allocation", "args": {"weights": w_inc}, "obs": {"status": "submitted", "gross_score": 86.5, "turnover_fee_deducted": 0.0}},
                ],
                public_score=86.5,
            )
            vbq.on_episode_complete(rec, mock_c)

        b = vbq.controller.buckets["calm_volatility"]
        assert b.primary_strategy == "STRAT_S1"

        # Unannounced calm drop (gross=18.9): VBQ must NOT quarantine immediately; it schedules pending_incumbent_verify!
        drop_rec = PublicEpisodeRecord(
            env_name="alpha_desk_v3",
            episode_idx=5,
            timestamp=6,
            task_prompt="Session",
            steps=[
                {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}},
                {"turn": 2, "tool": "submit_allocation", "args": {"weights": w_inc}, "obs": {"status": "submitted", "gross_score": 18.9, "turnover_fee_deducted": 0.0}},
            ],
            public_score=18.9,
        )
        vbq.on_episode_complete(drop_rec, mock_c)
        assert b.quarantined_strategies == []
        assert b.pending_incumbent_verify == "STRAT_S1"

        # Case A: Post-dip rebound where STRAT_S1 probes at 91.5 -> immediately retained with 0 quarantine!
        import copy as _cp
        vbq_rebound = _cp.deepcopy(vbq)
        reb_rec = PublicEpisodeRecord(
            env_name="alpha_desk_v3",
            episode_idx=6,
            timestamp=7,
            task_prompt="Session",
            steps=[
                {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}},
                {"turn": 2, "tool": "probe_strategy_execution", "args": {"strategy": "STRAT_S1"}, "obs": {"strategy": "STRAT_S1", "live_net_yield_estimate": 91.5, "desk_best_yield_estimate": 91.5}},
                {"turn": 3, "tool": "submit_allocation", "args": {"weights": w_inc}, "obs": {"status": "submitted", "gross_score": 86.6, "turnover_fee_deducted": 0.0}},
            ],
            public_score=76.6,
        )
        m_reb = vbq_rebound.on_episode_complete(reb_rec, mock_c)
        assert m_reb.get("vbq_incumbent_retained") is True
        assert vbq_rebound.controller.buckets["calm_volatility"].quarantined_strategies == []
        assert vbq_rebound.controller.buckets["calm_volatility"].probing_steps_remaining == 0

        # Case B: True reversal where STRAT_S1 probes at 18.2 while desk_best is 93.0 -> now quarantined!
        rev_rec = PublicEpisodeRecord(
            env_name="alpha_desk_v3",
            episode_idx=6,
            timestamp=7,
            task_prompt="Session",
            steps=[
                {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility", "liquidity_status": "orderly_normal"}},
                {"turn": 2, "tool": "probe_strategy_execution", "args": {"strategy": "STRAT_S1"}, "obs": {"strategy": "STRAT_S1", "live_net_yield_estimate": 18.2, "desk_best_yield_estimate": 93.0}},
                {"turn": 3, "tool": "submit_allocation", "args": {"weights": normalize_weights_v3({}, "STRAT_S3", "STRAT_S4")}, "obs": {"status": "submitted", "gross_score": 85.0, "turnover_fee_deducted": 4.0}},
            ],
            public_score=71.0,
        )
        vbq.on_episode_complete(rev_rec, mock_c)
        assert "STRAT_S1" in b.quarantined_strategies


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


def test_public_slb_scope_partitioning_covers_all_banking_l3_prefixes() -> None:
    """Verify 100% of public banking_l3 instructions map to distinct non-general workflow scopes."""
    from src.harnesses.sgeh_slb import ScopedSLBHarness, infer_public_slb_scope

    assert infer_public_slb_scope("Credit desk item LIM-711012 proposes a $3805.69 ceiling") == "banking_limit"
    assert infer_public_slb_scope("Work item LIM-871849: ACC-774095 would take $4307.35") == "banking_limit"
    assert infer_public_slb_scope("Adjudicate card event TXN-824136: Lumen Setting Group") == "banking_card_txn"
    assert infer_public_slb_scope("Payment file TRF-798725 on account ACC-875599: 4 lines totalling $19910") == "banking_batch"
    assert infer_public_slb_scope("Transfer desk item TRF-763157 covers $8033.73") == "banking_transfer"

    h = ScopedSLBHarness()
    h._get_scope("banking_limit").active_rule = "Basic verification max limit is $4000."
    # A card-txn task must NOT receive the banking_limit override!
    ctx_txn = h.build_context(PublicTaskSpec("slb_banking", 1, 1, "", "Adjudicate card event TXN-824136", []))
    assert ctx_txn == ""
    ctx_lim = h.build_context(PublicTaskSpec("slb_banking", 1, 1, "", "Credit desk item LIM-711012", []))
    assert "[banking_limit]" in ctx_lim
