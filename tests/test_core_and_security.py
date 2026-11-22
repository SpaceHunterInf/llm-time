"""Tier-0 ($0.00) unit tests for environments, cost ledger, harnesses, causal surgery, and dashboard security."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.run_causal_fork_v2 import run_causal_experiment
from scripts.run_iter1c_score_anchor import load_saved_iter1b_checkpoint, run_iter1c_experiment
from scripts.run_iter2_online_matrix import run_iter2_batch
from src.core.ledger import BudgetExceededError, CostLedger, mask_email_for_display
from src.core.llm_client import MeteredLLMClient
from src.dashboard.server import create_dashboard_server
from src.envs import AlphaDeskV2Env, make_env
from src.harnesses.base import PublicEpisodeRecord
from src.harnesses.scoped_cards import ScopedCardBookHarness, sanitize_scope_text
from src.runner import _build_acting_system_prompt, run_lifetime
from src.training.act3_lodo import LODO_FOLDS, build_reference_sft_examples_for_fold, compute_act3_grpo_reward


@pytest.mark.parametrize(
    ("stat_key", "drift_key", "prefix_len"),
    [
        ("alpha_desk:stationary", "alpha_desk:gradual_decay", 6),
        ("alpha_desk:stationary", "alpha_desk:regime_cycle", 6),
        ("life_companion:stationary", "life_companion:lifecycle", 6),
        ("coalition_ops:stationary", "coalition_ops:reversal", 6),
        ("alpha_desk_v2:stationary", "alpha_desk_v2:abrupt_reversal", 12),
        ("alpha_desk_v2:stationary", "alpha_desk_v2:gradual_decay", 12),
        ("alpha_desk_v2:stationary", "alpha_desk_v2:off_scope_shock", 12),
    ],
)
def test_mode_free_interface_and_preshift_prefix_identity_all_domains(
    stat_key: str, drift_key: str, prefix_len: int
) -> None:
    """Learner-visible task specs and pre-shift observations must be bit-identical and mode-free across all domains."""
    e_stat = make_env(stat_key, horizon=36, seed=101)
    e_drift = make_env(drift_key, horizon=36, seed=101)

    first_tool = {
        "alpha_desk": "inspect_market_tape",
        "alpha_desk_v2": "inspect_desk_brief",
        "life_companion": "read_user_dossier",
        "coalition_ops": "inspect_dispatch_ledger",
    }[stat_key.split(":")[0]]

    for ep in range(prefix_len):
        s1 = e_stat.start_episode(ep)
        s2 = e_drift.start_episode(ep)
        assert s1 == s2
        assert ":" not in s1.env_name
        for mode_tok in ("stationary", "decay", "reversal", "lifecycle", "cycle", "shock", "opportunity"):
            assert mode_tok not in s1.env_name.lower()

        o1 = e_stat.step_tool(first_tool, {})
        o2 = e_drift.step_tool(first_tool, {})
        assert o1.observation == o2.observation
        low_obs = json.dumps(o1.observation).lower()
        assert "t_star" not in low_obs
        assert "abrupt_reversal" not in low_obs


def test_opportunity_rise_zero_weight_incumbent_distribution_identity() -> None:
    """Under `opportunity_rise`, incumbent 0.65 ALPHA + 0.35 BETA portfolio must observe identical rewards pre & post t*."""
    e_base = AlphaDeskV2Env(mode="opportunity_rise", horizon=36, seed=102, t_star_override=12)
    incumbent_w = {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.35, "STRAT_GAMMA": 0.0, "STRAT_DELTA": 0.0}

    for ep in range(12, 24):
        e_pre_equiv = AlphaDeskV2Env(mode="opportunity_rise", horizon=36, seed=102, t_star_override=35)
        e_base.start_episode(ep)
        e_pre_equiv.start_episode(ep)
        e_base.step_tool("inspect_desk_brief", {})
        e_pre_equiv.step_tool("inspect_desk_brief", {})
        out_shift = e_base.step_tool("submit_allocation", {"weights": incumbent_w})
        out_still = e_pre_equiv.step_tool("submit_allocation", {"weights": incumbent_w})
        assert out_shift.public_score == out_still.public_score


def test_concurrent_inflight_budget_reservation(tmp_path: Path) -> None:
    """Concurrent in-flight reservations must block threads before overshooting `max_usd`."""
    ledger = CostLedger(results_root=tmp_path, run_id="test_conc", phase="dev", max_usd=0.010)
    r1 = ledger.acquire_reservation(0.004)
    r2 = ledger.acquire_reservation(0.004)
    with pytest.raises(BudgetExceededError):
        ledger.acquire_reservation(0.004)  # 0.008 + 0.004 = 0.012 > 0.010!
    ledger.release_reservation(r1)
    ledger.release_reservation(r2)
    assert ledger.in_flight_usd == 0.0


def test_scoped_cardbook_per_card_attribution_and_surgery(tmp_path: Path) -> None:
    """Only cards matching scope AND allocated >= 0.30 weight increment stats; utility excision preserves invariants."""
    ledger = CostLedger(results_root=tmp_path, run_id="test_cards", phase="dev", max_usd=1.0)
    client = MeteredLLMClient(model="mock", ledger=ledger)
    h = ScopedCardBookHarness(reflect_every=10)
    h.cards = [
        {
            "card_id": "C1",
            "kind": "invariant_or_hedge",
            "scope_condition": "volatility_regime_hint == 'elevated_volatility'",
            "target_strategy": "STRAT_DELTA",
            "recommended_weights": {
                "STRAT_ALPHA": 0.05,
                "STRAT_BETA": 0.05,
                "STRAT_GAMMA": 0.25,
                "STRAT_DELTA": 0.65,
            },
            "uses": 3,
            "mean_score": 84.0,
        },
        {
            "card_id": "C2",
            "kind": "calm_utility",
            "scope_condition": "volatility_regime_hint == 'calm_volatility'",
            "target_strategy": "STRAT_ALPHA",
            "recommended_weights": {
                "STRAT_ALPHA": 0.65,
                "STRAT_BETA": 0.25,
                "STRAT_GAMMA": 0.05,
                "STRAT_DELTA": 0.05,
            },
            "uses": 5,
            "mean_score": 76.0,
        },
    ]
    rec = PublicEpisodeRecord(
        env_name="alpha_desk_v2",
        episode_idx=0,
        timestamp=1,
        task_prompt="Session #1",
        steps=[
            {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility"}},
            {"turn": 2, "tool": "submit_allocation", "args": {"weights": {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.35}}, "obs": {"score": 88.0}},
        ],
        public_score=88.0,
    )
    h.on_episode_complete(rec, client)
    assert h.cards[0]["uses"] == 3 and h.cards[0]["mean_score"] == 84.0  # High-vol hedge untouched on calm episode!
    assert h.cards[1]["uses"] == 6 and h.cards[1]["mean_score"] == 78.0  # Attributed calm card updated ((380 + 88)/6)!

    sham, _ = h.make_sham_edited_copy(client, env_name="alpha_desk_v2")
    excised = sham.make_utility_excised_from_sham()
    assert len(excised.cards) == 2
    assert excised.cards[0] == sham.cards[0]
    assert excised.cards[1]["scope_condition"] == sham.cards[1]["scope_condition"]
    assert excised.cards[1]["target_strategy"] is None
    assert excised.cards[1]["uses"] == 0 and excised.cards[1]["mean_score"] is None
    assert all(v is None for v in excised.cards[1]["recommended_weights"].values())


def test_iteration_1b_single_variable_prompt_isolation(tmp_path: Path) -> None:
    """Assert actor prompts at t* and curator prompts at t*+2 vary ONLY calm allocation weights & stats."""
    assert "prob" not in sanitize_scope_text("calm_volatility without probing or probe_strategy_execution").lower()

    ledger = CostLedger(results_root=tmp_path, run_id="test_iso", phase="dev", max_usd=1.0)
    client = MeteredLLMClient(model="mock", ledger=ledger)
    env = AlphaDeskV2Env(mode="abrupt_reversal", horizon=40, seed=102, t_star_override=13)
    base = ScopedCardBookHarness(reflect_every=3)

    # Simulate 13 pre-shift episodes (13 % 3 != 0, testing non-multiple-of-3 fork offset!)
    for ep in range(13):
        is_hv = ep % 4 == 2
        w = (
            {"STRAT_ALPHA": 0.05, "STRAT_BETA": 0.05, "STRAT_GAMMA": 0.25, "STRAT_DELTA": 0.65}
            if is_hv
            else {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.25, "STRAT_GAMMA": 0.05, "STRAT_DELTA": 0.05}
        )
        rec = PublicEpisodeRecord(
            env_name="alpha_desk_v2",
            episode_idx=ep,
            timestamp=ep + 1,
            task_prompt=f"Session #{ep + 1}",
            steps=[
                {
                    "turn": 1,
                    "tool": "inspect_desk_brief",
                    "args": {},
                    "obs": {"volatility_regime_hint": "elevated_volatility" if is_hv else "calm_volatility"},
                },
                {"turn": 2, "tool": "submit_allocation", "args": {"weights": w}, "obs": {"score": 84.0 if is_hv else 76.0}},
            ],
            public_score=84.0 if is_hv else 76.0,
        )
        base.on_episode_complete(rec, client, force_curate=(ep == 12))

    base.set_post_fork_schedule()
    h_ret = base.clone()
    h_sham, _ = base.make_sham_edited_copy(client, env_name="alpha_desk_v2")
    h_exc = h_sham.make_utility_excised_from_sham()

    assert len(h_ret._history) == len(h_sham._history) == len(h_exc._history) == 13
    assert h_ret.fork_offset == h_sham.fork_offset == h_exc.fork_offset == 13

    spec_t = env.start_episode(13)
    for arm_h in (h_ret, h_sham, h_exc):
        ctx = arm_h.build_context(spec_t).lower()
        assert "probe" not in ctx and "probing" not in ctx

    p_sham = _build_acting_system_prompt(spec_t.domain_prompt, spec_t.tool_schemas, h_sham.build_context(spec_t))
    p_exc = _build_acting_system_prompt(spec_t.domain_prompt, spec_t.tool_schemas, h_exc.build_context(spec_t))

    sham_lines = p_sham.splitlines()
    exc_lines = p_exc.splitlines()
    assert len(sham_lines) == len(exc_lines)
    diff_indices = [i for i, (a, b) in enumerate(zip(sham_lines, exc_lines)) if a != b]
    assert len(diff_indices) == 1, f"Expected exactly 1 differing line (calm card), got {diff_indices}"
    d_idx = diff_indices[0]
    assert "[calm_utility |" in sham_lines[d_idx] and "[calm_utility |" in exc_lines[d_idx]
    # Verify identical scope condition in that line
    assert sham_lines[d_idx].split("]: IF ")[1].split(" -> ")[0] == exc_lines[d_idx].split("]: IF ")[1].split(" -> ")[0]

    # Feed identical post-fork episodes t*=13, 14, 15 and check curation schedule + curator inputs at t*+2 (ep=15)
    for post_ep in (13, 14, 15):
        post_rec = PublicEpisodeRecord(
            env_name="alpha_desk_v2",
            episode_idx=post_ep,
            timestamp=post_ep + 1,
            task_prompt=f"Session #{post_ep + 1}",
            steps=[
                {"turn": 1, "tool": "inspect_desk_brief", "args": {}, "obs": {"volatility_regime_hint": "calm_volatility"}},
                {
                    "turn": 2,
                    "tool": "submit_allocation",
                    "args": {"weights": {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.35}},
                    "obs": {"score": 28.0},
                },
            ],
            public_score=28.0,
        )
        for arm_h in (h_ret, h_sham, h_exc):
            arm_h._history.append(post_rec)
            assert arm_h.should_curate_now() is (post_ep == 15)

    sys_s, usr_s = h_sham.curate_input_messages()
    sys_e, usr_e = h_exc.curate_input_messages()
    assert sys_s == sys_e
    trace_marker = "Recent 3 session traces:\n"
    assert usr_s.split(trace_marker)[1] == usr_e.split(trace_marker)[1]
    cards_s = json.loads(usr_s.split("Current cards:\n")[1].split("\n\nRecent 3")[0])
    cards_e = json.loads(usr_e.split("Current cards:\n")[1].split("\n\nRecent 3")[0])
    assert cards_s[1] == cards_e[1]  # invariant_or_hedge card 100% identical!
    assert cards_s[0]["card_id"] == cards_e[0]["card_id"]
    assert cards_s[0]["kind"] == cards_e[0]["kind"] == "calm_utility"
    assert cards_s[0]["scope_condition"] == cards_e[0]["scope_condition"]
    assert cards_e[0]["target_strategy"] is None and cards_e[0]["uses"] == 0 and cards_e[0]["mean_portfolio_score"] is None


def test_iteration_1c_score_visibility_single_variable_isolation(tmp_path: Path) -> None:
    """Verify Score-Shown vs Score-Hidden forks from saved checkpoint with 0-char length diff and single-field isolation."""
    real_results = Path(__file__).resolve().parents[1] / "results"
    t_star, base_h, _ = load_saved_iter1b_checkpoint(
        results_root=real_results, source_run_id="iter1b_clean_fork_n4", seed=101
    )
    assert t_star == 15
    h_shown = base_h.make_weights_excised_score_shown()
    h_hidden = base_h.make_weights_excised_score_hidden()
    h_retained = base_h.clone()

    env = AlphaDeskV2Env(mode="abrupt_reversal", horizon=40, seed=101, t_star_override=t_star)
    spec = env.start_episode(t_star)
    p_shown = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_shown.build_context(spec))
    p_hidden = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_hidden.build_context(spec))
    p_retained = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_retained.build_context(spec))

    assert "limit 1 probe per session" in p_shown and "limit 1 probe per session" in p_hidden
    assert len(p_shown) == len(p_hidden)  # '72.9' vs 'null' -> exact 0-char length diff!
    sh_lines = p_shown.splitlines()
    hi_lines = p_hidden.splitlines()
    diffs = [i for i, (a, b) in enumerate(zip(sh_lines, hi_lines)) if a != b]
    assert diffs == [i for i, l in enumerate(sh_lines) if "[calm_utility |" in l]
    # Replacing `mean_portfolio_score=72.9` with `mean_portfolio_score=null` makes the two prompts byte-identical!
    assert p_shown.replace("mean_portfolio_score=72.9", "mean_portfolio_score=null") == p_hidden
    assert "mean_portfolio_score=72.9" in p_retained

    # Verify curator input at t*+2 also differs ONLY in `mean_portfolio_score`
    cs_sh, cu_sh = h_shown.curate_input_messages()
    cs_hi, cu_hi = h_hidden.curate_input_messages()
    assert cs_sh == cs_hi
    assert cu_sh.replace('"mean_portfolio_score": 72.9', '"mean_portfolio_score": null') == cu_hi

    agg = run_iter1c_experiment(
        run_id="mock_1c",
        seeds=[101],
        model="mock",
        post_window=4,
        max_usd=1.0,
        workers=2,
        thinking_budget=0,
        results_root=tmp_path,
        dotenv_path=None,
        source_run_id=None,
    )
    assert agg["total_run_usd"] == 0.0
    assert "drift_score_shown_minus_score_hidden" in agg["iter1c_contrasts_mean_and_95ci"]


def test_counterbalanced_latin_square_and_iter2_offline(tmp_path: Path) -> None:
    """Verify 1:1:1:1 strategy rotation, 50/50 hedge-overlap balance, prefix identity, and $0.00 Iter 2 runner."""
    inc_set = set()
    overlaps = []
    for s in (201, 202, 203, 204):
        e_stat = AlphaDeskV2Env(mode="stationary", horizon=36, seed=s, counterbalance=True)
        e_rev = AlphaDeskV2Env(mode="abrupt_reversal", horizon=36, seed=s, counterbalance=True)
        e_opp = AlphaDeskV2Env(mode="opportunity_rise", horizon=36, seed=s, counterbalance=True)
        e_ops = AlphaDeskV2Env(mode="opportunity_stationary", horizon=36, seed=s, counterbalance=True)
        rm = e_stat.private_role_map()
        inc_set.add(rm["inc_1"])
        overlaps.append(rm["hedge_overlaps_calm_winner"])

        for ep in range(e_stat.t_star):
            assert e_stat.start_episode(ep) == e_rev.start_episode(ep)
            assert e_opp.start_episode(ep) == e_ops.start_episode(ep)
            assert e_stat.step_tool("inspect_desk_brief", {}).observation == e_rev.step_tool("inspect_desk_brief", {}).observation
            w_ref = rm["reference_incumbent_weights"]
            assert e_opp.step_tool("submit_allocation", {"weights": w_ref}).public_score == e_ops.step_tool("submit_allocation", {"weights": w_ref}).public_score

        # Post-t* opportunity_rise still keeps reference incumbent distribution identical!
        for ep in range(e_opp.t_star, e_opp.t_star + 4):
            e_opp.start_episode(ep)
            e_ops.start_episode(ep)
            e_opp.step_tool("inspect_desk_brief", {})
            e_ops.step_tool("inspect_desk_brief", {})
            w_ref = rm["reference_incumbent_weights"]
            assert e_opp.step_tool("submit_allocation", {"weights": w_ref}).public_score == e_ops.step_tool("submit_allocation", {"weights": w_ref}).public_score

    assert len(inc_set) == 4  # Exact 1:1:1:1 balance across all 4 strategy names!
    assert overlaps.count(True) == 2 and overlaps.count(False) == 2  # Exact 50/50 hedge-overlap balance!

    rep = run_iter2_batch(
        run_id="mock_iter2",
        seeds=[201],
        model="mock",
        post_window=4,
        max_usd=1.0,
        workers=2,
        thinking_budget=0,
        results_root=tmp_path,
        dotenv_path=None,
    )
    assert rep["total_run_usd"] == 0.0
    assert rep["per_seed"][0]["modes"]["off_scope_shock"]["sgeh_hazard"]["alarms_fired"] == 0


def test_act3_lodo_zero_holdout_leakage_and_proper_brier_reward() -> None:
    """Verify each LODO fold strictly excludes its held-out domain, matches live tool schemas, and uses proper Brier scoring."""
    env_map = {
        "alpha_desk_v2": "alpha_desk_v2:abrupt_reversal",
        "life_companion": "life_companion:lifecycle",
        "coalition_ops": "coalition_ops:reversal",
    }
    for fold_name, cfg in LODO_FOLDS.items():
        rows = build_reference_sft_examples_for_fold(fold_name, n_seeds=4)
        assert len(rows) == 8
        for r in rows:
            dom = r["domain"]
            assert dom != cfg["eval_domain"]
            assert dom in cfg["train_domains"]
            comp = json.loads(r["messages"][-1]["content"])
            check_env = make_env(env_map[dom], horizon=18, seed=500)
            spec = check_env.start_episode(0)
            valid_tools = {t["name"] for t in spec.tool_schemas}
            assert comp["tool"] in valid_tools
            step_out = check_env.step_tool(comp["tool"], comp["args"])
            assert "error" not in step_out.observation

    r_calibrated = compute_act3_grpo_reward(
        arm="arm_c_epistemic_brier_grpo",
        normalized_utility=0.80,
        invariant_violation=False,
        predicted_p_ref_high=0.05,
        actual_ref_event_high=0,
        probed_already_falsified_incumbent=False,
    )
    r_overconf = compute_act3_grpo_reward(
        arm="arm_c_epistemic_brier_grpo",
        normalized_utility=0.80,
        invariant_violation=False,
        predicted_p_ref_high=0.95,
        actual_ref_event_high=0,
        probed_already_falsified_incumbent=True,
    )
    assert r_calibrated.total_reward > r_overconf.total_reward


def test_causal_fork_runner_offline_mock(tmp_path: Path) -> None:
    agg = run_causal_experiment(
        run_id="mock_fork",
        seeds=[101],
        model="mock",
        post_window=4,
        max_usd=1.0,
        workers=2,
        thinking_budget=128,
        results_root=tmp_path,
        dotenv_path=None,
    )
    assert agg["n_seeds"] == 1
    assert agg["total_run_usd"] == 0.0
    assert "drift_excised_minus_sham" in agg["contrasts_mean_and_95ci"]


@pytest.mark.parametrize("harness_name", ["blind", "window", "srn", "skillbook", "scoped_cards", "sgeh"])
def test_offline_mock_lifetimes_v2(tmp_path: Path, harness_name: str) -> None:
    ledger = CostLedger(results_root=tmp_path, run_id="mock_v2", phase="dev", max_usd=1.0)
    summary, logs = run_lifetime(
        env_spec="alpha_desk_v2:abrupt_reversal",
        harness_name=harness_name,
        model="mock",
        seed=101,
        horizon=6,
        ledger=ledger,
    )
    assert summary.episodes == 6
    assert len(logs) == 6
    assert summary.invariant_violation_rate == 0.0


def test_alphadesk_v3_multishift_stateful_dp_and_shelf_offline(tmp_path: Path) -> None:
    """Verify AlphaDesk-v0.3 6-way rotation, pre-W1 prefix identity, stateful L1 turnover, DP oracle, and $0.00 mock runner."""
    from scripts.run_alphadesk_v3_matrix import run_v3_batch
    from src.envs.alpha_desk_v3 import AlphaDeskV3Env, STRATEGIES_V3

    r0_set = set()
    partial_first_flags = []
    for s in range(401, 407):
        e_stat = AlphaDeskV3Env(mode="stationary", horizon=84, seed=s)
        e_t1 = AlphaDeskV3Env(mode="multi_abrupt_recurrence", horizon=84, seed=s)
        e_t2 = AlphaDeskV3Env(mode="mixed_decay_opportunity", horizon=84, seed=s)
        meta = e_t1.private_schedule_metadata()
        r0_set.add(meta["role_to_strat"]["r0"])
        partial_first_flags.append(meta["partial_first"])
        assert len(meta["window_ranges"]) == 6
        assert len(meta["shift_boundaries"]) == 5

        # Mode-free public interface + bit-identical W1 prefix between Stationary and Track 1!
        w1_end = meta["shift_boundaries"][0]
        w_test = {k: (0.55 if i == 0 else (0.45 if i == 1 else 0.0)) for i, k in enumerate(STRATEGIES_V3)}
        for ep in range(w1_end):
            sp_s = e_stat.start_episode(ep)
            sp_1 = e_t1.start_episode(ep)
            sp_2 = e_t2.start_episode(ep)
            assert sp_s.env_name == "alpha_desk_v3" and ":" not in sp_s.env_name
            assert sp_s == sp_1 == sp_2
            b_s = e_stat.step_tool("inspect_desk_brief", {}).observation
            b_1 = e_t1.step_tool("inspect_desk_brief", {}).observation
            assert b_s == b_1
            # Preview tool must match exact turnover fee without mutating holdings
            prev_chk = e_t1.step_tool("inspect_risk_and_turnover", {"candidate_weights": w_test}).observation
            assert prev_chk["compliant"] is True
            out_s = e_stat.step_tool("submit_allocation", {"weights": w_test})
            out_1 = e_t1.step_tool("submit_allocation", {"weights": w_test})
            assert out_s.public_score == out_1.public_score
            assert abs(out_1.observation["turnover_fee_deducted"] - prev_chk["turnover_fee_pts"]) <= 0.02
            # On ep >= 1, holding identical `w_test` must incur 0.0 turnover!
            if ep >= 1:
                assert out_1.observation["turnover_fee_deducted"] == 0.0

        # Verify Track 1 contains BOTH a full recurrence window AND a partial recurrence window
        reg_tags = {e_t1._calm_role_means_and_label(lo)[1] for lo, _hi in meta["window_ranges"]}
        assert any("full_rec" in t for t in reg_tags)
        assert any("partial_rec" in t for t in reg_tags)

    assert len(r0_set) == 6  # Exact 1:1:1:1:1:1 6-strategy rotation!
    assert partial_first_flags.count(True) == 3 and partial_first_flags.count(False) == 3

    rep = run_v3_batch(
        run_id="mock_v3",
        modes=["multi_abrupt_recurrence", "mixed_decay_opportunity"],
        seeds=[401],
        methods=["skillbook", "scoped_cards_v3", "sgeh_hazard_noshelf", "sgeh_hazard_shelf"],
        horizon=36,
        model="mock",
        max_usd=1.0,
        workers=4,
        thinking_budget=0,
        results_root=tmp_path,
        dotenv_path=None,
    )
    assert rep["total_run_usd"] == 0.0
    assert len(rep["by_mode_summary"]["multi_abrupt_recurrence"]["sgeh_hazard_shelf"]["per_window_utility_pct"]) == 6


def test_dashboard_security_controls(tmp_path: Path) -> None:
    """Verify localhost-only binding, GET-only method allow-list, CSP nonces, traversal defense, and PII masking."""
    assert mask_email_for_display("gs06monitor@gmail.com") == "gs***@gmail.com"
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_dashboard_server(tmp_path, host="0.0.0.0", port=0)

    (tmp_path / ".env").write_text("GEMINI_API_KEY=SUPER_SECRET_DO_NOT_LEAK\n", encoding="utf-8")
    srv = create_dashboard_server(tmp_path, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            csp = resp.headers.get("Content-Security-Policy", "")
            assert "nonce-" in csp
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert b'href="roadmap_dashboard.html"' in resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/roadmap_dashboard.html") as resp:
            original = resp.read()
            assert hashlib.sha256(original).hexdigest() == "bc5adfa1ecff3305a50a4ace4ef87fc965b5447a946a1fdde99c11c373b00103"
            # The original embeds one stylesheet and one script. Both must work with CSP.
            csp = resp.headers["Content-Security-Policy"]
            for tag in (b"style", b"script"):
                block, = re.findall(b"<" + tag + b">(.*?)</" + tag + b">", original, re.DOTALL)
                digest = base64.b64encode(hashlib.sha256(block).digest()).decode("ascii")
                assert f"'sha256-{digest}'" in csp

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/telemetry") as resp:
            body = resp.read().decode("utf-8")
            assert "SUPER_SECRET_DO_NOT_LEAK" not in body
            assert "gs06monitor@gmail.com" not in body

        post_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/telemetry", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(post_req)
        assert exc_info.value.code == 405

        with pytest.raises(urllib.error.HTTPError) as trav_err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/static/../../.env")
        assert trav_err.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_cross_model_autonomy_fork_invariants_offline(tmp_path: Path) -> None:
    """Verify shared W1 fork identity, equal-loss freeze vs. reversal calibration, superset prompt, & $0 mock run."""
    from scripts.run_cross_model_autonomy_fork import (
        build_frozen_arm_states_at_fork,
        generate_shared_w1_acquisition,
        run_autonomy_batch,
    )
    from src.envs.alpha_desk_v3 import AlphaDeskV3ForkEnv

    r0_set = set()
    for sd in range(501, 507):
        acq_hi = generate_shared_w1_acquisition(seed=sd, prefix_family="high_a", t_fork=14, h_post=16)
        r0_set.add(acq_hi["role_to_strat"]["r0"])
        arms = build_frozen_arm_states_at_fork(acq_hi)

        e_rev = AlphaDeskV3ForkEnv(
            fork_mode="abrupt_reversal",
            seed=sd,
            t_fork=14,
            h_post=16,
            initial_post_holdings=acq_hi["fork_holdings"],
        )
        e_frz = AlphaDeskV3ForkEnv(
            fork_mode="freeze_matched_loss",
            seed=sd,
            t_fork=14,
            h_post=16,
            initial_post_holdings=acq_hi["fork_holdings"],
        )
        e_rev.set_fork_state(acq_hi["fork_holdings"], acq_hi["fork_nav"])
        e_frz.set_fork_state(acq_hi["fork_holdings"], acq_hi["fork_nav"])

        spec_rev = e_rev.start_episode(14)
        spec_frz = e_frz.start_episode(14)
        assert spec_rev.domain_prompt == spec_frz.domain_prompt

        # Verify Factual-Ledger has ZERO prescriptive keywords and is an exact prefix of Directive-SGEH!
        fact_ctx = arms["factual_ledger_v3"].build_context(spec_rev)
        dir_ctx = arms["sgeh_directive_v3"].build_context(spec_rev)
        assert dir_ctx.startswith(fact_ctx)
        assert len(dir_ctx) > len(fact_ctx)
        assert "probe_strategy_execution" not in fact_ctx
        assert "recommended_weights" not in fact_ctx
        assert "LIQUIDITY SAFEGUARD" not in fact_ctx

        # Verify equal apparent loss (~18.9 vs 19.0) when holding W1 incumbent at t_fork = 14!
        m_rev, _, _ = e_rev._expected_payoffs(14)
        m_frz, _, _ = e_frz._expected_payoffs(14)
        loss_rev = sum(acq_hi["fork_holdings"][s] * m_rev[s] for s in acq_hi["fork_holdings"])
        loss_frz = sum(acq_hi["fork_holdings"][s] * m_frz[s] for s in acq_hi["fork_holdings"])
        assert abs(loss_rev - loss_frz) <= 0.25

        # Bellman DP oracle holds fork_holdings during freeze_matched_loss (0 turnover), switches on abrupt_reversal!
        assert e_frz._post_dp_oracle["weights_by_ep"][14] == acq_hi["fork_holdings"]
        assert e_rev._post_dp_oracle["weights_by_ep"][14] != acq_hi["fork_holdings"]

    assert len(r0_set) == 6  # Exact 1:1:1:1:1:1 Latin-square across Seeds 501..506

    rep = run_autonomy_batch(
        run_id="mock_autonomy",
        models=["mock"],
        arms=["scoped_cards_v3", "factual_ledger_v3", "sgeh_directive_v3"],
        fork_modes=["stationary", "abrupt_reversal", "freeze_matched_loss", "opportunity_stationary", "opportunity_rise"],
        seeds=[501],
        t_fork=14,
        h_post=8,
        max_usd=1.0,
        workers=3,
        thinking_budget=0,
        results_root=tmp_path,
        dotenv_path=None,
    )
    assert rep["total_run_usd"] == 0.0
    assert "did_abrupt_reversal_directive_minus_factual" in rep["paired_contrasts"]["mock"]["did_summary"]
