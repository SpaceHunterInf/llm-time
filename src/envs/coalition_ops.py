"""CoalitionOps: Multi-actor strategic contracting under silent Foe <-> Friend reversals."""

from __future__ import annotations

import random
from typing import Any

from src.envs.base import ParadigmEnv, PublicTaskSpec, StepOutcome

PARTNERS = ("VANGUARD_CORP", "MERIDIAN_LOGISTICS", "HELIOS_MATERIALS")
POSTURES = ("cooperative_joint", "escrow_hedged", "defensive_reroute")
PROBE_COST = 18.0  # Paid active diagnostic audit cost (creates true VOI trade-off vs memory)


class CoalitionOpsEnv(ParadigmEnv):
    """Strategic supply-network environment where partner incentive alignment silently reverses.

    Payoff structure (before probe cost, for matching partner's hidden alignment):
      - `ally` (positive-sum partner): `cooperative_joint`=96, `escrow_hedged`=58, `defensive_reroute`=22
      - `rival` (opportunistic defector): `cooperative_joint`=12, `escrow_hedged`=62, `defensive_reroute`=90
      - `neutral`: `cooperative_joint`=54, `escrow_hedged`=84, `defensive_reroute`=52
    Invariant rule: Cannot use `cooperative_joint` when `reserve_buffer < 0.25` (SLA solvency rule -> 0 score).
    """

    env_name = "coalition_ops"
    max_turns_per_episode = 6

    def __init__(self, *, mode: str = "utility_reversal", horizon: int = 36, seed: int = 42) -> None:
        if mode not in ("stationary", "utility_reversal"):
            raise ValueError(f"Unsupported CoalitionOps mode: {mode}")
        self.mode = mode
        self.horizon = int(horizon)
        self.seed = int(seed)

        self._episodes: list[dict[str, Any]] = []
        for ep in range(self.horizon):
            rng = random.Random(self.seed * 17_011 + ep * 157 + 29)
            assigned_partner = PARTNERS[ep % len(PARTNERS)]
            # Keep sufficient reserve buffer on most steps, with occasional tight-reserve invariant tests
            reserve_buffer = round(0.18 if (ep % 9 == 8) else rng.uniform(0.32, 0.85), 2)
            probe_flip = rng.random() < 0.08  # 92% accurate diagnostic audit signal
            noise = rng.gauss(0.0, 2.5)
            self._episodes.append(
                {
                    "partner": assigned_partner,
                    "reserve_buffer": reserve_buffer,
                    "probe_flip": probe_flip,
                    "noise": noise,
                }
            )

        self._cur_ep = 0
        self._inspected_order = False
        self._probes_used = 0
        self._done = True

    def num_episodes(self) -> int:
        return self.horizon

    def _alignments(self, ep: int) -> tuple[dict[str, str], str]:
        regime_1 = {
            "VANGUARD_CORP": "ally",
            "MERIDIAN_LOGISTICS": "rival",
            "HELIOS_MATERIALS": "neutral",
        }
        if self.mode == "stationary":
            return regime_1, "coalition_phase_1"

        t1 = self.horizon // 3
        t2 = (2 * self.horizon) // 3
        if ep < t1:
            return regime_1, "coalition_phase_1"
        if ep < t2:
            # Foe <-> Friend full utility reversal between Vanguard and Meridian!
            return (
                {
                    "VANGUARD_CORP": "rival",
                    "MERIDIAN_LOGISTICS": "ally",
                    "HELIOS_MATERIALS": "ally",
                },
                "coalition_phase_2_reversal",
            )
        return (
            {
                "VANGUARD_CORP": "neutral",
                "MERIDIAN_LOGISTICS": "ally",
                "HELIOS_MATERIALS": "rival",
            },
            "coalition_phase_3_realignment",
        )

    @staticmethod
    def _base_payoff(alignment: str, posture: str, reserve_buffer: float) -> tuple[float, bool]:
        if reserve_buffer < 0.25 and posture == "cooperative_joint":
            return 0.0, True
        table = {
            "ally": {"cooperative_joint": 96.0, "escrow_hedged": 58.0, "defensive_reroute": 22.0},
            "rival": {"cooperative_joint": 12.0, "escrow_hedged": 62.0, "defensive_reroute": 90.0},
            "neutral": {"cooperative_joint": 54.0, "escrow_hedged": 84.0, "defensive_reroute": 52.0},
        }
        return table[alignment][posture], False

    def start_episode(self, episode_idx: int) -> PublicTaskSpec:
        self._cur_ep = int(episode_idx)
        self._inspected_order = False
        self._probes_used = 0
        self._done = False
        partner = self._episodes[self._cur_ep]["partner"]

        domain_prompt = (
            "You are the Strategic Operations Director in CoalitionOps, managing critical supply dispatches "
            f"across three counterparties ({', '.join(PARTNERS)}).\n"
            "Available contract postures: `cooperative_joint`, `escrow_hedged`, `defensive_reroute`.\n"
            "INVARIANT SOLVENCY RULE (always enforced):\n"
            "- Never select `cooperative_joint` when `reserve_buffer < 0.25` (triggers immediate SLA default: Score 0/100).\n"
            "STRATEGIC NOTE:\n"
            "- Counterparties may act as cooperative allies (rewarding `cooperative_joint`), opportunistic rivals "
            "(requiring `defensive_reroute`), or neutral parties (`escrow_hedged`).\n"
            "- Boardroom ownership and incentives can shift silently over time. You may optionally run "
            f"`audit_counterparty` (costs {PROBE_COST:.0f} points) to inspect live alignment clues before committing."
        )
        task_prompt = (
            f"Dispatch Cycle #{episode_idx + 1}: Counterparty `{partner}` is on the critical path. "
            "Inspect the dispatch ledger (and optionally audit the counterparty if uncertain), then commit "
            "your posture via `execute_contract`."
        )
        tool_schemas = [
            {
                "name": "inspect_dispatch_ledger",
                "description": "Read active counterparty, current reserve_buffer, and SLA solvency threshold.",
                "args": {},
            },
            {
                "name": "audit_counterparty",
                "description": f"Paid diagnostic probe (-{PROBE_COST:.0f} pts): run compliance/incentive audit on a partner.",
                "args": {"partner": "Partner ID"},
            },
            {
                "name": "execute_contract",
                "description": "Terminal action: commit contract posture for this cycle's counterparty.",
                "args": {
                    "partner": "Must match active counterparty",
                    "posture": "One of cooperative_joint, escrow_hedged, defensive_reroute",
                },
            },
        ]
        return PublicTaskSpec(
            env_name=self.env_name,
            episode_idx=episode_idx,
            timestamp=episode_idx + 1,
            domain_prompt=domain_prompt,
            task_prompt=task_prompt,
            tool_schemas=tool_schemas,
        )

    def step_tool(self, tool_name: str, tool_args: dict[str, Any]) -> StepOutcome:
        if self._done:
            raise RuntimeError("Episode already finished.")
        ep_data = self._episodes[self._cur_ep]
        alignments, phase_label = self._alignments(self._cur_ep)

        if tool_name == "inspect_dispatch_ledger":
            self._inspected_order = True
            return StepOutcome(
                observation={
                    "cycle": self._cur_ep + 1,
                    "active_counterparty": ep_data["partner"],
                    "reserve_buffer": ep_data["reserve_buffer"],
                    "min_reserve_for_cooperative_joint": 0.25,
                },
                done=False,
            )

        if tool_name == "audit_counterparty":
            partner = str(tool_args.get("partner", ep_data["partner"])).strip().upper()
            if partner not in PARTNERS:
                return StepOutcome(observation={"error": f"Unknown partner '{partner}'."}, done=False)
            self._probes_used += 1
            true_align = alignments[partner]
            signal_map = {
                "ally": "joint_capacity_pledged_cooperative",
                "rival": "opportunistic_margin_extraction_detected",
                "neutral": "arms_length_standard_terms",
            }
            observed_signal = signal_map[true_align]
            if ep_data["probe_flip"]:
                observed_signal = "arms_length_standard_terms"
            return StepOutcome(
                observation={
                    "audited_partner": partner,
                    "audit_fee_deducted": PROBE_COST,
                    "live_behavioral_indicator": observed_signal,
                },
                done=False,
            )

        if tool_name == "execute_contract":
            self._done = True
            partner = str(tool_args.get("partner", ep_data["partner"])).strip().upper()
            posture = str(tool_args.get("posture", "")).strip().lower()
            reserve = ep_data["reserve_buffer"]

            phase1_align = {"VANGUARD_CORP": "ally", "MERIDIAN_LOGISTICS": "rival", "HELIOS_MATERIALS": "neutral"}
            true_align = alignments[ep_data["partner"]]
            stale_align = phase1_align[ep_data["partner"]]

            valid_postures = [p for p in POSTURES if not (reserve < 0.25 and p == "cooperative_joint")]
            best_posture = max(valid_postures, key=lambda p: self._base_payoff(true_align, p, reserve)[0])
            stale_posture = max(valid_postures, key=lambda p: self._base_payoff(stale_align, p, reserve)[0])
            oracle_score = self._base_payoff(true_align, best_posture, reserve)[0]

            if partner != ep_data["partner"] or posture not in POSTURES:
                return StepOutcome(
                    observation={"status": "invalid_contract_args", "score": 0.0},
                    done=True,
                    public_score=0.0,
                    private_eval={
                        "regime": phase_label,
                        "expected_score": 0.0,
                        "oracle_expected_score": oracle_score,
                        "normalized_utility": 0.0,
                        "stale_policy_match": False,
                        "optimal_policy_match": False,
                        "invariant_violation": True,
                        "probes_used": self._probes_used,
                    },
                )

            base, violated = self._base_payoff(true_align, posture, reserve)
            probe_deduction = self._probes_used * PROBE_COST
            uninspected_pen = 0.0 if self._inspected_order else 15.0
            expected = 0.0 if violated else max(0.0, base - probe_deduction - uninspected_pen)
            realized = 0.0 if violated else min(100.0, max(0.0, round(expected + ep_data["noise"], 1)))

            shifted = true_align != stale_align
            return StepOutcome(
                observation={"status": "executed", "posture": posture, "score": realized},
                done=True,
                public_score=realized,
                private_eval={
                    "regime": phase_label,
                    "expected_score": round(expected, 2),
                    "oracle_expected_score": oracle_score,
                    "normalized_utility": round(expected / max(1.0, oracle_score), 4),
                    "stale_policy_match": shifted and (posture == stale_posture) and (stale_posture != best_posture),
                    "optimal_policy_match": posture == best_posture,
                    "invariant_violation": violated,
                    "probes_used": self._probes_used,
                },
            )

        return StepOutcome(observation={"error": f"Unknown tool '{tool_name}'."}, done=False)
