"""Harness registry."""

from __future__ import annotations

from src.harnesses.base import EpistemicHarness, PublicEpisodeRecord
from src.harnesses.baselines import (
    BlindHarness,
    SkillBookHarness,
    SlidingWindowHarness,
    SRNHarness,
)
from src.harnesses.procedure_harness import (
    SLBFactualLogHarness,
    SLBTransferredProcedureHarness,
    SelfAuthoredProcedureHarness,
)
from src.harnesses.scoped_cards import ScopedCardBookHarness
from src.harnesses.sgeh import SurpriseGatedEpistemicHarness
from src.harnesses.sgeh_v2 import PeriodicProbeHarness, ScopedSGEHHardened
from src.harnesses.sgeh_v3 import (
    DirectiveSGEHV3Harness,
    FactualLedgerV3Harness,
    ScopedCardBookV3Harness,
    ScopedSGEHV3,
)
from src.harnesses.sgeh_slb import ScopedSLBHarness


def make_harness(name: str) -> EpistemicHarness:
    mapping = {
        "blind": BlindHarness,
        "window": SlidingWindowHarness,
        "srn": SRNHarness,
        "skillbook": SkillBookHarness,
        "scoped_cards": ScopedCardBookHarness,
        "scoped_cards_v3": ScopedCardBookV3Harness,
        "factual_ledger_v3": FactualLedgerV3Harness,
        "editable_proc_v3": lambda: SelfAuthoredProcedureHarness(mode="editable"),
        "frozen_proc_v3": lambda: SelfAuthoredProcedureHarness(mode="frozen"),
        "decomposed_proc_v3": lambda: SelfAuthoredProcedureHarness(mode="editable", action_agnostic_schema=True),
        "frozen_decomposed_proc_v3": lambda: SelfAuthoredProcedureHarness(mode="frozen", action_agnostic_schema=True),
        "sgeh_directive_v3": DirectiveSGEHV3Harness,
        "sgeh_vbq_v3": lambda: DirectiveSGEHV3Harness(verify_before_quarantine=True),
        "periodic_probe": PeriodicProbeHarness,
        "sgeh": SurpriseGatedEpistemicHarness,
        "sgeh_slb": ScopedSLBHarness,
        "factual_slb": SLBFactualLogHarness,
        "transferred_proc_slb": SLBTransferredProcedureHarness,
        "sgeh_surprise": lambda: ScopedSGEHHardened(enable_hazard=False),
        "sgeh_surprise_shelf": lambda: ScopedSGEHV3(enable_hazard=False, enable_shelf=True),
        "sgeh_hazard": lambda: ScopedSGEHHardened(enable_hazard=True),
        "sgeh_hazard_noshelf": lambda: ScopedSGEHV3(enable_hazard=True, enable_shelf=False),
        "sgeh_hazard_shelf": lambda: ScopedSGEHV3(enable_hazard=True, enable_shelf=True),
    }
    if name not in mapping:
        raise ValueError(f"Unknown harness '{name}'. Valid: {sorted(mapping)}")
    return mapping[name]()


__all__ = [
    "BlindHarness",
    "EpistemicHarness",
    "PublicEpisodeRecord",
    "SRNHarness",
    "SkillBookHarness",
    "SlidingWindowHarness",
    "SurpriseGatedEpistemicHarness",
    "make_harness",
]
