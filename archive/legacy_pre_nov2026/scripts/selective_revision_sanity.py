"""Regenerate the CPU-only reference report; no credentials or network needed."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from llm_rl_engine.analysis.selective_revision import ReferenceModel, decision_values, posterior


def make_case(structure: str, successes: int, cost: float) -> dict:
    model = ReferenceModel(structure=structure)
    observations = {"A": [1] * successes + [0] * (5 - successes), "B": [], "C": []}
    result = posterior(model, observations)
    result.pop("joint")
    result["decision"] = decision_values(result["posterior_high"]["B"], cost=cost)
    public = asdict(model)
    if structure == "stable":
        public["hazard"] = 0.0
    public.update({
        "observations": observations,
        "observation_counts": {
            key: {"successes": sum(values), "failures": len(values) - sum(values)}
            for key, values in observations.items()
        },
        "safe_utility": 0.65, "diagnostic_cost": cost,
    })
    return {"public_inputs": public, "reference_targets": result}


def main() -> None:
    worked = [make_case(structure, 0, 0.04) for structure in ("shared", "independent")]
    grid = [make_case(structure, successes, cost)
            for structure in ("shared", "independent", "stable")
            for successes in range(6) for cost in (0.0, 0.04, 0.20)]
    payload = {
        "schema_version": 1,
        "kind": "enumerated mathematical diagnostic, not model experiment",
        "worked_example": worked, "diagnostic_grid": grid,
    }
    root = Path(__file__).resolve().parents[1] / "reports"
    root.mkdir(exist_ok=True)
    json_path = root / "selective_revision_reference.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# Selective revision: exact reference", "",
        "CPU-only mathematical check. No LLM calls, training, or sampled performance evaluation.", "",
        "## Assumptions", "",
        "A/B/C independently start high with probability .99; rates are .2 or .8. "
        "The prior summarizes earlier evidence; no raw prehistory is asserted. One possible "
        "reset occurs before the observation batch with probability .1. Shared A/B resets "
        "share an event but redraw their values independently. Independent resets have matched "
        "marginal hazards. C cannot reset. Stable explicitly means a known zero hazard; it "
        "is not a secretly labelled no-event realization under a nonzero hazard.", "",
        "A diagnostic reveals B's rate, not its next reward. After diagnosis choose the higher "
        "expected-utility option and charge the diagnostic cost once. The alternative pays .65. "
        "Ties within 1e-12 prefer safe, then B, then diagnosis.", "",
        "## Worked example: five failures on A", "",
        "| Structure | Forecast A | Forecast B | Forecast C | B utility | Safe utility | Diagnostic utility | Best action |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in worked:
        r = case["reference_targets"]
        means, d = r["predictive_mean"], r["decision"]
        v = d["action_values"]
        lines.append(f"| {case['public_inputs']['structure']} | {means['A']:.6f} | "
                     f"{means['B']:.6f} | {means['C']:.6f} | {v['choose_B']:.6f} | "
                     f"{v['choose_safe']:.6f} | {v['diagnose_B']:.6f} | {d['optimal_action']} |")
    lines += ["", "The reference predicts the same A outcome in both cases, preserves C, "
              "and recommends different B decisions. This establishes a discriminating task, "
              "not that an LLM fails it or that training helps.", "",
              "## Diagnostic grid", "",
              "54 enumerated cases: six A success counts, three structures, three diagnostic "
              "costs. These are not independent episodes. Full inputs and targets are stored "
              "separately in the adjacent JSON; reference targets must not leak into ordinary prompts.", "",
              "| Structure | A successes / 5 | Cost | Forecast B | Best action |", "|---|---:|---:|---:|---|"]
    for case in grid:
        p, r = case["public_inputs"], case["reference_targets"]
        lines.append(f"| {p['structure']} | {p['observation_counts']['A']['successes']} | "
                     f"{p['diagnostic_cost']:.2f} | {r['predictive_mean']['B']:.6f} | "
                     f"{r['decision']['optimal_action']} |")
    lines += ["", "## Reproduction and limits", "", "```bash",
              "uv run python scripts/selective_revision_sanity.py",
              "uv run pytest -q tests/test_selective_revision.py", "```", "",
              "The observer enumerates event/state pairs and normalizes log likelihoods. "
              "Impossible observations raise an error. It accepts observations of all three "
              "components, but supports one pre-batch event only and fixed A/B/C roles. "
              "Label permutation, learned dependency structure, repeated changes, and actual "
              "post-diagnostic model behavior remain follow-up work. Forecasts condition on "
              "the stated model, never the simulator's actual hidden event.", ""]
    (root / "selective_revision_reference.md").write_text("\n".join(lines))
    print(f"Wrote {json_path} and markdown report ({len(grid)} diagnostic cases).")


if __name__ == "__main__":
    main()
