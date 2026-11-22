"""Offline checks for task 006 procedure acquisition. No network."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from single_life.gemini_adapter import TransportError, Usage
from single_life.procedure_client import (
    INPUT_TOKEN_CAP,
    ProcedureGeneration,
    ScriptedProcedureClient,
    attempt_cap_usd,
    completion_cap_usd,
    token_cost,
)
from single_life.procedure_dsl import (
    ACTION_ORDER,
    FROZEN_CALLS,
    INSTRUCTION,
    LOCAL_ALGORITHM,
    RANDOM_ALGORITHM,
    REVISION_CALLS,
    WHOLE_ALGORITHM,
    CacheLearner,
    ProcedureParseError,
    build_prompt,
    parse_procedure,
    type_at,
)
from single_life.procedure_progress import ProcedureLedger
from single_life.procedure_runner import run_lifetime, run_reference
from single_life.procedure_world import load_config, utility_tenths
import single_life.procedure_runner as runner_module


ROOT = Path(__file__).resolve().parents[1]
PAIR = Fraction(3, 10)
ROUTE = Fraction(1, 2)


def tenths(value: Fraction) -> int:
    scaled = value * 10
    if scaled.denominator != 1:
        raise AssertionError(value)
    return scaled.numerator


def procedure_text(mapping: dict[str, list[str]], rationale: str = "note", evidence: list[int] | None = None) -> str:
    return json.dumps(
        {
            "evidence_ids": [] if evidence is None else evidence,
            "procedure": mapping,
            "rationale": rationale,
        }
    )


CONSTANT = {"T1": ["A|pair"], "T2": ["B|pair"], "T3": ["A|pair"]}


def generation(text: str, *, usage: str = "available") -> ProcedureGeneration | str:
    if usage == "available":
        return text
    return ProcedureGeneration(
        text=text,
        usage=Usage("unknown", None, None, None, None, None),
        finish_reason="STOP",
        model_version="scripted",
        elapsed_seconds=0.0,
        provider_metadata={
            "candidate_count": 1,
            "finishReason": "STOP",
            "modelVersion": "scripted",
            "part_thought_flags": [False],
            "responseId": None,
            "usageMetadata": {"candidatesTokenCount": 20},
        },
        counted_input_tokens=128,
    )


class ProcedureAcquisitionTests(unittest.TestCase):
    def test_tables_match_task_004_arithmetic_and_registered_file(self) -> None:
        self.assertEqual(type_at(1), "T1")
        self.assertEqual(type_at(2), "T2")
        self.assertEqual(type_at(3), "T3")
        self.assertEqual(type_at(43), "T1")
        self.assertEqual(utility_tenths("stable", 1, "T1", "A|pair"), tenths(ROUTE + PAIR))
        self.assertEqual(utility_tenths("stable", 96, "T2", "B|single"), tenths(ROUTE))
        self.assertEqual(utility_tenths("shared", 42, "T1", "B|pair"), tenths(PAIR))
        self.assertEqual(utility_tenths("shared", 43, "T1", "B|pair"), tenths(Fraction(-2, 5)))
        self.assertEqual(utility_tenths("shared", 43, "T1", "A|pair"), tenths(ROUTE + Fraction(-2, 5)))
        self.assertEqual(utility_tenths("route", 43, "T1", "B|pair"), tenths(ROUTE + PAIR))
        self.assertEqual(utility_tenths("route", 43, "T1", "A|pair"), tenths(PAIR))
        self.assertEqual(utility_tenths("route", 43, "T3", "A|pair"), tenths(ROUTE + PAIR))
        self.assertEqual(utility_tenths("trap", 42, "T1", "A|pair"), tenths(ROUTE + PAIR))
        self.assertEqual(utility_tenths("trap", 43, "T1", "A|single"), tenths(Fraction(1, 2)))
        self.assertEqual(utility_tenths("trap", 43, "T1", "A|pair"), tenths(Fraction(1, 10)))
        self.assertEqual(utility_tenths("trap", 43, "T2", "A|pair"), tenths(Fraction(4, 5)))
        self.assertEqual(utility_tenths("trap", 43, "T2", "A|single"), tenths(Fraction(1, 5)))
        self.assertIsInstance(utility_tenths("shared", 43, "T1", "B|pair"), int)
        load_config(ROOT / "configs" / "006_worlds.json")
        self.assertEqual(completion_cap_usd(), Decimal("2.82624"))
        self.assertEqual(attempt_cap_usd(), token_cost(8192, 2048))

    def test_parser_and_source_boundaries(self) -> None:
        parsed = parse_procedure(procedure_text(CONSTANT, evidence=[]), cutoff=0)
        self.assertEqual(parsed["procedure"]["T1"], ["A|pair"])
        with self.assertRaises(ProcedureParseError):
            parse_procedure(procedure_text(CONSTANT, evidence=[1]), cutoff=0)
        with self.assertRaises(ProcedureParseError):
            parse_procedure(procedure_text(CONSTANT, evidence=[True]), cutoff=4)  # type: ignore[list-item]
        with self.assertRaises(ProcedureParseError):
            parse_procedure(procedure_text({"T1": ["A|pair", "A|pair"], "T2": ["B|pair"], "T3": ["A|pair"]}), cutoff=0)
        with self.assertRaises(ProcedureParseError):
            parse_procedure(procedure_text(CONSTANT, rationale="x" * 1001), cutoff=0)
        source = Path(runner_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("llm_rl_engine", source)
        self.assertNotIn("evaluator_cycle_means", source)
        self.assertNotIn("change_at", INSTRUCTION)
        self.assertNotIn("SENTINEL_PRIVATE", INSTRUCTION)
        prompt = build_prompt(action_index=1, horizon=96, observations=[], procedure=None, rationale=None)
        self.assertNotIn("SENTINEL_PRIVATE", prompt)

    def test_interpreter_reset_frozen_schedule_and_saved_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = {
                "T1": ["A|single"],
                "T2": ["A|single", "A|pair", "B|single"],
                "T3": ["A|pair"],
            }
            second = {
                "T1": ["B|pair"],
                "T2": ["A|single", "A|pair", "B|single"],
                "T3": ["A|pair"],
            }
            client = ScriptedProcedureClient(
                [procedure_text(first), procedure_text(second, evidence=[1, 2, 12])]
            )
            ledger = self.ledger(root / "progress.json")
            result = run_lifetime(
                world="stable",
                condition="revision",
                run_dir=root / "revision",
                client=client,
                ledger=ledger,
                horizon=16,
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.call_indices, [1, 13])
            self.assertNotIn(43, result.call_indices)
            self.assertEqual(result.events[13]["type"], "T2")
            self.assertEqual(result.events[13]["action"], "A|single")
            self.assertEqual(result.events[0]["utility_tenths"], utility_tenths("stable", 1, "T1", "A|single"))
            prompt = json.loads((root / "revision" / "responses" / "c013-a1.json").read_text(encoding="utf-8"))
            payload = json.loads(prompt["prompt"].split("public_payload:\n", 1)[1])
            self.assertEqual(payload["evidence_cutoff"], 12)
            self.assertEqual([item["index"] for item in payload["observations"]], list(range(1, 13)))
            self.assertEqual(set(payload["observations"][0]), {"action", "index", "type", "utility_tenths"})
            self.assertIsNone(payload.get("future_types"))
            self.assertNotIn("SENTINEL_PRIVATE", prompt["prompt"])
            self.assertNotIn("table_phase", (root / "revision" / "public_log.jsonl").read_text(encoding="utf-8"))
            self.assertIn("table_phase", (root / "revision" / "private_log.jsonl").read_text(encoding="utf-8"))
            manifest = json.loads((root / "revision" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["baseline_definition"])
            self.assertNotIn("UCB", (root / "revision" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse((root / "revision" / "checkpoint.json").exists())
            usage = prompt["provider_metadata"]["usageMetadata"]
            self.assertEqual(usage["promptTokenCount"], 100)
            self.assertEqual(prompt["usage"]["cost_usd"], format(token_cost(100, 20), "f"))

            frozen = ScriptedProcedureClient([procedure_text(CONSTANT), procedure_text(CONSTANT), procedure_text(CONSTANT), procedure_text({"T1": ["B|single"], "T2": ["B|single"], "T3": ["B|single"]})])
            frozen_result = run_lifetime(
                world="stable",
                condition="frozen",
                run_dir=root / "frozen",
                client=frozen,
                horizon=96,
            )
            self.assertEqual(frozen.generate_calls, 3)
            self.assertEqual(frozen_result.call_indices, list(FROZEN_CALLS))
            self.assertEqual(
                [item["deployment_index"] for item in frozen_result.artifacts],
                [1, 13, 25],
            )
            self.assertEqual(frozen_result.events[24]["procedure_hash"], frozen_result.events[95]["procedure_hash"])
            self.assertTrue(all(event["action"].endswith("|pair") for event in frozen_result.events[24:]))

    def test_future_perturbation_stops_before_changed_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replies = [procedure_text(CONSTANT) for _ in range(8)]
            stable = ScriptedProcedureClient(list(replies))
            shared = ScriptedProcedureClient(list(replies))
            stable_result = run_lifetime(world="stable", condition="revision", run_dir=root / "stable", client=stable)
            shared_result = run_lifetime(world="shared", condition="revision", run_dir=root / "shared", client=shared)
            self.assertEqual(stable_result.call_indices, list(REVISION_CALLS))
            self.assertEqual(
                [(event["action"], event["utility_tenths"]) for event in stable_result.events[:42]],
                [(event["action"], event["utility_tenths"]) for event in shared_result.events[:42]],
            )
            self.assertEqual(stable.prompts[3], shared.prompts[3])
            self.assertEqual(stable_result.events[42]["action"], shared_result.events[42]["action"])
            self.assertNotEqual(stable_result.events[42]["utility_tenths"], shared_result.events[42]["utility_tenths"])
            self.assertNotEqual(stable.prompts[4], shared.prompts[4])
            self.assertNotIn("stable", stable.prompts[0])
            self.assertNotIn("shared", shared.prompts[0])
            self.assertIsNone(json.loads(stable.prompts[0].split("public_payload:\n", 1)[1])["current_procedure"])

    def test_invalid_transport_cap_and_unknown_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retry = ScriptedProcedureClient(
                [procedure_text(CONSTANT, evidence=[1]), procedure_text(CONSTANT)]
            )
            ledger = self.ledger(root / "retry.json")
            retried = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "retry",
                client=retry,
                ledger=ledger,
                horizon=1,
            )
            self.assertEqual(retried.status, "completed")
            self.assertEqual(retried.committed_index, 1)
            self.assertEqual(ledger.attempts, 2)
            self.assertEqual(ledger.retries, 1)
            self.assertEqual(ledger.completed_smoke_ticks, 1)
            self.assertEqual(retry.generate_calls, 2)

            aborted = run_lifetime(
                world="stable",
                condition="revision",
                run_dir=root / "abort",
                client=ScriptedProcedureClient(["nope", "still"]),
                horizon=16,
            )
            self.assertEqual(aborted.status, "aborted")
            self.assertEqual(aborted.committed_index, 0)
            self.assertEqual(aborted.reason, "invalid_procedure")
            self.assertFalse(aborted.stop_task)

            later = ScriptedProcedureClient(
                [procedure_text(CONSTANT), "nope", "still"]
            )
            later_result = run_lifetime(
                world="stable",
                condition="revision",
                run_dir=root / "later",
                client=later,
                horizon=16,
            )
            self.assertEqual(later_result.committed_index, 12)
            self.assertEqual(later_result.reason, "invalid_procedure")
            self.assertEqual(len(read_events(root / "later")), 12)

            transport = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "transport",
                client=ScriptedProcedureClient([TransportError("down"), TransportError("down")]),
                horizon=1,
            )
            self.assertEqual(transport.committed_index, 0)
            self.assertEqual(transport.reason, "transport_failed")
            self.assertFalse(transport.stop_task)

            auth = ScriptedProcedureClient(
                [
                    TransportError("denied", auth_or_quota=True),
                    procedure_text(CONSTANT),
                ]
            )
            recovered = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "auth-ok",
                client=auth,
                horizon=1,
            )
            self.assertEqual(recovered.status, "completed")
            self.assertFalse(recovered.stop_task)
            stopped = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "auth-stop",
                client=ScriptedProcedureClient(
                    [TransportError("denied", auth_or_quota=True), TransportError("denied", auth_or_quota=True)]
                ),
                horizon=1,
            )
            self.assertTrue(stopped.stop_task)

            capped = ScriptedProcedureClient([procedure_text(CONSTANT)])
            capped.token_count = INPUT_TOKEN_CAP + 1
            capped_result = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "cap",
                client=capped,
                horizon=1,
            )
            self.assertEqual(capped.generate_calls, 0)
            self.assertEqual(capped_result.reason, "input_cap")
            saved_prompt = json.loads((root / "cap" / "responses" / "c001-cap.json").read_text(encoding="utf-8"))["prompt"]
            self.assertLess(len(saved_prompt) // 4, INPUT_TOKEN_CAP)

            unknown_ledger = self.ledger(root / "unknown.json")
            unknown = run_lifetime(
                world="stable",
                condition="smoke",
                run_dir=root / "unknown",
                client=ScriptedProcedureClient([generation(procedure_text(CONSTANT), usage="unknown")]),
                ledger=unknown_ledger,
                horizon=1,
            )
            self.assertEqual(unknown.committed_index, 1)
            self.assertEqual(unknown_ledger.unknown_attempts, 1)
            self.assertEqual(unknown_ledger.unknown_reserved, attempt_cap_usd())
            self.assertEqual(unknown_ledger.known_spend, Decimal("0"))

    def test_references_react_only_to_observed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = run_reference(world="stable", algorithm=LOCAL_ALGORITHM, run_dir=root / "local")
            whole = run_reference(world="stable", algorithm=WHOLE_ALGORITHM, run_dir=root / "whole")
            random_a = run_reference(world="stable", algorithm=RANDOM_ALGORITHM, run_dir=root / "random-a")
            random_b = run_reference(world="route", algorithm=RANDOM_ALGORITHM, run_dir=root / "random-b")
            self.assertEqual(sum(event["utility_tenths"] for event in local.events), 720)
            self.assertEqual(
                [event["action"] for event in local.events],
                [event["action"] for event in whole.events],
            )
            self.assertEqual(
                [event["action"] for event in random_a.events],
                [event["action"] for event in random_b.events],
            )
            self.assertEqual(json.loads((root / "local" / "manifest.json").read_text(encoding="utf-8"))["usage_estimate_usd"], "0")
            route_local = run_reference(world="route", algorithm=LOCAL_ALGORITHM, run_dir=root / "route-local")
            route_whole = run_reference(world="route", algorithm=WHOLE_ALGORITHM, run_dir=root / "route-whole")
            self.assertEqual(route_local.events[43]["type"], "T2")
            self.assertEqual(route_local.events[43]["action"], "B|pair")
            self.assertEqual(route_whole.events[43]["action"], "A|single")

            def hidden_improvement(index: int, typ: str, action: str) -> int:
                if typ == "T1" and action == "A|single" and index >= 43:
                    return 9
                return utility_tenths("stable", 1, typ, action)

            stayed = run_reference(
                world="stable",
                algorithm=LOCAL_ALGORITHM,
                run_dir=root / "stayed",
                utility_of=hidden_improvement,
            )
            t1 = [event["action"] for event in stayed.events if event["type"] == "T1"]
            self.assertEqual(t1[4:], ["A|pair"] * (len(t1) - 4))

            def observed_drop(index: int, typ: str, action: str) -> int:
                if typ == "T1" and action == "A|pair" and index >= 43:
                    return 1
                return utility_tenths("stable", 1, typ, action)

            dropped = run_reference(
                world="stable",
                algorithm=LOCAL_ALGORITHM,
                run_dir=root / "dropped",
                utility_of=observed_drop,
            )
            dropped_t1 = [event["action"] for event in dropped.events if event["type"] == "T1"]
            self.assertEqual(dropped_t1[14], "A|pair")
            self.assertEqual(dropped_t1[15], "A|single")

        learner = CacheLearner(whole=False)
        self.assertEqual(learner.choose("T1"), ACTION_ORDER[0])
        learner.observe("T1", "A|pair", 8)
        learner.observe("T2", "B|pair", 8)
        learner.observe("T1", "A|pair", 1)
        self.assertEqual(learner.cache["T2"], {"B|pair": 8})
        self.assertEqual(learner.cache["T1"], {"A|pair": 1})
        broad = CacheLearner(whole=True)
        broad.observe("T1", "A|pair", 8)
        broad.observe("T2", "B|pair", 8)
        broad.observe("T1", "A|pair", 1)
        self.assertEqual(broad.cache["T2"], {})
        self.assertEqual(broad.cache["T1"], {"A|pair": 1})

    def test_progress_separates_ticks_calls_and_reserves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            ledger = ProcedureLedger(
                path,
                planned_smoke_calls=1,
                planned_revision_calls=1,
                planned_frozen_calls=1,
                planned_smoke_ticks=1,
                planned_revision_ticks=1,
                planned_frozen_ticks=1,
                planned_references=3,
                attempt_cap=attempt_cap_usd(),
                completion_cap=completion_cap_usd(),
                heartbeat_seconds=0.05,
                printer=lambda _line: None,
            )
            ledger.mark_references(3)
            ledger.locate(world="stable", condition="revision", tick=1)
            ledger.begin_attempt()
            ledger.start_heartbeat()
            time.sleep(0.2)
            self.assertEqual(ledger.attempts, 0)
            self.assertGreaterEqual(len(ledger.lines), 2)
            ledger.finish_attempt(usage=None, failed=False)
            ledger.complete_call()
            ledger.complete_tick()
            ledger.locate(world="stable", condition="frozen", tick=2)
            ledger.complete_call()
            ledger.complete_tick()
            self.assertEqual(ledger.snapshot()["completed_revision_calls"], 1)
            self.assertEqual(ledger.snapshot()["completed_frozen_ticks"], 1)
            self.assertLessEqual(ledger.snapshot()["percent"], 99)
            ledger.locate(world="stable", condition="smoke", tick=1)
            ledger.complete_call()
            ledger.complete_tick()
            ledger.finish("complete")
            snap = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(snap["percent"], 100)
            self.assertEqual(snap["status"], "complete")
            self.assertIn("ticks ", snap["last_line"])
            self.assertIn("calls ", snap["last_line"])
            self.assertIn("rev ", snap["last_line"])
            self.assertIn("frz ", snap["last_line"])
            self.assertIn("projected_final=", snap["last_line"])
            ledger.unknown_reserved = completion_cap_usd()
            with self.assertRaises(RuntimeError):
                ledger.begin_attempt()

    def ledger(self, path: Path) -> ProcedureLedger:
        return ProcedureLedger(
            path,
            planned_smoke_calls=2,
            planned_revision_calls=8,
            planned_frozen_calls=3,
            planned_smoke_ticks=2,
            planned_revision_ticks=96,
            planned_frozen_ticks=96,
            planned_references=0,
            attempt_cap=attempt_cap_usd(),
            completion_cap=completion_cap_usd(),
            printer=lambda _line: None,
        )


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (path / "public_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
