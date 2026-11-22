"""Offline checks for the Gemini note-retention diagnostic. No network."""

from __future__ import annotations

import ast
import io
import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path

from single_life.baselines import make_policy
from single_life.contracts import CheckpointError, ContractError
from single_life.fixtures import BanditEvaluator, load_spec, spec_from_dict
from single_life.gemini_adapter import (
    THINKING_LEVEL,
    GeminiClient,
    GeminiNotePolicy,
    NoteCarryUpdater,
    NoteHandoff,
    ScriptedClient,
    TransportError,
    Usage,
    attempt_cap_usd,
    completion_cap_usd,
    parse_model_json,
    pricing_record,
    scripted_reply,
    token_cost,
)
from single_life.identity import dependency_identity
from single_life.pilot import (
    ensure_fresh,
    load_world_dict,
    run_note_lifetime,
    smoke_dict,
    with_agent,
)
from single_life.progress import ProgressLedger
from single_life.runner import _fresh_state, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "SENTINEL_PRIVATE_do_not_leak",
    "oracle_best_arm",
    "private-w1-a",
    "private-w2",
    "private-later",
    "innovation-005",
    "regime_id",
    "written_at_decision",
    "unreachable",
    "evaluator_private",
    "0.8",
    "0.2",
    "0.5",
    "PROGRESS",
    "cap_usd",
    "thinking_level",
    "gemini",
)


def bandit_spec(**kwargs):
    horizon = kwargs.get("horizon", 2)
    segments = kwargs.get("segments")
    if segments is None:
        segments = [
            {
                "start_index": 1,
                "end_index": horizon,
                "probabilities": {"arm_a": 0.8, "arm_b": 0.2, "arm_c": 0.5},
                "regime_id": kwargs.get("regime_id", "private-shared"),
                "unreachable_metadata": {
                    "oracle_best_arm": "arm_a",
                    "sentinel": "SENTINEL_PRIVATE_do_not_leak 005",
                },
            }
        ]
    data = {
        "action_ids": ["arm_a", "arm_b", "arm_c"],
        "agent": kwargs.get("agent", "llm-note"),
        "agent_seed": "agent-test",
        "artifact_period": horizon,
        "artifact_window": 12,
        "compute_budget": 1000,
        "evaluator_private_note": "SENTINEL_PRIVATE_do_not_leak 005",
        "experiment_id": "005-gemini-lifetime",
        "fixture_id": kwargs.get("fixture_id", "005-test"),
        "horizon": horizon,
        "innovation_seed": kwargs.get("innovation_seed", "innovation-test"),
        "interaction_budget": horizon,
        "lifetime_id": kwargs.get("lifetime_id", "life-test"),
        "protocol_version": "single-life-v0.2",
        "schedule_seed": "schedule-test",
        "segments": segments,
        "ucb_window": 12,
    }
    return spec_from_dict(data)


def reply(action: str, note: str) -> str:
    return json.dumps({"action": action, "strategy_note": note})


class GeminiPilotTests(unittest.TestCase):
    def test_worlds_match_the_registered_tables(self) -> None:
        stable = load_world_dict(ROOT, "w1")
        changing = load_world_dict(ROOT, "w2")
        self.assertEqual(stable["ucb_window"], 12)
        self.assertEqual(changing["ucb_window"], 12)
        self.assertEqual(stable["horizon"], 48)
        self.assertEqual(changing["horizon"], 48)
        self.assertEqual(stable["lifetime_id"], "life-005-w1")
        self.assertEqual(changing["lifetime_id"], "life-005-w2")
        self.assertEqual(stable["segments"][0]["probabilities"], {"arm_a": 0.8, "arm_b": 0.2, "arm_c": 0.5})
        self.assertEqual(
            [segment["end_index"] for segment in changing["segments"]],
            [16, 34, 48],
        )
        self.assertEqual(
            changing["segments"][1]["probabilities"],
            {"arm_a": 0.2, "arm_b": 0.8, "arm_c": 0.5},
        )
        for blob in (json.dumps(stable), json.dumps(changing)):
            lowered = blob.lower()
            self.assertNotIn("stable", lowered)
            self.assertNotIn("change", lowered)
        recent = spec_from_dict(with_agent(changing, "llm-recent"))
        persistent = spec_from_dict(with_agent(changing, "llm-note"))
        self.assertEqual(recent.segments, persistent.segments)
        self.assertEqual(recent.lifetime_id, persistent.lifetime_id)
        self.assertEqual(recent.innovation_seed, persistent.innovation_seed)
        self.assertNotEqual(recent.agent_name, persistent.agent_name)
        smoke = spec_from_dict(smoke_dict(ROOT))
        self.assertEqual(smoke.horizon, 4)
        self.assertEqual(smoke.lifetime_id, "life-005-smoke")
        self.assertNotEqual(smoke.innovation_seed, stable["innovation_seed"])
        self.assertEqual(smoke.segments[0].probabilities["arm_a"], 0.8)
        self.assertEqual(format(completion_cap_usd(), "f"), "9.633792")
        self.assertEqual(THINKING_LEVEL, "low")
        self.assertIsNone(pricing_record()["thinking_budget_integer"])
        self.assertEqual(dependency_identity(ROOT)["runtime_dependencies"], [])
        with self.assertRaises(ContractError):
            make_policy("llm-note", 12)

    def test_prompts_are_allowlisted_and_memory_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = spec_from_dict(with_agent(load_world_dict(ROOT, "w1"), "llm-note"))
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            result, policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="allowlist",
                carry=True,
                world="w1",
                condition="persistent",
                memory_condition="persistent-note",
                client=ScriptedClient([reply("arm_a", "hello-note")]),
                ledger=ledger,
                phase="pilot",
                stop_after=1,
            )
            self.assertEqual(result.status, "stopped_for_checkpoint", result.error_private)
            self.assertEqual(len(policy.prompts), 1)
            payload = json.loads(policy.prompts[0].split("public_payload:\n", 1)[1])
            self.assertIsNone(payload["retained_strategy_note"])
            self.assertEqual(payload["observations"], [])
            self.assertEqual(payload["permitted_action_ids"], ["arm_a", "arm_b", "arm_c"])
            blob = "\n".join(policy.prompts)
            for token in FORBIDDEN:
                self.assertNotIn(token, blob)
            progress = (root / "progress.json").read_text(encoding="utf-8")
            self.assertNotIn("hello-note", progress)
            self.assertNotIn("SENTINEL_PRIVATE_do_not_leak", progress)
            self.assertNotIn("private-w1-a", progress)
            self.assertNotIn("public_payload", progress)

    def test_note_is_carried_only_by_the_persistent_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=2)
            persistent_ledger = ProgressLedger(root / "p.json", planned_smoke=0, planned_pilot=4)
            recent_ledger = ProgressLedger(root / "r.json", planned_smoke=0, planned_pilot=4)
            replies = [reply("arm_b", "use-arm_b"), reply("arm_a", "later-note")]
            persistent, persistent_policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "persistent",
                run_id="persistent",
                carry=True,
                world="w1",
                condition="persistent",
                memory_condition="persistent-note",
                client=ScriptedClient(list(replies)),
                ledger=persistent_ledger,
                phase="pilot",
            )
            recent, recent_policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "recent",
                run_id="recent",
                carry=False,
                world="w1",
                condition="recent",
                memory_condition="recent-only",
                client=ScriptedClient(list(replies)),
                ledger=recent_ledger,
                phase="pilot",
            )
            self.assertEqual(persistent.status, "completed", persistent.error_private)
            self.assertEqual(recent.status, "completed", recent.error_private)
            persistent_payload = json.loads(persistent_policy.prompts[1].split("public_payload:\n", 1)[1])
            recent_payload = json.loads(recent_policy.prompts[1].split("public_payload:\n", 1)[1])
            self.assertEqual(persistent_payload["retained_strategy_note"], "use-arm_b")
            self.assertIsNone(recent_payload["retained_strategy_note"])
            self.assertIsNone(json.loads(persistent_policy.prompts[0].split("public_payload:\n", 1)[1])["retained_strategy_note"])
            recorded = json.loads((root / "recent" / "responses" / "d001-a1.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["parsed_note"], "use-arm_b")
            artifacts = json.loads((root / "recent" / "artifacts.json").read_text(encoding="utf-8"))
            self.assertFalse(any(item.get("kind") == "strategy-note" for item in artifacts))
            carried = json.loads((root / "persistent" / "artifacts.json").read_text(encoding="utf-8"))
            notes = [item for item in carried if item.get("kind") == "strategy-note"]
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["deployment_decision_index"], 2)
            self.assertEqual(notes[0]["creation_cutoff"], 0)
            self.assertEqual(notes[0]["content"]["strategy_note"], "use-arm_b")
            self.assertNotIn("use-arm_b", (root / "p.json").read_text(encoding="utf-8"))
            self.assertEqual(persistent_ledger.completed_pilot, 2)
            self.assertEqual(len(BanditEvaluator(spec).reveal_calls), 0)

    def test_long_note_is_truncated_before_the_next_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=2)
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            _result, policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="truncate",
                carry=True,
                world="w1",
                condition="persistent",
                memory_condition="persistent-note",
                client=ScriptedClient([reply("arm_a", "Z" * 1500), reply("arm_a", "next")]),
                ledger=ledger,
                phase="pilot",
            )
            saved = json.loads((root / "run" / "responses" / "d001-a1.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["parser_status"], "truncated")
            self.assertEqual(len(saved["parsed_note"]), 1000)
            payload = json.loads(policy.prompts[1].split("public_payload:\n", 1)[1])
            self.assertEqual(payload["retained_strategy_note"], "Z" * 1000)
            self.assertNotIn("Z" * 1001, policy.prompts[1])

    def test_retry_costs_once_and_does_not_score_a_second_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=1, agent="llm-recent")
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            responses = root / "run" / "responses"

            class OrderEvaluator(BanditEvaluator):
                def reveal(self, action_index: int, action_id: str):
                    self.files_before_score = list(responses.glob("d*-a*.json"))
                    return super().reveal(action_index, action_id)

            evaluator = OrderEvaluator(spec)
            result, _policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="retry",
                carry=False,
                world="w1",
                condition="recent",
                memory_condition="recent-only",
                client=ScriptedClient(["not-json", reply("arm_a", "kept")]),
                ledger=ledger,
                phase="pilot",
                evaluator=evaluator,
            )
            self.assertEqual(result.status, "completed", result.error_private)
            self.assertEqual(len(evaluator.reveal_calls), 1)
            self.assertEqual(len(evaluator.files_before_score), 2)
            self.assertEqual(ledger.attempts, 2)
            self.assertEqual(ledger.retries, 1)
            self.assertEqual(ledger.completed_pilot, 1)
            self.assertEqual(ledger.known_spend, token_cost(100, 20) * 2)
            self.assertEqual(len(result.observed_events), 1)
            self.assertEqual(result.public_failures, [])

    def test_invalid_action_consumes_one_tick_and_reveals_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=1)
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            evaluator = BanditEvaluator(spec)
            result, _policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="invalid",
                carry=True,
                world="w1",
                condition="persistent",
                memory_condition="persistent-note",
                client=ScriptedClient(["nope", "still-nope"]),
                ledger=ledger,
                phase="pilot",
                evaluator=evaluator,
            )
            self.assertEqual(result.status, "completed", result.error_private)
            self.assertEqual(evaluator.reveal_calls, [])
            self.assertEqual(len(result.public_failures), 1)
            self.assertEqual(result.public_failures[0]["reward_revealed"], False)
            self.assertEqual(ledger.attempts, 2)
            self.assertEqual(ledger.retries, 1)
            self.assertEqual(ledger.completed_pilot, 1)
            self.assertEqual(result.observed_events, [])

    def test_transport_failure_does_not_take_a_second_environment_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=2)
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            evaluator = BanditEvaluator(spec)
            result, policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="transport",
                carry=True,
                world="w1",
                condition="persistent",
                memory_condition="persistent-note",
                client=ScriptedClient([TransportError("down"), TransportError("down")]),
                ledger=ledger,
                phase="pilot",
                evaluator=evaluator,
            )
            self.assertEqual(result.status, "actor_failure")
            self.assertEqual(evaluator.reveal_calls, [])
            self.assertEqual(result.committed_index, 0)
            self.assertEqual(ledger.attempts, 2)
            self.assertEqual(ledger.retries, 1)
            self.assertEqual(ledger.completed_pilot, 1)
            self.assertFalse(policy.stop_task)
            self.assertEqual(ledger.unknown_reserved, attempt_cap_usd() * 2)

    def test_auth_is_retried_once_and_then_stops_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=2)
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            evaluator = BanditEvaluator(spec)
            _result, policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="auth",
                carry=False,
                world="w1",
                condition="recent",
                memory_condition="recent-only",
                client=ScriptedClient(
                    [
                        TransportError("denied", status=401, auth_or_quota=True),
                        TransportError("denied", status=401, auth_or_quota=True),
                    ]
                ),
                ledger=ledger,
                phase="pilot",
                evaluator=evaluator,
            )
            self.assertTrue(policy.stop_task)
            self.assertEqual(evaluator.reveal_calls, [])
            self.assertEqual(ledger.attempts, 2)
            self.assertEqual(ledger.retries, 1)

    def test_unknown_usage_keeps_the_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=1, agent="llm-recent")
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=4)
            result, _policy = run_note_lifetime(
                root=ROOT,
                spec=spec,
                run_dir=root / "run",
                run_id="unknown",
                carry=False,
                world="w1",
                condition="recent",
                memory_condition="recent-only",
                client=ScriptedClient([scripted_reply("arm_c", "note", usage="unknown")]),
                ledger=ledger,
                phase="pilot",
            )
            self.assertEqual(result.status, "completed", result.error_private)
            self.assertEqual(ledger.known_spend, 0)
            self.assertEqual(ledger.unknown_attempts, 1)
            self.assertEqual(ledger.unknown_reserved, attempt_cap_usd())
            saved = json.loads((root / "run" / "responses" / "d001-a1.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["usage"]["status"], "unknown")
            self.assertIsNone(saved["usage"]["cost_usd"])

    def test_sealed_futures_do_not_change_the_shared_prefix(self) -> None:
        def segments(later_probs: dict[str, float], regime: str) -> list[dict]:
            return [
                {
                    "start_index": 1,
                    "end_index": 3,
                    "probabilities": {"arm_a": 0.8, "arm_b": 0.2, "arm_c": 0.5},
                    "regime_id": "private-shared",
                    "unreachable_metadata": {"oracle_best_arm": "arm_a", "sentinel": "SENTINEL_PRIVATE_do_not_leak 005"},
                },
                {
                    "start_index": 4,
                    "end_index": 6,
                    "probabilities": later_probs,
                    "regime_id": regime,
                    "unreachable_metadata": {"oracle_best_arm": "arm_b", "sentinel": "SENTINEL_PRIVATE_do_not_leak 005"},
                },
            ]

        prompts = []
        for index, (probs, regime) in enumerate(
            (
                ({"arm_a": 0.8, "arm_b": 0.2, "arm_c": 0.5}, "private-later-a"),
                ({"arm_a": 0.2, "arm_b": 0.8, "arm_c": 0.5}, "private-later-b"),
            )
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = bandit_spec(
                    horizon=6,
                    segments=segments(probs, regime),
                    lifetime_id="life-paired",
                    innovation_seed="innovation-paired",
                    agent="llm-recent",
                )
                ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=8)
                _result, policy = run_note_lifetime(
                    root=ROOT,
                    spec=spec,
                    run_dir=root / "run",
                    run_id=f"future-{index}",
                    carry=False,
                    world="w2",
                    condition="recent",
                    memory_condition="recent-only",
                    client=ScriptedClient([reply("arm_a", "plain-note") for _ in range(6)]),
                    ledger=ledger,
                    phase="pilot",
                )
                prompts.append(list(policy.prompts))
                blob = "\n".join(policy.prompts)
                for token in FORBIDDEN:
                    self.assertNotIn(token, blob)
        self.assertEqual(prompts[0][:4], prompts[1][:4])
        self.assertEqual(len(prompts[0]), 6)

    def test_progress_ledger_accounts_retry_unknown_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            lines: list[str] = []
            ledger = ProgressLedger(
                path,
                planned_smoke=4,
                planned_pilot=4,
                printer=lines.append,
                heartbeat_seconds=3600,
            )
            known = Usage("available", 100, 20, 0, 120, token_cost(100, 20))
            ledger.set_phase("smoke")
            ledger.locate(world="w1", condition="recent", step=1)
            ledger.begin_attempt()
            ledger.finish_attempt(usage=known, failed=False)
            ledger.complete_decision()
            ledger.locate(world="w1", condition="recent", step=2)
            ledger.begin_attempt()
            ledger.finish_attempt(usage=known, failed=True)
            ledger.begin_attempt()
            ledger.finish_attempt(usage=known, failed=False)
            ledger.complete_decision()
            ledger.locate(world="w1", condition="recent", step=3)
            ledger.begin_attempt()
            ledger.finish_attempt(usage=None, failed=False)
            ledger.complete_decision()
            ledger.locate(world="w1", condition="recent", step=4)
            ledger.begin_attempt()
            ledger.finish_attempt(usage=known, failed=True)
            ledger.complete_decision()
            self.assertEqual(ledger.attempts, 5)
            self.assertEqual(ledger.retries, 1)
            self.assertEqual(ledger.failures, 2)
            self.assertEqual(ledger.known_spend, token_cost(100, 20) * 4)
            self.assertEqual(ledger.unknown_reserved, attempt_cap_usd())
            self.assertEqual(ledger.completed_smoke, 4)
            self.assertNotEqual(ledger.snapshot()["percent"], 100)
            ledger.finish("failed")
            self.assertLessEqual(ledger.snapshot()["percent"], 99)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("SECRET-VALUE", text)
            self.assertNotIn("strategy_note", text)
            self.assertNotIn("public_payload", text)
            self.assertTrue(any(line.startswith("PROGRESS [") for line in lines))
            self.assertNotIn("100%", "\n".join(lines))

            done = ProgressLedger(
                Path(tmp) / "done.json",
                planned_smoke=1,
                planned_pilot=0,
                printer=lambda _line: None,
            )
            done.set_phase("smoke")
            done.locate(world="w1", condition="recent", step=1)
            done.begin_attempt()
            done.finish_attempt(usage=known, failed=False)
            done.complete_decision()
            self.assertEqual(done.snapshot()["percent"], 99)
            done.finish("complete")
            self.assertEqual(done.snapshot()["percent"], 100)

    def test_heartbeat_does_not_issue_another_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines: list[str] = []
            ledger = ProgressLedger(
                Path(tmp) / "progress.json",
                planned_smoke=1,
                planned_pilot=1,
                printer=lines.append,
                heartbeat_seconds=0.05,
            )
            ledger.start_heartbeat()
            try:
                ledger.begin_attempt()
                time.sleep(0.2)
                self.assertEqual(ledger.attempts, 0)
                self.assertGreaterEqual(sum("pending" in line for line in lines), 2)
            finally:
                ledger.finish_attempt(
                    usage=Usage("available", 100, 20, 0, 120, token_cost(100, 20)),
                    failed=False,
                )
                ledger.stop_heartbeat()

    def test_checkpoint_rejects_the_gemini_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = bandit_spec(horizon=1)
            ledger = ProgressLedger(root / "progress.json", planned_smoke=0, planned_pilot=1, printer=lambda _line: None)
            policy = GeminiNotePolicy(
                carry_note=True,
                client=ScriptedClient([]),
                handoff=NoteHandoff(True),
                responses_dir=root / "responses",
                ledger=ledger,
                world="w1",
                condition="persistent",
            )
            path = root / "checkpoint.json"
            with self.assertRaises(CheckpointError):
                save_checkpoint(path, spec, policy, NoteCarryUpdater(NoteHandoff(True)), _fresh_state(spec))
            self.assertFalse(path.exists())

    def test_client_redacts_the_key_and_skips_thoughts(self) -> None:
        secret = "SECRET-VALUE"
        captured = []

        def fail(request, timeout):
            del timeout
            captured.append(request)
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "no",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"SECRET-VALUE"}'),
            )

        with self.assertRaises(TransportError) as caught:
            GeminiClient(secret, opener=fail).generate("hello")
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, captured[0].full_url)
        self.assertIn(secret, captured[0].headers["X-goog-api-key"])

        def succeed(request, timeout):
            del timeout
            self.assertNotIn(secret, request.full_url)
            body = {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"thought": True, "text": "hidden-thought"},
                                {"text": '{"action":"arm_a","strategy_note":"seen"}'},
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 7,
                    "thoughtsTokenCount": 3,
                    "totalTokenCount": 21,
                },
                "modelVersion": "gemini-3.5-flash",
            }

            class Response:
                def read(self) -> bytes:
                    return json.dumps(body).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, *args) -> bool:
                    return False

            return Response()

        generation = GeminiClient(secret, opener=succeed).generate("hello")
        self.assertNotIn("hidden-thought", generation.text)
        self.assertEqual(generation.usage.cost_usd, token_cost(11, 10))
        action, note, truncated = parse_model_json(generation.text, ("arm_a", "arm_b", "arm_c"))
        self.assertEqual((action, note, truncated), ("arm_a", "seen", False))

        def missing(request, timeout):
            del request, timeout

            class Response:
                def read(self) -> bytes:
                    return json.dumps(
                        {"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
                    ).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, *args) -> bool:
                    return False

            return Response()

        unknown = GeminiClient(secret, opener=missing).generate("hello")
        self.assertEqual(unknown.usage.status, "unknown")
        self.assertIsNone(unknown.usage.cost_usd)

    def test_parser_accepts_a_fence_and_rejects_a_bad_action(self) -> None:
        action, note, truncated = parse_model_json(
            '```json\n{"action": "arm_b", "strategy_note": "n"}\n```',
            ("arm_a", "arm_b", "arm_c"),
        )
        self.assertEqual((action, note, truncated), ("arm_b", "n", False))
        with self.assertRaises(Exception):
            parse_model_json('{"action": "nope", "strategy_note": "n"}', ("arm_a",))

    def test_source_does_not_import_the_parent_engine(self) -> None:
        root = ROOT / "src" / "single_life"
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    self.assertFalse(name.startswith("llm_rl_engine"), path.name)
                    self.assertNotIn("google.genai", name)

    def test_nonempty_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            path.mkdir()
            (path / "manifest.json").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_fresh(path)


if __name__ == "__main__":
    unittest.main()
