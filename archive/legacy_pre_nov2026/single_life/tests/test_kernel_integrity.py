"""Regressions for review findings R1–R4. Infrastructure checks, not a method result."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from single_life import PROTOCOL_VERSION
from single_life.artifacts import NullUpdater
from single_life.baselines import make_policy
from single_life.cli import main
from single_life.contracts import (
    ArtifactVersion,
    ContractError,
    PublicView,
    bind_public_view,
    canonical_dumps,
    content_hash,
    sha256_text,
)
from single_life.fixtures import BanditEvaluator, load_spec, spec_from_dict
from single_life.runner import CHECKPOINT_V1_FORMAT, load_checkpoint, run_lifetime, semantic_stream
from test_causal_kernel import _event, _spec_body

ROOT = Path(__file__).resolve().parents[1]


def _view_body(decision, events, artifacts=None, actions=("arm_a", "arm_b")):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "lifetime_id": "life",
        "decision_index": decision,
        "permitted_action_ids": list(actions),
        "public_current_observation": None,
        "observed_events": [event.to_dict() for event in events],
        "accessible_artifacts": [] if artifacts is None else artifacts,
        "remaining_public_resources": {
            "interactions_remaining": 8,
            "compute_budget_remaining": 8,
        },
    }


class BackdatedUpdater:
    def apply(self, view, store):
        if view.decision_index != 4:
            return None
        event = view.observed_events[0]
        store.create(
            artifact_id="late",
            version="v1",
            kind="configuration",
            parent_version=None,
            creation_cutoff=1,
            supporting_observed_event_ids=[event.event_id],
            content={"arm_means": {"arm_a": 1}},
        )
        store.deploy("v1", 2)


class HonestLateUpdater:
    def apply(self, view, store):
        if view.decision_index != 4:
            return None
        event = view.observed_events[0]
        store.create(
            artifact_id="late",
            version="v1",
            kind="configuration",
            parent_version=None,
            creation_cutoff=1,
            supporting_observed_event_ids=[event.event_id],
            content={"kept": True},
        )
        store.deploy("v1")


class ForgedSupportUpdater:
    def apply(self, view, store):
        if view.decision_index != 2:
            return None
        store.create(
            artifact_id="forged",
            version="v1",
            kind="configuration",
            parent_version=None,
            creation_cutoff=1,
            supporting_observed_event_ids=["not-observed"],
            content={"arm_means": {}},
        )


class DirectForgeUpdater:
    """Bypass deploy() and append a backdated artifact object."""

    def apply(self, view, store):
        if view.decision_index != 3 or not view.observed_events:
            return None
        event = view.observed_events[0]
        content = {"arm_means": {"arm_a": 1}}
        forged = {
            "artifact_id": "forged",
            "version": "v9",
            "kind": "configuration",
            "parent_version": None,
            "creation_cutoff": 1,
            "creation_decision_index": 2,
            "supporting_observed_event_ids": [event.event_id],
            "content": content,
            "content_hash": content_hash(content),
            "deployment_decision_index": 2,
        }
        store.versions.append(ArtifactVersion.from_dict(forged))


class BoomAfterReward:
    def __init__(self) -> None:
        self.selects = 0

    def select(self, view, rng):
        del rng
        self.selects += 1
        return "arm_a"

    def observe(self, events):
        if events and events[-1].action_index >= 2:
            raise RuntimeError("after observed reward")

    def state_dict(self):
        return {}

    def load_state_dict(self, data):
        if data != {}:
            raise ContractError(str(data))


class IntegrityTests(unittest.TestCase):
    def test_future_observation_rejected_at_serialized_boundary(self):
        body = _view_body(2, [_event(60, "arm_a", 1)])
        with self.assertRaises(ContractError):
            PublicView.from_dict(body)
        loose = PublicView(
            protocol_version=PROTOCOL_VERSION,
            lifetime_id="life",
            decision_index=2,
            permitted_action_ids=("arm_a", "arm_b"),
            public_current_observation=None,
            observed_events=(_event(60, "arm_a", 1),),
            accessible_artifacts=(),
            remaining_public_resources={
                "interactions_remaining": 8,
                "compute_budget_remaining": 8,
            },
        )
        with self.assertRaises(ContractError):
            bind_public_view(loose)
        legal_gap = _view_body(4, [_event(1, "arm_a", 1), _event(3, "arm_b", 0)])
        parsed = PublicView.from_dict(legal_gap)
        self.assertEqual([event.action_index for event in parsed.observed_events], [1, 3])
        with self.assertRaises(ContractError):
            PublicView.from_dict(_view_body(4, [_event(2, "arm_a", 1), _event(1, "arm_a", 0)]))
        with self.assertRaises(ContractError):
            PublicView.from_dict(
                _view_body(3, [_event(1, "arm_a", 1), _event(1, "arm_b", 0, lifetime="other")])
            )
        with self.assertRaises(ContractError):
            PublicView.from_dict(_view_body(3, [_event(1, "not-an-arm", 1)]))
        spec = _spec_body(horizon=3, interaction_budget=3, compute_budget=3)
        result = run_lifetime(spec, updater=NullUpdater())
        self.assertEqual(result.status, "completed")
        for payload in result.actor_inputs + result.updater_inputs:
            PublicView.from_dict(json.loads(payload))
        tampered = json.loads(result.actor_inputs[1])
        tampered["observed_events"] = [_event(60, "arm_a", 1).to_dict()]
        with self.assertRaises(ContractError):
            PublicView.from_dict(tampered)

    def test_backdated_deployment_rejected_and_stale_evidence_kept(self):
        spec = _spec_body(horizon=4, interaction_budget=4, compute_budget=4, agent="random")
        evaluator = BanditEvaluator(spec)
        rejected = run_lifetime(spec, updater=BackdatedUpdater(), evaluator=evaluator)
        self.assertEqual(rejected.status, "updater_failure")
        self.assertEqual(rejected.artifacts, [])
        self.assertEqual([event["action_index"] for event in rejected.observed_events], [1, 2, 3])
        self.assertEqual([call[0] for call in evaluator.reveal_calls], [1, 2, 3])
        self.assertNotIn(2, [artifact.get("deployment_decision_index") for artifact in rejected.artifacts])
        honest_evaluator = BanditEvaluator(spec)
        honest = run_lifetime(spec, updater=HonestLateUpdater(), evaluator=honest_evaluator)
        self.assertEqual(honest.status, "completed")
        self.assertEqual(honest.artifacts[0]["creation_cutoff"], 1)
        self.assertEqual(honest.artifacts[0]["creation_decision_index"], 4)
        self.assertEqual(honest.artifacts[0]["deployment_decision_index"], 4)
        self.assertEqual(json.loads(honest.actor_inputs[1])["accessible_artifacts"], [])
        self.assertEqual(
            json.loads(honest.actor_inputs[3])["accessible_artifacts"][0]["deployment_decision_index"],
            4,
        )
        forged = run_lifetime(spec, updater=ForgedSupportUpdater(), evaluator=BanditEvaluator(spec))
        self.assertEqual(forged.status, "updater_failure")
        self.assertEqual(forged.observed_events[0]["action_index"], 1)
        direct = run_lifetime(spec, updater=DirectForgeUpdater(), evaluator=BanditEvaluator(spec))
        self.assertEqual(direct.status, "updater_failure")
        self.assertEqual(direct.artifacts, [])

    def test_inconsistent_checkpoint_rejected_before_reveal(self):
        spec = _spec_body(horizon=8, interaction_budget=8, compute_budget=5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            run_lifetime(spec, stop_after=4, checkpoint_path=checkpoint)
            good = run_lifetime(spec, resume_path=checkpoint)
            self.assertEqual(good.status, "compute_budget_exhausted")
            self.assertEqual(good.committed_index, 5)
            self.assertEqual(good.compute_units_consumed, 5)
            envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
            envelope["state"]["remaining_compute"] = spec.compute_budget
            envelope["sha256"] = sha256_text(canonical_dumps(envelope["state"]))
            checkpoint.write_text(canonical_dumps(envelope), encoding="utf-8")
            evaluator = BanditEvaluator(spec)
            with self.assertRaises(Exception) as caught:
                run_lifetime(spec, resume_path=checkpoint, evaluator=evaluator)
            self.assertIn("compute accounting", str(caught.exception))
            self.assertEqual(evaluator.reveal_calls, [])
            v1 = {
                "format": CHECKPOINT_V1_FORMAT,
                "complete": True,
                "sha256": "0" * 64,
                "state": {"protocol_version": "single-life-v0.1"},
            }
            v1_path = Path(directory) / "v1.json"
            v1_path.write_text(canonical_dumps(v1), encoding="utf-8")
            evaluator = BanditEvaluator(spec)
            with self.assertRaises(Exception) as caught:
                run_lifetime(spec, resume_path=v1_path, evaluator=evaluator)
            self.assertIn("v0.1", str(caught.exception))
            self.assertEqual(evaluator.reveal_calls, [])
            self.assertEqual(load_checkpoint.__name__, "load_checkpoint")

    def test_failures_before_action_and_after_reward(self):
        spec = _spec_body(horizon=4, interaction_budget=4, compute_budget=4)
        before = BanditEvaluator(spec)

        class BoomUpdater:
            def apply(self, view, store):
                del store
                if view.decision_index == 3:
                    raise RuntimeError("before action")

        result = run_lifetime(spec, updater=BoomUpdater(), evaluator=before)
        self.assertEqual(result.status, "updater_failure")
        self.assertEqual([event["action_index"] for event in result.observed_events], [1, 2])
        self.assertEqual([call[0] for call in before.reveal_calls], [1, 2])
        self.assertFalse(result.public_abort["tick_consumed"])
        self.assertFalse(result.public_abort["reward_revealed"])
        after_policy = BoomAfterReward()
        after_evaluator = BanditEvaluator(spec)
        after = run_lifetime(
            spec, policy=after_policy, updater=NullUpdater(), evaluator=after_evaluator
        )
        self.assertEqual(after.status, "post_observation_failure")
        self.assertEqual([event["action_index"] for event in after.observed_events], [1, 2])
        self.assertEqual(after.observed_events[1]["observed_outcome"] in (0, 1), True)
        self.assertEqual([call[0] for call in after_evaluator.reveal_calls], [1, 2])
        self.assertEqual(after_policy.selects, 2)
        self.assertTrue(after.public_abort["reward_revealed"])
        self.assertTrue(after.public_abort["tick_consumed"])

    def test_cli_preserves_prior_directory_and_failed_prefix(self):
        spec = _spec_body(horizon=6, interaction_budget=6, compute_budget=6, agent="random")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(canonical_dumps(spec.to_dict()), encoding="utf-8")
            prior = root / "partial-run"
            prior.mkdir()
            marker = "prior unfinished run evidence\n"
            (prior / "public_log.jsonl").write_text(marker, encoding="utf-8")

            def forbid_run(*args, **kwargs):
                del args, kwargs
                raise AssertionError("fresh run started on a nonempty directory")

            with patch("single_life.cli.run_lifetime", forbid_run):
                code = main(["run", "--config", str(config), "--run-dir", str(prior)])
            self.assertEqual(code, 2)
            self.assertEqual((prior / "public_log.jsonl").read_text(encoding="utf-8"), marker)

            crashed = root / "crashed-run"

            class DurabilityCheck:
                def apply(self, view, store):
                    del store
                    if view.decision_index == 1:
                        manifest = json.loads((crashed / "manifest.json").read_text(encoding="utf-8"))
                        if manifest["status"] != "running":
                            raise AssertionError(manifest["status"])
                        if not (crashed / "config.json").is_file():
                            raise AssertionError("config missing before the first outcome")
                    if view.decision_index == 3:
                        text = (crashed / "public_log.jsonl").read_text(encoding="utf-8")
                        if '"action_index":1' not in text or '"action_index":2' not in text:
                            raise AssertionError("completed steps were not flushed")
                        raise RuntimeError("injected updater failure")

            original = __import__("single_life.cli", fromlist=["run_lifetime"]).run_lifetime

            def injected(cfg, **kwargs):
                kwargs["updater"] = DurabilityCheck()
                return original(cfg, **kwargs)

            with patch("single_life.cli.run_lifetime", injected):
                code = main(["run", "--config", str(config), "--run-dir", str(crashed)])
            self.assertEqual(code, 1)
            manifest = json.loads((crashed / "manifest.json").read_text(encoding="utf-8"))
            public = (crashed / "public_log.jsonl").read_text(encoding="utf-8")
            private = (crashed / "private_log.jsonl").read_text(encoding="utf-8")
            self.assertEqual(manifest["status"], "updater_failure")
            self.assertIn('"action_index":1', public)
            self.assertIn('"action_index":2', public)
            self.assertNotIn('"action_index":3', public)
            self.assertIn("injected updater failure", private)
            self.assertNotIn("injected updater failure", public)
            refused = main(["run", "--config", str(config), "--run-dir", str(crashed)])
            self.assertEqual(refused, 2)
            self.assertIn('"action_index":1', (crashed / "public_log.jsonl").read_text(encoding="utf-8"))

            rewarded = root / "rewarded-run"
            policy = BoomAfterReward()

            def injected_policy(cfg, **kwargs):
                kwargs["policy"] = policy
                kwargs["updater"] = NullUpdater()
                return original(cfg, **kwargs)

            with patch("single_life.cli.run_lifetime", injected_policy):
                code = main(["run", "--config", str(config), "--run-dir", str(rewarded)])
            self.assertEqual(code, 1)
            rewarded_public = (rewarded / "public_log.jsonl").read_text(encoding="utf-8")
            self.assertEqual(rewarded_public.count('"record_type":"observed_event"'), 2)
            self.assertIn('"action_index":2', rewarded_public)
            self.assertEqual(policy.selects, 2)
            rewarded_manifest = json.loads((rewarded / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(rewarded_manifest["status"], "post_observation_failure")

    def test_resume_matrix_and_independent_process(self):
        cases = (
            ("boundary", self._boundary_spec(), 61),
            ("deployment", _spec_body(horizon=8, interaction_budget=8, compute_budget=20, artifact_period=4, artifact_window=8), 5),
            ("invalid", _spec_body(horizon=8, interaction_budget=8, compute_budget=20), 4),
        )
        for agent_name in ("random", "ucb", "artifact"):
            for label, spec, stop in cases:
                with self.subTest(agent=agent_name, case=label):
                    spec = spec_from_dict({**spec.to_dict(), "agent": agent_name})
                    policy = make_policy(agent_name, spec.ucb_window)
                    resumed_policy = make_policy(agent_name, spec.ucb_window)
                    forced = None
                    if label == "invalid":
                        forced = ["arm_a", "arm_a", "not-an-arm"] + ["arm_a"] * 5
                    with tempfile.TemporaryDirectory() as directory:
                        checkpoint = Path(directory) / "checkpoint.json"
                        run_lifetime(
                            spec,
                            policy=policy,
                            stop_after=stop,
                            checkpoint_path=checkpoint,
                            forced_actions=forced,
                        )
                        resumed = run_lifetime(
                            spec,
                            policy=resumed_policy,
                            resume_path=checkpoint,
                            forced_actions=forced,
                        )
                    full_policy = make_policy(agent_name, spec.ucb_window)
                    full = run_lifetime(spec, policy=full_policy, forced_actions=forced)
                    self.assertEqual(semantic_stream(resumed), semantic_stream(full))
                    if label == "deployment":
                        self.assertTrue(any(item["deployment_decision_index"] == 5 for item in full.artifacts))
                    if label == "invalid":
                        self.assertTrue(any(item["decision_index"] == 3 for item in full.public_failures))
                    if label == "boundary":
                        self.assertGreaterEqual(full.committed_index, 61)

        process_spec = _spec_body(horizon=8, interaction_budget=8, compute_budget=20, agent="random")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.json"
            spec_path = directory / "spec.json"
            out = directory / "resumed.json"
            spec_path.write_text(canonical_dumps(process_spec.to_dict()), encoding="utf-8")
            run_lifetime(process_spec, stop_after=4, checkpoint_path=checkpoint)
            full = run_lifetime(process_spec)
            script = (
                "import json, sys\n"
                "from pathlib import Path\n"
                "from single_life.fixtures import spec_from_dict\n"
                "from single_life.runner import run_lifetime\n"
                "spec = spec_from_dict(json.loads(Path(sys.argv[1]).read_text()))\n"
                "result = run_lifetime(spec, resume_path=sys.argv[2])\n"
                "Path(sys.argv[3]).write_text(json.dumps(result.semantic_stream()))\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script, str(spec_path), str(checkpoint), str(out)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), semantic_stream(full))

    def _boundary_spec(self):
        raw = load_spec(ROOT / "configs" / "dev_continuing.json").to_dict()
        first, second = raw["segments"][0], dict(raw["segments"][1])
        second["end_index"] = 65
        raw.update(horizon=65, interaction_budget=65, compute_budget=200, segments=[first, second])
        return spec_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
