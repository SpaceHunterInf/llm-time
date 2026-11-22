"""Protocol integrity checks for the task-001 causal kernel.

These tests establish the runner boundary. They are not a performance claim.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from single_life.artifacts import ArtifactStore, NullUpdater, ScriptedMeanUpdater
from single_life.baselines import (
    ArtifactConfigAgent,
    RandomAgent,
    SlidingWindowUCB,
    choose_ucb,
)
from single_life.cli import main
from single_life.contracts import (
    ArtifactVersion,
    CheckpointError,
    ContractError,
    EvidenceError,
    ObservedEvent,
    PublicView,
    canonical_dumps,
    content_hash,
    sha256_text,
)
from single_life.fixtures import (
    BanditEvaluator,
    innovation_unit,
    load_spec,
    spec_from_dict,
)
from single_life.runner import load_checkpoint, run_lifetime, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "SENTINEL_PRIVATE_do_not_leak"
BANNED_KEYS = {
    "chosen_probability",
    "evaluator_diagnostics",
    "evaluator_private_note",
    "evaluator_state",
    "git_commit",
    "innovation_ref",
    "innovation_u",
    "oracle_best_arm",
    "probabilities",
    "regime_id",
    "schedule_position",
    "schedule_seed",
    "source_sha256",
    "unreachable_metadata",
    "worktree_dirty",
}


def _event(index: int, action: str, reward: float, lifetime: str = "life") -> ObservedEvent:
    return ObservedEvent(
        event_id=f"{lifetime}:event:{index}",
        action_index=index,
        chosen_action=action,
        observed_outcome=reward,
        public_observation=None,
        actual_interaction_cost=1,
    )


def _view(events, artifacts=(), decision_index=None, actions=("arm_a", "arm_b")) -> PublicView:
    decision = len(events) + 1 if decision_index is None else decision_index
    return PublicView(
        protocol_version="single-life-v0.2",
        lifetime_id="life",
        decision_index=decision,
        permitted_action_ids=tuple(actions),
        public_current_observation=None,
        observed_events=tuple(events),
        accessible_artifacts=tuple(artifacts),
        remaining_public_resources={
            "interactions_remaining": 10,
            "compute_budget_remaining": 10,
        },
    )


def _spec_body(**overrides):
    body = {
        "experiment_id": "001-causal-kernel",
        "fixture_id": "unit",
        "lifetime_id": "life-unit",
        "action_ids": ["arm_a", "arm_b", "arm_c"],
        "horizon": 6,
        "interaction_budget": 6,
        "compute_budget": 100,
        "schedule_seed": "schedule-unit",
        "innovation_seed": "innovation-unit",
        "agent_seed": "agent-unit",
        "agent": "random",
        "ucb_window": 30,
        "artifact_period": 10,
        "artifact_window": 30,
        "protocol_version": "single-life-v0.2",
        "evaluator_private_note": f"{SENTINEL} unit",
        "segments": [
            {
                "start_index": 1,
                "end_index": 6,
                "probabilities": {"arm_a": 0.0, "arm_b": 1.0, "arm_c": 0.5},
                "regime_id": "regime-unit",
                "unreachable_metadata": {"note": SENTINEL, "oracle_best_arm": "arm_b"},
            }
        ],
    }
    body.update(overrides)
    if "segments" not in overrides and "horizon" in overrides:
        body["segments"][0]["end_index"] = overrides["horizon"]
    return spec_from_dict(body)


def _keys(value, found):
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)


def _assert_clean(test, payload: str):
    test.assertNotIn(SENTINEL, payload)
    found = []
    _keys(json.loads(payload), found)
    leaked = sorted(set(found) & BANNED_KEYS)
    test.assertEqual(leaked, [])


class RecordingUpdater:
    def __init__(self, inner):
        self.inner = inner
        self.views = []

    def apply(self, view, store):
        if type(view) is not PublicView:
            raise AssertionError(type(view))
        self.views.append(view)
        return self.inner.apply(view, store)


class RecordingPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.views = []

    def select(self, view, rng):
        if type(view) is not PublicView:
            raise AssertionError(type(view))
        self.views.append(view)
        return self.inner.select(view, rng)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, data):
        self.inner.load_state_dict(data)


class BadAction:
    def __init__(self, decision_index, value):
        self.decision_index = decision_index
        self.value = value

    def select(self, view, rng):
        del rng
        if view.decision_index == self.decision_index:
            return self.value
        return "arm_a"

    def state_dict(self):
        return {}

    def load_state_dict(self, data):
        if data != {}:
            raise ContractError("bad policy state")


class KernelTests(unittest.TestCase):
    def test_development_fixture_shape(self):
        continuing = load_spec(ROOT / "configs" / "dev_continuing.json")
        stationary = load_spec(ROOT / "configs" / "dev_stationary.json")
        self.assertEqual(continuing.horizon, 180)
        self.assertEqual(stationary.horizon, 180)
        self.assertEqual(continuing.action_ids, ("arm_a", "arm_b", "arm_c"))
        self.assertEqual([segment.start_index for segment in continuing.segments], [1, 61, 121])
        self.assertEqual(len(stationary.segments), 1)
        self.assertEqual(
            continuing.segments[0].probabilities, stationary.segments[0].probabilities
        )
        self.assertEqual(continuing.innovation_seed, stationary.innovation_seed)
        self.assertEqual(continuing.lifetime_id, stationary.lifetime_id)
        self.assertNotEqual(continuing.schedule_seed, stationary.schedule_seed)
        for spec in (continuing, stationary):
            for segment in spec.segments:
                self.assertEqual(set(segment.probabilities), set(spec.action_ids))

    def test_contracts_reject_invalid_boundary_inputs(self):
        event = _event(1, "arm_a", 1).to_dict()
        with self.subTest("extra field"):
            broken = dict(event)
            broken["regime_id"] = "regime-early"
            with self.assertRaises(ContractError):
                ObservedEvent.from_dict(broken)
        with self.subTest("non-finite reward"):
            for value in (float("nan"), float("inf"), float("-inf"), True):
                broken = dict(event)
                broken["observed_outcome"] = value
                with self.assertRaises(ContractError):
                    ObservedEvent.from_dict(broken)
        with self.subTest("public view extra field"):
            payload = json.loads(_view([_event(1, "arm_a", 1)]).canonical_json())
            payload["regime_id"] = "regime-early"
            with self.assertRaises(ContractError):
                PublicView.from_dict(payload)
        with self.subTest("config extra field"):
            raw = _spec_body().to_dict()
            raw["oracle_best_arm"] = "arm_a"
            with self.assertRaises(ContractError):
                spec_from_dict(raw)
        with self.subTest("tampered artifact hash"):
            content = {"arm_means": {"arm_a": 1.0}}
            raw = {
                "artifact_id": "life:means",
                "version": "v1",
                "kind": "empirical-arm-means",
                "parent_version": None,
                "creation_cutoff": 1,
                "creation_decision_index": 2,
                "supporting_observed_event_ids": ["life:event:1"],
                "content": content,
                "content_hash": "0" * 64,
                "deployment_decision_index": 2,
            }
            with self.assertRaises(ContractError):
                ArtifactVersion.from_dict(raw)
            raw["content_hash"] = content_hash(content)
            raw["deployment_decision_index"] = 1
            with self.assertRaises(EvidenceError):
                ArtifactVersion.from_dict(raw)

    def test_evidence_references_and_deployment_order(self):
        store = ArtifactStore()
        store.bind_decision(9, [_event(4, "arm_a", 1), _event(8, "arm_b", 0)])
        with self.subTest("missing evidence"):
            with self.assertRaises(EvidenceError):
                store.create(
                    artifact_id="life:means",
                    version="v1",
                    kind="empirical-arm-means",
                    parent_version=None,
                    creation_cutoff=8,
                    supporting_observed_event_ids=["missing"],
                    content={"arm_means": {}},
                )
        with self.subTest("future evidence"):
            with self.assertRaises(EvidenceError):
                store.create(
                    artifact_id="life:means",
                    version="v1",
                    kind="empirical-arm-means",
                    parent_version=None,
                    creation_cutoff=5,
                    supporting_observed_event_ids=["life:event:8"],
                    content={"arm_means": {}},
                )
        created = store.create(
            artifact_id="life:means",
            version="v1",
            kind="empirical-arm-means",
            parent_version=None,
            creation_cutoff=4,
            supporting_observed_event_ids=["life:event:4"],
            content={"arm_means": {"arm_a": 1.0}},
        )
        with self.subTest("deploy at cutoff"):
            with self.assertRaises(EvidenceError):
                store.deploy(created.version, created.creation_cutoff)
        with self.subTest("unknown version"):
            with self.assertRaises(EvidenceError):
                store.deploy("v9")
        store.deploy(created.version)
        self.assertEqual(created.creation_decision_index, 9)
        self.assertEqual(store.by_version("v1").deployment_decision_index, 9)
        with self.subTest("child before parent"):
            with self.assertRaises(EvidenceError):
                store.create(
                    artifact_id="life:means",
                    version="v2",
                    kind="empirical-arm-means",
                    parent_version="v1",
                    creation_cutoff=4,
                    supporting_observed_event_ids=["life:event:4"],
                    content={"arm_means": {"arm_a": 1.0}},
                )
        with self.subTest("missing parent"):
            with self.assertRaises(EvidenceError):
                store.create(
                    artifact_id="life:means",
                    version="v3",
                    kind="empirical-arm-means",
                    parent_version="missing",
                    creation_cutoff=8,
                    supporting_observed_event_ids=["life:event:4"],
                    content={"arm_means": {}},
                )

    def test_counting_bug_prefix_supplied_repeatedly(self):
        events = [_event(1, "arm_a", 1), _event(2, "arm_a", 0), _event(3, "arm_b", 1)]
        agent = SlidingWindowUCB(window=50)
        fresh = SlidingWindowUCB(window=50)
        for _ in range(3):
            agent.observe(events)
        agent.observe(events + events + events)
        self.assertEqual(agent.unique_count, 3)
        self.assertEqual(agent.counts_by_arm(), {"arm_a": 2, "arm_b": 1})
        view = _view(events, actions=("arm_a", "arm_b"))
        self.assertEqual(
            agent.select(view, random.Random(0)),
            fresh.select(view, random.Random(0)),
        )
        spec = _spec_body(horizon=3, interaction_budget=3, agent="ucb", artifact_period=10)
        policy = SlidingWindowUCB(window=spec.ucb_window)
        result = run_lifetime(spec, policy=policy, updater=NullUpdater())
        self.assertEqual(result.status, "completed")
        self.assertEqual(policy.unique_count, 3)
        self.assertEqual(sum(policy.counts_by_arm().values()), 3)

    def test_ucb_window_is_global_time(self):
        self.assertEqual(choose_ucb((), 1, 5, ["arm_b", "arm_a"]), "arm_a")
        tied = [_event(1, "arm_a", 1), _event(2, "arm_b", 1)]
        self.assertEqual(choose_ucb(tied, 3, 5, ["arm_a", "arm_b"]), "arm_a")
        events = [
            _event(1, "arm_b", 0),
            _event(2, "arm_a", 1),
            _event(3, "arm_b", 1),
            _event(4, "arm_a", 0),
        ]
        self.assertEqual(choose_ucb(events, 5, 2, ["arm_a", "arm_b"]), "arm_b")
        self.assertEqual(choose_ucb(events, 5, 4, ["arm_a", "arm_b"]), "arm_a")
        conflicting = [_event(1, "arm_a", 1), _event(1, "arm_a", 0)]
        with self.assertRaises(ContractError):
            choose_ucb(conflicting, 2, 5, ["arm_a", "arm_b"])

    def test_artifact_content_changes_the_next_action(self):
        spec = _spec_body(
            horizon=5,
            interaction_budget=5,
            agent="artifact",
            artifact_period=4,
            artifact_window=10,
            segments=[
                {
                    "start_index": 1,
                    "end_index": 5,
                    "probabilities": {"arm_a": 0.0, "arm_b": 1.0, "arm_c": 0.0},
                    "regime_id": "regime-tiny",
                    "unreachable_metadata": {"note": SENTINEL, "oracle_best_arm": "arm_b"},
                }
            ],
        )
        updater = RecordingUpdater(ScriptedMeanUpdater(spec.artifact_period, spec.artifact_window))
        policy = RecordingPolicy(ArtifactConfigAgent())
        result = run_lifetime(
            spec,
            policy=policy,
            updater=updater,
            forced_actions=["arm_b", "arm_b", "arm_b", "arm_a", None],
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.actions[4]["action"], "arm_b")
        actor = json.loads(result.actor_inputs[4])
        updater_input = json.loads(result.updater_inputs[4])
        self.assertEqual(updater_input["accessible_artifacts"], [])
        self.assertEqual(len(actor["accessible_artifacts"]), 1)
        artifact = actor["accessible_artifacts"][0]
        self.assertEqual(artifact["creation_cutoff"], 4)
        self.assertEqual(artifact["deployment_decision_index"], 5)
        self.assertEqual(set(artifact["content"]), {"arm_means", "evidence_count", "window", "window_definition"})
        self.assertLess(
            max(
                int(event_id.rsplit(":", 1)[1])
                for event_id in artifact["supporting_observed_event_ids"]
            ),
            5,
        )
        self.assertGreater(artifact["content"]["arm_means"]["arm_b"], artifact["content"]["arm_means"]["arm_a"])
        visible = PublicView.from_dict(actor)
        hidden = PublicView(
            protocol_version=visible.protocol_version,
            lifetime_id=visible.lifetime_id,
            decision_index=visible.decision_index,
            permitted_action_ids=visible.permitted_action_ids,
            public_current_observation=visible.public_current_observation,
            observed_events=visible.observed_events,
            accessible_artifacts=(),
            remaining_public_resources=visible.remaining_public_resources,
        )
        rng = random.Random(0)
        self.assertEqual(ArtifactConfigAgent().select(visible, rng), "arm_b")
        self.assertEqual(ArtifactConfigAgent().select(hidden, rng), "arm_a")
        self.assertEqual(type(policy.views[-1]), PublicView)
        self.assertEqual(policy.views[-1].decision_index, 5)
        self.assertEqual(type(updater.views[4]), PublicView)

    def test_future_perturbation_and_unreachable_metadata(self):
        raw = load_spec(ROOT / "configs" / "dev_continuing.json").to_dict()
        raw["horizon"] = 70
        raw["interaction_budget"] = 70
        raw["compute_budget"] = 1000
        prefix = raw["segments"][0]
        future_a = {
            "start_index": 61,
            "end_index": 70,
            "probabilities": {"arm_a": 0.0, "arm_b": 0.0, "arm_c": 0.0},
            "regime_id": "regime-future-a",
            "unreachable_metadata": {"note": SENTINEL, "oracle_best_arm": "arm_a"},
        }
        future_b = {
            "start_index": 61,
            "end_index": 70,
            "probabilities": {"arm_a": 1.0, "arm_b": 1.0, "arm_c": 1.0},
            "regime_id": "regime-future-b",
            "unreachable_metadata": {
                "note": f"{SENTINEL} future-only",
                "oracle_best_arm": "arm_c",
                "decoy_unreached": {"start_index": 1000, "label": "not-installed"},
            },
        }
        spec_a = spec_from_dict(
            {**raw, "fixture_id": "future-a", "schedule_seed": "schedule-future-a", "segments": [prefix, future_a]}
        )
        spec_b = spec_from_dict(
            {
                **raw,
                "fixture_id": "future-b",
                "schedule_seed": "schedule-future-b",
                "evaluator_private_note": f"{SENTINEL} other-note",
                "segments": [prefix, future_b],
            }
        )
        result_a = run_lifetime(spec_a)
        result_b = run_lifetime(spec_b)
        self.assertEqual(result_a.updater_inputs[:61], result_b.updater_inputs[:61])
        self.assertEqual(result_a.actor_inputs[:61], result_b.actor_inputs[:61])
        self.assertEqual(result_a.actions[:61], result_b.actions[:61])
        rewards_a = [event["observed_outcome"] for event in result_a.observed_events if event["action_index"] <= 60]
        rewards_b = [event["observed_outcome"] for event in result_b.observed_events if event["action_index"] <= 60]
        self.assertEqual(rewards_a, rewards_b)
        deployed_a = [item for item in result_a.artifacts if item["deployment_decision_index"] <= 61]
        deployed_b = [item for item in result_b.artifacts if item["deployment_decision_index"] <= 61]
        self.assertEqual(deployed_a, deployed_b)
        self.assertTrue(any(item["deployment_decision_index"] == 61 for item in deployed_a))
        self.assertNotEqual(
            result_a.observed_events[60]["observed_outcome"],
            result_b.observed_events[60]["observed_outcome"],
        )
        for event_a, event_b in zip(result_a.private_events[:60], result_b.private_events[:60]):
            self.assertEqual(event_a["innovation_ref"], event_b["innovation_ref"])
            self.assertEqual(
                event_a["evaluator_diagnostics"]["innovation_u"],
                event_b["evaluator_diagnostics"]["innovation_u"],
            )

        metadata_only = spec_from_dict(
            {
                **spec_a.to_dict(),
                "fixture_id": "future-meta",
                "schedule_seed": "schedule-metadata-only",
                "evaluator_private_note": f"{SENTINEL} metadata-only",
                "segments": [
                    {
                        **prefix,
                        "unreachable_metadata": {
                            "note": f"{SENTINEL} prefix-meta",
                            "oracle_best_arm": "arm_c",
                        },
                    },
                    {
                        **future_a,
                        "unreachable_metadata": {
                            "note": f"{SENTINEL} future-meta",
                            "oracle_best_arm": "arm_b",
                            "decoy_unreached": {"start_index": 900},
                        },
                    },
                ],
            }
        )
        result_c = run_lifetime(metadata_only)
        self.assertEqual(result_a.actor_inputs, result_c.actor_inputs)
        self.assertEqual(result_a.updater_inputs, result_c.updater_inputs)
        self.assertEqual(result_a.actions, result_c.actions)
        self.assertEqual(result_a.observed_events, result_c.observed_events)
        self.assertEqual(result_a.artifacts, result_c.artifacts)
        self.assertNotEqual(result_a.private_events, result_c.private_events)

    def test_information_boundary_reaches_actor_and_updater(self):
        spec = load_spec(ROOT / "configs" / "dev_continuing.json")
        raw = spec.to_dict()
        first, second = raw["segments"][0], dict(raw["segments"][1])
        second["end_index"] = 70
        shortened = spec_from_dict(
            {**raw, "horizon": 70, "interaction_budget": 70, "segments": [first, second]}
        )
        updater = RecordingUpdater(ScriptedMeanUpdater(shortened.artifact_period, shortened.artifact_window))
        policy = RecordingPolicy(ArtifactConfigAgent())
        result = run_lifetime(shortened, policy=policy, updater=updater)
        self.assertGreaterEqual(len(policy.views), 61)
        self.assertGreaterEqual(len(updater.views), 61)
        self.assertEqual(len(policy.views), len(result.actor_inputs))
        self.assertEqual(len(updater.views), len(result.updater_inputs))
        for payload in result.actor_inputs + result.updater_inputs:
            _assert_clean(self, payload)
        for view in policy.views + updater.views:
            self.assertIsNone(view.public_current_observation)
            self.assertFalse(hasattr(view, "spec"))
            _assert_clean(self, view.canonical_json())
        private_text = canonical_dumps(result.private_events)
        self.assertIn(SENTINEL, private_text)
        self.assertIn("regime_id", private_text)
        self.assertIn("oracle_best_arm", private_text)

    def test_hidden_change_persists_without_a_boundary_signal(self):
        spec = load_spec(ROOT / "configs" / "dev_continuing.json")
        policy = SlidingWindowUCB(window=spec.ucb_window)
        result = run_lifetime(spec, policy=policy)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.committed_index, 180)
        self.assertEqual(policy.unique_count, 180)
        self.assertIn(f"{spec.lifetime_id}:event:1", policy._by_id)
        for decision in (61, 121):
            payload = json.loads(result.actor_inputs[decision - 1])
            self.assertEqual(payload["decision_index"], decision)
            self.assertEqual(payload["lifetime_id"], spec.lifetime_id)
            self.assertIsNone(payload["public_current_observation"])
            self.assertEqual(len(payload["observed_events"]), decision - 1)
            self.assertEqual(payload["observed_events"][-1]["action_index"], decision - 1)
            self.assertEqual(payload["permitted_action_ids"], ["arm_a", "arm_b", "arm_c"])
            deployed = [
                item["deployment_decision_index"] for item in payload["accessible_artifacts"]
            ]
            self.assertIn(11, deployed)
            self.assertTrue(all(item <= decision for item in deployed))
            _assert_clean(self, result.actor_inputs[decision - 1])
            _assert_clean(self, result.updater_inputs[decision - 1])
        early = [
            item
            for item in result.artifacts
            if item["deployment_decision_index"] is not None and item["deployment_decision_index"] < 61
        ]
        self.assertTrue(early)
        self.assertTrue(any(item["parent_version"] == "v1" for item in result.artifacts))

    def test_pairing_prechange_outcomes_ignore_future_schedule(self):
        continuing = load_spec(ROOT / "configs" / "dev_continuing.json")
        stationary = load_spec(ROOT / "configs" / "dev_stationary.json")
        actions = [("arm_a", "arm_b", "arm_c")[index % 3] for index in range(180)]
        left = BanditEvaluator(continuing)
        right = BanditEvaluator(stationary)
        for index, action in enumerate(actions, start=1):
            if index > 60:
                break
            reward_left, private_left = left.reveal(index, action)
            reward_right, private_right = right.reveal(index, action)
            self.assertEqual(reward_left, reward_right)
            self.assertEqual(private_left.innovation_ref, private_right.innovation_ref)
            self.assertEqual(len(left.reveal_calls), index)
            self.assertEqual(left.reveal_calls[-1][1], action)
        run_left = run_lifetime(continuing, forced_actions=actions, updater=NullUpdater())
        run_right = run_lifetime(stationary, forced_actions=actions, updater=NullUpdater())
        self.assertEqual(
            [event["observed_outcome"] for event in run_left.observed_events[:60]],
            [event["observed_outcome"] for event in run_right.observed_events[:60]],
        )
        later = [
            run_left.observed_events[index]["observed_outcome"]
            != run_right.observed_events[index]["observed_outcome"]
            for index in range(60, 120)
        ]
        self.assertTrue(any(later))
        shifted = spec_from_dict({**continuing.to_dict(), "schedule_seed": "schedule-shifted"})
        for index in range(1, 61):
            for action in continuing.action_ids:
                self.assertEqual(
                    innovation_unit(continuing.innovation_seed, continuing.lifetime_id, index, action),
                    innovation_unit(shifted.innovation_seed, shifted.lifetime_id, index, action),
                )
        other_life = innovation_unit(continuing.innovation_seed, "other-life", 1, "arm_a")
        this_life = innovation_unit(continuing.innovation_seed, continuing.lifetime_id, 1, "arm_a")
        self.assertNotEqual(other_life, this_life)
        same_actions = run_lifetime(shifted, forced_actions=actions, updater=NullUpdater())
        self.assertEqual(run_left.observed_events, same_actions.observed_events)

    def test_resume_matches_uninterrupted_stream(self):
        for agent_name in ("random", "ucb", "artifact"):
            with self.subTest(agent=agent_name):
                spec = _spec_body(
                    agent=agent_name,
                    horizon=20,
                    interaction_budget=20,
                    artifact_period=4,
                    artifact_window=10,
                    segments=[
                        {
                            "start_index": 1,
                            "end_index": 20,
                            "probabilities": {"arm_a": 0.2, "arm_b": 0.8, "arm_c": 0.5},
                            "regime_id": "regime-resume",
                            "unreachable_metadata": {"note": SENTINEL},
                        }
                    ],
                )
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "partial" / "checkpoint.json"
                    run_lifetime(spec, stop_after=8, checkpoint_path=checkpoint)
                    resumed = run_lifetime(spec, resume_path=checkpoint)
                uninterrupted = run_lifetime(spec)
                self.assertEqual(resumed.semantic_stream(), uninterrupted.semantic_stream())
                encoded = json.dumps(resumed.semantic_stream())
                self.assertNotIn("written_at", encoded)
                self.assertNotIn(str(checkpoint.parent), encoded)

    def test_incomplete_checkpoint_is_rejected_without_replay(self):
        spec = _spec_body(horizon=8, interaction_budget=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            run_lifetime(spec, stop_after=4, checkpoint_path=path)
            valid = json.loads(path.read_text(encoding="utf-8"))
            cases = {
                "flag": {**valid, "complete": False},
                "truncated": None,
                "hash": {**valid, "sha256": "0" * 64},
                "missing-key": None,
                "ambiguous-next": None,
            }
            missing = json.loads(canonical_dumps(valid["state"]))
            del missing["agent_rng"]
            cases["missing-key"] = {**valid, "state": missing}
            ambiguous = json.loads(canonical_dumps(valid["state"]))
            ambiguous["next_decision_index"] = 99
            cases["ambiguous-next"] = {
                **valid,
                "state": ambiguous,
                "sha256": sha256_text(canonical_dumps(ambiguous)),
            }
            for name, envelope in cases.items():
                with self.subTest(name=name):
                    bad = Path(directory) / f"{name}.json"
                    if name == "truncated":
                        bad.write_text("{", encoding="utf-8")
                    else:
                        bad.write_text(canonical_dumps(envelope), encoding="utf-8")
                    evaluator = BanditEvaluator(spec)
                    with self.assertRaises(CheckpointError):
                        run_lifetime(spec, resume_path=bad, evaluator=evaluator)
                    self.assertEqual(evaluator.reveal_calls, [])
            self.assertIsInstance(load_checkpoint(path), dict)

    def test_invalid_action_budget_and_evaluator_failure(self):
        spec = _spec_body(horizon=4, interaction_budget=4)
        with self.subTest("invalid string"):
            evaluator = BanditEvaluator(spec)
            result = run_lifetime(
                spec,
                policy=BadAction(2, "not-an-arm"),
                evaluator=evaluator,
                updater=NullUpdater(),
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual([call[0] for call in evaluator.reveal_calls], [1, 3, 4])
            self.assertEqual(len(evaluator.reveal_calls), len({call[0] for call in evaluator.reveal_calls}))
            self.assertEqual(result.public_failures[0]["decision_index"], 2)
            self.assertTrue(result.public_failures[0]["tick_consumed"])
            self.assertFalse(result.public_failures[0]["reward_revealed"])
            self.assertNotIn(2, [event["action_index"] for event in result.observed_events])
        with self.subTest("non-string action"):
            evaluator = BanditEvaluator(spec)
            result = run_lifetime(
                spec,
                policy=BadAction(2, None),
                evaluator=evaluator,
                updater=NullUpdater(),
            )
            self.assertEqual([call[0] for call in evaluator.reveal_calls], [1, 3, 4])
            self.assertIsNone(result.public_failures[0]["attempted_action"])
        with self.subTest("interaction budget"):
            limited = _spec_body(horizon=10, interaction_budget=3)
            evaluator = BanditEvaluator(limited)
            result = run_lifetime(limited, evaluator=evaluator, updater=NullUpdater())
            self.assertEqual(result.status, "budget_exhausted")
            self.assertEqual(result.committed_index, 3)
            self.assertEqual(len(evaluator.reveal_calls), 3)
            self.assertNotIn(4, [call[0] for call in evaluator.reveal_calls])
        with self.subTest("compute budget"):
            limited = _spec_body(horizon=10, interaction_budget=10, compute_budget=2)
            evaluator = BanditEvaluator(limited)
            result = run_lifetime(limited, evaluator=evaluator, updater=NullUpdater())
            self.assertEqual(result.status, "compute_budget_exhausted")
            self.assertEqual(len(evaluator.reveal_calls), 2)
            self.assertEqual(result.committed_index, 2)
        with self.subTest("evaluator failure"):
            faulty = _spec_body(horizon=6, interaction_budget=6)
            evaluator = BanditEvaluator(faulty, fault_decisions={3})
            result = run_lifetime(faulty, evaluator=evaluator, updater=NullUpdater())
            self.assertEqual(result.status, "evaluator_failure")
            self.assertEqual(result.committed_index, 2)
            self.assertEqual([call[0] for call in evaluator.reveal_calls], [1, 2])
            self.assertFalse(result.public_abort["tick_consumed"])
            self.assertFalse(result.public_abort["reward_revealed"])
            self.assertNotIn(3, [event["action_index"] for event in result.observed_events])
            self.assertTrue(result.error_private)

    def test_package_has_no_parent_or_provider_imports(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src" / "single_life").rglob("*.py")
        )
        for banned in (
            "llm_rl_engine",
            "openai",
            "anthropic",
            "dotenv",
            "generativeai",
            "deepseek",
            "ToolSandbox",
            "toolsandbox",
        ):
            self.assertNotIn(banned, source)
        for name in sys.modules:
            self.assertNotIn("llm_rl_engine", name)
            self.assertNotIn("openai", name)
            self.assertNotIn("dotenv", name)

    def test_cli_writes_separated_logs(self):
        spec = _spec_body(horizon=5, interaction_budget=5, agent="artifact", artifact_period=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(canonical_dumps(spec.to_dict()), encoding="utf-8")
            code = main(
                [
                    "run",
                    "--config",
                    str(config),
                    "--run-dir",
                    str(root / "run"),
                    "--summary",
                    str(root / "summary.json"),
                    "--example-decision",
                    "1",
                    "--example-out",
                    str(root / "example.json"),
                ]
            )
            self.assertEqual(code, 0)
            public = (root / "run" / "public_log.jsonl").read_text(encoding="utf-8")
            private = (root / "run" / "private_log.jsonl").read_text(encoding="utf-8")
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertNotIn(SENTINEL, public)
            self.assertIn(SENTINEL, private)
            self.assertEqual(manifest["actual_usage"]["api_calls"], 0)
            self.assertIsNone(manifest["actual_usage"]["api_usd"])
            self.assertIsNone(manifest["actual_usage"]["tokens"])
            self.assertIsNone(manifest["performance_claim"])
            self.assertIsNone(summary["performance_claim"])
            self.assertEqual(summary["status"], "completed")
            _assert_clean(self, (root / "example.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
