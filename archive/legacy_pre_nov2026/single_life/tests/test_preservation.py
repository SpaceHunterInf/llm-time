"""Regressions for review findings R2a, R3a, R4a, and R4b."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from single_life.artifacts import ArtifactStore, NullUpdater, ScriptedMeanUpdater
from single_life.baselines import RandomAgent
from single_life.cli import atomic_write_text, main
from single_life.contracts import ContractError, canonical_dumps, content_hash
from single_life.fixtures import BanditEvaluator
from single_life.runner import CHECKPOINT_V2_FORMAT, run_lifetime, semantic_stream
from test_causal_kernel import _event, _spec_body


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MutateThenRaise:
    def __init__(self, inner: ScriptedMeanUpdater) -> None:
        self.inner = inner

    def apply(self, view, store):
        if view.decision_index == 3 and store.versions:
            means = store.versions[0].content["arm_means"]
            means[next(iter(means))] = 0.123456
            raise RuntimeError("mutation should already have been rejected")
        return self.inner.apply(view, store)


class ReplaceThenContinue:
    def __init__(self, inner: ScriptedMeanUpdater) -> None:
        self.inner = inner

    def apply(self, view, store):
        if view.decision_index == 3 and store.versions:
            old = store.versions[0]
            content = old.to_dict()["content"]
            arm = next(iter(content["arm_means"]))
            content["arm_means"][arm] = 0.123456
            replacement = dict(old.to_dict())
            replacement["content"] = content
            replacement["content_hash"] = content_hash(content)
            from single_life.contracts import ArtifactVersion

            store.versions[0] = ArtifactVersion.from_dict(replacement)
            return None
        return self.inner.apply(view, store)


class SneakyUpdater(ScriptedMeanUpdater):
    pass


class SneakyPolicy(RandomAgent):
    pass


class PreservationTests(unittest.TestCase):
    def _period_one(self):
        return _spec_body(
            horizon=4,
            interaction_budget=4,
            compute_budget=8,
            agent="random",
            artifact_period=1,
            artifact_window=4,
        )

    def test_historical_artifact_bytes_survive_mutation(self):
        spec = self._period_one()
        inner = ScriptedMeanUpdater(spec.artifact_period, spec.artifact_window)
        raised = run_lifetime(spec, updater=MutateThenRaise(inner))
        self.assertEqual(raised.status, "updater_failure")
        self.assertEqual(raised.committed_index, 2)
        seen = json.loads(raised.actor_inputs[1])["accessible_artifacts"][0]
        final = raised.artifacts[0]
        self.assertEqual(canonical_dumps(seen), canonical_dumps(final))
        self.assertEqual(content_hash(final["content"]), final["content_hash"])
        self.assertNotEqual(next(iter(final["content"]["arm_means"].values())), 0.123456)

        replaced = run_lifetime(
            spec, updater=ReplaceThenContinue(ScriptedMeanUpdater(1, 4))
        )
        self.assertEqual(replaced.status, "updater_failure")
        seen = json.loads(replaced.actor_inputs[1])["accessible_artifacts"][0]
        final = replaced.artifacts[0]
        self.assertEqual(canonical_dumps(seen), canonical_dumps(final))
        self.assertEqual(content_hash(final["content"]), final["content_hash"])

        store = ArtifactStore()
        store.bind_decision(2, [_event(1, "arm_a", 1)])
        supplied = {"arm_means": {"arm_a": 1.0}}
        store.create(
            artifact_id="life:means",
            version="v1",
            kind="empirical-arm-means",
            parent_version=None,
            creation_cutoff=1,
            supporting_observed_event_ids=["life:event:1"],
            content=supplied,
        )
        supplied["arm_means"]["arm_a"] = 0.0
        stored = store.by_version("v1")
        self.assertEqual(stored.content["arm_means"]["arm_a"], 1.0)
        exported = stored.to_dict()
        exported["content"]["arm_means"]["arm_a"] = 9
        self.assertEqual(store.by_version("v1").content["arm_means"]["arm_a"], 1.0)

    def test_component_identity_rejects_mismatch_before_reveal(self):
        spec = _spec_body(
            horizon=8,
            interaction_budget=8,
            compute_budget=20,
            agent="random",
            artifact_period=2,
            artifact_window=4,
        )
        evaluator = BanditEvaluator(spec)
        with self.assertRaises(ContractError):
            run_lifetime(spec, updater=ScriptedMeanUpdater(3, 1), evaluator=evaluator)
        self.assertEqual(evaluator.reveal_calls, [])
        evaluator = BanditEvaluator(spec)
        with self.assertRaises(ContractError):
            run_lifetime(spec, updater=SneakyUpdater(2, 4), evaluator=evaluator)
        self.assertEqual(evaluator.reveal_calls, [])
        evaluator = BanditEvaluator(spec)
        with self.assertRaises(ContractError):
            run_lifetime(spec, policy=SneakyPolicy(), evaluator=evaluator)
        self.assertEqual(evaluator.reveal_calls, [])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            run_lifetime(spec, stop_after=4, checkpoint_path=checkpoint)
            evaluator = BanditEvaluator(spec)
            with self.assertRaises(ContractError):
                run_lifetime(
                    spec,
                    updater=ScriptedMeanUpdater(3, 1),
                    resume_path=checkpoint,
                    evaluator=evaluator,
                )
            self.assertEqual(evaluator.reveal_calls, [])
            v2 = {
                "format": CHECKPOINT_V2_FORMAT,
                "complete": True,
                "sha256": "0" * 64,
                "state": {"protocol_version": "single-life-v0.2"},
            }
            v2_path = Path(directory) / "v2.json"
            v2_path.write_text(canonical_dumps(v2), encoding="utf-8")
            evaluator = BanditEvaluator(spec)
            with self.assertRaises(Exception) as caught:
                run_lifetime(spec, resume_path=v2_path, evaluator=evaluator)
            self.assertIn("single-life-checkpoint-v2", str(caught.exception))
            self.assertEqual(evaluator.reveal_calls, [])
            resumed = run_lifetime(spec, resume_path=checkpoint)
            full = run_lifetime(spec)
            self.assertEqual(semantic_stream(resumed), semantic_stream(full))
            self.assertEqual(
                [item["deployment_decision_index"] for item in resumed.artifacts],
                [item["deployment_decision_index"] for item in full.artifacts],
            )

    def test_interrupted_write_keeps_the_committed_prefix(self):
        spec = _spec_body(horizon=4, interaction_budget=4, compute_budget=8, agent="random")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(canonical_dumps(spec.to_dict()), encoding="utf-8")
            run_dir = root / "interrupted"
            original = atomic_write_text

            def interrupt(path: Path, text: str) -> None:
                if path.name == "public_log.jsonl" and '"action_index":2' in text:
                    raise KeyboardInterrupt("interruption before replace")
                original(path, text)

            with patch("single_life.cli.atomic_write_text", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    main(["run", "--config", str(config), "--run-dir", str(run_dir)])
            public = (run_dir / "public_log.jsonl").read_text(encoding="utf-8")
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn('"action_index":1', public)
            self.assertNotIn('"action_index":2', public)
            self.assertEqual(manifest["status"], "running")
            self.assertNotEqual(manifest["status"], "completed")
            self.assertEqual(manifest["actual_usage"]["observed_events"], 1)
            self.assertEqual(_sha256(run_dir / "public_log.jsonl"), manifest["hashes"]["public_log_sha256"])
            self.assertEqual(_sha256(run_dir / "private_log.jsonl"), manifest["hashes"]["private_log_sha256"])
            self.assertEqual(_sha256(run_dir / "artifacts.json"), manifest["hashes"]["artifacts_sha256"])
            self.assertFalse(any(path.suffix == ".partial" for path in run_dir.iterdir()))

    def test_preflight_rejects_collisions_without_running(self):
        spec = _spec_body(horizon=4, interaction_budget=4, compute_budget=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(canonical_dumps(spec.to_dict()), encoding="utf-8")
            marker = root / "marker.txt"
            marker.write_text("unchanged\n", encoding="utf-8")

            def forbid(*args, **kwargs):
                del args, kwargs
                raise AssertionError("run_lifetime was called")

            run_dir = root / "run"
            run_dir.mkdir()
            link = root / "summary-link.jsonl"
            link.symlink_to(run_dir / "public_log.jsonl")
            before = sorted(path.name for path in run_dir.iterdir())
            with patch("single_life.cli.run_lifetime", forbid):
                code = main(
                    [
                        "run",
                        "--config",
                        str(config),
                        "--run-dir",
                        str(run_dir),
                        "--summary",
                        str(link),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(sorted(path.name for path in run_dir.iterdir()), before)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged\n")

            previous = os.getcwd()
            os.chdir(root)
            try:
                with patch("single_life.cli.run_lifetime", forbid):
                    code = main(
                        [
                            "run",
                            "--config",
                            "config.json",
                            "--run-dir",
                            "fresh",
                            "--summary",
                            "fresh/public_log.jsonl",
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertFalse((root / "fresh").exists())
                with patch("single_life.cli.run_lifetime", forbid):
                    code = main(
                        [
                            "run",
                            "--config",
                            "config.json",
                            "--run-dir",
                            "other",
                            "--example-decision",
                            "0",
                            "--example-out",
                            "example.json",
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertFalse((root / "other").exists())
                blocker = root / "not-a-directory"
                blocker.write_text("x", encoding="utf-8")
                with patch("single_life.cli.run_lifetime", forbid):
                    code = main(
                        [
                            "run",
                            "--config",
                            "config.json",
                            "--run-dir",
                            "elsewhere",
                            "--summary",
                            "not-a-directory/summary.json",
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertEqual(blocker.read_text(encoding="utf-8"), "x")
            finally:
                os.chdir(previous)

            summary = root / "summary.json"
            example = root / "example.json"
            valid = root / "valid-run"
            code = main(
                [
                    "run",
                    "--config",
                    str(config),
                    "--run-dir",
                    str(valid),
                    "--summary",
                    str(summary),
                    "--example-decision",
                    "1",
                    "--example-out",
                    str(example),
                ]
            )
            self.assertEqual(code, 0)
            saved = json.loads(summary.read_text(encoding="utf-8"))
            manifest = json.loads((valid / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["hashes"], manifest["hashes"])
            self.assertEqual(_sha256(valid / "public_log.jsonl"), saved["hashes"]["public_log_sha256"])
            self.assertEqual(_sha256(valid / "private_log.jsonl"), saved["hashes"]["private_log_sha256"])
            self.assertEqual(_sha256(valid / "artifacts.json"), saved["hashes"]["artifacts_sha256"])
            self.assertTrue(example.is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged\n")


if __name__ == "__main__":
    unittest.main()
