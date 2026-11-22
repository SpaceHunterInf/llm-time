"""Commander review probes; reports gaps without modifying implementation or old runs.
Run from single_life/: uv run --offline --no-sync python reviews/001-kernel-probes.py
"""
from __future__ import annotations
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from single_life.artifacts import NullUpdater
from single_life.cli import main
from single_life.contracts import PublicView, ObservedEvent, canonical_dumps, sha256_text
from single_life.fixtures import load_spec, spec_from_dict
from single_life.runner import run_lifetime

ROOT = Path(__file__).resolve().parents[1]

def spec(**overrides):
    raw = load_spec(ROOT / "configs/dev_continuing.json").to_dict()
    raw.update(horizon=8, interaction_budget=8, compute_budget=8, agent="random")
    raw["segments"] = [{**raw["segments"][0], "end_index": 8}]
    raw.update(overrides)
    return spec_from_dict(raw)

def event(index):
    return ObservedEvent(f"life:event:{index}", index, "arm_a", 1, None, 1).to_dict()

def public_probe():
    body = {
        "protocol_version": "single-life-v0.1", "lifetime_id": "life",
        "decision_index": 2, "permitted_action_ids": ["arm_a", "arm_b"],
        "public_current_observation": None, "observed_events": [event(60)],
        "accessible_artifacts": [],
        "remaining_public_resources": {"interactions_remaining": 8, "compute_budget_remaining": 8},
    }
    try:
        parsed = PublicView.from_dict(body)
        return {"accepted_future_observation": True, "decision": parsed.decision_index,
                "observed_action_indexes": [e.action_index for e in parsed.observed_events]}
    except ValueError as exc:
        return {"accepted_future_observation": False, "error": str(exc)}

class BackdatedUpdater:
    def apply(self, view, store):
        if view.decision_index == 4:
            e = view.observed_events[0]
            store.create(artifact_id="late", version="v1", kind="configuration",
                         parent_version=None, creation_cutoff=1,
                         supporting_observed_event_ids=[e.event_id],
                         content={"arm_means": {"arm_a": 1}},
                         catalog={e.event_id: e})
            store.deploy("v1", 2)

def artifact_probe():
    try:
        result = run_lifetime(spec(), updater=BackdatedUpdater())
        at2 = json.loads(result.actor_inputs[1])["accessible_artifacts"]
        at4 = json.loads(result.actor_inputs[3])["accessible_artifacts"]
        return {"accepted_backdated_deployment": True,
                "artifacts_seen_at_2": len(at2), "artifacts_seen_at_4": len(at4),
                "recorded_deployment": result.artifacts[0]["deployment_decision_index"]}
    except ValueError as exc:
        return {"accepted_backdated_deployment": False, "error": str(exc)}

def checkpoint_probe(directory):
    cfg = spec(compute_budget=5)
    checkpoint = directory / "checkpoint.json"
    run_lifetime(cfg, stop_after=4, checkpoint_path=checkpoint)
    good = run_lifetime(cfg, resume_path=checkpoint)
    envelope = json.loads(checkpoint.read_text())
    envelope["state"]["remaining_compute"] = cfg.compute_budget
    envelope["sha256"] = sha256_text(canonical_dumps(envelope["state"]))
    checkpoint.write_text(canonical_dumps(envelope))
    try:
        bad = run_lifetime(cfg, resume_path=checkpoint)
        return {"accepted_inconsistent_budget": True,
                "valid_checkpoint_final_index": good.committed_index,
                "inconsistent_checkpoint_final_index": bad.committed_index,
                "status": bad.status, "compute_used": bad.compute_units_consumed,
                "interactions_used": bad.interactions_consumed}
    except ValueError as exc:
        return {"accepted_inconsistent_budget": False, "error": str(exc)}

class CrashingUpdater:
    def apply(self, view, store):
        if view.decision_index == 3:
            raise RuntimeError("review injected updater failure")

def crash_probe(directory):
    config = directory / "config.json"
    config.write_text(canonical_dumps(spec().to_dict()))
    run_dir = directory / "crashed-run"
    import single_life.cli as cli
    original = cli.run_lifetime
    def injected(cfg):
        return original(cfg, updater=CrashingUpdater())
    caught = None
    try:
        with patch.object(cli, "run_lifetime", injected):
            main(["run", "--config", str(config), "--run-dir", str(run_dir)])
    except RuntimeError as exc:
        caught = str(exc)
    return {"exception": caught, "run_directory_exists": run_dir.exists(),
            "manifest_exists": (run_dir / "manifest.json").exists(),
            "public_log_exists": (run_dir / "public_log.jsonl").exists()}

def overwrite_probe(directory):
    config = directory / "overwrite-config.json"
    config.write_text(canonical_dumps(spec().to_dict()))
    run_dir = directory / "partial-run"
    run_dir.mkdir()
    public = run_dir / "public_log.jsonl"
    marker = "prior unfinished run evidence\n"
    public.write_text(marker)
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["run", "--config", str(config), "--run-dir", str(run_dir)])
    return {"exit_code": code, "prior_log_preserved": public.read_text() == marker}

def main_probes():
    with tempfile.TemporaryDirectory(prefix="single-life-review-") as tmp:
        directory = Path(tmp)
        return {"public_cutoff": public_probe(), "artifact_deployment": artifact_probe(),
                "checkpoint_accounting": checkpoint_probe(directory),
                "uncaught_crash": crash_probe(directory),
                "partial_run_overwrite": overwrite_probe(directory)}

if __name__ == "__main__":
    print(json.dumps(main_probes(), indent=2, sort_keys=True))
