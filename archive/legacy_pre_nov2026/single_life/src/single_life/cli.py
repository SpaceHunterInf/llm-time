"""Command line for one offline fixture lifetime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from single_life.baselines import UCB_DEFINITION
from single_life.contracts import (
    assert_separated_streams,
    canonical_dumps,
    sha256_bytes,
    sha256_text,
)
from single_life.fixtures import load_spec
from single_life.identity import dependency_identity, source_identity
from single_life.runner import clock_fields, run_lifetime

AGENT_IDS = {
    "random": "random-v1",
    "ucb": "sliding-window-ucb-v1",
    "artifact": "scripted-artifact-agent-v1",
    "llm-recent": "gemini-note-recent-v1",
    "llm-note": "gemini-note-persistent-v1",
}
AUTHORITATIVE_NAMES = (
    "config.json",
    "public_log.jsonl",
    "private_log.jsonl",
    "artifacts.json",
    "manifest.json",
)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` only after the new bytes are complete.

    An interruption while writing the temporary file leaves ``path`` unchanged.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="single-life")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one offline fixture lifetime")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--run-dir", required=True, type=Path)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--summary", type=Path, default=None)
    run_parser.add_argument("--example-decision", type=int, default=None)
    run_parser.add_argument("--example-out", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_command(args)


def run_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    spec = load_spec(args.config)
    refusal = _preflight(args, spec)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    run_dir = args.run_dir
    config_bytes = args.config.read_bytes()
    code = source_identity(project_root)
    dependencies = dependency_identity(project_root)
    journal = _RunJournal(
        run_dir=run_dir,
        spec=spec,
        run_id=args.run_id or run_dir.name,
        config_bytes=config_bytes,
        code=code,
        dependencies=dependencies,
    )
    journal.write(status="running", result=None)
    try:
        result = run_lifetime(spec, on_commit=journal.commit)
    except Exception as exc:
        journal.write_exception(exc)
        print(f"status=aborted lifetime={spec.lifetime_id} performance_claim=none", file=sys.stderr)
        return 1
    journal.write(status=result.status, result=result)
    manifest = journal.manifest
    manifest_text = journal.manifest_text
    if args.example_decision is not None:
        if args.example_out is None:
            print("--example-decision requires --example-out", file=sys.stderr)
            return 2
        index = args.example_decision - 1
        if index < 0 or index >= len(result.actor_inputs):
            print("example decision was not reached", file=sys.stderr)
            return 2
        args.example_out.parent.mkdir(parents=True, exist_ok=True)
        args.example_out.write_text(result.actor_inputs[index] + "\n", encoding="utf-8")
    if args.summary is not None:
        summary = {
            "experiment_id": spec.experiment_id,
            "status": result.status,
            "failure_status": manifest["failure_status"],
            "performance_claim": None,
            "note": "Offline infrastructure smoke. Not an adaptation or method result.",
            "protocol_version": manifest["protocol_version"],
            "clock_convention": manifest["clock_convention"],
            "clock_limitation": manifest["clock_limitation"],
            "fixture_id": spec.fixture_id,
            "lifetime_id": spec.lifetime_id,
            "agent_id": manifest["agent_id"],
            "updater_id": manifest["updater_id"],
            "seeds": manifest["seeds"],
            "horizon": spec.horizon,
            "observed_events": len(result.observed_events),
            "artifacts_deployed": sum(
                1 for artifact in result.artifacts if artifact["deployment_decision_index"] is not None
            ),
            "source": code,
            "dependency_identity": dependencies,
            "config_sha256": manifest["config_sha256"],
            "spec_sha256": manifest["spec_sha256"],
            "hashes": manifest["hashes"],
            "manifest_sha256": sha256_text(manifest_text),
            "run_dir": str(run_dir),
            "baseline_definition": UCB_DEFINITION,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        " ".join(
            [
                f"status={result.status}",
                f"lifetime={spec.lifetime_id}",
                f"events={len(result.observed_events)}",
                f"artifacts={len(result.artifacts)}",
                f"committed_index={result.committed_index}",
                f"public_log_sha256={manifest['hashes']['public_log_sha256']}",
                f"performance_claim=none",
            ]
        )
    )
    if result.status in {"completed", "budget_exhausted", "stopped_for_checkpoint"}:
        return 0
    return 1


def _public_lines(result) -> list[str]:
    by_decision: dict[int, list[str]] = {}
    for artifact in result.artifacts:
        deployment = artifact["deployment_decision_index"]
        if deployment is None:
            continue
        created = dict(artifact)
        created["record_type"] = "artifact_created"
        deployed = {
            "record_type": "artifact_deployed",
            "artifact_id": artifact["artifact_id"],
            "version": artifact["version"],
            "deployment_decision_index": deployment,
            "creation_cutoff": artifact["creation_cutoff"],
            "content_hash": artifact["content_hash"],
        }
        by_decision.setdefault(deployment, []).append(canonical_dumps(created))
        by_decision[deployment].append(canonical_dumps(deployed))
    for event in result.observed_events:
        line = dict(event)
        line["record_type"] = "observed_event"
        by_decision.setdefault(event["action_index"], []).append(canonical_dumps(line))
    for failure in result.public_failures:
        by_decision.setdefault(failure["decision_index"], []).append(canonical_dumps(failure))
    lines: list[str] = []
    for decision in sorted(by_decision):
        lines.extend(by_decision[decision])
    return lines


def _preflight(args: argparse.Namespace, spec) -> str | None:
    if args.example_decision is not None and args.example_out is None:
        return "--example-decision requires --example-out"
    if args.example_decision is not None and (
        args.example_decision < 1 or args.example_decision > spec.horizon
    ):
        return (
            f"example decision {args.example_decision} is outside 1..{spec.horizon}"
        )
    run_dir = args.run_dir
    if _impossible_path(run_dir):
        return f"impossible run directory {run_dir}"
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        return f"refusing nonempty run directory {run_dir}"
    reserved = {_identity(run_dir / name) for name in AUTHORITATIVE_NAMES}
    config_id = _identity(args.config)
    claimed: dict[Path, str] = {}
    for label, path in (("summary", args.summary), ("example", args.example_out)):
        if path is None:
            continue
        if _impossible_path(path):
            return f"impossible {label} path {path}"
        if path.exists():
            return f"refusing to overwrite {path}"
        ident = _identity(path)
        if ident == config_id or ident in reserved:
            return f"{label} collides with an authoritative path: {path}"
        previous = claimed.get(ident)
        if previous is not None:
            return f"{label} collides with {previous}: {path}"
        claimed[ident] = label
    return None


def _identity(path: Path) -> Path:
    return path.expanduser().resolve()


def _impossible_path(path: Path) -> bool:
    for parent in path.expanduser().absolute().parents:
        if not parent.exists():
            continue
        return not parent.is_dir()
    return False


class _RunJournal:
    """Durable local log. A hard stop can still leave status=running."""

    def __init__(self, run_dir, spec, run_id, config_bytes, code, dependencies) -> None:
        self.run_dir = run_dir
        self.spec = spec
        self.run_id = run_id
        self.config_bytes = config_bytes
        self.code = code
        self.dependencies = dependencies
        self.manifest: dict[str, Any] = {}
        self.manifest_text = ""
        self.public_text = ""

    def commit(self, state) -> None:
        self._persist(
            status="running",
            artifacts=[artifact.to_dict() for artifact in state.store.versions],
            observed_events=[event.to_dict() for event in state.observed],
            private_events=[event.to_dict() for event in state.private],
            public_failures=list(state.failures),
            public_abort=None,
            error_private=None,
            interactions_consumed=state.interactions_consumed,
            compute_units_consumed=state.compute_units_consumed,
            spec_sha256=self.spec.fingerprint(),
        )

    def write(self, status: str, result) -> None:
        if result is None:
            self._persist(
                status=status,
                artifacts=[],
                observed_events=[],
                private_events=[],
                public_failures=[],
                public_abort=None,
                error_private=None,
                interactions_consumed=0,
                compute_units_consumed=0,
                spec_sha256=self.spec.fingerprint(),
            )
            return
        self._persist(
            status=status,
            artifacts=list(result.artifacts),
            observed_events=list(result.observed_events),
            private_events=list(result.private_events),
            public_failures=list(result.public_failures),
            public_abort=result.public_abort,
            error_private=result.error_private,
            interactions_consumed=result.interactions_consumed,
            compute_units_consumed=result.compute_units_consumed,
            spec_sha256=result.spec_sha256,
        )

    def write_exception(self, exc: BaseException) -> None:
        private_text = (self.run_dir / "private_log.jsonl").read_text(encoding="utf-8")
        detail = canonical_dumps(
            {
                "record_type": "unhandled_failure",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        private_text = private_text + detail + "\n"
        atomic_write_text(self.run_dir / "private_log.jsonl", private_text)
        if self.manifest:
            self.manifest["status"] = "aborted"
            self.manifest["failure_status"] = "aborted"
            self.manifest["completion_reason"] = "aborted"
            self.manifest["hashes"]["private_log_sha256"] = sha256_text(private_text)
            self.manifest_text = canonical_dumps(self.manifest)
            atomic_write_text(self.run_dir / "manifest.json", self.manifest_text)

    def _persist(
        self,
        *,
        status: str,
        artifacts: list[dict[str, Any]],
        observed_events: list[dict[str, Any]],
        private_events: list[dict[str, Any]],
        public_failures: list[dict[str, Any]],
        public_abort: dict[str, Any] | None,
        error_private: str | None,
        interactions_consumed: int,
        compute_units_consumed: int,
        spec_sha256: str,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.run_dir / "config.json", self.config_bytes)
        view = _LogView(artifacts, observed_events, public_failures)
        public_lines = _public_lines(view)
        private_lines = [canonical_dumps(event) for event in private_events]
        if public_abort is not None:
            public_lines.append(canonical_dumps(public_abort))
            private_lines.append(
                canonical_dumps(
                    {
                        "record_type": public_abort["failure_kind"],
                        "decision_index": public_abort["decision_index"],
                        "message": error_private,
                    }
                )
            )
        public_text = "".join(line + "\n" for line in public_lines)
        private_text = "".join(line + "\n" for line in private_lines)
        artifacts_text = canonical_dumps(artifacts)
        atomic_write_text(self.run_dir / "public_log.jsonl", public_text)
        atomic_write_text(self.run_dir / "private_log.jsonl", private_text)
        atomic_write_text(self.run_dir / "artifacts.json", artifacts_text)
        manifest = {
            **clock_fields(),
            "run_id": self.run_id,
            "experiment_id": self.spec.experiment_id,
            "status": status,
            "agent_id": AGENT_IDS[self.spec.agent_name],
            "updater_id": "scripted-mean-updater-v1",
            "config_id": self.spec.fixture_id,
            "config_sha256": sha256_bytes(self.config_bytes),
            "spec_sha256": spec_sha256,
            "code": self.code,
            "dependency_identity": self.dependencies,
            "seeds": {
                "schedule_seed": self.spec.schedule_seed,
                "innovation_seed": self.spec.innovation_seed,
                "agent_seed": self.spec.agent_seed,
                "lifetime_id": self.spec.lifetime_id,
            },
            "resource_caps": {
                "interaction_budget": self.spec.interaction_budget,
                "compute_budget": self.spec.compute_budget,
                "horizon": self.spec.horizon,
            },
            "actual_usage": {
                "interactions_consumed": interactions_consumed,
                "compute_units_consumed": compute_units_consumed,
                "observed_events": len(observed_events),
                "artifacts": len(artifacts),
                "api_calls": 0,
                "api_usd": None,
                "tokens": None,
            },
            "paths": {
                "public_log": "public_log.jsonl",
                "private_log": "private_log.jsonl",
                "artifacts": "artifacts.json",
                "config": "config.json",
            },
            "hashes": {
                "public_log_sha256": sha256_text(public_text),
                "private_log_sha256": sha256_text(private_text),
                "artifacts_sha256": sha256_text(artifacts_text),
            },
            "completion_reason": status,
            "failure_status": None if status in {"running", "completed"} else status,
            "performance_claim": None,
            "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "baseline_definition": UCB_DEFINITION,
            "preservation_limit": (
                "The authoritative committed prefix is the last manifest whose hashes "
                "match public_log.jsonl, private_log.jsonl, and artifacts.json. Each of "
                "those files is replaced atomically, so an interruption during the next "
                "write leaves the previous file intact. A crash between replacing one "
                "file and the next is an inconsistent directory and is not a successful "
                "run; it is not replayed. Arbitrary disk or power failure is not covered."
            ),
        }
        manifest_text = canonical_dumps(manifest)
        assert_separated_streams(manifest, public_text)
        atomic_write_text(self.run_dir / "manifest.json", manifest_text)
        self.manifest = manifest
        self.manifest_text = manifest_text
        self.public_text = public_text


class _LogView:
    def __init__(self, artifacts, observed_events, public_failures) -> None:
        self.artifacts = artifacts
        self.observed_events = observed_events
        self.public_failures = public_failures
