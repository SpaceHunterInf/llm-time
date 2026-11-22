"""Source identity for run manifests. Dirty worktree state is recorded, not hidden."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from single_life.contracts import sha256_bytes


def source_identity(project_root: Path) -> dict[str, Any]:
    git_root = _git_root(project_root)
    commit = None
    worktree_dirty = None
    single_life_porcelain: list[str] = []
    if git_root is not None:
        commit = _git(git_root, ["rev-parse", "HEAD"])
        porcelain = _git(git_root, ["status", "--porcelain"])
        worktree_dirty = bool(porcelain.strip())
        scoped = _git(
            git_root,
            ["status", "--porcelain", "--untracked-files=all", "--", "single_life"],
        )
        single_life_porcelain = [line for line in scoped.splitlines() if line.strip()]
    return {
        "git_commit": commit,
        "worktree_dirty": worktree_dirty,
        "single_life_porcelain": single_life_porcelain,
        "source_sha256": _tree_hash(project_root / "src" / "single_life"),
        "pyproject_sha256": sha256_bytes((project_root / "pyproject.toml").read_bytes()),
    }


def dependency_identity(project_root: Path) -> dict[str, Any]:
    lock_path = project_root / "uv.lock"
    lock_hash = sha256_bytes(lock_path.read_bytes()) if lock_path.is_file() else None
    return {
        "manager": "uv",
        "runtime_dependencies": [],
        "build_backend": "hatchling",
        "python": platform.python_version(),
        "uv_lock_sha256": lock_hash,
    }


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in files:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _git_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if args[:2] == ["rev-parse", "HEAD"] else completed.stdout
