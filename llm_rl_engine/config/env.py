from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None, *, override: bool = True) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if path is None:
        path = Path.cwd() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value