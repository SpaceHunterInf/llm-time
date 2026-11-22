#!/usr/bin/env python3
"""Export a 100% pre-rendered, self-contained offline HTML snapshot of the Research Portal.

Note on IDE HTML Previewers:
- All prose, walkthroughs, tables, MathML equations, and SVG visualizations are pre-rendered in Python
  (`src/dashboard/renderer.py`) so the file displays 100% of its content even with JavaScript disabled.
- Intentionally omits `<meta http-equiv="Content-Security-Policy">` from the offline `.html` file so VS Code /
  Antigravity IDE HTML previewers can inject preview stylesheets without CSP collisions (while `src/dashboard/server.py`
  enforces strict per-request HTTP CSP headers over `127.0.0.1`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.renderer import render_dashboard_html  # noqa: E402
from src.dashboard.telemetry import collect_dashboard_telemetry  # noqa: E402


def build_static_snapshot(workspace_root: Path, out_path: Path) -> Path:
    """Pre-render the entire HTML + SVG + MathML portal with inlined CSS & progressive-enhancement JS."""
    static_dir = workspace_root / "src" / "dashboard" / "static"
    roadmap = static_dir / "roadmap_dashboard.html"
    if out_path.name == roadmap.name or out_path.resolve() == roadmap.resolve():
        raise ValueError("Choose a meeting filename distinct from the preserved roadmap_dashboard.html.")
    css = (static_dir / "styles.css").read_text(encoding="utf-8")
    js = (static_dir / "app.js").read_text(encoding="utf-8")

    telemetry = collect_dashboard_telemetry(workspace_root)
    rendered = render_dashboard_html(
        telemetry,
        inline_css=css,
        inline_js=js,
        csp_nonce=None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    companion = out_path.with_name(roadmap.name)
    archived_page = roadmap.read_bytes()
    if not companion.exists():
        with companion.open("xb") as dest:
            dest.write(archived_page)
    elif companion.read_bytes() != archived_page:
        raise FileExistsError(f"{companion} already contains a different roadmap.")
    out_path.write_text(rendered, encoding="utf-8")

    # Keep `src/dashboard/static/index.html` pre-rendered too so direct IDE preview of `index.html` is never blank!
    index_preview_path = static_dir / "index.html"
    if static_dir.is_dir() and out_path.resolve() != index_preview_path.resolve():
        index_preview_path.write_text(rendered, encoding="utf-8")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 100% pre-rendered standalone supervisor dashboard HTML")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "supervisor_dashboard.html",
        help="Output standalone HTML file path",
    )
    args = parser.parse_args()
    target = build_static_snapshot(ROOT, args.out.resolve())
    print(f"Exported 100% pre-rendered standalone dashboard snapshot to {target}")


if __name__ == "__main__":
    main()
