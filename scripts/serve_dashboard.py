#!/usr/bin/env python3
"""Launch the localhost-only (`127.0.0.1`) PI & Supervisor Observatory Dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.server import create_dashboard_server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RSI-Paradigm-Shift Observatory on 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Localhost TCP port (default: 8765)")
    args = parser.parse_args()

    srv = create_dashboard_server(ROOT, host="127.0.0.1", port=args.port)
    print(f"Dashboard listening strictly on http://127.0.0.1:{args.port}")
    print(f"SSH tunnel tip: ssh -L {args.port}:127.0.0.1:{args.port} xz479@<server-host>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
