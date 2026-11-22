"""Localhost-only (`127.0.0.1`) hardened HTTP telemetry server for the PI/Supervisor Dashboard.

Security Architecture Notes (`mandatory-secure-web-skills`):
- Binds strictly to `127.0.0.1` or `localhost` (refuses `0.0.0.0` or external interfaces; intended for SSH tunnel viewing).
- Serves 100% Python pre-rendered HTML + SVG + MathML (`src/dashboard/renderer.py`) with strict `html.escape()` encoding.
- Enforces HTTP method allow-list (`GET` only -> `405` on `POST`, `PUT`, `DELETE`, `PATCH`, `OPTIONS`, `TRACE`).
- Injects per-request cryptographic CSP nonces (`default-src 'none'; script-src 'self' 'nonce-...'; style-src 'self' 'nonce-...'`),
  `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Permissions-Policy`, and `Cache-Control: no-store`.
- Strict static file allow-list + canonical `os.sep` boundary verification preventing path traversal.
- Zero `.env` or secret access; PII email addresses masked (`gs***@gmail.com`).
# TODO(security): If ever deployed beyond single-user `127.0.0.1` SSH loopback, add OAuth2/OIDC + MFA
# authentication, Argon2/CSRF session protections, per-client rate limiting, and TLS termination.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlparse

from src.dashboard.renderer import render_dashboard_html
from src.dashboard.telemetry import collect_dashboard_telemetry

ALLOWED_STATIC_FILES: dict[str, tuple[str, str]] = {
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/roadmap_dashboard.html": ("roadmap_dashboard.html", "text/html; charset=utf-8"),
}

# The 21 November export is served byte-for-byte, including its two inline assets.
ROADMAP_SCRIPT_HASH = "'sha256-fhl5RXQUjSqjNCL7I0/OuD1HplfRDGidMJAQET3dB/8='"
ROADMAP_STYLE_HASH = "'sha256-pjEqIFiE3MXJeRxhVhhN+oahWjZxVDkNqwlUTUjBnrM='"


def resolve_confined_static(static_root: Path, filename: str) -> Path:
    """Resolve `filename` strictly inside `static_root` using trailing `os.sep` boundary check."""
    base = static_root.resolve()
    clean_name = os.path.basename(filename)
    candidate = (base / clean_name).resolve()
    if not str(candidate).startswith(str(base) + os.sep):
        raise PermissionError("Static path escaped sandbox boundary.")
    return candidate


def make_handler(workspace_root: Path):
    static_dir = (Path(__file__).resolve().parent / "static").resolve()

    class HardenedDashboardHandler(BaseHTTPRequestHandler):
        server_version = "RSIDashboard/3.0"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            # Minimal safe request logging without query strings, headers, or tokens.
            return

        def _send_security_headers(
            self, content_type: str, nonce: str | None = None, *, original_roadmap: bool = False
        ) -> None:
            if nonce or original_roadmap:
                script_source = ROADMAP_SCRIPT_HASH if original_roadmap else f"'nonce-{nonce}'"
                style_source = ROADMAP_STYLE_HASH if original_roadmap else f"'nonce-{nonce}'"
                csp = (
                    "default-src 'none'; "
                    f"script-src 'self' {script_source}; "
                    f"style-src 'self' {style_source}; "
                    "img-src 'self' data:; "
                    "font-src 'self'; "
                    "connect-src 'self'; "
                    "object-src 'none'; "
                    "base-uri 'none'; "
                    "frame-ancestors 'none'"
                )
            else:
                csp = (
                    "default-src 'none'; "
                    "object-src 'none'; "
                    "base-uri 'none'; "
                    "frame-ancestors 'none'"
                )
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Security-Policy", csp)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header("Cache-Control", "no-store")

        def _reject_method(self) -> None:
            body = b'{"error":"method_not_allowed"}'
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self._send_security_headers("application/json; charset=utf-8")
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self._reject_method()

        def do_PUT(self) -> None:  # noqa: N802
            self._reject_method()

        def do_DELETE(self) -> None:  # noqa: N802
            self._reject_method()

        def do_PATCH(self) -> None:  # noqa: N802
            self._reject_method()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._reject_method()

        def do_TRACE(self) -> None:  # noqa: N802
            self._reject_method()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path

            try:
                if route == "/":
                    nonce = secrets.token_urlsafe(18)
                    telemetry = collect_dashboard_telemetry(workspace_root)
                    html_text = render_dashboard_html(telemetry, csp_nonce=nonce)
                    payload = html_text.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self._send_security_headers("text/html; charset=utf-8", nonce=nonce)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if route in ALLOWED_STATIC_FILES:
                    fname, mime = ALLOWED_STATIC_FILES[route]
                    fpath = resolve_confined_static(static_dir, fname)
                    payload = fpath.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self._send_security_headers(mime, original_roadmap=(route == "/roadmap_dashboard.html"))
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if route == "/api/telemetry":
                    data = collect_dashboard_telemetry(workspace_root)
                    raw_json = (
                        json.dumps(data)
                        .replace("</", "<\\/")
                        .replace("\u2028", "\\u2028")
                        .replace("\u2029", "\\u2029")
                    )
                    payload = raw_json.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self._send_security_headers("application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                not_found = b'{"error":"not_found"}'
                self.send_response(HTTPStatus.NOT_FOUND)
                self._send_security_headers("application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(not_found)))
                self.end_headers()
                self.wfile.write(not_found)
            except Exception:  # noqa: BLE001
                err = b'{"error":"internal_error"}'
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self._send_security_headers("application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

    return HardenedDashboardHandler


def create_dashboard_server(
    workspace_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a localhost-bound dashboard HTTP server (`0.0.0.0` is strictly forbidden)."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Dashboard server MUST bind strictly to 127.0.0.1 or localhost.")
    handler_cls = make_handler(workspace_root.resolve())
    return ThreadingHTTPServer((host, int(port)), handler_cls)
