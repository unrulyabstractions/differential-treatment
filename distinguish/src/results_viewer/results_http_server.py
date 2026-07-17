"""Local HTTP server for the results viewer UI (stdlib only, binds 127.0.0.1).

Routes:
    /            -> the single-page viewer (assets/viewer.html)
    /assets/*    -> static css/js from the package assets dir
    /api/index   -> RunsIndex JSON (rebuilt per request, so new runs appear live)
    /runs/*      -> files under the runs root (PNG plots, section JSONs)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.common.logging_utils import log
from src.results_viewer.runs_indexer import build_runs_index

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
}


class ResultsViewerHandler(BaseHTTPRequestHandler):
    """Request handler bound to a runs root by serve_results_viewer()."""

    runs_root: Path  # injected as a class attribute by serve_results_viewer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send_path(_ASSETS_DIR / "viewer.html")
        if path.startswith("/assets/"):
            return self._send_under(_ASSETS_DIR, path[len("/assets/") :])
        if path == "/api/index":
            body = json.dumps(build_runs_index(self.runs_root).to_dict())
            return self._send_bytes(body.encode(), _CONTENT_TYPES[".json"])
        if path.startswith("/runs/"):
            return self._send_under(self.runs_root, path[len("/runs/") :])
        self.send_error(404)

    def _send_under(self, root: Path, relative: str) -> None:
        """Serve a file below `root`, refusing path traversal."""
        target = (root / relative.lstrip("/")).resolve()
        if root.resolve() not in target.parents:
            return self.send_error(403)
        self._send_path(target)

    def _send_path(self, target: Path) -> None:
        if not target.is_file():
            return self.send_error(404)
        content_type = _CONTENT_TYPES.get(target.suffix)
        if content_type is None:
            return self.send_error(415)
        self._send_bytes(target.read_bytes(), content_type)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Quiet per-request logging; errors still surface via send_error."""


def serve_results_viewer(runs_root: Path, port: int) -> None:
    """Serve the viewer over `runs_root` on 127.0.0.1:`port` (blocks)."""
    bound = type(
        "BoundResultsViewerHandler",
        (ResultsViewerHandler,),
        {"runs_root": runs_root.resolve()},
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), bound)
    log(f"results viewer: http://127.0.0.1:{port}/  (runs root: {runs_root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("results viewer: stopped")
    finally:
        server.server_close()
