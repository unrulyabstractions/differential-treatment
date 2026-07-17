"""Serve the local results-viewer UI over runs/.

Usage:
    uv run python scripts/serve_results_viewer.py [--runs-root runs] \\
        [--port 8765] [--no-open]

Opens a browser tab (unless --no-open) with a sidebar to switch between every
dataset run: verdict tables with evidence bars, every plot PNG, and links to the
raw section JSONs.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.results_viewer.results_http_server import serve_results_viewer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open", action="store_true", help="do not open a browser tab"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_open:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    serve_results_viewer(args.runs_root, args.port)


if __name__ == "__main__":
    main()
