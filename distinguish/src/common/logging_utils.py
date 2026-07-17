"""Minimal timestamped logging for pipeline progress."""

from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)
