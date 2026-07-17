"""Minimal .env loader (no python-dotenv dependency).

Loads KEY=VALUE lines from a .env file into os.environ without overriding
variables that are already set. API keys never appear in artifacts or logs.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: Path | None = None) -> list[str]:
    """Load .env (default: cwd or repo root) into os.environ.

    Returns the list of variable NAMES that were loaded (never values).
    """
    candidates = [path] if path else [Path.cwd() / ".env", Path(__file__).parents[3] / ".env"]
    loaded: list[str] = []
    for candidate in candidates:
        if candidate is None or not Path(candidate).exists():
            continue
        for line in Path(candidate).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
        break  # first existing file wins
    return loaded
