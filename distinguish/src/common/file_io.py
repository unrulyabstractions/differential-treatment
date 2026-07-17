"""JSON and path utilities (lean port of queering-nlp-bias src/common/file_io.py)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_timestamp() -> str:
    """Current timestamp string suitable for run directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Save data as pretty, UTF-8 JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)
        f.write("\n")


def load_json(path: Path) -> dict | list:
    """Load a JSON file, raising a clear error when missing or invalid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
