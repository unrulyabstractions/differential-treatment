"""JSON / JSONL I/O utilities for pipeline artifacts.

Ported from queering-nlp-bias `src/common/file_io.py`, extended with JSONL
helpers (stage traces and per-response records are JSONL streams).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Keys whose multiline string values are stored as arrays of lines on disk,
# purely so artifacts stay pleasant to read and diff.
READABLE_TEXT_KEYS = ("text", "raw_text", "trace", "prompt_text", "response_text")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_text_readable(obj):
    """Recursively convert long text fields to arrays of lines for readability."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in READABLE_TEXT_KEYS and isinstance(v, str) and "\n" in v:
                # Convert multiline text to array of lines
                result[k] = v.split("\n")
            else:
                result[k] = _make_text_readable(v)
        return result
    elif isinstance(obj, list):
        return [_make_text_readable(item) for item in obj]
    else:
        return obj


def save_json(data, path: Path, readable_text: bool = True) -> None:
    """Save dictionary as pretty JSON."""
    if readable_text:
        data = _make_text_readable(data)
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)


def _restore_text_fields(obj):
    """Recursively restore text fields from arrays back to strings."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in READABLE_TEXT_KEYS and isinstance(v, list):
                # Join array of lines back to string
                result[k] = "\n".join(v)
            else:
                result[k] = _restore_text_fields(v)
        return result
    elif isinstance(obj, list):
        return [_restore_text_fields(item) for item in obj]
    else:
        return obj


def load_json(path: Path, default: dict | list | None = None) -> dict | list:
    """Load JSON file with extensive error recovery.

    Handles empty files, trailing/double commas, truncated JSON (attempts
    repair), BOM markers, and encoding issues.

    Args:
        path: Path to JSON file
        default: Default value if file is empty/missing (None = raise error)

    Returns:
        Parsed JSON data with text fields restored
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    # Read file content
    try:
        with open(path, encoding="utf-8") as f:
            s = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(path, encoding="latin-1") as f:
            s = f.read()

    # Handle empty file
    s = s.strip()
    if not s:
        if default is not None:
            return default
        raise ValueError(f"Empty JSON file: {path}")

    # Remove BOM if present
    if s.startswith("﻿"):
        s = s[1:]

    # Valid JSON must be parsed byte-for-byte as stored — repair heuristics
    # run ONLY after a parse failure, because the comma-fixing regexes are
    # string-unaware and would corrupt legitimate content (e.g. an LLM reply
    # containing "[1, 2, ]" inside a string).
    try:
        data = json.loads(s)
        return _restore_text_fields(data)
    except json.JSONDecodeError as e:
        for repaired in (_fix_comma_glitches(s), _attempt_json_repair(s)):
            if repaired is None or repaired == s:
                continue
            try:
                data = json.loads(repaired)
                print(f"  [Warning] Repaired malformed JSON: {path}")
                return _restore_text_fields(data)
            except json.JSONDecodeError:
                continue

        # If we have a default, use it
        if default is not None:
            print(f"  [Warning] Failed to parse JSON, using default: {path}")
            return default

        # Provide helpful error message
        raise ValueError(
            f"Invalid JSON in {path} at line {e.lineno}, col {e.colno}: {e.msg}\n"
            f"Context: ...{s[max(0, e.pos - 30):e.pos + 30]}..."
        ) from e


def _fix_comma_glitches(s: str) -> str:
    """Comma-fixing heuristics for hand-edited files (post-failure only)."""
    # Remove double/multiple commas (e.g., "a",, "b" -> "a", "b")
    s = re.sub(r",(\s*,)+", ",", s)
    # Remove trailing commas before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Remove leading commas after [ or {
    return re.sub(r"([{\[])(\s*),", r"\1\2", s)


def _attempt_json_repair(s: str) -> str | None:
    """Attempt to repair truncated/malformed JSON.

    Returns repaired string or None if repair not possible.
    """
    s = s.strip()
    if not s:
        return None

    # Count brackets to detect truncation
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    # If balanced, no repair needed (error is elsewhere)
    if open_braces == 0 and open_brackets == 0:
        return None

    repaired = s

    # Handle incomplete string at end (unclosed quote)
    # Count quotes - if odd, we have an unclosed string
    quote_count = repaired.count('"') - repaired.count('\\"')
    if quote_count % 2 == 1:
        repaired = repaired + '"'

    # Handle trailing colon (incomplete key-value pair)
    if re.search(r":\s*$", repaired):
        repaired = repaired + "null"

    # Handle trailing comma
    repaired = re.sub(r",\s*$", "", repaired)

    # Add missing closing brackets/braces in the order they were opened
    closings = []
    for char in repaired:
        if char == "{":
            closings.append("}")
        elif char == "}" and closings and closings[-1] == "}":
            closings.pop()
        elif char == "[":
            closings.append("]")
        elif char == "]" and closings and closings[-1] == "]":
            closings.pop()

    # Reverse to get correct closing order
    repaired += "".join(reversed(closings))

    return repaired


# =============================================================================
# JSONL helpers (streaming per-record artifacts: responses, scores, traces)
# =============================================================================


def append_jsonl(record: dict, path: Path) -> None:
    """Append a single record to a JSONL file (creates parent dirs)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write all records to a JSONL file, replacing any existing content."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Path, default: list | None = None) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping blank/corrupt lines.

    Corrupt lines are reported but never fatal: a partially-written trailing
    line (e.g. from an interrupted run) should not lose the rest of the file.
    """
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [Warning] Skipping corrupt JSONL line {line_num} in {path}")
    return records
