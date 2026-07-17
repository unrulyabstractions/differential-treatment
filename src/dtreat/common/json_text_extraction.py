"""Robust extraction of JSON payloads from LLM replies.

LLMs wrap JSON in prose, code fences, or thinking blocks; these helpers pull
out the first well-formed array/object instead of trusting the whole reply.
`strip_thinking_content` is ported from the base repo's llm_response_parsing.
"""

from __future__ import annotations

import json
from typing import Any


def strip_thinking_content(response: str) -> str:
    """Strip content before the last </think> tag (reasoning-model preambles)."""
    text = response
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences, keeping their contents."""
    if "```" not in text:
        return text
    parts = text.split("```")
    # fenced content sits at odd indices; strip a leading language tag line
    chunks = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            first_newline = part.find("\n")
            if first_newline != -1 and part[:first_newline].strip().isalpha():
                part = part[first_newline + 1 :]
        chunks.append(part)
    return "".join(chunks)


def _scan_balanced(text: str, open_char: str, close_char: str) -> str | None:
    """Return the first balanced {...} or [...] span, respecting strings."""
    start = text.find(open_char)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : position + 1]
        start = text.find(open_char, start + 1)
    return None


def extract_first_json_array(reply: str) -> list[Any] | None:
    """Best-effort parse of the first JSON array in an LLM reply."""
    text = strip_code_fences(strip_thinking_content(reply))
    span = _scan_balanced(text, "[", "]")
    if span is None:
        return None
    try:
        parsed = json.loads(span)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def extract_first_json_object(reply: str) -> dict[str, Any] | None:
    """Best-effort parse of the first JSON object in an LLM reply."""
    text = strip_code_fences(strip_thinking_content(reply))
    span = _scan_balanced(text, "{", "}")
    if span is None:
        return None
    try:
        parsed = json.loads(span)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
