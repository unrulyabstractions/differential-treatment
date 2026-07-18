"""Parsing and normalization of helper-proposed axes."""

from __future__ import annotations

import re

from .hypothesis_schemas import HypothesisAxis


def parse_helper_axes(reply: str) -> list[HypothesisAxis]:
    """Parse helper JSON into validated axes (invalid entries are dropped)."""
    from dtreat.common.json_text_extraction import extract_first_json_array

    entries = extract_first_json_array(reply) or []
    axes = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        axis_id = normalize_axis_id(str(entry.get("axis_id", "")))
        question = str(entry.get("question", "")).strip()
        if not axis_id or not question:
            continue
        axes.append(
            HypothesisAxis(
                axis_id=axis_id,
                question=question,
                rationale=str(entry.get("rationale", "")).strip(),
                rubric=str(entry.get("rubric", "")).strip(),
                source="helper",
            )
        )
    return axes


def normalize_axis_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def seed_hypothesis_axis(text: str, index: int) -> HypothesisAxis:
    """Turn a configured seed hypothesis (free text) into an axis."""
    axis_id = normalize_axis_id("_".join(text.split()[:4])) or f"seed_axis_{index}"
    return HypothesisAxis(axis_id=axis_id, question=text.strip(), source="seed")


def dedupe_axes(axes: list[HypothesisAxis]) -> list[HypothesisAxis]:
    """Drop repeated axis ids and near-identical questions, keeping first."""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    unique = []
    for axis in axes:
        question_key = re.sub(r"\W+", " ", axis.question.lower()).strip()
        if axis.axis_id in seen_ids or question_key in seen_questions:
            continue
        seen_ids.add(axis.axis_id)
        seen_questions.add(question_key)
        unique.append(axis)
    return unique
