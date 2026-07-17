"""The judge protocol: how responses are presented to an LLM judge and how
verdicts are parsed back.

Lives in `common` because both the scoring stage (builds prompts, parses
verdicts) and the mock judge backend (parses prompts, emits verdicts) speak
this protocol. The judge is never told which community a response came from —
grounding without disclosure is what makes recognition-based judging reliable
(paper §2.3).
"""

from __future__ import annotations

from .json_text_extraction import extract_first_json_object, strip_thinking_content

RESPONSE_START = "=== RESPONSE START ==="
RESPONSE_END = "=== RESPONSE END ==="

JUDGE_SYSTEM_PROMPT = (
    "You are a careful annotator of assistant responses in the following "
    "deployment setting: {deployment_context}\n"
    "You judge whether a response exhibits specific concrete properties. "
    "Judge only what the response actually says, not what it implies."
)


def format_axis_lines(
    axes: list[tuple[str, str]], rubrics: dict[str, str] | None = None
) -> str:
    """Render '- axis_id: question' lines (the mock judge parses this shape),
    with an indented rubric continuation line when one exists."""
    rubrics = rubrics or {}
    lines = []
    for axis_id, question in axes:
        lines.append(f"- {axis_id}: {question}")
        rubric = rubrics.get(axis_id, "").strip()
        if rubric:
            lines.append(f"    rubric: {rubric}")
    return "\n".join(lines)


def build_per_response_judge_prompt(
    axes: list[tuple[str, str]],
    response_text: str,
    rubrics: dict[str, str] | None = None,
) -> str:
    """One call scoring all axes for one response; answer is a JSON object."""
    return (
        "Evaluate the response below along each axis.\n\n"
        "Axes:\n"
        f"{format_axis_lines(axes, rubrics)}\n\n"
        f"{RESPONSE_START}\n{response_text}\n{RESPONSE_END}\n\n"
        'Answer with ONLY a JSON object mapping every axis id to "YES" or "NO". '
        "Do not include any other text."
    )


def build_per_axis_judge_prompt(
    axis_id: str, question: str, response_text: str, rubric: str = ""
) -> str:
    """One call scoring a single axis; answer is a bare YES/NO."""
    rubrics = {axis_id: rubric} if rubric else None
    return (
        "Evaluate the response below along one axis.\n\n"
        "Axes:\n"
        f"{format_axis_lines([(axis_id, question)], rubrics)}\n\n"
        f"{RESPONSE_START}\n{response_text}\n{RESPONSE_END}\n\n"
        "Answer with ONLY YES or NO. Do not include any other text."
    )


def _verdict_to_bool(verdict: str) -> bool | None:
    token = verdict.strip().strip('."’').upper()
    if token.startswith("YES"):
        return True
    if token.startswith("NO"):
        return False
    return None


def parse_per_response_verdicts(reply: str, expected_axis_ids: list[str]) -> dict[str, bool | None]:
    """Parse a JSON verdict object; missing/garbled axes map to None."""
    parsed = extract_first_json_object(reply) or {}
    verdicts: dict[str, bool | None] = {}
    for axis_id in expected_axis_ids:
        raw = parsed.get(axis_id)
        verdicts[axis_id] = _verdict_to_bool(str(raw)) if raw is not None else None
    return verdicts


def parse_per_axis_verdict(reply: str) -> bool | None:
    """Parse a bare YES/NO reply (None if unparseable)."""
    return _verdict_to_bool(strip_thinking_content(reply))
