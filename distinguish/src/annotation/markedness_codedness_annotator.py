"""LLM annotation of identity signals in prompt texts (paper 3.2).

Estimates the per-prompt labels y^mark (markedness: the text explicitly
discloses the author's sexual orientation or gender identity) and y^code
(codedness in [0,1]: strength of implicit community signal) from text alone,
via batched OpenAI chat completions. Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from src.common.base_schema import BaseSchema
from src.common.logging_utils import log

_BATCH_SIZE = 30
_MAX_ATTEMPTS = 2  # one corrective retry on a malformed / misaligned reply
_SYSTEM_PROMPT = """\
You annotate user prompts written to an AI chatbot for LGBTQ+ identity
signals. The user message is a JSON array of objects {"i": index, "t": text}.
For each text, judge two things about its AUTHOR's self-presentation:

"m" - markedness, 0 or 1. Set 1 if and only if the text EXPLICITLY discloses
the author's own sexual orientation or gender identity through direct
self-identification, e.g. "as a trans woman", "I'm gay", "my same-sex
partner". Other people's identities, general LGBTQ+ subject matter, and
implicit hints do NOT count; different-sex spouse mentions ("my wife" from an
author of unknown gender) do NOT count.

"c" - codedness, a number in [0.0, 1.0]. The strength of IMPLICIT LGBTQ+
community signal: coded vocabulary, chosen family, "T"/HRT mentioned without
explanation, community spaces, pronoun-sharing norms. 0.0 means no implicit
signal at all; explicit disclosure by itself does not raise "c".

Reply with ONLY a JSON array, no code fences, exactly one object per input,
each ECHOING the input's "i": [{"i": <same index>, "m": 0 or 1, "c":
0.0-1.0, "why": "rationale under 8 words"}, ...]. Cover every index exactly once.
"""


@dataclass
class PromptLabelEstimate(BaseSchema):
    """Estimated identity-signal labels for one prompt text."""

    markedness: int  # y^mark: 1 iff explicit orientation/gender self-disclosure
    codedness: float  # y^code in [0,1]: implicit community-signal strength
    rationale: str  # the model's short justification


def annotate_markedness_codedness(
    texts: list[str], model_name: str = "gpt-5-mini"
) -> list[PromptLabelEstimate]:
    """Markedness/codedness estimate for every text, in input order."""
    log(f"Annotating {len(texts)} prompts for markedness/codedness via {model_name}")
    client = OpenAI()
    estimates: list[PromptLabelEstimate] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        # Items carry a stable index so labels are bound by id, never by reply
        # position — a reordered/dropped/substituted reply is caught, not
        # silently misassigned to the wrong prompt.
        payload = json.dumps(
            [{"i": i, "t": t} for i, t in enumerate(batch)], ensure_ascii=False
        )
        estimates.extend(_annotate_batch(client, model_name, payload, len(batch)))
    return estimates


def _annotate_batch(
    client: OpenAI, model_name: str, payload: str, expected_count: int
) -> list[PromptLabelEstimate]:
    """One batch with a bounded corrective retry on malformed/misaligned output."""
    last_error: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        # No temperature override: gpt-5 family models reject non-default values.
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        try:
            return _parse_estimates(response.choices[0].message.content, expected_count)
        except ValueError as error:
            last_error = error
    raise ValueError(f"Annotation batch failed after retries: {last_error}")


def _parse_estimates(
    content: str | None, expected_count: int
) -> list[PromptLabelEstimate]:
    """Validate the reply and reindex by echoed id; fail loudly on any mismatch."""
    if not isinstance(content, str):
        raise ValueError(
            f"Model returned empty/non-text reply (expected {expected_count}): "
            f"{content!r}"
        )
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    entries = json.loads(cleaned)
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(
            f"Expected a JSON array of {expected_count} annotations, got: {content!r}"
        )
    # Bind each estimate to its echoed index; the set of indices must be exactly
    # 0..expected_count-1 (no missing, extra, or duplicate) or we cannot trust
    # the alignment.
    by_index: dict[int, PromptLabelEstimate] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Annotation entry is not a JSON object: {entry!r}")
        index = entry.get("i")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"Annotation entry missing integer index 'i': {entry!r}")
        if index in by_index:
            raise ValueError(f"Duplicate annotation index {index}")
        by_index[index] = _parse_entry(entry)
    if set(by_index) != set(range(expected_count)):
        raise ValueError(
            f"Annotation indices {sorted(by_index)} != 0..{expected_count - 1}"
        )
    return [by_index[i] for i in range(expected_count)]


def _parse_entry(entry: object) -> PromptLabelEstimate:
    """One validated estimate from one reply object; raise on any bad field."""
    if not isinstance(entry, dict):
        raise ValueError(f"Annotation entry is not a JSON object: {entry!r}")
    markedness = entry.get("m")
    codedness = entry.get("c")
    rationale = entry.get("why")
    if isinstance(markedness, bool) or markedness not in (0, 1):
        raise ValueError(f"markedness 'm' must be 0 or 1, got {markedness!r}")
    if (
        isinstance(codedness, bool)
        or not isinstance(codedness, int | float)
        or not 0.0 <= float(codedness) <= 1.0
    ):
        raise ValueError(f"codedness 'c' must be a number in [0,1], got {codedness!r}")
    if not isinstance(rationale, str):
        raise ValueError(f"rationale 'why' must be a string, got {rationale!r}")
    return PromptLabelEstimate(
        markedness=int(markedness),
        codedness=float(codedness),
        rationale=rationale,
    )
