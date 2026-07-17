"""Schemas for judge-scored responses (behavior characterizations)."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class ScoredResponse(BaseSchema):
    """One response's behavior characterization z = Lambda(y) (Eq 5).

    verdicts maps axis_id -> True/False; axes the judge failed to answer
    are listed in unparsed_axes and excluded from statistics rather than
    silently coerced.
    """

    response_id: str
    prompt_id: str
    community: str
    instruction_id: str
    refused: bool
    verdicts: dict[str, bool] = field(default_factory=dict)
    unparsed_axes: list[str] = field(default_factory=list)
    raw_judge_reply: str = ""


@dataclass
class ScoringManifest(BaseSchema):
    """Stage-4 summary: coverage, judge accounting, parse health."""

    judge_model: str
    judge_mode: str
    axis_ids: list[str]
    scored_responses: int
    skipped_refusals: int
    judge_calls: int
    unparsed_verdicts: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
