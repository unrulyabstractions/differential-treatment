"""Schemas for collected target-LLM responses."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema
from dtreat.llm.chat_types import ChatUsage


@dataclass
class ResponseRecord(BaseSchema):
    """One sampled response y_i ~ LLM(. | x) (Eq 4), with full provenance.

    Refusals are kept (refused=True): who gets refused is treatment signal,
    and stage 5 reports refusal rates alongside the axes.
    """

    response_id: str
    prompt_id: str
    community: str
    instruction_id: str
    sample_index: int
    seed: int
    model: str
    text: str
    finish_reason: str = "stop"
    refused: bool = False
    usage: ChatUsage = field(default_factory=ChatUsage)


@dataclass
class CollectionManifest(BaseSchema):
    """Stage-3 summary: what was collected, what failed, what it cost."""

    target_model: str
    samples_per_prompt: int
    temperature: float
    expected_responses: int
    collected_responses: int
    failed_requests: int
    refusals: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    responses_by_community: dict[str, int] = field(default_factory=dict)
