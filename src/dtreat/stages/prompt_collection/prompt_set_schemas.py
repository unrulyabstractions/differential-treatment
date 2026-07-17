"""Schemas for community prompt sets and the stage-1 artifact."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class PromptRecord(BaseSchema):
    """One prompt from a community, annotated with its underlying instruction
    (the iota(x) mapping of Eq 2 — what is being asked, independent of voice)."""

    prompt_id: str
    text: str
    instruction_id: str


@dataclass
class CommunityPromptFile(BaseSchema):
    """On-disk format of a community's collected prompts (input to stage 1)."""

    community: str
    domain: str
    prompts: list[PromptRecord] = field(default_factory=list)


@dataclass
class InstructionFrequency(BaseSchema):
    """Frequency of one instruction in both sets (the f_X(i) of Eq 2)."""

    instruction_id: str
    target_count: int
    baseline_count: int
    target_fraction: float
    baseline_fraction: float


@dataclass
class ComparabilityReport(BaseSchema):
    """Instruction-comparability check between the two prompt sets (Eq 3).

    total_variation_distance is 0.5 * sum_i |f_target(i) - f_baseline(i)|;
    chi-square tests independence of instruction frequency from community.
    """

    total_variation_distance: float
    chi2_statistic: float
    chi2_p_value: float
    max_allowed_tv_distance: float
    passed: bool
    frequencies: list[InstructionFrequency] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PromptStageArtifact(BaseSchema):
    """Stage-1 output: both validated prompt sets + the comparability report."""

    target_set: CommunityPromptFile
    baseline_set: CommunityPromptFile
    comparability: ComparabilityReport
