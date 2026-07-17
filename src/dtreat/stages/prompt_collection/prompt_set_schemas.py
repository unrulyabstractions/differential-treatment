"""Schemas for community prompt sets and the stage-1 artifact."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class PromptRecord(BaseSchema):
    """One prompt from a community, annotated with its underlying instruction
    (the iota(x) mapping of Eq 2 — what is being asked, independent of voice).

    instruction_id may be empty on input when the config uses extraction
    (`annotate_instructions: "extract"`); stage 1 then fills it in.
    """

    prompt_id: str
    text: str
    instruction_id: str = ""
    instruction_phrase: str = ""  # free-text phrase before canonicalization
    instruction_source: str = "provided"  # "provided" | "extracted"


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
class InstructionMatchOutcome(BaseSchema):
    """Kept/dropped counts for one instruction id during frequency matching."""

    instruction_id: str
    kept_per_side: int
    dropped_target: int
    dropped_baseline: int


@dataclass
class FrequencyMatchingReport(BaseSchema):
    """What frequency matching kept and dropped (empty when disabled)."""

    enabled: bool = False
    outcomes: list[InstructionMatchOutcome] = field(default_factory=list)
    dropped_prompt_ids: list[str] = field(default_factory=list)

    def total_dropped(self) -> int:
        return len(self.dropped_prompt_ids)


@dataclass
class PromptStageArtifact(BaseSchema):
    """Stage-1 output: both validated (possibly annotated/matched) prompt sets,
    the comparability report, and what frequency matching dropped."""

    target_set: CommunityPromptFile
    baseline_set: CommunityPromptFile
    comparability: ComparabilityReport
    annotator_model: str = ""
    matching: FrequencyMatchingReport = field(default_factory=FrequencyMatchingReport)
