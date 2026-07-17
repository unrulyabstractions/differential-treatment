"""Schemas for runs/{dataset}/summary.json: every comparison, one record."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict


@dataclass
class DimensionTiming(BaseSchema):
    """Wall-clock seconds one section took (main + explorations)."""

    comparison: str
    dimension: str
    seconds: float


@dataclass
class PromptSetSummary(BaseSchema):
    """Provenance of one side of a comparison."""

    name: str
    display_name: str
    group: str
    n_prompts: int
    n_authors: int


@dataclass
class ComparisonSummary(BaseSchema):
    """Outcome of one manifest comparison."""

    name: str
    expectation: str
    target: PromptSetSummary
    baseline: PromptSetSummary
    n_tests: int
    n_significant: int
    overall_distinguishable: bool
    verdicts: list[DimensionVerdict] = field(default_factory=list)


@dataclass
class DatasetRunSummary(BaseSchema):
    """Everything summary.json carries about a completed dataset run."""

    run_name: str
    dataset_name: str
    dataset_path: str
    created_at: str
    dimensions_run: list[str]
    comparisons: list[ComparisonSummary] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)  # variant (reason)
    timings: list[DimensionTiming] = field(default_factory=list)
