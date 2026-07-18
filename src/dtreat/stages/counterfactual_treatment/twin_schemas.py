"""Schemas for counterfactual voice-swapped twins and the paired analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class TwinPair(BaseSchema):
    """One prompt and its voice-swapped twin: same underlying request, the
    community voice flipped (the counterfactual unit, cf. name-swapping in
    arXiv:2410.19803 adapted to implicit voice)."""

    pair_id: str
    original_prompt_id: str
    original_community: str
    twin_voice: str  # the community whose voice the twin is written in
    original_text: str
    twin_text: str
    instruction_id: str = ""
    content_preserved: bool = True  # rewriter validation verdict
    rewriter_model: str = ""


@dataclass
class CounterfactualAxisResult(BaseSchema):
    """Net voice effect on one axis, holding request content fixed.

    delta = mean over pairs of (rate under target voice − rate under
    baseline voice); p from a paired sign-flip permutation test.
    """

    axis_id: str
    question: str
    n_pairs: int
    rate_target_voice: float
    rate_baseline_voice: float
    delta: float
    p_value: float
    q_value: float
    significant: bool
    # per-direction means: effect on target-authored content vs
    # baseline-authored content (their forward/reverse netting idea)
    delta_on_target_content: float | None = None
    delta_on_baseline_content: float | None = None
    naturalistic_delta: float | None = None  # same axis, unpaired design


@dataclass
class CounterfactualReport(BaseSchema):
    """Stage-6 output: causal voice effects + agreement with the
    naturalistic (distribution-matched) design."""

    target_community: str
    baseline_community: str
    n_pairs: int = 0
    n_twins_flagged: int = 0  # content-preservation failures (excluded)
    axes: list[CounterfactualAxisResult] = field(default_factory=list)
    # Pearson correlation of counterfactual vs naturalistic deltas over
    # axes present in both designs (None if < 3 shared axes)
    naturalistic_correlation: float | None = None
    n_permutations: int = 0
    fdr_alpha: float = 0.05
    seed: int = 0

    def significant_axes(self) -> list[CounterfactualAxisResult]:
        return [axis for axis in self.axes if axis.significant]
