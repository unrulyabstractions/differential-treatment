"""Schemas for the stage-5 analysis report."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema
from dtreat.stages.prompt_distinguishability.distinguish_report_schemas import (
    InputOutputComparison,
)


@dataclass
class AxisResult(BaseSchema):
    """Everything measured for one hypothesized axis of treatment."""

    axis_id: str
    question: str
    n_target: int  # responses with a valid verdict on this axis
    n_baseline: int
    rate_target: float  # z-hat_j^target (Eq 9)
    rate_baseline: float  # z-hat_j^baseline
    delta: float  # Delta_j (Eq 10)
    p_value: float  # permutation p-value (§4.5.1)
    q_value: float  # Benjamini–Hochberg adjusted
    significant: bool  # retained after FDR control
    info_bits: float  # I_j mutual information (Eq 13)
    insufficient_data: bool = False
    # inter-judge agreement for this axis (from judge calibration, when a
    # panel ran): low agreement means the axis wording is unreliable and its
    # verdict should be read with caution (2410.19803: LMRA reliability is
    # dimension-dependent)
    judge_kappa: float | None = None
    low_judge_agreement: bool = False


@dataclass
class MethodBreakdown(BaseSchema):
    """How one hypothesis-generation method's axes fared downstream."""

    method: str
    n_axes: int
    n_significant: int
    total_info_bits: float
    mean_abs_delta: float
    significant_axes: list[str] = field(default_factory=list)


@dataclass
class InstructionStratumGap(BaseSchema):
    """One axis's gap WITHIN one instruction stratum (2410.19803: bias
    concentrates in specific tasks and dilutes in the aggregate)."""

    instruction_id: str
    axis_id: str
    n_target: int
    n_baseline: int
    delta: float


@dataclass
class RefusalAnalysis(BaseSchema):
    """Refusals are treatment too: rates per community + exact test."""

    target_refusals: int
    target_total: int
    baseline_refusals: int
    baseline_total: int
    target_rate: float
    baseline_rate: float
    fisher_p_value: float


@dataclass
class ClassifierTwoSampleResult(BaseSchema):
    """C2ST (§4.5.3): held-out accuracy lower-bounds community separability."""

    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    majority_baseline: float
    n_train: int
    n_test: int
    n_dropped_incomplete: int
    above_chance: bool


@dataclass
class PromptBehaviorRates(BaseSchema):
    """Per-prompt mean behavior (the beta_x of Eq 6, kept for auditability)."""

    prompt_id: str
    community: str
    n_responses: int
    rates: dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisReport(BaseSchema):
    """Stage-5 output: the complete differential-treatment picture."""

    target_community: str
    baseline_community: str
    axes: list[AxisResult] = field(default_factory=list)
    d_pi_bits_significant_axes: float | None = None  # Eq 12 over retained axes
    d_pi_bits_all_axes: float | None = None
    c2st: ClassifierTwoSampleResult | None = None
    refusals: RefusalAnalysis | None = None
    prompt_rates: list[PromptBehaviorRates] = field(default_factory=list)
    # largest within-stratum gaps for significant axes (top strata by |delta|)
    instruction_strata: list[InstructionStratumGap] = field(default_factory=list)
    method_breakdown: list[MethodBreakdown] = field(default_factory=list)
    input_output: InputOutputComparison | None = None
    n_permutations: int = 0
    permutation_unit: str = "prompt"
    fdr_alpha: float = 0.05
    epsilon: float = 0.01
    seed: int = 0

    def significant_axes(self) -> list[AxisResult]:
        return [axis for axis in self.axes if axis.significant]
