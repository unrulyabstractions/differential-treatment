"""Schemas for input-side distinguishability results and the input-vs-output
comparison (bridging paper/distinguishability.pdf into this pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class InputDimensionVerdict(BaseSchema):
    """One statistical test from the distinguish pipeline (lexical, syntactic,
    semantic, distributional, topical, interactional, usage)."""

    dimension: str
    variant: str
    statistic_name: str
    statistic_value: float
    p_value: float | None = None
    significant: bool | None = None


@dataclass
class InputDistinguishabilityReport(BaseSchema):
    """How separable the two communities' PROMPT sets are (input side)."""

    run_dir: str
    n_tests: int = 0
    n_significant: int = 0
    overall_distinguishable: bool = False
    best_c2st_accuracy: float | None = None  # max distributional held-out acc
    best_c2st_variant: str = ""
    verdicts: list[InputDimensionVerdict] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)


@dataclass
class InputOutputComparison(BaseSchema):
    """Input legibility vs output treatment: does the model act on what the
    prompts reveal? (answers distinguishability paper §5.2 'LLM behavioral
    study' with the differential-treatment pipeline's measurements)

    signal_usage = (output_acc - 0.5) / (input_acc - 0.5), clipped at 0:
    the fraction of the input's community signal that survives into behavior
    separability. ~0 = model ignores the cues; ~1 = behavior is as separable
    as the prompts themselves.
    """

    input_c2st_accuracy: float | None = None
    input_n_significant: int = 0
    input_n_tests: int = 0
    output_c2st_accuracy: float | None = None
    output_significant_axes: int = 0
    output_total_axes: int = 0
    output_d_pi_bits: float | None = None
    signal_usage: float | None = None
    interpretation: str = ""
