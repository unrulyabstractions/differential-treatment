"""Lexical distinguishability dimension: calibrated marked words + BH FDR.

Answers the paper's Section 3.3 lexical question: which individual words are
used at reliably different rates between the target and baseline prompt sets,
after the hybrid prior calibrates away general-English frequency effects. The
default prior mode is Mickel et al. 2025's per-side calibration (Algorithm 3),
which suppresses common/register words while keeping genuine signature words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet
from src.common.run_config import LexicalConfig
from src.lexical.marked_words_analyzer import MarkedWord, compute_marked_words_table

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext


@dataclass
class LexicalResult(BaseSchema):
    """Corpus-level test outcome plus per-word tables for both sets."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    # Config echo — mode knobs first so a report is self-describing.
    prior_calibration: str  # "mickel" (per-side calibration) | "fixed"
    significance: str  # "bh_fdr" | "raw_z"
    min_word_count: int
    reference_corpus: str  # "wordfreq:<lang>" or the JSON corpus path used
    reference_prior_weight: float  # mickel: count-space weight on the reference
    calibration_constant_mode: str  # "auto" (per-corpus search) | "fixed"
    calibration_constant: float  # mickel: the C actually used (r_i = C*w_p/w_gi)
    english_prior_weight: float  # fixed mode
    prior_strength: float  # fixed mode
    fdr_alpha: float
    top_words_reported: int
    calibration_plots: bool  # emit lexical/calibration/ justification plots
    # Test outcome
    vocabulary_size: int
    total_tokens_target: int
    total_tokens_baseline: int
    n_significant_words: int
    n_significant_raw_z: int  # |z|>=1.96 count, always tracked (BH-independent)
    min_p_adjusted: float | None  # None when the vocabulary is empty
    # The MotS calibration word set W: register anchors, colored apart in plots.
    calibration_words: list[str] = field(default_factory=list)
    marked_words_target: list[MarkedWord] = field(default_factory=list)
    marked_words_baseline: list[MarkedWord] = field(default_factory=list)
    # Full vocabulary table, |z| descending; drives the volcano plot.
    all_words: list[MarkedWord] = field(default_factory=list)
    # Runner contract: variants skipped for missing API keys (none here — the
    # test is count-based and fully offline).
    skipped_variants: list[str] = field(default_factory=list)
    # Raw texts + author ids for the calibration plots' permutation null. These
    # are private (leading underscore -> excluded from to_dict/serialization);
    # the plot call happens in-process right after compute, so they are present.
    _pooled_texts: list[str] = field(default_factory=list)
    _pooled_labels: list[int] = field(default_factory=list)
    _pooled_author_ids: list[str] = field(default_factory=list)

    def to_verdicts(self) -> list[DimensionVerdict]:
        top_target = ", ".join(w.word for w in self.marked_words_target[:3]) or "(none)"
        top_baseline = (
            ", ".join(w.word for w in self.marked_words_baseline[:3]) or "(none)"
        )
        rule = (
            f"FDR {self.fdr_alpha}"
            if self.significance == "bh_fdr"
            else "raw |z| >= 1.96"
        )
        # When strict correction yields nothing, name the raw-z survivors so the
        # exploratory signal is never hidden behind a bare "0" (BH is
        # underpowered on sparse per-word counts — see docs/FINDINGS.md).
        raw_z_note = ""
        if self.n_significant_words == 0 and self.n_significant_raw_z > 0:
            raw_z_hits = ", ".join(
                w.word for w in self.all_words if w.significant_raw_z
            )[:80]
            raw_z_note = (
                f" 0 pass {rule}, but {self.n_significant_raw_z} pass raw "
                f"|z|>=1.96: {raw_z_hits}."
            )
        detail = (
            f"{self.n_significant_words} of {self.vocabulary_size} words pass "
            f"{rule} ({self.prior_calibration} prior); "
            f"marked for {self.target_label}: {top_target}; "
            f"marked for {self.baseline_label}: {top_baseline}.{raw_z_note}"
        )
        return [
            DimensionVerdict(
                dimension="lexical",
                test_name="calibrated_marked_words",
                variant="",
                statistic_name="n_significant_words",
                statistic_value=float(self.n_significant_words),
                p_value=self.min_p_adjusted,
                significant=self.n_significant_words > 0,
                detail=detail,
            )
        ]


def compute_lexical(
    target: PromptSet,
    baseline: PromptSet,
    config: LexicalConfig,
    context: PipelineContext,
) -> LexicalResult:
    """Run the calibrated marked-words test between the target and baseline sets.

    `context` is part of the uniform dimension signature; this test is
    deterministic and count-based, so it needs no embeddings or RNG.
    """
    log(
        f"lexical: marked words for '{target.name}' vs '{baseline.name}' "
        f"({config.prior_calibration} prior, {config.significance})"
    )
    table = compute_marked_words_table(
        target.texts,
        baseline.texts,
        min_word_count=config.min_word_count,
        reference_corpus=config.reference_corpus,
        fdr_alpha=config.fdr_alpha,
        prior_calibration=config.prior_calibration,
        reference_prior_weight=config.reference_prior_weight,
        calibration_constant=config.calibration_constant,
        english_prior_weight=config.english_prior_weight,
        prior_strength=config.prior_strength,
        significance=config.significance,
    )
    # table.words is |z|-descending, and the sign of z routes each word to the
    # set it is marked for, so per-set truncation keeps the strongest evidence.
    marked_target = [w for w in table.words if w.z_score > 0]
    marked_baseline = [w for w in table.words if w.z_score < 0]
    min_p_adjusted = min((w.p_adjusted for w in table.words), default=None)
    log(
        f"lexical: {table.n_significant_words}/{table.vocabulary_size} words "
        f"significant ({config.significance})"
    )
    # Author ids qualified per side so the permutation null (which needs one
    # label per author) never merges same-named authors from opposite sets.
    author_ids = [f"a:{target.name}:{a}" for a in target.author_ids] + [
        f"b:{baseline.name}:{a}" for a in baseline.author_ids
    ]
    result = LexicalResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        prior_calibration=config.prior_calibration,
        significance=config.significance,
        min_word_count=config.min_word_count,
        reference_corpus=config.reference_corpus,
        reference_prior_weight=config.reference_prior_weight,
        calibration_constant_mode=table.calibration_constant_mode,
        calibration_constant=table.calibration_constant,
        english_prior_weight=config.english_prior_weight,
        prior_strength=config.prior_strength,
        fdr_alpha=config.fdr_alpha,
        top_words_reported=config.top_words_reported,
        calibration_plots=config.calibration_plots,
        vocabulary_size=table.vocabulary_size,
        total_tokens_target=table.total_tokens_target,
        total_tokens_baseline=table.total_tokens_baseline,
        n_significant_words=table.n_significant_words,
        n_significant_raw_z=table.n_significant_raw_z,
        min_p_adjusted=min_p_adjusted,
        calibration_words=table.calibration_words,
        marked_words_target=marked_target[: config.top_words_reported],
        marked_words_baseline=marked_baseline[: config.top_words_reported],
        all_words=table.words,
    )
    result._pooled_texts = list(target.texts) + list(baseline.texts)
    result._pooled_labels = [1] * len(target.texts) + [0] * len(baseline.texts)
    result._pooled_author_ids = author_ids
    return result
