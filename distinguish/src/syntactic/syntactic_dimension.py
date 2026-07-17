"""Syntactic dimension: NeuroBiber features compared via smoothed log-odds.

Per the paper draft (3.3.2), NeuroBiber yields a small fixed inventory of
binary features, so frequency calibration is unnecessary and a plain
Haldane-Anscombe smoothed log-odds ratio per feature suffices. The draft
applies BH correction only to the lexical dimension, but we still control FDR
here across the fixed 96-feature inventory so that "n significant features"
stays honest under multiple testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet
from src.common.run_config import SyntacticConfig
from src.common.stats_utils import benjamini_hochberg, two_sided_p_from_z
from src.syntactic.neurobiber_extractor import NeurobiberExtractor

if TYPE_CHECKING:  # context is unused at runtime: the test is fully analytic
    from src.pipeline.pipeline_context import PipelineContext


def readable_feature_name(feature_name: str) -> str:
    """NeuroBiber labels are 'BIN_<CODE>'; the prefix is noise for humans."""
    return feature_name.removeprefix("BIN_")


@dataclass
class FeatureContrast(BaseSchema):
    """One NeuroBiber feature's prevalence contrast between the two sets."""

    feature_name: str
    prevalence_target: float  # fraction of target prompts with the feature, [0, 1]
    prevalence_baseline: float
    count_target: int
    count_baseline: int
    log_odds: float  # > 0 leans target, < 0 leans baseline
    z_score: float
    p_value: float
    p_adjusted: float
    significant: bool


@dataclass
class SyntacticResult(BaseSchema):
    """Log-odds contrasts for the full NeuroBiber feature inventory."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    model_name: str
    smoothing_count: float
    fdr_alpha: float
    top_features_reported: int
    n_significant_features: int
    # All features, sorted by |log_odds| descending.
    feature_contrasts: list[FeatureContrast] = field(default_factory=list)
    # Per-prompt count of active NeuroBiber features (extractor row sums): the
    # per-group distribution of stylistic richness, one int per prompt.
    features_per_prompt_target: list[int] = field(default_factory=list)
    features_per_prompt_baseline: list[int] = field(default_factory=list)
    # Runner contract: variants skipped for missing API keys (none here — the
    # NeuroBiber extractor is fully local).
    skipped_variants: list[str] = field(default_factory=list)

    def to_verdicts(self) -> list[DimensionVerdict]:
        leans_target = [
            c.feature_name for c in self.feature_contrasts if c.log_odds > 0
        ]
        leans_baseline = [
            c.feature_name for c in self.feature_contrasts if c.log_odds < 0
        ]
        top_target = (
            ", ".join(readable_feature_name(n) for n in leans_target[:3]) or "none"
        )
        top_baseline = (
            ", ".join(readable_feature_name(n) for n in leans_baseline[:3]) or "none"
        )
        min_p = min((c.p_adjusted for c in self.feature_contrasts), default=1.0)
        detail = (
            f"{self.n_significant_features}/{len(self.feature_contrasts)} NeuroBiber "
            f"features differ at BH FDR {self.fdr_alpha}; "
            f"top {self.target_label}: {top_target}; "
            f"top {self.baseline_label}: {top_baseline}"
        )
        return [
            DimensionVerdict(
                dimension="syntactic",
                test_name="neurobiber_log_odds",
                variant="",
                statistic_name="n_significant_features",
                statistic_value=float(self.n_significant_features),
                p_value=float(min_p),
                significant=self.n_significant_features > 0,
                detail=detail,
            )
        ]


def compute_syntactic(
    target: PromptSet,
    baseline: PromptSet,
    config: SyntacticConfig,
    context: PipelineContext,
) -> SyntacticResult:
    """Contrast the sets on NeuroBiber features via smoothed log-odds ratios."""
    extractor = NeurobiberExtractor(config.model_name, config.batch_size)
    features_target = extractor.extract(target.texts)
    features_baseline = extractor.extract(baseline.texts)
    extractor.cleanup()

    n_target, n_baseline = len(target.prompts), len(baseline.prompts)
    count_target = features_target.sum(axis=0).astype(float)
    count_baseline = features_baseline.sum(axis=0).astype(float)
    # Row sums: number of active features per prompt (cheap; already extracted).
    per_prompt_target = features_target.sum(axis=1).astype(int).tolist()
    per_prompt_baseline = features_baseline.sum(axis=1).astype(int).tolist()

    # Haldane-Anscombe smoothing keeps the log-odds finite when a feature is
    # absent (or universal) in one set, which is common at prompt-set sizes.
    s = config.smoothing_count
    log_odds = np.log(
        ((count_target + s) / (n_target - count_target + s))
        / ((count_baseline + s) / (n_baseline - count_baseline + s))
    )
    # Features absent from both sets (or present in every prompt of both) carry
    # no contrast; unequal set sizes would otherwise give them spurious log-odds.
    degenerate = ((count_target == 0) & (count_baseline == 0)) | (
        (count_target == n_target) & (count_baseline == n_baseline)
    )
    log_odds[degenerate] = 0.0
    standard_error = np.sqrt(
        1.0 / (count_target + s)
        + 1.0 / (n_target - count_target + s)
        + 1.0 / (count_baseline + s)
        + 1.0 / (n_baseline - count_baseline + s)
    )
    z_scores = log_odds / standard_error
    p_values = two_sided_p_from_z(z_scores)
    p_adjusted, rejected = benjamini_hochberg(p_values, config.fdr_alpha)

    contrasts = [
        FeatureContrast(
            feature_name=extractor.feature_names[j],
            prevalence_target=float(count_target[j] / n_target),
            prevalence_baseline=float(count_baseline[j] / n_baseline),
            count_target=int(count_target[j]),
            count_baseline=int(count_baseline[j]),
            log_odds=float(log_odds[j]),
            z_score=float(z_scores[j]),
            p_value=float(p_values[j]),
            p_adjusted=float(p_adjusted[j]),
            significant=bool(rejected[j]),
        )
        for j in np.argsort(-np.abs(log_odds))
    ]
    n_significant = int(rejected.sum())
    log(f"Syntactic: {n_significant}/{len(contrasts)} features significant")

    return SyntacticResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=n_target,
        n_prompts_baseline=n_baseline,
        model_name=config.model_name,
        smoothing_count=config.smoothing_count,
        fdr_alpha=config.fdr_alpha,
        top_features_reported=config.top_features_reported,
        n_significant_features=n_significant,
        feature_contrasts=contrasts,
        features_per_prompt_target=per_prompt_target,
        features_per_prompt_baseline=per_prompt_baseline,
    )
