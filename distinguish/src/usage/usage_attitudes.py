"""Usage & attitudes section: interaction-context scales, target vs baseline.

Paper section 5.2 / C3: do the two populations report different chatbot-use
frequencies and attitudes? Every scale is compared at the AUTHOR level — each
author's recorded (>0) values are averaged first — because prompts within an
author repeat the same survey answers, so prompt-level tests would fake
independence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import mannwhitneyu

from src.common.base_schema import BaseSchema
from src.common.dataset_annotations import DOMAINS
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet
from src.common.run_config import UsageConfig
from src.common.stats_utils import benjamini_hochberg

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

USAGE_SCALES = [  # InteractionContext fields compared between the sets
    "llm_freq",
    "professional_freq",
    "general_freq",
    "aversion",
    "satisfaction",
]
DOMAIN_SCALES = ["llm_freq", "professional_freq", "aversion", "satisfaction"]
MIN_AUTHORS_PER_SIDE = 3  # below this the Mann-Whitney test is skipped


@dataclass
class ScaleTest(BaseSchema):
    """Mann-Whitney comparison of one context scale over author means."""

    scale: str
    n_target_authors: int  # authors with at least one recorded (>0) value
    n_baseline_authors: int
    mean_target: float | None  # mean of author means; None when unannotated
    mean_baseline: float | None
    mann_whitney_u: float | None  # None when the test was skipped
    p_value: float | None
    p_adjusted: float | None
    rank_biserial: float | None
    significant: bool | None


@dataclass
class ScaleDomainMean(BaseSchema):
    """Author-mean average of one scale within one survey domain."""

    scale: str
    domain: str
    n_target_authors: int
    n_baseline_authors: int
    mean_target: float | None
    mean_baseline: float | None


@dataclass
class UsageResult(BaseSchema):
    """Per-scale Mann-Whitney outcomes plus per-domain profile means."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    fdr_alpha: float
    n_scales_tested: int
    n_significant_scales: int
    scale_tests: list[ScaleTest] = field(default_factory=list)
    domain_means: list[ScaleDomainMean] = field(default_factory=list)
    # Runner contract: variants skipped for missing API keys (none here — the
    # tests are deterministic rank statistics over the recorded context).
    skipped_variants: list[str] = field(default_factory=list)

    def to_verdicts(self) -> list[DimensionVerdict]:
        return [
            DimensionVerdict(
                dimension="usage",
                test_name="mann_whitney",
                variant=test.scale,
                statistic_name="rank_biserial",
                statistic_value=float(test.rank_biserial or 0.0),
                p_value=test.p_adjusted,
                significant=test.significant,
                detail=self._describe(test),
            )
            for test in self.scale_tests
        ]

    def _describe(self, test: ScaleTest) -> str:
        if test.p_value is None:
            return (
                f"{test.scale} unannotated: {test.n_target_authors} target / "
                f"{test.n_baseline_authors} baseline authors with recorded "
                f"values (need >= {MIN_AUTHORS_PER_SIDE} per side)."
            )
        return (
            f"{test.scale}: author means {test.mean_target:.2f} "
            f"({self.target_label}) vs {test.mean_baseline:.2f} "
            f"({self.baseline_label}); r = {test.rank_biserial:+.2f}, "
            f"adjusted p = {test.p_adjusted:.3g}."
        )


def _author_means(prompt_set: PromptSet, scale: str, domain: str = "") -> list[float]:
    """Each author's mean recorded (>0) value for one scale, optionally per domain."""
    per_author: dict[str, list[int]] = {}
    for record in prompt_set.prompts:
        if domain and record.context.domain != domain:
            continue
        value = getattr(record.context, scale)
        if value > 0:
            per_author.setdefault(record.author_id, []).append(value)
    return [float(np.mean(values)) for values in per_author.values()]


def _test_scale(
    scale: str, target_means: list[float], baseline_means: list[float]
) -> ScaleTest:
    """Two-sided Mann-Whitney U on author means, skipped when underpowered."""
    n_target, n_baseline = len(target_means), len(baseline_means)
    test = ScaleTest(
        scale=scale,
        n_target_authors=n_target,
        n_baseline_authors=n_baseline,
        mean_target=float(np.mean(target_means)) if target_means else None,
        mean_baseline=float(np.mean(baseline_means)) if baseline_means else None,
        mann_whitney_u=None,
        p_value=None,
        p_adjusted=None,
        rank_biserial=None,
        significant=None,
    )
    if n_target < MIN_AUTHORS_PER_SIDE or n_baseline < MIN_AUTHORS_PER_SIDE:
        return test
    u_statistic, p_value = mannwhitneyu(
        target_means, baseline_means, alternative="two-sided"
    )
    test.mann_whitney_u = float(u_statistic)
    test.p_value = 1.0 if np.isnan(p_value) else float(p_value)  # NaN: all tied
    # Wendt's rank-biserial from the target-side U: r > 0 when baseline author
    # means rank higher, r < 0 when target author means rank higher.
    test.rank_biserial = float(1.0 - 2.0 * u_statistic / (n_target * n_baseline))
    return test


def _domain_profile(target: PromptSet, baseline: PromptSet) -> list[ScaleDomainMean]:
    """Author-mean averages per (scale, domain) cell for the profile plot."""
    rows = []
    for scale in DOMAIN_SCALES:
        for domain in DOMAINS:
            target_means = _author_means(target, scale, domain)
            baseline_means = _author_means(baseline, scale, domain)
            rows.append(
                ScaleDomainMean(
                    scale=scale,
                    domain=domain,
                    n_target_authors=len(target_means),
                    n_baseline_authors=len(baseline_means),
                    mean_target=float(np.mean(target_means)) if target_means else None,
                    mean_baseline=(
                        float(np.mean(baseline_means)) if baseline_means else None
                    ),
                )
            )
    return rows


def compute_usage(
    target: PromptSet,
    baseline: PromptSet,
    config: UsageConfig,
    context: PipelineContext,
) -> UsageResult:
    """Compare the interaction-context scales between the two prompt sets.

    `context` is part of the uniform section signature; the tests are
    deterministic rank statistics, so it needs no embeddings or RNG.
    """
    log(f"usage: context scales for '{target.name}' vs '{baseline.name}'")
    tests = [
        _test_scale(scale, _author_means(target, scale), _author_means(baseline, scale))
        for scale in USAGE_SCALES
    ]
    tested = [t for t in tests if t.p_value is not None]
    adjusted, rejected = benjamini_hochberg(
        np.array([t.p_value for t in tested]), config.fdr_alpha
    )
    for test, p_adjusted, is_discovery in zip(tested, adjusted, rejected, strict=True):
        test.p_adjusted = float(p_adjusted)
        test.significant = bool(is_discovery)
    n_significant = sum(bool(t.significant) for t in tests)
    log(
        f"usage: {n_significant}/{len(tested)} tested scales significant "
        f"at FDR {config.fdr_alpha}"
    )
    return UsageResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        fdr_alpha=config.fdr_alpha,
        n_scales_tested=len(tested),
        n_significant_scales=n_significant,
        scale_tests=tests,
        domain_means=_domain_profile(target, baseline),
    )
