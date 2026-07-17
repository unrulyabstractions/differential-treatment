"""Conditional distinguishability: is the target/baseline difference topic-choice
or coded style?

Marginal (aggregate) distinguishability pools all prompts. Conditional
distinguishability measures the SAME dimension WITHIN strata of a content variable
Z (domain / topic / provenance) and aggregates. Comparing the two separates *what*
the groups talk about from *how* they talk about it (the paper's markedness /
codedness thesis):

  * marginal separable, conditional ~null  -> the difference is which topics each
    group raises (topic choice); conditioning removes it.
  * conditional stays separable, or exceeds the marginal (a Simpson reversal where
    each stratum separates but the pooled sets do not) -> genuine coded signal
    that survives holding topic fixed.

Per (test, variant) verdict the conditional aggregate is the stratum-size weighted
mean statistic plus a Fisher-combined p-value over the strata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict

if TYPE_CHECKING:  # runtime import would cycle via src.pipeline
    from src.common.dataset_tables import PromptDataset
    from src.pipeline.pipeline_context import PipelineContext

_P_FLOOR = 1e-4  # keep Fisher's -2 ln p finite for a stratum that fully rejects


@dataclass
class StratumResult(BaseSchema):
    """One stratum's within-stratum run of the dimension."""

    value: str  # the conditioning-variable value, e.g. "MH"
    n_prompts_target: int
    n_prompts_baseline: int
    verdicts: list[DimensionVerdict] = field(default_factory=list)


@dataclass
class ConditionalVerdict(BaseSchema):
    """Marginal vs conditional distinguishability for one (test, variant)."""

    test_name: str
    variant: str
    statistic_name: str
    marginal_value: float
    marginal_p: float | None
    marginal_significant: bool | None
    # Stratum-size weighted mean of the within-stratum statistic.
    conditional_value: float
    # Fisher-combined p over strata (H0: no distinguishability in any stratum).
    conditional_p: float | None
    n_significant_strata: int
    n_strata: int
    # "topic-choice" | "survives" | "revealed" | "inconclusive"
    interpretation: str


@dataclass
class ConditionalResult(BaseSchema):
    """Conditional-distinguishability breakdown for one section x variable."""

    section: str
    conditioning_variable: str
    strata: list[StratumResult] = field(default_factory=list)
    conditional_verdicts: list[ConditionalVerdict] = field(default_factory=list)
    skipped_strata: list[str] = field(default_factory=list)  # "value (reason)"


def stratum_values(
    dataset: PromptDataset, comparison_cohorts: tuple[str, str], variable: str
) -> list[str]:
    """Distinct recorded values of `variable` present in either cohort's prompts.

    Unrecorded rows (empty string / 0 topic id) are dropped — they cannot define a
    content stratum.
    """
    prompts = dataset.prompts
    mask = prompts["cohort"].isin(list(comparison_cohorts))
    raw = prompts.loc[mask, variable]
    values = {str(v).strip() for v in raw if str(v).strip() not in ("", "0", "nan")}
    return sorted(values)


def _fisher_combine(p_values: list[float]) -> float | None:
    """Fisher's method: combine per-stratum p into one. None if no p available."""
    usable = [min(max(p, _P_FLOOR), 1.0) for p in p_values if p is not None]
    if not usable:
        return None
    statistic = -2.0 * float(np.sum(np.log(usable)))
    return float(stats.chi2.sf(statistic, df=2 * len(usable)))


def _interpret(
    marginal_significant: bool | None,
    marginal_value: float,
    conditional_value: float,
    conditional_p: float | None,
    alpha: float,
) -> str:
    """Label how conditioning changes the distinguishability of this verdict."""
    conditional_significant = conditional_p is not None and conditional_p <= alpha
    # Statistics here are oriented so higher = more distinguishable.
    if conditional_value > marginal_value + 1e-9 and conditional_significant:
        return "revealed"  # Simpson: separates within strata, weaker when pooled
    if conditional_significant:
        return "survives"  # coded signal beyond the conditioning variable
    if marginal_significant and not conditional_significant:
        return "topic-choice"  # marginal difference explained by the variable
    return "inconclusive"


def compute_conditional(
    dataset: PromptDataset,
    target_cohort: str,
    baseline_cohort: str,
    variable: str,
    section: str,
    section_compute: Callable,
    section_config: object,
    context: PipelineContext,
    marginal_verdicts: list[DimensionVerdict],
    min_prompts_per_side: int,
) -> ConditionalResult:
    """Run `section` within each stratum of `variable`, aggregate vs the marginal."""
    result = ConditionalResult(section=section, conditioning_variable=variable)
    values = stratum_values(dataset, (target_cohort, baseline_cohort), variable)
    prompts = dataset.prompts

    for value in values:
        stratum_mask = prompts[variable].astype(str).str.strip() == value
        target = dataset.prompt_set(target_cohort, prompt_mask=stratum_mask)
        baseline = dataset.prompt_set(baseline_cohort, prompt_mask=stratum_mask)
        n_t, n_b = len(target.prompts), len(baseline.prompts)
        if min(n_t, n_b) < min_prompts_per_side:
            result.skipped_strata.append(f"{value} ({n_t}/{n_b} prompts)")
            continue
        section_result = section_compute(target, baseline, section_config, context)
        result.strata.append(
            StratumResult(
                value=value,
                n_prompts_target=n_t,
                n_prompts_baseline=n_b,
                verdicts=section_result.to_verdicts(),
            )
        )

    result.conditional_verdicts = _aggregate(marginal_verdicts, result.strata)
    return result


def _aggregate(
    marginal_verdicts: list[DimensionVerdict],
    strata: list[StratumResult],
    alpha: float = 0.05,
) -> list[ConditionalVerdict]:
    """Weighted-mean statistic + Fisher-combined p per (test, variant) verdict."""
    if not strata:
        return []
    conditional: list[ConditionalVerdict] = []
    for marginal in marginal_verdicts:
        key = (marginal.test_name, marginal.variant)
        weighted_sum, weight_total, p_values, n_sig = 0.0, 0.0, [], 0
        for stratum in strata:
            match = next(
                (v for v in stratum.verdicts if (v.test_name, v.variant) == key),
                None,
            )
            if match is None:
                continue
            weight = float(min(stratum.n_prompts_target, stratum.n_prompts_baseline))
            weighted_sum += weight * match.statistic_value
            weight_total += weight
            p_values.append(match.p_value)
            if match.significant:
                n_sig += 1
        if weight_total == 0:
            continue
        conditional_value = weighted_sum / weight_total
        conditional_p = _fisher_combine(p_values)
        conditional.append(
            ConditionalVerdict(
                test_name=marginal.test_name,
                variant=marginal.variant,
                statistic_name=marginal.statistic_name,
                marginal_value=marginal.statistic_value,
                marginal_p=marginal.p_value,
                marginal_significant=marginal.significant,
                conditional_value=conditional_value,
                conditional_p=conditional_p,
                n_significant_strata=n_sig,
                n_strata=len(strata),
                interpretation=_interpret(
                    marginal.significant,
                    marginal.statistic_value,
                    conditional_value,
                    conditional_p,
                    alpha,
                ),
            )
        )
    return conditional
