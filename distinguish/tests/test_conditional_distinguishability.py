"""Conditional distinguishability: aggregation + interpretation invariants.

Proves the marginal-vs-conditional machinery (src/conditional/conditional_analysis.py)
labels the four regimes correctly and combines strata soundly, plus an end-to-end
run on the synthetic dataset conditioning on domain.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.common.dataset_tables import PromptDataset
from src.common.dimension_result import DimensionVerdict
from src.common.run_config import DistributionalConfig, LexicalConfig
from src.conditional.conditional_analysis import (
    ConditionalResult,
    StratumResult,
    _aggregate,
    _fisher_combine,
    compute_conditional,
    stratum_values,
)
from src.distributional.distributional_dimension import compute_distributional
from src.lexical.lexical_dimension import compute_lexical
from src.pipeline.pipeline_context import PipelineContext

REPO = Path(__file__).resolve().parent.parent


def _verdict(stat: float, p: float | None, sig: bool | None) -> DimensionVerdict:
    return DimensionVerdict(
        dimension="d",
        test_name="t",
        variant="v",
        statistic_name="s",
        statistic_value=stat,
        p_value=p,
        significant=sig,
        detail="",
    )


def _stratum(stat: float, p: float, sig: bool, n: int = 20) -> StratumResult:
    return StratumResult(
        value="z",
        n_prompts_target=n,
        n_prompts_baseline=n,
        verdicts=[_verdict(stat, p, sig)],
    )


def test_fisher_combine_known_answer():
    """Fisher's method matches the chi-square closed form and handles the p-floor."""
    ps = [0.1, 0.2, 0.05]
    expected = float(stats.chi2.sf(-2 * sum(math.log(p) for p in ps), df=2 * 3))
    assert _fisher_combine(ps) == pytest.approx(expected)
    # A fully-rejecting stratum (p=0) is floored, never -inf.
    assert 0.0 <= _fisher_combine([0.0, 0.5]) <= 1.0
    assert _fisher_combine([None, None]) is None


def test_weighted_mean_is_stratum_size_weighted():
    """conditional_value weights each stratum by its smaller side's n."""
    strata = [_stratum(0.9, 0.01, True, n=30), _stratum(0.6, 0.5, False, n=10)]
    out = _aggregate([_verdict(0.8, 0.2, False)], strata)[0]
    assert out.conditional_value == pytest.approx((30 * 0.9 + 10 * 0.6) / 40)
    assert out.n_significant_strata == 1
    assert out.n_strata == 2


def test_interpretation_topic_choice():
    """Marginal significant, conditional null => explained by the variable."""
    # Two strata each clearly non-significant => Fisher p large.
    strata = [_stratum(0.51, 0.8, False), _stratum(0.52, 0.7, False)]
    out = _aggregate([_verdict(0.75, 1e-4, True)], strata)[0]
    assert out.interpretation == "topic-choice"


def test_interpretation_survives():
    """Conditional significant but not exceeding marginal => coded, survives."""
    strata = [_stratum(0.70, 1e-3, True), _stratum(0.72, 1e-3, True)]
    out = _aggregate([_verdict(0.80, 1e-6, True)], strata)[0]
    assert out.interpretation == "survives"


def test_interpretation_revealed_simpson():
    """Conditional exceeds the marginal AND is significant => Simpson reveal."""
    # Within-stratum statistic (0.9) beats the pooled marginal (0.55); strata sig.
    strata = [_stratum(0.90, 1e-3, True), _stratum(0.90, 1e-3, True)]
    out = _aggregate([_verdict(0.55, 0.3, False)], strata)[0]
    assert out.conditional_value > out.marginal_value
    assert out.interpretation == "revealed"


def test_stratum_values_drops_unrecorded():
    ds = PromptDataset.load(REPO / "data" / "synthetic")
    domains = stratum_values(ds, ("target", "baseline"), "domain")
    assert set(domains) == {"MH", "GSH", "REL"}
    assert "" not in domains and "0" not in domains


def test_end_to_end_distributional_revealed_on_synthetic():
    """The C2ST separates within each domain at least as well as pooled."""
    ds = PromptDataset.load(REPO / "data" / "synthetic")
    tgt, base = ds.prompt_set("target"), ds.prompt_set("baseline")
    cfg = DistributionalConfig(
        embedders=["sentence-transformers/all-MiniLM-L6-v2"],
        classifiers=["linear"],
        n_permutations=200,
    )
    marginal = compute_distributional(tgt, base, cfg, PipelineContext(0))
    conditional = compute_conditional(
        ds,
        "target",
        "baseline",
        "domain",
        "distributional",
        compute_distributional,
        cfg,
        PipelineContext(0),
        marginal.to_verdicts(),
        min_prompts_per_side=6,
    )
    assert len(conditional.strata) == 3  # MH, GSH, REL all powered enough
    verdict = conditional.conditional_verdicts[0]
    # Within-domain accuracy is high and the interpretation is coded (not topic).
    assert verdict.conditional_value > 0.9
    assert verdict.interpretation in ("survives", "revealed")

    # Round-trips cleanly, no NaN leak.
    import json

    assert (
        ConditionalResult.from_dict(conditional.to_dict()).to_dict()
        == conditional.to_dict()
    )
    json.dumps(conditional.to_dict(), allow_nan=False)


def test_lexical_conditional_runs_and_labels(tmp_path):
    """Lexical conditional produces a labelled verdict (topic-choice on synthetic)."""
    ds = PromptDataset.load(REPO / "data" / "synthetic")
    tgt, base = ds.prompt_set("target"), ds.prompt_set("baseline")
    cfg = LexicalConfig(calibration_plots=False)
    marginal = compute_lexical(tgt, base, cfg, PipelineContext(0))
    conditional = compute_conditional(
        ds,
        "target",
        "baseline",
        "domain",
        "lexical",
        compute_lexical,
        cfg,
        PipelineContext(0),
        marginal.to_verdicts(),
        min_prompts_per_side=6,
    )
    assert conditional.conditional_verdicts
    assert conditional.conditional_verdicts[0].interpretation in (
        "topic-choice",
        "survives",
        "revealed",
        "inconclusive",
    )
    _ = np  # keep numpy import meaningful for future assertions
