"""Proof-grade invariants for serialization + the shared statistics kernel.

These tests are built to FAIL if the code were subtly wrong, not merely to
exercise it:

* ROUND-TRIP: every BaseSchema result type is *produced for real* on
  data/synthetic (lexical, syntactic, semantic MiniLM, distributional linear,
  topical embedding, interactional embedding, usage) and must satisfy
  ``from_dict(to_dict()).to_dict() == to_dict()`` with no NaN/Inf leaking past
  ``json.dumps(..., allow_nan=False)``.  A fault-injection case plants real
  ``nan``/``inf`` into nested schemas to prove the canonicalizer's guard is
  what actually catches them.
* JSD: property-based bounds/symmetry/identity + exact scipy equivalence, plus
  the two disjoint/identical known answers (1 bit / 0 bit).
* permute_labels_by_author: property-based author-uniformity + author-label
  multiset preservation, a Monte-Carlo test that the induced permutation is
  *uniform* (an unbiased shuffle, not just any relabeling), and the mixed-label
  guard.
* benjamini_hochberg: exact match to scipy, monotonicity, the NaN guard, and a
  Monte-Carlo FDR-calibration test (empirical FDR under the global null stays
  at alpha, whereas uncorrected thresholding blows past it) plus known-answer
  recovery of planted discoveries.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import false_discovery_control

from src.common.run_config import (
    DistributionalConfig,
    InteractionalConfig,
    LexicalConfig,
    SemanticConfig,
    SyntacticConfig,
    TopicalConfig,
    UsageConfig,
)
from src.common.stats_utils import (
    benjamini_hochberg,
    jensen_shannon_divergence,
    permute_labels_by_author,
)
from src.distributional.distributional_dimension import compute_distributional
from src.interactional.interactional_dimension import compute_interactional
from src.lexical.lexical_dimension import compute_lexical
from src.pipeline.pipeline_context import PipelineContext
from src.semantic.semantic_dimension import compute_semantic
from src.syntactic.syntactic_dimension import (
    FeatureContrast,
    SyntacticResult,
    compute_syntactic,
)
from src.topical.topical_dimension import compute_topical
from src.usage.usage_attitudes import ScaleTest, UsageResult, compute_usage

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Fixture: produce EVERY result type for real on data/synthetic.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def produced_results(synthetic) -> dict[str, object]:
    """Run each dimension on the synthetic dataset and return the result objects.

    Heavy variants (OpenAI/Cohere/residual/ModernBERT/TopicGPT) are trimmed to
    the offline MiniLM / embedding backends so the suite is hermetic; the
    schemas produced are identical to a full run's.  Permutation budgets are
    cut because the round-trip contract is independent of the null size.
    """
    target = synthetic.prompt_set("target")
    baseline = synthetic.prompt_set("baseline")
    ctx = PipelineContext(random_seed=0)
    results: dict[str, object] = {}
    results["usage"] = compute_usage(target, baseline, UsageConfig(), ctx)
    results["lexical"] = compute_lexical(target, baseline, LexicalConfig(), ctx)
    results["syntactic"] = compute_syntactic(target, baseline, SyntacticConfig(), ctx)
    results["semantic"] = compute_semantic(
        target,
        baseline,
        SemanticConfig(text_embedders=[_MINILM], residual_models=[]),
        ctx,
    )
    results["distributional"] = compute_distributional(
        target,
        baseline,
        DistributionalConfig(
            embedders=[_MINILM], classifiers=["linear"], cv_folds=5, n_permutations=40
        ),
        ctx,
    )
    results["topical"] = compute_topical(
        target,
        baseline,
        TopicalConfig(
            assignment_backends=["embedding"],
            embedding_model=_MINILM,
            n_permutations=40,
        ),
        ctx,
    )
    results["interactional"] = compute_interactional(
        target,
        baseline,
        InteractionalConfig(
            annotation_backends=["embedding"],
            embedding_model=_MINILM,
            n_permutations=40,
        ),
        ctx,
    )
    ctx.cleanup()
    return results


_RESULT_NAMES = [
    "usage",
    "lexical",
    "syntactic",
    "semantic",
    "distributional",
    "topical",
    "interactional",
]


def _iter_scalars(obj):
    """Yield every scalar leaf of a nested to_dict() structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_scalars(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_scalars(v)
    else:
        yield obj


def _has_nonfinite_float(obj) -> bool:
    return any(
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
        for v in _iter_scalars(obj)
    )


# --------------------------------------------------------------------------- #
# (1) ROUND-TRIP for every produced result type.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _RESULT_NAMES)
def test_roundtrip_idempotent(produced_results, name):
    result = produced_results[name]
    payload = result.to_dict()
    reconstructed = type(result).from_dict(payload)
    assert reconstructed.to_dict() == payload, (
        f"{name}: from_dict(to_dict()).to_dict() != to_dict()"
    )


@pytest.mark.parametrize("name", _RESULT_NAMES)
def test_no_nan_inf_leak(produced_results, name):
    payload = produced_results[name].to_dict()
    # No raw non-finite float may survive canonicalization ...
    assert not _has_nonfinite_float(payload), f"{name}: non-finite float leaked"
    # ... which is exactly what lets strict JSON encoding succeed.
    json.dumps(payload, allow_nan=False)


def test_every_result_is_covered(produced_results):
    """Guard: the parametrization must exhaust what the fixture produces."""
    assert set(produced_results) == set(_RESULT_NAMES)


# --------------------------------------------------------------------------- #
# Fault injection: PLANTED nan/inf must be caught by the canonicalizer, not by
# luck-of-the-data.  A to_dict() that forgot to canonicalize would leak the
# float and allow_nan=False would raise -- so this proves the guard does work.
# --------------------------------------------------------------------------- #
def _syntactic_with_nonfinite() -> SyntacticResult:
    bad = FeatureContrast(
        feature_name="BIN_x",
        prevalence_target=float("nan"),
        prevalence_baseline=0.0,
        count_target=0,
        count_baseline=0,
        log_odds=float("inf"),
        z_score=float("-inf"),
        p_value=float("nan"),
        p_adjusted=1.0,
        significant=False,
    )
    return SyntacticResult(
        target_name="t",
        baseline_name="b",
        target_label="T",
        baseline_label="B",
        n_prompts_target=1,
        n_prompts_baseline=1,
        model_name="m",
        smoothing_count=0.5,
        fdr_alpha=0.05,
        top_features_reported=20,
        n_significant_features=0,
        feature_contrasts=[bad],
    )


def _usage_with_nonfinite() -> UsageResult:
    bad = ScaleTest(
        scale="llm_freq",
        n_target_authors=0,
        n_baseline_authors=0,
        mean_target=float("nan"),
        mean_baseline=float("inf"),
        mann_whitney_u=None,
        p_value=None,
        p_adjusted=None,
        rank_biserial=None,
        significant=None,
    )
    return UsageResult(
        target_name="t",
        baseline_name="b",
        target_label="T",
        baseline_label="B",
        n_prompts_target=1,
        n_prompts_baseline=1,
        fdr_alpha=0.05,
        n_scales_tested=0,
        n_significant_scales=0,
        scale_tests=[bad],
    )


@pytest.mark.parametrize("factory", [_syntactic_with_nonfinite, _usage_with_nonfinite])
def test_canonicalizer_neutralizes_planted_nonfinite(factory):
    result = factory()
    payload = result.to_dict()
    # Non-finite floats became sentinel strings, so nothing non-finite remains.
    assert not _has_nonfinite_float(payload)
    scalars = set(_iter_scalars(payload))
    assert {"NaN", "Inf"} <= scalars or {"NaN", "-Inf"} <= scalars
    json.dumps(payload, allow_nan=False)  # must not raise
    # And the round-trip is still idempotent through the sentinels.
    assert type(result).from_dict(payload).to_dict() == payload


# --------------------------------------------------------------------------- #
# (2) Jensen-Shannon divergence.
# --------------------------------------------------------------------------- #
@st.composite
def _prob_vector_pair(draw):
    n = draw(st.integers(min_value=2, max_value=8))
    entry = st.floats(
        min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False
    )
    p = np.array(draw(st.lists(entry, min_size=n, max_size=n)), dtype=float)
    q = np.array(draw(st.lists(entry, min_size=n, max_size=n)), dtype=float)
    assume(p.sum() > 1e-6 and q.sum() > 1e-6)
    return p, q


@settings(max_examples=300, deadline=None)
@given(_prob_vector_pair())
def test_jsd_bounds_symmetry_and_scipy_equivalence(pair):
    p, q = pair
    d = jensen_shannon_divergence(p, q)
    # Bounded in [0, 1] bits.
    assert -1e-12 <= d <= 1.0 + 1e-9
    # Symmetric.
    assert abs(d - jensen_shannon_divergence(q, p)) <= 1e-12
    # Exactly the squared scipy JS distance in base 2.
    assert abs(d - jensenshannon(p, q, base=2) ** 2) <= 1e-12


@settings(max_examples=300, deadline=None)
@given(_prob_vector_pair())
def test_jsd_zero_iff_equal(pair):
    p, q = pair
    # Identity direction: JSD(p, p) == 0.
    assert jensen_shannon_divergence(p, p) <= 1e-9
    pn, qn = p / p.sum(), q / q.sum()
    tv = 0.5 * float(np.abs(pn - qn).sum())
    d = jensen_shannon_divergence(p, q)
    if tv <= 1e-9:
        assert d <= 1e-9  # equal distributions -> zero divergence
    elif tv >= 0.05:
        assert d > 1e-9  # genuinely different -> strictly positive


def test_jsd_known_answers():
    # Disjoint support -> exactly 1 bit; identical -> 0.
    assert jensen_shannon_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert jensen_shannon_divergence(np.array([0.3, 0.7]), np.array([0.3, 0.7])) == 0.0


# --------------------------------------------------------------------------- #
# (3) permute_labels_by_author.
# --------------------------------------------------------------------------- #
@st.composite
def _author_fixture(draw, equal_counts=False):
    n_authors = draw(st.integers(min_value=2, max_value=8))
    names = [f"auth{i}" for i in range(n_authors)]
    author_label = {a: draw(st.integers(0, 1)) for a in names}
    fixed_k = draw(st.integers(1, 4)) if equal_counts else None
    author_ids: list[str] = []
    labels: list[int] = []
    for a in names:
        k = fixed_k if equal_counts else draw(st.integers(1, 4))
        author_ids.extend([a] * k)
        labels.extend([author_label[a]] * k)
    order = draw(st.permutations(range(len(author_ids))))
    author_ids = [author_ids[i] for i in order]
    labels = [labels[i] for i in order]
    seed = draw(st.integers(0, 2**31 - 1))
    return author_ids, np.array(labels, dtype=np.int64), author_label, seed


def _author_label_map(author_ids, labels) -> dict[str, int]:
    m: dict[str, int] = {}
    for a, label in zip(author_ids, labels, strict=True):
        m[a] = int(label)
    return m


@settings(max_examples=300, deadline=None)
@given(_author_fixture())
def test_permute_preserves_uniformity_and_author_multiset(fixture):
    author_ids, labels, author_label, seed = fixture
    permuted = permute_labels_by_author(author_ids, labels, np.random.default_rng(seed))
    # Shape + dtype preserved.
    assert permuted.shape == labels.shape
    assert permuted.dtype == labels.dtype
    # Each author still carries a single label (uniformity).
    permuted_map = _author_label_map(author_ids, permuted)
    for a in set(author_ids):
        vals = {int(permuted[i]) for i in range(len(author_ids)) if author_ids[i] == a}
        assert len(vals) == 1
    # The AUTHOR-level label multiset is preserved (it is a permutation).
    assert sorted(permuted_map.values()) == sorted(author_label.values())


@settings(max_examples=200, deadline=None)
@given(_author_fixture(equal_counts=True))
def test_permute_preserves_prompt_multiset_when_counts_equal(fixture):
    author_ids, labels, _author_label, seed = fixture
    permuted = permute_labels_by_author(author_ids, labels, np.random.default_rng(seed))
    # With equal prompts-per-author the prompt-level multiset is also preserved.
    assert sorted(permuted.tolist()) == sorted(labels.tolist())


def test_permute_raises_on_mixed_label_author():
    with pytest.raises(ValueError):
        permute_labels_by_author(["A", "A"], np.array([0, 1]), np.random.default_rng(0))


def test_permute_is_a_uniform_shuffle():
    """Monte-Carlo: every author is equally likely to receive each label.

    A biased or off-by-one relabeling (still uniformity-preserving) would pin an
    author near 0 or 1; a genuine uniform permutation gives each author label=1
    with frequency m1/k.  6 authors, 3 labelled 1 -> expected 0.5 each.
    """
    authors = [c for c in "ABCDEF" for _ in range(2)]
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    assert int(labels.reshape(6, 2)[:, 0].sum()) == 3  # 3 of 6 authors are label 1
    rng = np.random.default_rng(7)
    n_trials = 5000
    hits = dict.fromkeys("ABCDEF", 0)
    for _ in range(n_trials):
        permuted = permute_labels_by_author(authors, labels, rng)
        m = _author_label_map(authors, permuted)
        assert sum(m.values()) == 3  # author-label multiset conserved every draw
        for a in hits:
            hits[a] += m[a]
    for a, count in hits.items():
        freq = count / n_trials
        assert abs(freq - 0.5) < 0.06, f"author {a} biased: P(label=1)={freq:.3f}"


# --------------------------------------------------------------------------- #
# (4) Benjamini-Hochberg.
# --------------------------------------------------------------------------- #
@st.composite
def _pvalues(draw):
    n = draw(st.integers(min_value=1, max_value=40))
    entry = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    return np.array(draw(st.lists(entry, min_size=n, max_size=n)), dtype=float)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_pvalues(), st.sampled_from([0.01, 0.05, 0.1, 0.2]))
def test_bh_matches_scipy_and_reject_rule(p_values, alpha):
    adjusted, rejected = benjamini_hochberg(p_values, alpha)
    assert np.allclose(adjusted, false_discovery_control(p_values, method="bh"))
    # `rejected` is exactly adjusted <= alpha.
    assert np.array_equal(rejected, adjusted <= alpha)


@settings(max_examples=300, deadline=None)
@given(_pvalues())
def test_bh_adjusted_is_monotone_in_raw_p(p_values):
    adjusted, _ = benjamini_hochberg(p_values, 0.05)
    order = np.argsort(p_values, kind="stable")
    ordered = adjusted[order]
    assert np.all(np.diff(ordered) >= -1e-12)  # non-decreasing with raw p


def test_bh_nan_guard_treats_nan_as_least_significant():
    with_nan = np.array([0.001, np.nan, 0.02])
    replaced = np.array([0.001, 1.0, 0.02])
    adj_nan, rej_nan = benjamini_hochberg(with_nan, 0.05)
    adj_ref, rej_ref = benjamini_hochberg(replaced, 0.05)
    # nan_to_num(nan=1.0): the NaN slot behaves exactly like a raw p of 1.0.
    assert np.allclose(adj_nan, adj_ref)
    assert np.array_equal(rej_nan, rej_ref)
    assert adj_nan[1] == 1.0 and not rej_nan[1]


def test_bh_empty_input():
    adjusted, rejected = benjamini_hochberg(np.array([]), 0.05)
    assert adjusted.size == 0 and rejected.size == 0


def test_bh_recovers_planted_discoveries():
    rng = np.random.default_rng(0)
    p_values = np.concatenate([np.full(3, 1e-8), rng.uniform(0.2, 1.0, 17)])
    _, rejected = benjamini_hochberg(p_values, 0.05)
    assert rejected[:3].all()  # the strong signals are discovered
    assert rejected.sum() == 3  # and nothing spurious


def test_bh_controls_fdr_under_global_null():
    """Monte-Carlo calibration: under the full null BH holds FDR at alpha,

    while uncorrected thresholding (reject p<=alpha) does not -- proving the
    correction is real, not a pass-through.  Under the global null every
    rejection is false, so FDR == P(at least one rejection).
    """
    alpha, m, n_trials = 0.1, 20, 2000
    rng = np.random.default_rng(1)
    bh_any = 0
    raw_any = 0
    for _ in range(n_trials):
        p_values = rng.uniform(0.0, 1.0, m)
        _, rejected = benjamini_hochberg(p_values, alpha)
        bh_any += int(rejected.any())
        raw_any += int((p_values <= alpha).any())
    bh_fdr = bh_any / n_trials
    raw_fdr = raw_any / n_trials
    # BH controls the empirical FDR at ~alpha (small Monte-Carlo slack) ...
    assert bh_fdr <= alpha + 0.03, f"BH failed FDR control: {bh_fdr:.3f}"
    # ... whereas the uncorrected baseline is far above it, so BH is doing work.
    assert 0.5 < raw_fdr <= 1.0
    assert raw_fdr > bh_fdr + 0.3
