"""Unit tests for permutation significance and Benjamini–Hochberg control."""

import numpy as np
import pytest

from dtreat.stages.treatment_analysis.permutation_significance import (
    benjamini_hochberg,
    permutation_p_values,
)


class TestBenjaminiHochberg:
    def test_textbook_example(self):
        # Classic worked example: m=6; only the two smallest survive alpha=0.05
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06])
        q, significant = benjamini_hochberg(p, alpha=0.05)
        assert significant.tolist() == [True, True, False, False, False, False]
        assert q[0] == pytest.approx(0.006)
        assert q[1] == pytest.approx(0.024)
        # q-values are monotone in p order
        assert all(q[i] <= q[i + 1] + 1e-12 for i in range(len(q) - 1))

    def test_all_null_rejects_nothing(self):
        q, significant = benjamini_hochberg(np.array([0.2, 0.5, 0.9]), alpha=0.05)
        assert not significant.any()
        assert (q <= 1.0).all()

    def test_empty_input(self):
        q, significant = benjamini_hochberg(np.array([]), alpha=0.05)
        assert len(q) == 0 and len(significant) == 0


def _one_axis_matrices(rates_target, rates_baseline, per_cluster=5):
    """Build sums/counts for one axis from per-cluster verdict rates."""
    rates = np.array(list(rates_target) + list(rates_baseline))
    counts = np.full((1, len(rates)), per_cluster, dtype=float)
    sums = (rates * per_cluster).reshape(1, -1)
    is_target = np.array([True] * len(rates_target) + [False] * len(rates_baseline))
    return sums, counts, is_target


class TestPermutationPValues:
    def test_strong_effect_is_significant(self):
        sums, counts, is_target = _one_axis_matrices([1.0] * 10, [0.0] * 10)
        p, delta = permutation_p_values(sums, counts, is_target, 500, seed=1)
        assert delta[0] == pytest.approx(1.0)
        assert p[0] <= 3 / 501  # essentially the add-one floor

    def test_no_effect_is_not_significant(self):
        rng = np.random.default_rng(0)
        rates = rng.uniform(0.3, 0.7, size=20)
        sums, counts, is_target = _one_axis_matrices(rates[:10], rates[10:])
        p, _delta = permutation_p_values(sums, counts, is_target, 500, seed=2)
        assert p[0] > 0.05

    def test_p_values_bounded_by_add_one_rule(self):
        sums, counts, is_target = _one_axis_matrices([1.0] * 4, [0.0] * 4)
        p, _ = permutation_p_values(sums, counts, is_target, 100, seed=3)
        assert 0 < p[0] <= 1
        assert p[0] >= 1 / 101

    def test_deterministic_given_seed(self):
        sums, counts, is_target = _one_axis_matrices([0.8] * 6, [0.4] * 6)
        p1, _ = permutation_p_values(sums, counts, is_target, 300, seed=42)
        p2, _ = permutation_p_values(sums, counts, is_target, 300, seed=42)
        assert p1[0] == p2[0]

    def test_axis_with_empty_side_gets_p_one(self):
        sums = np.array([[3.0, 2.0, 0.0, 0.0]])
        counts = np.array([[5.0, 5.0, 0.0, 0.0]])  # baseline clusters have no verdicts
        is_target = np.array([True, True, False, False])
        p, delta = permutation_p_values(sums, counts, is_target, 100, seed=0)
        assert p[0] == 1.0
        assert np.isnan(delta[0])


class TestFdrCalibration:
    """L1 realism: empirical FDR under a global null stays near alpha."""

    def test_false_positive_rate_under_null(self):
        rng = np.random.default_rng(123)
        false_flags = 0
        trials = 60
        is_target = np.array([True] * 12 + [False] * 12)
        for trial in range(trials):
            # 5 independent null axes per trial
            sums5 = np.vstack([
                (rng.binomial(5, rng.uniform(0.4, 0.6), size=24)).astype(float)
                for _ in range(5)
            ])
            counts5 = np.full_like(sums5, 5.0)
            p, _ = permutation_p_values(sums5, counts5, is_target, 200, seed=trial)
            _q, significant = benjamini_hochberg(p, alpha=0.05)
            false_flags += int(significant.any())
        # Family-wise false-flag rate under BH at alpha=0.05 with independent
        # nulls should be ~5%; allow generous slack for 60 trials
        assert false_flags / trials <= 0.15
