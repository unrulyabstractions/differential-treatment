"""Unit tests for the information measures against the paper's worked example."""

import numpy as np
import pytest

from dtreat.common.discrete_information import (
    binary_entropy,
    community_axis_information,
    kl_divergence_bits,
    normalize_profile,
)


class TestPaperWorkedExample:
    """§4.5.3 case study: z_t = [1/3, 2/3, 1/3, 2/3], z_b = [1, 0, 0, 1/3]."""

    def test_d_pi_matches_paper(self):
        z_target = np.array([1 / 3, 2 / 3, 1 / 3, 2 / 3])
        z_baseline = np.array([1.0, 0.0, 0.0, 1 / 3])
        d_pi = kl_divergence_bits(
            normalize_profile(z_target, epsilon=0.01),
            normalize_profile(z_baseline, epsilon=0.01),
        )
        assert d_pi == pytest.approx(2.37, abs=0.005)

    def test_profiles_match_paper_rounding(self):
        profile = normalize_profile(np.array([1 / 3, 2 / 3, 1 / 3, 2 / 3]), 0.01)
        assert np.round(profile, 2).tolist() == [0.17, 0.33, 0.17, 0.33]
        profile_b = normalize_profile(np.array([1.0, 0.0, 0.0, 1 / 3]), 0.01)
        assert np.round(profile_b, 2).tolist() == [0.74, 0.01, 0.01, 0.25]

    def test_information_ranking_matches_paper(self):
        # Paper table: I_1 = 0.46, I_2 = 0.46, I_3 = 0.19, I_4 = 0.08
        assert community_axis_information(1 / 3, 1.0) == pytest.approx(0.46, abs=0.005)
        assert community_axis_information(2 / 3, 0.0) == pytest.approx(0.46, abs=0.005)
        assert community_axis_information(1 / 3, 0.0) == pytest.approx(0.19, abs=0.005)
        assert community_axis_information(2 / 3, 1 / 3) == pytest.approx(0.08, abs=0.005)


class TestBinaryEntropy:
    def test_bounds_and_symmetry(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0
        assert binary_entropy(0.5) == pytest.approx(1.0)
        assert binary_entropy(0.2) == pytest.approx(binary_entropy(0.8))


class TestKlDivergence:
    def test_zero_for_identical(self):
        p = np.array([0.2, 0.3, 0.5])
        assert kl_divergence_bits(p, p) == pytest.approx(0.0)

    def test_infinite_when_support_missing(self):
        assert kl_divergence_bits(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == float("inf")

    def test_zero_p_terms_contribute_nothing(self):
        assert kl_divergence_bits(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(1.0)


class TestNormalizeProfile:
    def test_rejects_all_zero_without_epsilon(self):
        with pytest.raises(ValueError):
            normalize_profile(np.zeros(3))

    def test_epsilon_gives_uniform_for_all_zero(self):
        assert normalize_profile(np.zeros(3), 0.01) == pytest.approx(np.ones(3) / 3)


class TestMutualInformation:
    def test_zero_when_rates_equal(self):
        assert community_axis_information(0.4, 0.4) == pytest.approx(0.0, abs=1e-12)

    def test_maximal_when_deterministic_and_opposite(self):
        assert community_axis_information(1.0, 0.0) == pytest.approx(1.0)

    def test_unbalanced_weights(self):
        # With all weight on target, no information about community remains
        assert community_axis_information(0.9, 0.1, weight_target=1.0) == pytest.approx(0.0)
