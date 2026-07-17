"""L1 realism: statistics exercised on synthetic score matrices with known
effect sizes — no LLMs anywhere. Validates power, calibration, and C2ST
behavior beyond what single mock runs can show."""

import numpy as np

from dtreat.stages.response_scoring.scored_response_schemas import ScoredResponse
from dtreat.stages.treatment_analysis.classifier_two_sample_test import run_c2st
from dtreat.stages.treatment_analysis.permutation_significance import (
    benjamini_hochberg,
    permutation_p_values,
)
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    build_cluster_matrices,
    build_permutation_clusters,
)

AXES = ["a1", "a2", "a3"]


def synthetic_scored(rates_target, rates_baseline, n_prompts, samples, seed):
    """Generate ScoredResponse records from per-axis Bernoulli rates."""
    rng = np.random.default_rng(seed)
    records = []
    for community, rates, prefix in (
        ("lgbtq", rates_target, "t"),
        ("cishet", rates_baseline, "b"),
    ):
        for prompt_index in range(n_prompts):
            for sample_index in range(samples):
                verdicts = {
                    axis: bool(rng.random() < rate)
                    for axis, rate in zip(AXES, rates, strict=True)
                }
                records.append(
                    ScoredResponse(
                        response_id=f"{prefix}{prompt_index}~s{sample_index}",
                        prompt_id=f"{prefix}{prompt_index}",
                        community=community,
                        instruction_id="i",
                        refused=False,
                        verdicts=verdicts,
                    )
                )
    return records


def analyze(records, n_permutations=400, seed=0, unit="prompt"):
    clusters = build_permutation_clusters(records, AXES, unit)
    sums, counts, is_target = build_cluster_matrices(clusters, AXES, "lgbtq")
    p, deltas = permutation_p_values(sums, counts, is_target, n_permutations, seed)
    _q, significant = benjamini_hochberg(p, 0.05)
    return p, deltas, significant


class TestPower:
    def test_large_effect_detected_at_moderate_n(self):
        records = synthetic_scored([0.8, 0.5, 0.5], [0.2, 0.5, 0.5], 20, 3, seed=1)
        _p, deltas, significant = analyze(records)
        assert significant[0] and deltas[0] > 0.4
        assert not significant[1] and not significant[2]

    def test_power_grows_with_sample_size(self):
        detections_small, detections_large = 0, 0
        for seed in range(12):
            small = synthetic_scored([0.65, 0.5, 0.5], [0.45, 0.5, 0.5], 8, 2, seed=seed)
            large = synthetic_scored([0.65, 0.5, 0.5], [0.45, 0.5, 0.5], 40, 4, seed=seed)
            detections_small += int(analyze(small, 300, seed)[2][0])
            detections_large += int(analyze(large, 300, seed)[2][0])
        assert detections_large >= detections_small
        assert detections_large >= 10  # Δ=0.2 with n=320/side is near-certain


class TestCalibration:
    def test_null_rarely_flags(self):
        flags = 0
        for seed in range(30):
            records = synthetic_scored([0.5, 0.3, 0.7], [0.5, 0.3, 0.7], 15, 3, seed=seed)
            flags += int(analyze(records, 300, seed)[2].any())
        assert flags / 30 <= 0.15

    def test_response_level_permutation_available(self):
        records = synthetic_scored([0.9, 0.5, 0.5], [0.1, 0.5, 0.5], 10, 3, seed=5)
        _p, _deltas, significant = analyze(records, unit="response")
        assert significant[0]


class TestC2st:
    def _features_labels(self, rate_target, rate_baseline, n, seed):
        rng = np.random.default_rng(seed)
        features = np.vstack([
            rng.random((n, 3)) < rate_target,
            rng.random((n, 3)) < rate_baseline,
        ]).astype(float)
        labels = np.array([True] * n + [False] * n)
        return features, labels

    def test_separable_data_beats_chance(self):
        features, labels = self._features_labels([0.9, 0.1, 0.5], [0.1, 0.9, 0.5], 150, 0)
        result = run_c2st(features, labels, 0.3, seed=0, n_dropped_incomplete=0)
        assert result.above_chance and result.accuracy > 0.75

    def test_identical_data_near_chance(self):
        features, labels = self._features_labels([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 150, 1)
        result = run_c2st(features, labels, 0.3, seed=1, n_dropped_incomplete=0)
        assert not result.above_chance

    def test_degenerate_data_returns_none(self):
        features = np.zeros((6, 3))
        labels = np.array([True] * 6)  # single class
        assert run_c2st(features, labels, 0.3, seed=0, n_dropped_incomplete=0) is None
