"""Classifier two-sample test (paper §4.5.3).

Trains a logistic-regression classifier to predict the community label from a
response's behavior characterization; held-out accuracy lower-bounds how
separable the model's behavior toward the two communities is in principle.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .analysis_report_schemas import ClassifierTwoSampleResult


def run_c2st(
    feature_matrix: np.ndarray,
    is_target: np.ndarray,
    test_fraction: float,
    seed: int,
    n_dropped_incomplete: int,
) -> ClassifierTwoSampleResult | None:
    """C2ST on complete behavior vectors; None when data is degenerate.

    above_chance is True when the Wilson 95% CI lower bound exceeds the
    majority-class baseline — a conservative "behavior is separable" call.
    """
    n_samples = len(is_target)
    n_classes = len(np.unique(is_target))
    if n_samples < 10 or n_classes < 2:
        return None

    try:
        features_train, features_test, labels_train, labels_test = train_test_split(
            feature_matrix,
            is_target,
            test_size=test_fraction,
            random_state=seed,
            stratify=is_target,
        )
    except ValueError:
        # stratified split impossible (tiny minority class) — C2ST undefined
        return None
    if len(np.unique(labels_train)) < 2 or len(labels_test) == 0:
        return None

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(features_train, labels_train)
    accuracy = float(classifier.score(features_test, labels_test))

    ci_low, ci_high = _wilson_interval(accuracy, len(labels_test))
    ci_low, ci_high = float(ci_low), float(ci_high)
    majority = float(max(np.mean(is_target), 1.0 - np.mean(is_target)))

    return ClassifierTwoSampleResult(
        accuracy=accuracy,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        majority_baseline=majority,
        n_train=len(labels_train),
        n_test=len(labels_test),
        n_dropped_incomplete=n_dropped_incomplete,
        above_chance=ci_low > majority,
    )


def _wilson_interval(rate: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = float(norm.ppf(0.5 + confidence / 2.0))
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(rate * (1.0 - rate) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
