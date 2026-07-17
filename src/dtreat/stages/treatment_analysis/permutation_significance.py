"""Permutation testing + Benjamini–Hochberg FDR control (paper §4.5.1).

For each axis j we ask whether an effect as large as |Delta_j| could arise by
chance: permute the community labels (removing any real signal) and recompute
the effect under each permutation to obtain a null distribution.

Labels are permuted at the *prompt* level by default: all responses to one
prompt move together, respecting that responses within a prompt are correlated
(a response-level permutation would treat them as independent and overstate
significance).
"""

from __future__ import annotations

import numpy as np


def permutation_p_values(
    per_cluster_sums: np.ndarray,
    per_cluster_counts: np.ndarray,
    is_target_cluster: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided permutation p-values for the per-axis rate gaps.

    Args:
        per_cluster_sums: [n_axes, n_clusters] verdict-sums per cluster
        per_cluster_counts: [n_axes, n_clusters] valid-verdict counts per cluster
        is_target_cluster: [n_clusters] observed community labels
        n_permutations: label permutations to draw
        seed: RNG seed (permutations are shared across axes, preserving
            cross-axis dependence)

    Returns:
        (p_values [n_axes], observed_deltas [n_axes]); axes with no valid
        verdicts on either side get p = 1.0 and delta = nan.
    """
    n_axes, n_clusters = per_cluster_sums.shape
    if is_target_cluster.shape != (n_clusters,):
        raise ValueError("is_target_cluster length must match cluster axis")

    observed = _deltas_for_labels(per_cluster_sums, per_cluster_counts, is_target_cluster)

    rng = np.random.default_rng(seed)
    exceed_counts = np.zeros(n_axes, dtype=int)
    valid = ~np.isnan(observed)
    for _ in range(n_permutations):
        permuted = rng.permutation(is_target_cluster)
        null_deltas = _deltas_for_labels(per_cluster_sums, per_cluster_counts, permuted)
        with np.errstate(invalid="ignore"):
            exceed = np.abs(null_deltas) >= np.abs(observed) - 1e-12
        # A permutation that yields an undefined delta (all verdicts on one
        # side) counts as exceeding: it cannot contradict the null.
        exceed = np.where(np.isnan(null_deltas), True, exceed)
        exceed_counts += exceed.astype(int)

    # add-one correction keeps p in (0, 1] and unbiased under the null
    p_values = (1.0 + exceed_counts) / (n_permutations + 1.0)
    p_values = np.where(valid, p_values, 1.0)
    return p_values, observed


def _deltas_for_labels(
    sums: np.ndarray, counts: np.ndarray, is_target: np.ndarray
) -> np.ndarray:
    """Rate gap target-minus-baseline per axis for one labeling (nan if a side
    has no valid verdicts)."""
    target_counts = counts[:, is_target].sum(axis=1)
    baseline_counts = counts[:, ~is_target].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_rates = sums[:, is_target].sum(axis=1) / target_counts
        baseline_rates = sums[:, ~is_target].sum(axis=1) / baseline_counts
    deltas = target_rates - baseline_rates
    deltas[(target_counts == 0) | (baseline_counts == 0)] = np.nan
    return deltas


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up procedure.

    Returns (q_values, significant_mask). q_i = min_{j >= rank(i)} m * p_(j) / j,
    capped at 1; significant where q <= alpha.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p_values)
    ranked = p_values[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_values = np.empty(m)
    q_values[order] = q_sorted
    return q_values, q_values <= alpha
