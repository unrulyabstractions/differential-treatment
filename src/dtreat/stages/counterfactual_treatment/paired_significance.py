"""Paired sign-flip permutation test for counterfactual voice effects.

Each pair contributes d_i = rate(axis | target voice) − rate(axis | baseline
voice) for the SAME request content. Under the null (voice has no effect),
the sign of each d_i is exchangeable; flipping signs at random builds the
null distribution of the mean paired difference.
"""

from __future__ import annotations

import numpy as np


def sign_flip_p_values(
    paired_diffs: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided sign-flip p-values per axis.

    Args:
        paired_diffs: [n_axes, n_pairs] per-pair rate differences
            (NaN where a pair has no valid verdicts for that axis)
        n_permutations: sign patterns to draw (shared across axes)
        seed: RNG seed

    Returns:
        (p_values [n_axes], observed mean diffs [n_axes]); axes with no
        valid pairs get p = 1.0 and mean = nan.
    """
    n_axes, n_pairs = paired_diffs.shape
    valid_mask = ~np.isnan(paired_diffs)
    observed = np.full(n_axes, np.nan)
    for j in range(n_axes):
        if valid_mask[j].any():
            observed[j] = float(np.nanmean(paired_diffs[j]))

    rng = np.random.default_rng(seed)
    exceed_counts = np.zeros(n_axes, dtype=int)
    diffs_filled = np.where(valid_mask, paired_diffs, 0.0)
    valid_counts = valid_mask.sum(axis=1)
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=n_pairs)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_means = (diffs_filled * signs).sum(axis=1) / valid_counts
        exceed = np.abs(null_means) >= np.abs(observed) - 1e-12
        exceed_counts += np.where(np.isnan(observed), 0, exceed.astype(int))

    p_values = np.where(
        np.isnan(observed),
        1.0,
        (1.0 + exceed_counts) / (n_permutations + 1.0),
    )
    return p_values, observed
