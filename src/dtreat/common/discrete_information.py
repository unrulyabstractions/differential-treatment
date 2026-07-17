"""Discrete information measures used by the treatment analysis stage.

Implements exactly the quantities in the paper (equations noted per function).
All entropies/divergences are in bits (log base 2), matching the paper's
worked example (D_pi = 2.37 bits).
"""

from __future__ import annotations

import numpy as np


def binary_entropy(p: float) -> float:
    """h(p) = -p log2 p - (1-p) log2 (1-p), with h(0) = h(1) = 0.  (Eq 14)"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def normalize_profile(rates: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """Treatment profile pi^C = norm(z-hat^C).  (Eq 11)

    Args:
        rates: per-axis expected behavior rates z-hat (values in [0, 1])
        epsilon: added to every axis before normalizing, to avoid zero
            support (the paper uses epsilon = 0.01 before computing D_pi)
    """
    shifted = np.asarray(rates, dtype=float) + epsilon
    total = shifted.sum()
    if total <= 0:
        raise ValueError("Cannot normalize an all-zero profile; use epsilon > 0")
    return shifted / total


def kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Relative entropy H(p || q) in bits between two discrete distributions.

    Used for D_pi = H(pi^target || pi^baseline).  (Eq 12)
    Terms with p_i = 0 contribute 0; q_i = 0 with p_i > 0 yields inf.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def community_axis_information(
    rate_target: float,
    rate_baseline: float,
    weight_target: float = 0.5,
) -> float:
    """Mutual information I_j = I(C; Z_j) between community and axis verdict.

    I_j = h(w_t * z_t + w_b * z_b) - [w_t * h(z_t) + w_b * h(z_b)]

    With equal weights this is exactly Eq 13. Weights should be the
    empirical community proportions when the prompt sets are unbalanced.
    """
    if not 0.0 <= weight_target <= 1.0:
        raise ValueError(f"weight_target must be in [0,1], got {weight_target}")
    weight_baseline = 1.0 - weight_target
    mixed = weight_target * rate_target + weight_baseline * rate_baseline
    conditional = weight_target * binary_entropy(rate_target) + weight_baseline * binary_entropy(
        rate_baseline
    )
    # Clamp tiny negative values from float error: MI is nonnegative
    return max(0.0, binary_entropy(mixed) - conditional)
