"""Shared statistics: FDR correction, divergences, permutation testing.

Every dimension reuses these instead of rolling its own; scipy provides the
core procedures (Benjamini-Hochberg, Jensen-Shannon) so we only wrap them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import jensenshannon
from scipy.stats import false_discovery_control, norm


def two_sided_p_from_z(z_scores: NDArray[np.floating]) -> NDArray[np.floating]:
    """Two-sided p-values for an array of z-scores."""
    return 2.0 * norm.sf(np.abs(z_scores))


def benjamini_hochberg(
    p_values: NDArray[np.floating], alpha: float
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Benjamini-Hochberg FDR control.

    Returns:
        (adjusted_p_values, rejected) where rejected marks discoveries at `alpha`.
    """
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values, np.zeros(0, dtype=bool)
    # A NaN p-value (degenerate word) must not abort false_discovery_control;
    # treat it as the least significant possible.
    p_values = np.nan_to_num(p_values, nan=1.0)
    adjusted = false_discovery_control(p_values, method="bh")
    return adjusted, adjusted <= alpha


def jensen_shannon_divergence(
    p: NDArray[np.floating], q: NDArray[np.floating]
) -> float:
    """Jensen-Shannon divergence in bits (base 2), in [0, 1].

    scipy's jensenshannon returns the JS *distance* (the square root).
    """
    return float(jensenshannon(p, q, base=2) ** 2)


def permutation_p_value(observed: float, null_values: NDArray[np.floating]) -> float:
    """One-sided permutation p-value with the standard +1 correction."""
    null_values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null_values >= observed)) / (1 + null_values.size))


def permute_labels_by_author(
    author_ids: list[str],
    labels: NDArray[np.integer],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    """Permute binary set labels at the author level.

    All prompts by one author keep a common label; which authors carry which
    label is shuffled. This respects the dependence structure the paper's
    author-level cross-validation guards against.
    """
    labels = np.asarray(labels)
    unique_authors = sorted(set(author_ids))
    author_label: dict[str, int] = {}
    for author, label in zip(author_ids, labels, strict=True):
        previous = author_label.setdefault(author, int(label))
        if previous != int(label):
            raise ValueError(
                f"Author '{author}' carries both labels; qualify author ids per "
                "set (see qualified_author_ids) before permuting"
            )
    shuffled_values = rng.permutation([author_label[a] for a in unique_authors])
    permuted_map = dict(zip(unique_authors, shuffled_values, strict=True))
    return np.array([permuted_map[a] for a in author_ids], dtype=labels.dtype)


def interleave_texts(
    texts_a: list[str], texts_b: list[str]
) -> tuple[list[str], Callable[[list], tuple[list, list]]]:
    """Alternate two lists into one, plus a splitter to undo it.

    Used when a nondeterministic annotator processes texts in batches: pooling
    the two sides into alternating order spreads batch-level noise evenly, so
    it cannot masquerade as a between-set difference.
    """
    pooled: list[str] = []
    origin: list[int] = []
    iters = [iter(texts_a), iter(texts_b)]
    lengths = [len(texts_a), len(texts_b)]
    taken = [0, 0]
    side = 0
    while taken[0] < lengths[0] or taken[1] < lengths[1]:
        if taken[side] < lengths[side]:
            pooled.append(next(iters[side]))
            origin.append(side)
            taken[side] += 1
        side = 1 - side

    def unpool(values: list) -> tuple[list, list]:
        split: tuple[list, list] = ([], [])
        for value, where in zip(values, origin, strict=True):
            split[where].append(value)
        return split

    return pooled, unpool
