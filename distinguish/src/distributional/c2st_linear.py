"""Linear classifier two-sample test (C2ST) on embedding vectors.

Lopez-Paz & Oquab (2017): train a classifier to tell the two samples apart;
its held-out accuracy lower-bounds the Bayes-optimal separability (0.5 means
indistinguishable). Authors contribute several prompts each, so folds group whole authors (never
splitting a person across train/test) and the permutation null reassigns set
labels at the author level — otherwise within-author correlation inflates
significance. Fold assignment is a fixed label-agnostic shuffle (not the input
order) so the observed labelling gets no privileged fold balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.common.base_schema import BaseSchema
from src.common.stats_utils import permutation_p_value, permute_labels_by_author

# Fixed seed for the fold shuffle: decorrelating fold assignment from the
# set-aligned author order (the calibration fix) must NOT depend on the
# permutation rng, or the observed statistic would wobble seed to seed. A
# constant seed keeps the observed accuracy a deterministic property of the data.
_FOLD_SHUFFLE_SEED = 0


def _author_grouped_folds(
    author_ids: list[str], n_splits: int
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Whole-author CV folds, assigned from a fixed label-agnostic shuffled order.

    Authors group into folds (a person's prompts never straddle train/test), but
    the assignment is a shuffled round-robin under a FIXED seed. Using the input
    row order instead — which arrives set-aligned (all target authors, then all
    baseline) — hands the observed labelling systematically label-balanced folds
    while permuted labellings get imbalanced ones, biasing the permutation null
    low (anti-conservative p-values). A fixed shuffle removes that confound and
    keeps the observed statistic seed-independent; the same folds serve the
    observed statistic and every permutation.
    """
    unique_authors = list(dict.fromkeys(author_ids))
    order = np.random.default_rng(_FOLD_SHUFFLE_SEED).permutation(len(unique_authors))
    fold_of_author = {
        unique_authors[order[k]]: k % n_splits for k in range(len(unique_authors))
    }
    author_fold = np.array([fold_of_author[a] for a in author_ids])
    return [
        (np.where(author_fold != f)[0], np.where(author_fold == f)[0])
        for f in range(n_splits)
    ]


@dataclass
class LinearC2stOutcome(BaseSchema):
    """Pooled held-out C2ST accuracy plus its author-level permutation null."""

    accuracy: float
    fold_accuracies: list[float]
    p_value: float
    n_permutations: int
    # The full null distribution is kept so the histogram plot (and anyone
    # auditing the p-value) never needs to re-run the permutations.
    null_accuracies: list[float] = field(default_factory=list)
    # Pooled held-out decision scores (predict_proba of class 1) with their
    # true labels, prompt-aligned, kept for the ROC curve plot.
    scores: list[float] = field(default_factory=list)
    true_labels: list[int] = field(default_factory=list)


def _pooled_heldout_scores(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    folds: list[tuple[NDArray[np.integer], NDArray[np.integer]]],
) -> NDArray[np.floating]:
    """Held-out P(class 1) for every prompt, pooled over the folds."""
    scores = np.zeros(len(labels), dtype=float)
    for train_index, test_index in folds:
        train_labels = labels[train_index]
        if np.unique(train_labels).size < 2:
            # A degenerate permutation can hand every training author the same
            # set label; the only consistent score is certainty in that class.
            scores[test_index] = float(train_labels[0])
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(embeddings[train_index], train_labels)
            scores[test_index] = model.predict_proba(embeddings[test_index])[:, 1]
    return scores


def _accuracies_from_scores(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    folds: list[tuple[NDArray[np.integer], NDArray[np.integer]]],
) -> tuple[float, list[float]]:
    """Pooled and per-fold accuracy at sklearn's 0.5 decision threshold."""
    correct = (scores > 0.5).astype(labels.dtype) == labels
    fold_accuracies = [float(correct[test].mean()) for _, test in folds]
    return float(correct.mean()), fold_accuracies


def run_linear_c2st(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    author_ids: list[str],
    cv_folds: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> LinearC2stOutcome:
    """Author-grouped CV accuracy with an author-level permutation p-value."""
    labels = np.asarray(labels)
    n_splits = min(cv_folds, len(set(author_ids)))
    # One fixed fold structure, assigned from a shuffled (label-agnostic) author
    # order so the observed labelling gets no privileged fold balance vs the
    # permutations. Authors never move between folds; observed and every permuted
    # statistic share these splits and differ only in how labels attach to authors.
    folds = _author_grouped_folds(author_ids, n_splits)

    scores = _pooled_heldout_scores(embeddings, labels, folds)
    accuracy, fold_accuracies = _accuracies_from_scores(scores, labels, folds)

    null_accuracies: list[float] = []
    for _ in tqdm(range(n_permutations), desc="c2st permutations", leave=False):
        permuted = permute_labels_by_author(author_ids, labels, rng)
        null_scores = _pooled_heldout_scores(embeddings, permuted, folds)
        null_accuracy, _ = _accuracies_from_scores(null_scores, permuted, folds)
        null_accuracies.append(null_accuracy)

    return LinearC2stOutcome(
        accuracy=accuracy,
        fold_accuracies=fold_accuracies,
        p_value=permutation_p_value(accuracy, np.asarray(null_accuracies)),
        n_permutations=n_permutations,
        null_accuracies=null_accuracies,
        scores=[float(s) for s in scores],
        true_labels=[int(label) for label in labels],
    )
