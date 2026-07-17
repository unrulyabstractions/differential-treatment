"""Proof-grade validity suite for the linear classifier two-sample test.

Target: `src.distributional.c2st_linear.run_linear_c2st` (Lopez-Paz & Oquab
C2ST with author-grouped CV and an author-level permutation null).

These tests do NOT merely check that the function returns a float. They PROVE
the statistical contract of a valid two-sample test and would FAIL if the code
were subtly wrong:

  * NULL CALIBRATION  — under H0 (target & baseline drawn from the SAME
    author-generating distribution) the permutation p-values are ~Uniform(0,1)
    (KS), the false-positive rate is ~alpha (binomial CI), and held-out
    accuracy is ~chance. Anti-conservative p-values (a broken null) would fail.
  * POWER             — under a real mean shift the test rejects with high
    probability and accuracy >> chance.
  * NO AUTHOR LEAK    — with near-identical within-author rows split 50/50 over
    labels, author-grouped CV stays at chance; a leaky (row-level) split on the
    SAME data would score ~1.0. This proves the grouping is load-bearing.
  * DETERMINISM       — a fixed seed reproduces accuracy + the full null exactly.
  * +1 CORRECTION     — the permutation p-value floors at 1/(n_perm+1).

Author-level dependence is the design premise of the test ("authors contribute
several prompts each"), so the null draws each author as a bag of correlated
prompts from a common generative process — the correct H0 for an author-level
permutation test.

Run: `uv run pytest tests/test_linear_c2st_validity.py -q`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import binomtest, kstest
from sklearn.model_selection import KFold

from src.common.stats_utils import permutation_p_value
from src.distributional.c2st_linear import (
    _accuracies_from_scores,
    _pooled_heldout_scores,
    run_linear_c2st,
)

# --- shared null-calibration configuration -------------------------------
# Scale mirrors a real target-vs-baseline comparison in this repo's synthetic
# dataset (12 authors per set x 4 prompts); we use 10 authors x 5 prompts so
# the default cv_folds=5 splits authors evenly.
N_REP = 300
N_PERM = 99  # permutation p-value floor = 1/(N_PERM+1) = 0.01
CV_FOLDS = 5
AUTHORS_PER_SET = 10
PROMPTS_PER_AUTHOR = 5
DIM = 16
AUTHOR_SIG = 1.5  # within-author correlation (the test's design premise)
ROW_NOISE = 1.0


@dataclass
class C2stData:
    embeddings: NDArray[np.floating]
    labels: NDArray[np.integer]
    author_ids: list[str]


def make_authored_data(
    rng: np.random.Generator,
    *,
    set_shift: float = 0.0,
    author_sig: float = AUTHOR_SIG,
    row_noise: float = ROW_NOISE,
    authors_per_set: int = AUTHORS_PER_SET,
    prompts_per_author: int = PROMPTS_PER_AUTHOR,
    dim: int = DIM,
    shift_dims: int = 4,
) -> C2stData:
    """Two label sets of authored prompts.

    Each author gets a signature center (author_sig controls within-author
    correlation); each prompt is center + row_noise. `set_shift` moves the
    label-1 (target) set's mean by `set_shift` on the first `shift_dims`
    coordinates. set_shift=0 => identical distributions => true H0.
    """
    embeddings: list[NDArray[np.floating]] = []
    labels: list[int] = []
    author_ids: list[str] = []
    shift = np.zeros(dim)
    shift[:shift_dims] = set_shift
    for set_index, label in ((0, 0), (1, 1)):
        set_mean = shift if label == 1 else np.zeros(dim)
        for a in range(authors_per_set):
            center = set_mean + rng.normal(0.0, author_sig, dim)
            for _ in range(prompts_per_author):
                embeddings.append(center + rng.normal(0.0, row_noise, dim))
                labels.append(label)
                author_ids.append(f"s{set_index}_a{a}")
    return C2stData(
        np.asarray(embeddings, dtype=float),
        np.asarray(labels, dtype=int),
        author_ids,
    )


def make_near_identical_authors(
    rng: np.random.Generator,
    *,
    authors_per_set: int = 8,
    prompts_per_author: int = 5,
    dim: int = 12,
    within_author_noise: float = 1e-4,
) -> C2stData:
    """Authors whose rows are near-identical, labels 50/50 but INDEPENDENT of
    the embeddings (a null labelling). A row-level (leaky) split could memorise
    each author's location and its label; an author-grouped split cannot.
    """
    embeddings: list[NDArray[np.floating]] = []
    labels: list[int] = []
    author_ids: list[str] = []
    for set_index, label in ((0, 0), (1, 1)):
        for a in range(authors_per_set):
            center = rng.normal(0.0, 3.0, dim)  # spread-out, label-independent
            for _ in range(prompts_per_author):
                embeddings.append(center + rng.normal(0.0, within_author_noise, dim))
                labels.append(label)
                author_ids.append(f"s{set_index}_a{a}")
    return C2stData(
        np.asarray(embeddings, dtype=float),
        np.asarray(labels, dtype=int),
        author_ids,
    )


# --- (1) NULL CALIBRATION -------------------------------------------------
# These encode the validity contract: under a true H0 the author-level
# permutation p-values are ~Uniform and FPR@alpha ~= alpha. They caught a real
# anti-conservative bug (the fold split was built from the set-aligned author
# order, giving the observed labelling privileged fold balance); fixed by
# assigning folds from a FIXED label-agnostic shuffle in _author_grouped_folds.


@pytest.fixture(scope="module")
def null_run() -> dict[str, NDArray[np.floating]]:
    """Run the C2ST N_REP times under H0 (identical distributions)."""
    p_values = np.empty(N_REP)
    accuracies = np.empty(N_REP)
    for i in range(N_REP):
        rng = np.random.default_rng(10_000 + i)
        data = make_authored_data(rng, set_shift=0.0)
        outcome = run_linear_c2st(
            data.embeddings, data.labels, data.author_ids, CV_FOLDS, N_PERM, rng
        )
        p_values[i] = outcome.p_value
        accuracies[i] = outcome.accuracy
    return {"p_values": p_values, "accuracies": accuracies}


def test_null_heldout_accuracy_is_chance(null_run):
    """Under H0 the held-out C2ST accuracy centers on chance (0.5)."""
    mean_acc = float(null_run["accuracies"].mean())
    assert 0.45 <= mean_acc <= 0.55, f"mean null accuracy {mean_acc:.3f} not ~0.5"


def test_null_pvalues_are_uniform(null_run):
    """A valid test's null p-values are ~Uniform(0,1) (KS, lenient alpha=0.01)."""
    ks_p = float(kstest(null_run["p_values"], "uniform").pvalue)
    assert ks_p > 0.01, (
        f"null p-values reject uniformity (KS p={ks_p:.4f}); "
        f"median p={np.median(null_run['p_values']):.3f} "
        "(a miscalibrated permutation null)"
    )


def test_null_false_positive_rate_is_alpha(null_run):
    """FPR at alpha=0.05 sits inside a 99% binomial CI around 0.05."""
    n_reject = int((null_run["p_values"] <= 0.05).sum())
    ci = binomtest(n_reject, N_REP, 0.05).proportion_ci(0.99)
    fpr = n_reject / N_REP
    assert ci.low <= 0.05 <= ci.high, (
        f"false-positive rate {fpr:.3f} ({n_reject}/{N_REP}) excludes 0.05; "
        f"99% CI [{ci.low:.3f}, {ci.high:.3f}]"
    )


def test_null_pvalues_respect_floor(null_run):
    """No null p-value falls below the +1-correction floor 1/(N_PERM+1)."""
    floor = 1.0 / (N_PERM + 1)
    assert null_run["p_values"].min() >= floor - 1e-12


# --- (2) POWER ------------------------------------------------------------


def test_power_under_mean_shift():
    """Under a clear mean shift the test rejects ~always with accuracy >> chance."""
    n_trials = 15
    rejections = 0
    accuracies = []
    floor_hits = 0
    floor = 1.0 / (N_PERM + 1)
    for t in range(n_trials):
        rng = np.random.default_rng(70_000 + t)
        data = make_authored_data(rng, set_shift=2.5)
        outcome = run_linear_c2st(
            data.embeddings, data.labels, data.author_ids, CV_FOLDS, N_PERM, rng
        )
        rejections += int(outcome.p_value <= 0.05)
        accuracies.append(outcome.accuracy)
        floor_hits += int(abs(outcome.p_value - floor) < 1e-12)
    mean_acc = float(np.mean(accuracies))
    assert rejections >= 14, (
        f"only {rejections}/{n_trials} rejections under a real shift"
    )
    assert mean_acc > 0.7, (
        f"mean accuracy {mean_acc:.3f} not >> chance under a real shift"
    )
    assert floor_hits >= 1, "a strong shift never drove p to its 1/(n_perm+1) floor"


# --- (3) NO AUTHOR LEAK ---------------------------------------------------


def test_no_author_leak_grouped_cv_stays_at_chance():
    """Near-identical within-author rows, labels independent of embeddings:
    author-grouped CV must stay at chance and NOT declare significance."""
    accuracies = []
    p_values = []
    for t in range(8):
        rng = np.random.default_rng(90_000 + t)
        data = make_near_identical_authors(rng)
        outcome = run_linear_c2st(
            data.embeddings, data.labels, data.author_ids, CV_FOLDS, N_PERM, rng
        )
        accuracies.append(outcome.accuracy)
        p_values.append(outcome.p_value)
    mean_acc = float(np.mean(accuracies))
    assert mean_acc < 0.65, (
        f"grouped-CV accuracy {mean_acc:.3f} is inflated on a null labelling — "
        "same-author rows appear to leak across folds"
    )
    # A leaky implementation would fire far more than alpha of the time.
    assert np.mean([p <= 0.05 for p in p_values]) <= 0.25


def test_leak_would_inflate_without_author_grouping():
    """Fault injection: feed the SAME near-identical data to the internal scorer
    with a ROW-LEVEL (leaky) split. Because each author's label is constant and
    its rows near-identical, memorisation drives accuracy to ~1.0 — proving the
    construction really can leak and that GroupKFold is what prevents it."""
    rng = np.random.default_rng(90_000)
    data = make_near_identical_authors(rng)
    leaky_folds = list(
        KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0).split(data.embeddings)
    )
    scores = _pooled_heldout_scores(data.embeddings, data.labels, leaky_folds)
    leaky_acc, _ = _accuracies_from_scores(scores, data.labels, leaky_folds)
    assert leaky_acc > 0.9, (
        f"row-level leak accuracy {leaky_acc:.3f} was expected ~1.0; the "
        "near-identical construction is not actually leakable, so the grouped "
        "test above proves nothing"
    )


# --- (4) DETERMINISM ------------------------------------------------------


def test_same_seed_reproduces_accuracy_and_null():
    """Identical inputs + identical seed => identical accuracy and full null."""
    rng_data = np.random.default_rng(123)
    data = make_authored_data(rng_data, set_shift=1.0)
    first = run_linear_c2st(
        data.embeddings,
        data.labels,
        data.author_ids,
        CV_FOLDS,
        N_PERM,
        np.random.default_rng(555),
    )
    second = run_linear_c2st(
        data.embeddings,
        data.labels,
        data.author_ids,
        CV_FOLDS,
        N_PERM,
        np.random.default_rng(555),
    )
    assert first.accuracy == second.accuracy
    assert first.fold_accuracies == second.fold_accuracies
    assert first.p_value == second.p_value
    assert np.array_equal(first.null_accuracies, second.null_accuracies)
    assert first.scores == second.scores


def test_different_seed_changes_null_not_observed():
    """The rng drives ONLY the permutation null: a different seed changes the
    null draws but leaves the (rng-independent) observed accuracy identical."""
    rng_data = np.random.default_rng(321)
    data = make_authored_data(rng_data, set_shift=1.0)
    a = run_linear_c2st(
        data.embeddings,
        data.labels,
        data.author_ids,
        CV_FOLDS,
        N_PERM,
        np.random.default_rng(1),
    )
    b = run_linear_c2st(
        data.embeddings,
        data.labels,
        data.author_ids,
        CV_FOLDS,
        N_PERM,
        np.random.default_rng(2),
    )
    assert a.accuracy == b.accuracy  # observed statistic is seed-independent
    assert not np.array_equal(a.null_accuracies, b.null_accuracies)


# --- (5) +1 CORRECTION / min p -------------------------------------------


def test_permutation_pvalue_plus_one_correction():
    """permutation_p_value uses the standard +1 correction with `>=` ties."""
    nulls = np.array([0.40, 0.50, 0.60])
    # observed strictly above every null => floor p = 1/(n+1)
    assert permutation_p_value(0.70, nulls) == pytest.approx(1 / (len(nulls) + 1))
    # one null ties/exceeds => (1 + 1) / (1 + 3)
    assert permutation_p_value(0.60, nulls) == pytest.approx(2 / 4)
    # all nulls >= observed => p = 1 (max)
    assert permutation_p_value(0.10, nulls) == pytest.approx(4 / 4)


def test_c2st_pvalue_floor_is_one_over_nperm_plus_one():
    """End-to-end: an UNAMBIGUOUSLY separable pair drives run_linear_c2st's
    p-value to exactly 1/(n_perm+1) — never 0 — proving the +1 correction is
    wired in. The shift dwarfs the within-author signature so the observed
    accuracy is ~1 and no label permutation (which scrambles set membership) can
    match it; a weaker shift can be matched by chance under author-correlated
    data now that the null is correctly (no longer anti-conservatively) calibrated.
    """
    rng = np.random.default_rng(2024)
    data = make_authored_data(rng, set_shift=8.0)
    outcome = run_linear_c2st(
        data.embeddings, data.labels, data.author_ids, CV_FOLDS, N_PERM, rng
    )
    assert outcome.accuracy > 0.95  # unambiguously separable
    assert outcome.p_value == pytest.approx(1.0 / (N_PERM + 1))
