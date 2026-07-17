"""Calibrated marked-words statistics (Monroe et al. 2008, "Fightin' Words").

Log-odds with an informative Dirichlet prior over per-word counts. Two prior
modes:

- ``"mickel"`` (default) reproduces the calibration of Mickel et al. 2025 ("More
  of the Same", Algorithm 3): a count-space hybrid prior
  ``P = (1 - w_ref)*pooled_counts + w_ref*reference_counts`` and *per-side*
  regularizers ``r_i = C * w_p / w_gi`` estimated over a calibration word set
  (top-20 prior/target/baseline words plus a fixed stopword list). Each side's
  effective Dirichlet pseudo-counts are ``P / r_i`` — data-driven, corpus-size
  adaptive shrinkage that collapses common/register words to z -> 0 while
  content words keep near-raw counts.
- ``"fixed"`` keeps a single symmetric prior of fixed strength (the legacy
  hybrid ``prior_strength * pi`` mode), retained for comparison runs.

Significance is either Benjamini-Hochberg FDR ("bh_fdr", default) or the raw
two-sided ``|z| >= 1.96`` rule of the MotS paper ("raw_z").

The reference side of the prior is configurable: "wordfreq:<lang>" built-in
frequency lists, or a JSON file mapping word -> relative frequency (a
domain-specific reference corpus).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from wordfreq import word_frequency

from src.common.base_schema import BaseSchema
from src.common.file_io import load_json
from src.common.stats_utils import benjamini_hochberg, two_sided_p_from_z
from src.common.text_tokenization import count_words

# Words the reference corpus has never seen still need prior mass: a hard zero
# would strip smoothing exactly where it matters most (rare or coined words).
_ZERO_REFERENCE_FREQUENCY_FLOOR = 1e-9
_WORDFREQ_PREFIX = "wordfreq:"

# MotS Algorithm 3 constants (calibrated_marked_words.py::get_log_odds):
# the 27 hardcoded common words that always join the calibration set, and the
# NLTK Brown corpus size used to lift wordfreq relative frequencies into a
# comparable count scale (their English prior is raw Brown counts, ~1.16M tokens).
CALIBRATION_STOPWORDS = frozenset(
    {
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "for",
        "is",
        "on",
        "that",
        "by",
        "this",
        "with",
        "i",
        "you",
        "it",
        "not",
        "or",
        "be",
        "are",
        "from",
        "at",
        "as",
        "your",
        "am",
        "an",
        "my",
    }
)
_BROWN_PSEUDO_CORPUS_TOKENS = 1_161_192
_RAW_Z_THRESHOLD = 1.96
_CALIBRATION_TOP_K = 20
# Auto-calibration (MotS find_optimal_alpha): search this C range for the
# largest value at which no register word clears raw |z|>=1.96.
_AUTO_C_RANGE = (0.005, 1.0)
_AUTO_C_ITERS = 24  # binary-search steps -> ~1e-7 resolution over the range
_AUTO_REGISTER_TOP_K = 40  # top English-frequency vocab words treated as register


@lru_cache(maxsize=8)
def _load_reference_corpus_file(path: str) -> dict[str, float]:
    """word -> relative frequency from a JSON reference corpus, loaded once."""
    table = load_json(Path(path))
    if not isinstance(table, dict):
        raise ValueError(
            f"Reference corpus must be a JSON object of word -> frequency: {path}"
        )
    return {str(word).lower(): float(freq) for word, freq in table.items()}


def reference_frequencies(
    vocabulary: list[str], reference_corpus: str
) -> NDArray[np.float64]:
    """Per-word relative frequency under the configured reference corpus.

    `reference_corpus` is "wordfreq:<lang>" (built-in frequency lists) or a
    filesystem path to a JSON file mapping word -> relative frequency. Words
    the corpus has never seen get a small floor instead of a hard zero.
    """
    if reference_corpus.startswith(_WORDFREQ_PREFIX):
        lang = reference_corpus[len(_WORDFREQ_PREFIX) :]
        frequencies = np.array([word_frequency(w, lang) for w in vocabulary])
    else:
        table = _load_reference_corpus_file(reference_corpus)
        frequencies = np.array([table.get(w, 0.0) for w in vocabulary])
    return np.maximum(frequencies, _ZERO_REFERENCE_FREQUENCY_FLOOR)


def reference_counts(
    vocabulary: list[str],
    reference_corpus: str,
    pseudo_corpus_tokens: int = _BROWN_PSEUDO_CORPUS_TOKENS,
) -> NDArray[np.float64]:
    """Reference frequencies scaled into a Brown-sized pseudo-count corpus.

    MotS mixes raw Brown counts with the topic counts in *count* space; wordfreq
    is a relative-frequency table, so multiply by the Brown token total to put
    the two sides on the same scale.
    """
    return reference_frequencies(vocabulary, reference_corpus) * pseudo_corpus_tokens


def hybrid_prior_counts(
    vocabulary: list[str],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    reference_corpus: str,
    reference_prior_weight: float,
) -> NDArray[np.float64]:
    """Count-space hybrid prior P = (1-w_ref)*pooled + w_ref*reference."""
    pooled = y_target + y_baseline
    reference = reference_counts(vocabulary, reference_corpus)
    return (1.0 - reference_prior_weight) * pooled + reference_prior_weight * reference


def calibration_word_indices(
    vocabulary: list[str],
    prior_counts: NDArray[np.float64],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    top_k: int = _CALIBRATION_TOP_K,
) -> NDArray[np.bool_]:
    """Boolean mask of the MotS calibration word set W over the vocabulary.

    W = (top-k prior words) ∩ (top-k target words) ∩ (top-k baseline words),
    unioned with the fixed stopword list — the words whose prior mass anchors
    the per-side regularizers.
    """
    vocab = np.asarray(vocabulary)

    def _top(values: NDArray[np.float64]) -> set[str]:
        # Only rank words the side actually uses (count > 0). Over the shared
        # vocabulary, absent words sit at count 0; when a side has fewer than
        # top_k words they would otherwise pad the top-k with words it never
        # used, polluting the register anchor set (MotS ranks each side's own
        # keys). On real corpora (>top_k words/side) this changes nothing.
        nonzero = np.flatnonzero(values > 0)
        ordered = nonzero[np.argsort(-values[nonzero])[:top_k]]
        return set(vocab[ordered])

    calibration = (
        _top(prior_counts) & _top(y_target) & _top(y_baseline)
    ) | CALIBRATION_STOPWORDS
    return np.isin(vocab, sorted(calibration))


def calibrated_side_alphas(
    vocabulary: list[str],
    prior_counts: NDArray[np.float64],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    calibration_constant: float,
    top_k: int = _CALIBRATION_TOP_K,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Mickel et al. Algorithm 3: per-side Dirichlet pseudo-counts P / r_i.

    r_i = C * w_p / w_gi with w_p / w_gi summed over the calibration word set,
    so side i's prior mass on calibration words is exactly w_gi / C: several
    times its own observed count, and automatically rescaled by corpus size.
    """
    in_w = calibration_word_indices(
        vocabulary, prior_counts, y_target, y_baseline, top_k
    )
    # If the calibration set is empty (a corpus with no stopwords and disjoint
    # top-k lists), anchor the regularizer on the whole vocabulary so w_p > 0 and
    # the alphas stay finite — a strongly-shrunk result instead of a NaN crash.
    anchor = in_w if in_w.any() else np.ones(len(vocabulary), dtype=bool)
    w_p = max(float(prior_counts[anchor].sum()), 1e-9)
    w_gt = max(float(y_target[anchor].sum()), 1.0)
    w_gb = max(float(y_baseline[anchor].sum()), 1.0)
    r_t = calibration_constant * w_p / w_gt
    r_b = calibration_constant * w_p / w_gb
    calibration_words = sorted(np.asarray(vocabulary)[in_w].tolist())
    return prior_counts / r_t, prior_counts / r_b, calibration_words


def _register_word_mask(
    vocabulary: list[str], reference_corpus: str
) -> NDArray[np.bool_]:
    """Words treated as register for auto-calibration: the fixed stopwords plus
    the highest general-English-frequency vocab words (catches contractions and
    function words the hardcoded list misses, e.g. "i'm", "you're")."""
    vocab = np.asarray(vocabulary)
    stopword = np.isin(vocab, sorted(CALIBRATION_STOPWORDS))
    # Contractions ("i'm", "i've", "don't") are pure function/pronoun forms on
    # any corpus — the frequency-list top-k can miss them, so flag them directly.
    contraction = np.array(["'" in w for w in vocabulary])
    freqs = reference_frequencies(vocabulary, reference_corpus)
    top_english = np.zeros(len(vocabulary), dtype=bool)
    top_english[np.argsort(-freqs)[:_AUTO_REGISTER_TOP_K]] = True
    return stopword | contraction | top_english


def auto_calibration_constant(
    vocabulary: list[str],
    prior_counts: NDArray[np.float64],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    register_mask: NDArray[np.bool_],
) -> float:
    """Largest C at which no register word clears raw |z|>=1.96 (MotS-style).

    z compresses toward 0 as C shrinks (stronger prior), so register
    significance is monotone in C — a clean binary search. Returns the low end
    of the range if even that leaves a register word significant (degenerate,
    tiny corpora), so the caller still gets the most conservative prior.
    """
    lo, hi = _AUTO_C_RANGE

    def register_clean(constant: float) -> bool:
        alpha_t, alpha_b, _ = calibrated_side_alphas(
            vocabulary, prior_counts, y_target, y_baseline, constant
        )
        _, z = _log_odds_z(y_target, y_baseline, alpha_t, alpha_b)
        if not register_mask.any():
            return True
        return bool(np.abs(z[register_mask]).max() < _RAW_Z_THRESHOLD)

    if not register_clean(lo):
        return lo
    if register_clean(hi):
        return hi
    for _ in range(_AUTO_C_ITERS):
        mid = 0.5 * (lo + hi)
        if register_clean(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _fixed_alpha(
    vocabulary: list[str],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    reference_corpus: str,
    english_prior_weight: float,
    prior_strength: float,
) -> NDArray[np.float64]:
    """Legacy symmetric prior: fixed-strength probability-space hybrid."""
    p_reference = reference_frequencies(vocabulary, reference_corpus)
    p_reference = p_reference / p_reference.sum()
    p_corpus = (y_target + y_baseline) / (y_target + y_baseline).sum()
    pi = english_prior_weight * p_reference + (1.0 - english_prior_weight) * p_corpus
    return prior_strength * pi


def _log_odds_z(
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    alpha_target: NDArray[np.float64],
    alpha_baseline: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Monroe et al. prior-adjusted log-odds delta and its z-score.

    `alpha_*` are per-side Dirichlet pseudo-counts (equal arrays in fixed mode,
    per-side arrays in mickel mode). n_t/n_b are in-vocabulary token totals so
    the underlying multinomial stays consistent with the counts used.
    """
    n_t, n_b = y_target.sum(), y_baseline.sum()
    a0_t, a0_b = alpha_target.sum(), alpha_baseline.sum()
    delta = np.log(
        (y_target + alpha_target) / (n_t + a0_t - y_target - alpha_target)
    ) - np.log(
        (y_baseline + alpha_baseline) / (n_b + a0_b - y_baseline - alpha_baseline)
    )
    variance = 1.0 / (y_target + alpha_target) + 1.0 / (y_baseline + alpha_baseline)
    return delta, delta / np.sqrt(variance)


def _build_vocabulary(
    texts_target: list[str], texts_baseline: list[str], min_word_count: int
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64], int, int]:
    """Shared vocabulary (combined count >= min) with per-side count vectors."""
    counts_target = count_words(texts_target)
    counts_baseline = count_words(texts_baseline)
    total_target = int(sum(counts_target.values()))
    total_baseline = int(sum(counts_baseline.values()))
    combined = counts_target + counts_baseline
    vocabulary = sorted(w for w, c in combined.items() if c >= min_word_count)
    y_t = np.array([counts_target[w] for w in vocabulary], dtype=float)
    y_b = np.array([counts_baseline[w] for w in vocabulary], dtype=float)
    return vocabulary, y_t, y_b, total_target, total_baseline


def marked_words_z_scores(
    texts_target: list[str],
    texts_baseline: list[str],
    *,
    min_word_count: int,
    reference_corpus: str,
    prior_calibration: str = "mickel",
    reference_prior_weight: float = 0.85,
    calibration_constant: float | str = "auto",
    english_prior_weight: float = 0.5,
    prior_strength: float = 500.0,
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64], list[str], float]:
    """Vocabulary, delta, z-scores, calibration word set, and the resolved C.

    Significance is *not* applied here — callers add BH-FDR or the raw-z rule.
    Exposed on its own so the calibration plots can re-score permuted label
    assignments and constant sweeps without rebuilding the significance logic.
    """
    vocabulary, y_t, y_b, _, _ = _build_vocabulary(
        texts_target, texts_baseline, min_word_count
    )
    if len(vocabulary) < 2:
        empty = np.zeros(0, dtype=float)
        return vocabulary, empty, empty, [], 0.0

    prior_counts = hybrid_prior_counts(
        vocabulary, y_t, y_b, reference_corpus, reference_prior_weight
    )
    resolved_constant = resolve_calibration_constant(
        calibration_constant, vocabulary, prior_counts, y_t, y_b, reference_corpus
    )
    alpha_t, alpha_b, calibration_words = calibrated_side_alphas(
        vocabulary, prior_counts, y_t, y_b, resolved_constant
    )
    if prior_calibration != "mickel":
        alpha = _fixed_alpha(
            vocabulary, y_t, y_b, reference_corpus, english_prior_weight, prior_strength
        )
        alpha_t = alpha_b = alpha
    delta, z_scores = _log_odds_z(y_t, y_b, alpha_t, alpha_b)
    return vocabulary, delta, z_scores, calibration_words, resolved_constant


def resolve_calibration_constant(
    calibration_constant: float | str,
    vocabulary: list[str],
    prior_counts: NDArray[np.float64],
    y_target: NDArray[np.float64],
    y_baseline: NDArray[np.float64],
    reference_corpus: str,
) -> float:
    """A float passes through; "auto" binary-searches C per corpus (MotS)."""
    if calibration_constant == "auto":
        register_mask = _register_word_mask(vocabulary, reference_corpus)
        return auto_calibration_constant(
            vocabulary, prior_counts, y_target, y_baseline, register_mask
        )
    return float(calibration_constant)


def significance_mask(
    z_scores: NDArray[np.float64], significance: str, fdr_alpha: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """(p_values, p_adjusted, rejected) under "bh_fdr" or "raw_z"."""
    p_values = two_sided_p_from_z(z_scores)
    if significance == "raw_z":
        rejected = np.abs(z_scores) >= _RAW_Z_THRESHOLD
        return p_values, p_values.copy(), rejected
    p_adjusted, rejected = benjamini_hochberg(p_values, fdr_alpha)
    return p_values, p_adjusted, rejected


@dataclass
class MarkedWord(BaseSchema):
    """One vocabulary word's calibrated log-odds test outcome."""

    word: str
    count_target: int
    count_baseline: int
    log_odds: float  # prior-adjusted delta; positive = marked for the target set
    z_score: float
    p_value: float
    p_adjusted: float
    significant: bool  # under the configured significance rule
    significant_raw_z: bool = False  # |z| >= 1.96, MotS's own rule, always tracked


@dataclass
class MarkedWordsTable(BaseSchema):
    """Per-word outcomes over the shared vocabulary, sorted by |z| descending."""

    vocabulary_size: int
    total_tokens_target: int
    total_tokens_baseline: int
    n_significant_words: int
    # Raw-z (|z|>=1.96, no correction) count, always reported so that a strict-
    # correction "0 significant" still surfaces the exploratory MotS signal —
    # BH is underpowered on sparse per-word data (short/small real corpora).
    n_significant_raw_z: int = 0
    prior_calibration: str = "mickel"
    significance: str = "bh_fdr"
    reference_prior_weight: float = 0.85
    # "auto" (per-corpus binary search) or a fixed number, as configured.
    calibration_constant_mode: str = "auto"
    # The C actually used (resolved from "auto" via the binary search).
    calibration_constant: float = 0.0
    # The MotS calibration word set W (register anchors) for this comparison.
    calibration_words: list[str] = field(default_factory=list)
    words: list[MarkedWord] = field(default_factory=list)


def compute_marked_words_table(
    texts_target: list[str],
    texts_baseline: list[str],
    *,
    min_word_count: int,
    reference_corpus: str,
    fdr_alpha: float,
    prior_calibration: str = "mickel",
    reference_prior_weight: float = 0.85,
    calibration_constant: float | str = "auto",
    english_prior_weight: float = 0.5,
    prior_strength: float = 500.0,
    significance: str = "bh_fdr",
) -> MarkedWordsTable:
    """Run the calibrated log-odds z-test for every shared-vocabulary word."""
    mode = "auto" if calibration_constant == "auto" else "fixed"
    vocabulary, y_t, y_b, total_target, total_baseline = _build_vocabulary(
        texts_target, texts_baseline, min_word_count
    )
    # A one-word multinomial admits no log-odds contrast (the Monroe delta is
    # log(x/0) - log(x/0) = NaN), so fewer than two words means no test.
    if len(vocabulary) < 2:
        return MarkedWordsTable(
            vocabulary_size=len(vocabulary),
            total_tokens_target=total_target,
            total_tokens_baseline=total_baseline,
            n_significant_words=0,
            prior_calibration=prior_calibration,
            significance=significance,
            reference_prior_weight=reference_prior_weight,
            calibration_constant_mode=mode,
        )

    _, delta, z_scores, calibration_words, resolved_c = marked_words_z_scores(
        texts_target,
        texts_baseline,
        min_word_count=min_word_count,
        reference_corpus=reference_corpus,
        prior_calibration=prior_calibration,
        reference_prior_weight=reference_prior_weight,
        calibration_constant=calibration_constant,
        english_prior_weight=english_prior_weight,
        prior_strength=prior_strength,
    )
    p_values, p_adjusted, rejected = significance_mask(
        z_scores, significance, fdr_alpha
    )
    raw_z_rejected = np.abs(z_scores) >= _RAW_Z_THRESHOLD

    order = np.argsort(-np.abs(z_scores))
    words = [
        MarkedWord(
            word=vocabulary[i],
            count_target=int(y_t[i]),
            count_baseline=int(y_b[i]),
            log_odds=float(delta[i]),
            z_score=float(z_scores[i]),
            p_value=float(p_values[i]),
            p_adjusted=float(p_adjusted[i]),
            significant=bool(rejected[i]),
            significant_raw_z=bool(raw_z_rejected[i]),
        )
        for i in order
    ]
    return MarkedWordsTable(
        vocabulary_size=len(vocabulary),
        total_tokens_target=total_target,
        total_tokens_baseline=total_baseline,
        n_significant_words=int(rejected.sum()),
        n_significant_raw_z=int(raw_z_rejected.sum()),
        prior_calibration=prior_calibration,
        significance=significance,
        reference_prior_weight=reference_prior_weight,
        calibration_constant_mode=mode,
        calibration_constant=resolved_c,
        calibration_words=calibration_words,
        words=words,
    )
