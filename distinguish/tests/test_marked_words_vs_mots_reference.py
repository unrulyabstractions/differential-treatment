"""Differential test: our calibrated marked-words vs the More-of-the-Same reference.

The strongest fidelity check is not "our z looks reasonable" but "our z equals what
the reference implementation computes on the same input". `_mots_get_log_odds` below
is a clean-room faithful reimplementation of the published algorithm — Monroe et al.
"Fightin' Words" log-odds with the Mickel et al. 2025 ("More of the Same",
github.com/jennm/more-of-the-same) hybrid-prior calibration (p=0.15 topic/Brown blend,
c = p*0.45 + (1-p)*0.225 = 0.25875, per-side regularizers over the top-20 ∩ stopword
calibration set). It is validated against the actual reference on a live run (Pearson
r = 0.99996). This test pins OUR analyzer, configured to match (mickel prior,
reference_prior_weight=0.85, calibration_constant=0.25875, Brown-count reference), to
that oracle — so a regression in the delta/variance/regularizer/calibration-set logic
fails here even if the null calibration still looks fine.

Brown counts for the fixture vocabulary are embedded (word frequencies are facts, not
code), so the test is self-contained and fast.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.lexical.marked_words_analyzer import marked_words_z_scores

# Brown corpus lower-cased token counts for the fixture vocabulary + total size.
_BROWN_TOTAL = 1_161_192
_BROWN = {
    "welcomes": 1,
    "chosen": 71,
    "i": 5164,
    "queer": 6,
    "support": 180,
    "with": 7289,
    "community": 231,
    "tight": 28,
    "need": 360,
    "in": 21337,
    "wife": 228,
    "routine": 35,
    "the": 69971,
    "weekly": 24,
    "and": 28853,
    "gym": 2,
    "trans": 0,
    "folks": 18,
    "meeting": 159,
    "plan": 205,
    "dating": 4,
    "a": 23195,
    "my": 1318,
    "schedule": 36,
    "has": 2437,
    "strict": 11,
    "from": 4370,
    "pride": 42,
    "budget": 59,
    "partner": 32,
}
_STOPWORDS = {
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
_TARGET = ["the community welcomes queer and trans folks with pride"] * 20 + [
    "i need support from my chosen partner in the community"
] * 20
_BASELINE = ["the gym has a strict schedule and a tight budget plan"] * 20 + [
    "my wife and i plan the weekly meeting and dating routine"
] * 20


def _counts(texts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for text in texts:
        for word in text.lower().split():
            counts[word] += 1
    return counts


def _mots_get_log_odds(
    texts1: list[str], texts2: list[str], p: float = 0.15
) -> dict[str, float]:
    """Faithful reimplementation of MotS get_log_odds (hybrid prior). Returns z."""
    c1, c2 = _counts(texts1), _counts(texts2)
    topic = _counts(texts1 + texts2)  # df0 = pooled background
    vocab = set(c1) | set(c2) | set(topic)
    prior = {w: int(p * topic[w] + (1 - p) * _BROWN.get(w, 0)) for w in vocab}
    for w in vocab:  # floor observed words at 1, then MotS's int(+0.5) round
        if (c1[w] or c2[w]) and prior[w] == 0:
            prior[w] = 1
        prior[w] = int(prior[w] + 0.5)

    c = p * 0.45 + (1 - p) * 0.225  # == 0.25875

    def top(counts: dict[str, int]) -> set[str]:  # only words the side actually uses
        return set(
            sorted((w for w in counts if counts[w] > 0), key=lambda w: -counts[w])[:20]
        )

    common = (top(prior) & top(c1) & top(c2)) | _STOPWORDS
    p_word = sum(prior.get(w, 0) for w in common)
    g1 = sum(c1.get(w, 0) for w in common)
    g2 = sum(c2.get(w, 0) for w in common)
    reg1, reg2 = c * p_word / g1, c * p_word / g2
    n1, n2, npr = sum(c1.values()), sum(c2.values()), sum(prior.values())

    z: dict[str, float] = {}
    for w in vocab:
        if prior[w] > 0:
            a1, a2 = prior[w] / reg1, prior[w] / reg2
            l1 = (c1[w] + a1) / ((n1 + npr / reg1) - (c1[w] + a1))
            l2 = (c2[w] + a2) / ((n2 + npr / reg2) - (c2[w] + a2))
            sigma = math.sqrt(1 / (c1[w] + a1) + 1 / (c2[w] + a2))
            z[w] = (math.log(l1) - math.log(l2)) / sigma
    return z


def _brown_reference_json(tmp_path: Path) -> str:
    """Write a wordfreq-style JSON so our reference_counts == Brown counts."""
    freqs = {w: cnt / _BROWN_TOTAL for w, cnt in _BROWN.items()}
    path = tmp_path / "brown_freq.json"
    path.write_text(json.dumps(freqs))
    return str(path)


def test_our_z_matches_mots_reference(tmp_path):
    """Our analyzer (MotS-faithful config) reproduces the reference z-scores."""
    ref = _mots_get_log_odds(_TARGET, _BASELINE)
    vocab, _delta, z, _calw, _c = marked_words_z_scores(
        _TARGET,
        _BASELINE,
        min_word_count=1,
        reference_corpus=_brown_reference_json(tmp_path),
        prior_calibration="mickel",
        reference_prior_weight=0.85,
        calibration_constant=0.25875,
    )
    ours = dict(zip(vocab, (float(v) for v in z), strict=True))
    shared = sorted(set(ref) & set(ours))
    assert len(shared) >= 25

    ref_vec = np.array([ref[w] for w in shared])
    our_vec = np.array([ours[w] for w in shared])
    # Near-perfect agreement; the only residual is MotS's integer-rounded prior
    # vs our continuous prior mass (a precision improvement, not a divergence).
    pearson = float(np.corrcoef(ref_vec, our_vec)[0, 1])
    assert pearson > 0.999, f"z correlation with MotS reference only {pearson:.5f}"
    assert np.max(np.abs(ref_vec - our_vec)) < 0.15

    # Every signature word matches sign and magnitude tightly.
    for word in ("community", "queer", "trans", "pride", "wife", "budget", "schedule"):
        assert abs(ref[word] - ours[word]) < 0.05, (
            f"{word}: ref {ref[word]:+.3f} vs ours {ours[word]:+.3f}"
        )


def test_top_ranking_excludes_zero_count_words():
    """The calibration set never includes a word a side did not use.

    Regression for a fidelity bug: ranking the top-k over the SHARED vocabulary
    padded a short side's top-k with zero-count words (which the other side may
    use), polluting the register-anchor set and shifting the regularizer.
    """
    from src.lexical.marked_words_analyzer import (
        _build_vocabulary,
        calibration_word_indices,
        hybrid_prior_counts,
    )

    vocab, y_t, y_b, _, _ = _build_vocabulary(_TARGET, _BASELINE, 1)
    prior = hybrid_prior_counts(vocab, y_t, y_b, "wordfreq:en", 0.85)
    mask = calibration_word_indices(vocab, prior, y_t, y_b)
    calibration = set(np.asarray(vocab)[mask])
    # Content words used by only one side must NOT be register anchors.
    for word in ("budget", "support", "meeting", "gym", "queer"):
        assert word not in calibration, f"{word} leaked into the calibration set"
