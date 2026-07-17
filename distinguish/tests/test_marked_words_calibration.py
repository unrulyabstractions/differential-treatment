"""Proof-grade tests for the calibrated marked-words test.

Target under test: ``src/lexical/marked_words_analyzer.py`` (Monroe et al. 2008
prior-adjusted log-odds z-test with the Mickel et al. 2025 per-side Dirichlet
calibration, plus BH-FDR / raw-z significance).

These tests are designed to FAIL if the statistics are subtly wrong, not merely
to confirm the code runs:

1. NULL CALIBRATION (Monte-Carlo, 200 random label splits of ONE corpus): the
   Monroe z is standard-normal under the null, so |z|>=1.96 fires at the nominal
   ~5% per-word rate while BH-FDR survivors collapse to ~0. Proves the per-word
   statistic is calibrated AND that BH actually corrects for multiplicity (a
   miscalibrated z, or a no-op BH, both break this).
2. SIGNAL RECOVERY (known-answer): planted signature words in one side surface as
   BH-significant with the correct sign; balanced register/function words and
   balanced background content do NOT. Proves sensitivity + specificity.
3. AUTO-CALIBRATION: ``register_clean(C)`` is monotone in C (so the binary search
   is well-posed) on both a real corpus (data/synthetic) and a controlled one;
   the resolved C sits at the clean/dirty boundary (largest clean C); and on
   data/synthetic the resolved C suppresses {i'm, i've, a, do, the}.
4. EMPTY-CALIBRATION NO-NaN (the fixed bug): two disjoint, stopword-free corpora
   make the calibration word set genuinely empty; the whole-vocabulary fallback
   anchor keeps the alphas finite. A fault-injection counterfactual shows the
   pre-fix path (anchor = the empty mask) would have produced inf/NaN.
5. BH == scipy: ``benjamini_hochberg`` reproduces
   ``scipy.stats.false_discovery_control(..., method="bh")`` exactly, including
   the NaN->1.0 substitution and the ``significance_mask`` integration.

Run: ``uv run pytest tests/test_marked_words_calibration.py -q``.
"""

from __future__ import annotations

import math
import string

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import false_discovery_control

from src.common.stats_utils import benjamini_hochberg, two_sided_p_from_z
from src.lexical.marked_words_analyzer import (
    _AUTO_C_RANGE,
    _RAW_Z_THRESHOLD,
    CALIBRATION_STOPWORDS,
    _build_vocabulary,
    _log_odds_z,
    _register_word_mask,
    calibrated_side_alphas,
    calibration_word_indices,
    compute_marked_words_table,
    hybrid_prior_counts,
    marked_words_z_scores,
    resolve_calibration_constant,
    significance_mask,
)

REFERENCE = "wordfreq:en"
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Corpus helpers (all seeded / deterministic)                                 #
# --------------------------------------------------------------------------- #
def _nonsense(n: int, prefix: str) -> list[str]:
    """`n` distinct letter-only nonsense tokens (never in the wordfreq list).

    Letter-only so the tokenizer keeps them whole (its regex strips digits).
    """
    letters = string.ascii_lowercase
    words = [prefix + letters[i // 26] + letters[i % 26] for i in range(n)]
    assert len(set(words)) == n
    return words


def _docs_from_distribution(
    words: list[str], probs: np.ndarray, n_docs: int, doclen: int, seed: int
) -> list[str]:
    """`n_docs` documents, each `doclen` iid draws from a fixed word distribution."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(words))
    return [
        " ".join(words[j] for j in rng.choice(idx, size=doclen, p=probs))
        for _ in range(n_docs)
    ]


def _register_clean_at(
    vocab: list[str],
    prior: np.ndarray,
    y_t: np.ndarray,
    y_b: np.ndarray,
    register_mask: np.ndarray,
    constant: float,
) -> bool:
    """Reproduce the predicate the auto search optimizes: no register word raw-z."""
    alpha_t, alpha_b, _ = calibrated_side_alphas(vocab, prior, y_t, y_b, constant)
    _, z = _log_odds_z(y_t, y_b, alpha_t, alpha_b)
    if not register_mask.any():
        return True
    return bool(np.abs(z[register_mask]).max() < _RAW_Z_THRESHOLD)


# --------------------------------------------------------------------------- #
# 1. NULL CALIBRATION                                                         #
# --------------------------------------------------------------------------- #
def test_null_calibration_raw_z_nominal_and_bh_controls_fdr():
    """Split one corpus (identical word distribution) into two random halves 200x.

    Under the null the Monroe z is ~N(0,1), so with a weak (non-shrinking) prior
    |z|>=1.96 fires at ~5% per word while BH survivors collapse to ~0. The weak
    fixed prior is deliberate: it is the regime in which the log-odds statistic is
    *supposed* to be calibrated, so a nominal 5% raw rate here is direct evidence
    the z machinery is correct; a broken variance term or a broken BH would move
    these numbers.
    """
    vocab_size = 60
    words = _nonsense(vocab_size, "qz")
    probs = np.full(vocab_size, 1.0 / vocab_size)
    # One shared corpus; every split draws its two halves from THE SAME docs, so
    # both sides are guaranteed to share the word distribution (a true null).
    docs = _docs_from_distribution(words, probs, n_docs=400, doclen=30, seed=1)

    weak_c = 50.0  # weak prior -> alpha << counts -> z ~ N(0,1)
    n_splits = 200
    raw_fracs, bh_fracs, z_stds = [], [], []
    total_raw, total_bh = 0, 0
    for t in range(n_splits):
        rng = np.random.default_rng(1000 + t)
        left = rng.random(len(docs)) < 0.5
        side_a = [d for d, keep in zip(docs, left, strict=True) if keep]
        side_b = [d for d, keep in zip(docs, left, strict=True) if not keep]
        _, _, z, _, _ = marked_words_z_scores(
            side_a,
            side_b,
            min_word_count=5,
            reference_corpus=REFERENCE,
            calibration_constant=weak_c,
        )
        assert np.isfinite(z).all()
        _, _, rejected = significance_mask(z, "bh_fdr", ALPHA)
        raw = np.abs(z) >= _RAW_Z_THRESHOLD
        raw_fracs.append(float(raw.mean()))
        bh_fracs.append(float(rejected.mean()))
        z_stds.append(float(z.std()))
        total_raw += int(raw.sum())
        total_bh += int(rejected.sum())

    mean_raw = float(np.mean(raw_fracs))
    mean_bh = float(np.mean(bh_fracs))
    mean_zstd = float(np.mean(z_stds))

    # (a) The raw-z per-word false-positive rate is the nominal ~5%.
    assert 0.035 <= mean_raw <= 0.075, f"raw-z FPR {mean_raw:.4f} not ~5%"
    # (b) The null z is standard-normal (calibration evidence).
    assert 0.9 <= mean_zstd <= 1.1, f"null z std {mean_zstd:.3f} not ~1"
    # (c) BH survivors ~ 0: the FDR correction all but eliminates discoveries.
    assert mean_bh < 0.006, f"BH survivor fraction {mean_bh:.4f} not ~0"
    # (d) BH is doing real work, not just underpowered: raw >> BH by an order+.
    assert mean_raw > 8 * max(mean_bh, 1e-9)
    # (e) Global-null sanity on absolute counts.
    assert total_raw > 300  # ~5% of 200*60 = 600 expected
    assert total_bh < 40  # complete-null BH: a small handful over 200 splits


def test_null_auto_calibration_is_no_less_conservative():
    """Production ('auto') calibration never inflates the null error rate.

    Auto deliberately over-shrinks, so its raw-z FPR must be <= the (already
    nominal) weak-prior FPR, and its BH survivors must stay ~0.
    """
    words = _nonsense(50, "nz")
    probs = np.full(50, 1.0 / 50)
    docs = _docs_from_distribution(words, probs, n_docs=300, doclen=30, seed=7)
    raw_fracs, bh_fracs = [], []
    for t in range(40):
        rng = np.random.default_rng(5000 + t)
        left = rng.random(len(docs)) < 0.5
        side_a = [d for d, k in zip(docs, left, strict=True) if k]
        side_b = [d for d, k in zip(docs, left, strict=True) if not k]
        _, _, z, _, _ = marked_words_z_scores(
            side_a,
            side_b,
            min_word_count=5,
            reference_corpus=REFERENCE,
            calibration_constant="auto",
        )
        assert np.isfinite(z).all()
        _, _, rejected = significance_mask(z, "bh_fdr", ALPHA)
        raw_fracs.append(float((np.abs(z) >= _RAW_Z_THRESHOLD).mean()))
        bh_fracs.append(float(rejected.mean()))
    assert float(np.mean(raw_fracs)) <= 0.075
    assert float(np.mean(bh_fracs)) < 0.01


# --------------------------------------------------------------------------- #
# 2. SIGNAL RECOVERY                                                          #
# --------------------------------------------------------------------------- #
def _signal_corpus(seed: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Target gets extra 'signature' words; register + background stay balanced."""
    background = _nonsense(40, "bg")
    signature = _nonsense(6, "sg")
    register = ["the", "of", "and", "a", "do", "i'm", "you", "it"]

    def make(n_docs: int, inject: bool, rng_seed: int) -> list[str]:
        r = np.random.default_rng(rng_seed)
        docs = []
        for _ in range(n_docs):
            toks = [background[j] for j in r.choice(len(background), size=20)]
            toks += [register[j] for j in r.choice(len(register), size=15)]
            if inject:
                for w in signature:
                    toks += [w] * r.poisson(0.6)
            docs.append(" ".join(toks))
        return docs

    target = make(60, inject=True, rng_seed=seed + 1)
    baseline = make(60, inject=False, rng_seed=seed + 2)
    return target, baseline, signature, register


def test_signal_recovery_surfaces_signatures_not_register():
    """Planted signatures surface BH-significant (correct sign); register do not."""
    target, baseline, signature, register = _signal_corpus(seed=11)
    table = compute_marked_words_table(
        target,
        baseline,
        min_word_count=3,
        reference_corpus=REFERENCE,
        fdr_alpha=ALPHA,
        calibration_constant="auto",
        significance="bh_fdr",
    )
    by_word = {w.word: w for w in table.words}

    # Every planted signature word is a BH discovery, marked FOR the target (z>0).
    for w in signature:
        assert w in by_word, f"signature {w} missing from vocabulary"
        mw = by_word[w]
        assert mw.significant, f"signature {w} not BH-significant (z={mw.z_score:.2f})"
        assert mw.z_score > 0, f"signature {w} has wrong sign (z={mw.z_score:.2f})"

    # Balanced register/function words are NOT flagged (specificity).
    for w in register:
        if w in by_word:
            assert not by_word[w].significant, f"register word {w} wrongly flagged"

    # Balanced background content is (almost entirely) not flagged either.
    bg_flagged = sum(
        by_word[w].significant for w in _nonsense(40, "bg") if w in by_word
    )
    assert bg_flagged <= 2, f"{bg_flagged} balanced background words wrongly flagged"

    # The signatures dominate the |z| ranking (table.words is |z|-descending).
    top6 = {w.word for w in table.words[:6]}
    assert set(signature) <= top6, f"signatures not at the top: {top6}"


# --------------------------------------------------------------------------- #
# 3. AUTO-CALIBRATION                                                         #
# --------------------------------------------------------------------------- #
def _controlled_register_corpus() -> tuple[
    list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Balanced, equal-size sides with 40+ real function words (a clean register)."""
    common = [
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
        "he",
        "she",
        "we",
        "they",
        "but",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "do",
        "does",
        "did",
        "an",
        "my",
        "me",
    ]
    common = list(dict.fromkeys(common))
    content_t = ["alpha", "beta", "gamma", "delta", "omega"]
    content_b = ["kappa", "lambda", "sigma", "theta", "rho"]

    def make(seed: int, extra: list[str]) -> list[str]:
        r = np.random.default_rng(seed)
        docs = []
        for _ in range(80):
            toks = [common[j] for j in r.choice(len(common), size=25)]
            for w in extra:
                toks += [w] * r.poisson(0.6)
            docs.append(" ".join(toks))
        return docs

    target = make(1, content_t)
    baseline = make(2, content_b)
    vocab, y_t, y_b, _, _ = _build_vocabulary(target, baseline, 2)
    prior = hybrid_prior_counts(vocab, y_t, y_b, REFERENCE, 0.6)
    rmask = _register_word_mask(vocab, REFERENCE)
    return vocab, list(vocab), y_t, y_b, prior, rmask


@pytest.mark.parametrize("which", ["synthetic", "controlled"])
def test_register_clean_is_monotone_in_C(synthetic, which):
    """register_clean(C) is monotone non-increasing in C -> the search is valid.

    Stronger prior (smaller C) shrinks z toward 0, so 'no register word crosses
    1.96' can only *stop* holding as C grows. If this predicate were not monotone,
    the binary search in auto_calibration_constant would be ill-posed.
    """
    if which == "synthetic":
        target = synthetic.prompt_set("target").texts
        baseline = synthetic.prompt_set("baseline").texts
        vocab, y_t, y_b, _, _ = _build_vocabulary(target, baseline, 2)
        prior = hybrid_prior_counts(vocab, y_t, y_b, REFERENCE, 0.6)
        rmask = _register_word_mask(vocab, REFERENCE)
    else:
        vocab, _, y_t, y_b, prior, rmask = _controlled_register_corpus()

    assert rmask.sum() > 0, "no register words to test monotonicity on"
    grid = np.linspace(_AUTO_C_RANGE[0], _AUTO_C_RANGE[1], 80)
    clean = [_register_clean_at(vocab, prior, y_t, y_b, rmask, float(c)) for c in grid]
    for i in range(len(clean) - 1):
        assert clean[i] >= clean[i + 1], (
            f"register_clean not monotone at C={grid[i]:.3f}->{grid[i + 1]:.3f} "
            f"({clean[i]}->{clean[i + 1]})"
        )


def test_auto_resolves_to_largest_clean_C_on_synthetic(synthetic):
    """resolve_calibration_constant('auto') lands at the clean/dirty boundary."""
    target = synthetic.prompt_set("target").texts
    baseline = synthetic.prompt_set("baseline").texts
    vocab, y_t, y_b, _, _ = _build_vocabulary(target, baseline, 2)
    prior = hybrid_prior_counts(vocab, y_t, y_b, REFERENCE, 0.6)
    rmask = _register_word_mask(vocab, REFERENCE)

    resolved = resolve_calibration_constant("auto", vocab, prior, y_t, y_b, REFERENCE)
    # The resolved value is interior to the search range (a real boundary here).
    assert _AUTO_C_RANGE[0] < resolved < _AUTO_C_RANGE[1]
    # It is clean; a clearly stronger prior is clean; a clearly weaker prior dirty.
    assert _register_clean_at(vocab, prior, y_t, y_b, rmask, resolved)
    assert _register_clean_at(vocab, prior, y_t, y_b, rmask, resolved * 0.5)
    assert not _register_clean_at(vocab, prior, y_t, y_b, rmask, 0.08)
    # Combined with monotonicity, this pins the largest-clean-C interpretation.


def test_auto_suppresses_register_words_on_synthetic(synthetic):
    """On data/synthetic the resolved C suppresses {i'm, i've, a, do, the}."""
    target = synthetic.prompt_set("target").texts
    baseline = synthetic.prompt_set("baseline").texts
    table = compute_marked_words_table(
        target,
        baseline,
        min_word_count=2,
        reference_corpus=REFERENCE,
        fdr_alpha=ALPHA,
        reference_prior_weight=0.6,
        calibration_constant="auto",
        significance="bh_fdr",
    )
    by_word = {w.word: w for w in table.words}
    for w in ["i'm", "i've", "a", "do", "the"]:
        assert w in by_word, f"{w} not in synthetic vocabulary"
        mw = by_word[w]
        assert abs(mw.z_score) < _RAW_Z_THRESHOLD, (
            f"register word {w} not suppressed at resolved C="
            f"{table.calibration_constant:.5f} (z={mw.z_score:.3f})"
        )
        assert not mw.significant_raw_z


# --------------------------------------------------------------------------- #
# 4. EMPTY-CALIBRATION NO-NaN (the fixed bug)                                 #
# --------------------------------------------------------------------------- #
def _disjoint_stopword_free_corpora() -> tuple[list[str], list[str]]:
    """Two corpora with disjoint vocabularies (>20 positive words each), no stopwords."""
    tw = _nonsense(25, "tt")
    bw = _nonsense(25, "bb")
    assert set(tw).isdisjoint(bw)
    assert not (set(tw) | set(bw)) & CALIBRATION_STOPWORDS
    r = np.random.default_rng(0)
    target = [" ".join(r.choice(tw, size=30)) for _ in range(40)]
    baseline = [" ".join(r.choice(bw, size=30)) for _ in range(40)]
    return target, baseline


def test_empty_calibration_set_stays_finite():
    """Disjoint, stopword-free corpora -> empty calibration set -> finite z (no NaN)."""
    target, baseline = _disjoint_stopword_free_corpora()
    vocab, y_t, y_b, _, _ = _build_vocabulary(target, baseline, 1)
    prior = hybrid_prior_counts(vocab, y_t, y_b, REFERENCE, 0.85)

    # The calibration word set is genuinely empty (the pre-fix crash trigger).
    in_w = calibration_word_indices(vocab, prior, y_t, y_b)
    assert not in_w.any(), "calibration set unexpectedly non-empty"

    # The fallback anchor keeps both sides' alphas finite and strictly positive.
    alpha_t, alpha_b, calib_words = calibrated_side_alphas(vocab, prior, y_t, y_b, 0.25)
    assert calib_words == []
    assert np.isfinite(alpha_t).all() and np.isfinite(alpha_b).all()
    assert (alpha_t > 0).all() and (alpha_b > 0).all()

    # End-to-end under BH: no crash, every z / p / p_adjusted finite.
    table = compute_marked_words_table(
        target,
        baseline,
        min_word_count=1,
        reference_corpus=REFERENCE,
        fdr_alpha=ALPHA,
        calibration_constant="auto",
        significance="bh_fdr",
    )
    assert table.calibration_words == []
    z = np.array([w.z_score for w in table.words])
    p = np.array([w.p_value for w in table.words])
    padj = np.array([w.p_adjusted for w in table.words])
    assert np.isfinite(z).all()
    assert np.isfinite(p).all()
    assert np.isfinite(padj).all()


def test_empty_calibration_fallback_is_load_bearing():
    """Fault injection: the pre-fix path (anchor = the empty mask) WOULD be non-finite.

    calibrated_side_alphas guards with `anchor = in_w if in_w.any() else ones`.
    Reproducing the unguarded computation on the empty mask must yield inf/NaN,
    proving the guard is what prevents the crash.
    """
    target, baseline = _disjoint_stopword_free_corpora()
    vocab, y_t, y_b, _, _ = _build_vocabulary(target, baseline, 1)
    prior = hybrid_prior_counts(vocab, y_t, y_b, REFERENCE, 0.85)
    in_w = calibration_word_indices(vocab, prior, y_t, y_b)
    assert not in_w.any()

    # Unguarded (pre-fix) regularizer: anchor on the empty mask.
    w_p_broken = float(prior[in_w].sum())  # 0.0 over an empty selection
    w_gt_broken = float(y_t[in_w].sum())
    assert w_p_broken == 0.0 and w_gt_broken == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r_broken = 0.25 * w_p_broken / max(w_gt_broken, 1.0)  # -> 0.0
        alpha_broken = prior / r_broken  # divide by zero
    assert not np.isfinite(alpha_broken).all(), "pre-fix path was unexpectedly finite"

    # The actual (guarded) function is finite on the very same input.
    alpha_t, alpha_b, _ = calibrated_side_alphas(vocab, prior, y_t, y_b, 0.25)
    assert np.isfinite(alpha_t).all() and np.isfinite(alpha_b).all()


# --------------------------------------------------------------------------- #
# 5. BH == scipy.stats.false_discovery_control                               #
# --------------------------------------------------------------------------- #
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=300,
    )
)
def test_bh_matches_scipy_exactly(pvals):
    """benjamini_hochberg reproduces scipy false_discovery_control(method='bh')."""
    p = np.asarray(pvals, dtype=float)
    adjusted, rejected = benjamini_hochberg(p, ALPHA)
    scipy_adjusted = false_discovery_control(p, method="bh")
    assert np.array_equal(adjusted, scipy_adjusted, equal_nan=True) or np.allclose(
        adjusted, scipy_adjusted, rtol=0, atol=0, equal_nan=True
    )
    assert np.array_equal(rejected, scipy_adjusted <= ALPHA)


def test_bh_nan_is_treated_as_least_significant():
    """A NaN p-value is substituted with 1.0 and then matches scipy exactly."""
    p = np.array([0.001, 0.2, np.nan, 0.04, 0.9])
    adjusted, rejected = benjamini_hochberg(p, ALPHA)
    reference = false_discovery_control(np.nan_to_num(p, nan=1.0), method="bh")
    assert np.allclose(adjusted, reference)
    # The NaN word must never be a discovery.
    assert not rejected[2]


def test_bh_known_answers():
    """Deterministic edge cases: no signal -> no rejections; clear signal -> rejects."""
    # Complete null (all large p): BH rejects nothing.
    _, rej = benjamini_hochberg(np.full(50, 0.9), ALPHA)
    assert rej.sum() == 0
    # One dominant signal among nulls: BH rejects it and adjusted p is monotone.
    p = np.concatenate([[1e-8], np.full(19, 0.8)])
    adjusted, rej = benjamini_hochberg(p, ALPHA)
    assert rej[0] and rej.sum() == 1
    # Adjusted p-values, taken in increasing raw-p order, are non-decreasing.
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_significance_mask_bh_matches_scipy_on_z():
    """significance_mask('bh_fdr') == scipy BH applied to the two-sided p-values."""
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(0, 1, 80), rng.normal(4, 1, 5)])
    p_values, p_adjusted, rejected = significance_mask(z, "bh_fdr", ALPHA)
    reference = false_discovery_control(two_sided_p_from_z(z), method="bh")
    assert np.allclose(p_values, two_sided_p_from_z(z))
    assert np.allclose(p_adjusted, reference)
    assert np.array_equal(rejected, reference <= ALPHA)


def test_significance_mask_raw_z_rule():
    """significance_mask('raw_z') is exactly the |z| >= 1.96 rule (no correction)."""
    z = np.array([-2.5, -1.96, -1.9, 0.0, 1.95, 1.96, 3.0])
    _, _, rejected = significance_mask(z, "raw_z", ALPHA)
    assert np.array_equal(rejected, np.abs(z) >= _RAW_Z_THRESHOLD)


# --- (6) MONROE CLOSED-FORM KNOWN-ANSWER ---------------------------------
# The distributional null tests prove the z is standard-normal under H0, but a
# bug that biases the point estimate or variance while preserving null normality
# would slip. This pins the Monroe delta AND variance to a hand-computed value.
def test_log_odds_z_matches_hand_computed_monroe():
    """_log_odds_z equals the Monroe et al. closed form on a tiny 2-word case."""
    y_t = np.array([3.0, 7.0])
    y_b = np.array([5.0, 5.0])
    a_t = np.array([1.0, 1.0])
    a_b = np.array([1.0, 1.0])
    n_t, n_b, a0 = 10.0, 10.0, 2.0

    def hand(i: int) -> tuple[float, float]:
        delta = math.log((y_t[i] + a_t[i]) / (n_t + a0 - y_t[i] - a_t[i])) - math.log(
            (y_b[i] + a_b[i]) / (n_b + a0 - y_b[i] - a_b[i])
        )
        var = 1.0 / (y_t[i] + a_t[i]) + 1.0 / (y_b[i] + a_b[i])
        return delta, delta / math.sqrt(var)

    delta, z = _log_odds_z(y_t, y_b, a_t, a_b)
    for i in range(2):
        exp_delta, exp_z = hand(i)
        assert delta[i] == pytest.approx(exp_delta, abs=1e-12)
        assert z[i] == pytest.approx(exp_z, abs=1e-12)
    # Concrete anchor values (independently arithmetic): word 0 leans baseline
    # (z<0), word 1 leans target (z>0), with the exact magnitudes below.
    assert z[0] == pytest.approx(-1.073819, abs=1e-5)
    assert z[1] == pytest.approx(1.283459, abs=1e-5)
