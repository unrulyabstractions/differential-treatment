"""PROOF-GRADE robustness suite for the offline dimensions.

Every test here is built to FAIL if the code were subtly wrong, not merely to
confirm "it runs and returns a float". The strategies are:

* Monte-Carlo NULL CALIBRATION -- run the estimator thousands of times under a
  true null and check the p-value / null distribution lands where theory says.
* KNOWN-ANSWER RECOVERY -- plant a signal (a marked word, a separable Gaussian,
  a scale gap) whose sign and significance are known, and demand it back.
* GROUND-TRUTH / CLOSED-FORM comparison -- JSD, BH, rank-biserial against
  independently computed values.
* FAULT INJECTION -- feed degenerate/adversarial inputs and require a clear
  error OR a sane result, never a crash, NaN, or a silent wrong number.
* SCALE -- 4000 prompts through compute_lexical + a MiniLM linear C2ST.

Run: `uv run pytest tests/test_offline_robustness_proofs.py -x -q`

Offline only (lexical, syntactic/neurobiber, distributional linear + MiniLM,
jsd, permutation, usage); no OpenAI/Cohere path is exercised.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path

# Force the HF stack offline: every model these tests touch (MiniLM, neurobiber)
# is cached, and no test should ever depend on the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pytest

from src.common.dataset_annotations import InteractionContext, PromptLabels
from src.common.dataset_tables import PromptDataset
from src.common.prompt_set_schema import (
    GROUP_BASELINE,
    GROUP_TARGET,
    AuthorProfile,
    PromptRecord,
    PromptSet,
)
from src.common.run_config import (
    DistributionalConfig,
    LexicalConfig,
    SyntacticConfig,
    UsageConfig,
)
from src.common.stats_utils import (
    benjamini_hochberg,
    jensen_shannon_divergence,
    permutation_p_value,
    permute_labels_by_author,
)
from src.distributional.c2st_linear import run_linear_c2st
from src.distributional.distributional_dimension import (
    _majority_chance,
    compute_distributional,
)
from src.lexical.lexical_dimension import compute_lexical
from src.lexical.marked_words_analyzer import (
    _AUTO_C_RANGE,
    _RAW_Z_THRESHOLD,
    _register_word_mask,
    marked_words_z_scores,
)
from src.pipeline.pipeline_context import PipelineContext
from src.syntactic.syntactic_dimension import compute_syntactic
from src.usage.usage_attitudes import MIN_AUTHORS_PER_SIDE, compute_usage

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _pset(
    name: str,
    group: str,
    texts: list[str],
    author_ids: list[str] | None = None,
    contexts: list[InteractionContext] | None = None,
) -> PromptSet:
    """Assemble a PromptSet from raw texts, bypassing validate() so degenerate
    inputs (empty strings, single prompts) can be tested directly."""
    n = len(texts)
    author_ids = author_ids or [f"{name}_a{i}" for i in range(n)]
    contexts = contexts or [InteractionContext() for _ in range(n)]
    prompts = [
        PromptRecord(
            prompt_id=f"{name}_p{i}",
            author_id=author_ids[i],
            text=texts[i],
            labels=PromptLabels(1 if group == GROUP_TARGET else 0, 0, 0.0),
            context=contexts[i],
        )
        for i in range(n)
    ]
    authors = [AuthorProfile(author_id=a) for a in dict.fromkeys(author_ids)]
    return PromptSet(name=name, group=group, authors=authors, prompts=prompts)


def _gaussian(n: int, dim: int, mean: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mean, scale=1.0, size=(n, dim)).astype(np.float64)


def _manual_jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Independent closed-form JSD in bits, to check the library wrapper."""
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ===========================================================================
# A. PURE-STATS PROOFS (permutation, jsd, BH, author permutation)
# ===========================================================================
def test_permutation_pvalue_exact_formula():
    """Closed-form check of the +1-corrected one-sided permutation p-value."""
    null = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # observed=2.5 -> null >= 2.5 is {3,4} -> (1+2)/(1+5) = 0.5
    assert permutation_p_value(2.5, null) == pytest.approx(3 / 6)
    # observed above every null -> minimal attainable p = 1/(B+1)
    assert permutation_p_value(99.0, null) == pytest.approx(1 / 6)
    # observed below every null -> p = (1+B)/(1+B) = 1.0 (ties count as >=)
    assert permutation_p_value(-1.0, null) == pytest.approx(1.0)
    # exact tie is counted (>=), never dropped
    assert permutation_p_value(2.0, null) == pytest.approx((1 + 3) / 6)


def test_permutation_pvalue_montecarlo_uniform():
    """MONTE-CARLO CALIBRATION: when observed and null are exchangeable draws
    from the same law, the permutation p-value is (super-)uniform, i.e.
    P(p<=alpha) ~= alpha. A p-value formula that were off (wrong tail, missing
    +1, > vs >=) would break this."""
    rng = np.random.default_rng(0)
    trials, b = 4000, 99
    ps = np.empty(trials)
    for t in range(trials):
        sample = rng.standard_normal(b + 1)
        ps[t] = permutation_p_value(sample[0], sample[1:])
    for alpha in (0.1, 0.25, 0.5):
        emp = float(np.mean(ps <= alpha))
        # binomial std over 4000 trials ~ 0.007; 0.03 is a comfortable band and
        # still tight enough to catch a mis-calibrated tail.
        assert emp == pytest.approx(alpha, abs=0.03), (alpha, emp)
    # a valid p-value must never be < the minimal grid value 1/(B+1)
    assert ps.min() >= 1 / (b + 1) - 1e-12


def test_jsd_ground_truth_and_bounds():
    """JSD against independently computed base-2 values, plus its [0,1] range
    and symmetry."""
    assert jensen_shannon_divergence(
        np.array([0.5, 0.5]), np.array([0.5, 0.5])
    ) == pytest.approx(0.0, abs=1e-12)
    # disjoint supports -> exactly 1 bit
    assert jensen_shannon_divergence(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(1.0, abs=1e-9)
    # closed-form intermediate value
    p, q = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    assert jensen_shannon_divergence(p, q) == pytest.approx(
        _manual_jsd_base2(p, q), abs=1e-9
    )
    assert _manual_jsd_base2(p, q) == pytest.approx(0.3112781, abs=1e-6)
    # random distributions: symmetric and bounded in [0,1]
    rng = np.random.default_rng(1)
    for _ in range(300):
        a = rng.random(6) + 1e-6
        b = rng.random(6) + 1e-6
        d_ab = jensen_shannon_divergence(a / a.sum(), b / b.sum())
        d_ba = jensen_shannon_divergence(b / b.sum(), a / a.sum())
        assert d_ab == pytest.approx(d_ba, abs=1e-9)
        assert -1e-9 <= d_ab <= 1.0 + 1e-9


def test_benjamini_hochberg_known_example_and_nan():
    """Known-answer BH rejection set + NaN-safety."""
    p = np.array([0.001, 0.5, 0.6, 0.7, 0.8])
    adjusted, rejected = benjamini_hochberg(p, alpha=0.05)
    assert list(rejected) == [True, False, False, False, False]
    assert adjusted[0] == pytest.approx(0.005, abs=1e-9)  # 0.001 * 5 / 1
    assert np.all(adjusted <= 1.0 + 1e-12)
    # a NaN p-value must be neutralised (treated as p=1), never crash BH
    p_nan = np.array([0.001, np.nan, 0.5])
    adj_nan, rej_nan = benjamini_hochberg(p_nan, alpha=0.05)
    assert not np.isnan(adj_nan).any()
    assert not rej_nan[1]  # the NaN entry is never a discovery
    # empty input is handled
    adj_e, rej_e = benjamini_hochberg(np.array([]), 0.05)
    assert adj_e.size == 0 and rej_e.size == 0


def test_benjamini_hochberg_controls_fdr_under_null():
    """MONTE-CARLO: under the global null (uniform p-values) BH at alpha rejects
    'any' with probability <= ~alpha (FDR = FWER here). A BH that forgot the
    rank scaling would over-reject and blow past the bound."""
    rng = np.random.default_rng(7)
    trials, m, alpha = 600, 20, 0.05
    any_reject = 0
    total_reject = 0
    for _ in range(trials):
        p = rng.random(m)
        _, rejected = benjamini_hochberg(p, alpha)
        any_reject += int(rejected.any())
        total_reject += int(rejected.sum())
    frac_any = any_reject / trials
    mean_false = total_reject / trials
    assert frac_any <= 0.10, frac_any  # ~alpha with slack
    assert mean_false <= 0.15, mean_false  # essentially no false discoveries


def test_permute_labels_by_author_invariants_and_fault():
    """The author-level permutation must (a) keep one label per author, (b)
    preserve the label multiset (it is a relabeling, not resampling), and (c)
    reject an author that already carries both labels."""
    rng = np.random.default_rng(3)
    # 4 authors, 3 prompts each; authors 0,1 -> label 1, authors 2,3 -> label 0
    authors = [f"au{i}" for i in range(4) for _ in range(3)]
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(50):
        permuted = permute_labels_by_author(authors, labels, rng)
        # (b) multiset preserved
        assert sorted(permuted.tolist()) == sorted(labels.tolist())
        # (a) every author's prompts share one label after permutation
        by_author: dict[str, set] = {}
        for a, lab in zip(authors, permuted, strict=True):
            by_author.setdefault(a, set()).add(int(lab))
        assert all(len(s) == 1 for s in by_author.values())
    # (c) fault injection: an author with conflicting labels is rejected
    with pytest.raises(ValueError):
        permute_labels_by_author(["x", "x"], np.array([1, 0]), rng)


def test_permute_labels_by_author_is_unbiased():
    """MONTE-CARLO: with k of n authors carrying label 1, each author ends up
    labelled 1 with probability k/n. A biased shuffle (e.g. permuting prompts
    instead of authors, or leaking the original assignment) fails this."""
    rng = np.random.default_rng(11)
    n_authors = 6
    authors = [f"g{i}" for i in range(n_authors)]  # one prompt per author
    labels = np.array([1, 1, 1, 0, 0, 0])  # k=3, n=6 -> expected 0.5 each
    trials = 4000
    hits = np.zeros(n_authors)
    for _ in range(trials):
        permuted = permute_labels_by_author(authors, labels, rng)
        hits += permuted
    freq = hits / trials
    assert np.allclose(freq, 0.5, atol=0.04), freq


# ===========================================================================
# B. DISTRIBUTIONAL C2ST PROOFS (synthetic embeddings, no model needed)
# ===========================================================================
def test_majority_chance_reflects_class_imbalance():
    """chance_level is the majority-class rate, not a hard-coded 0.5."""
    assert _majority_chance(np.array([1] * 10 + [0] * 200)) == pytest.approx(200 / 210)
    assert _majority_chance(np.array([1, 1, 0, 0])) == 0.5
    assert _majority_chance(np.array([], dtype=int)) == 0.5
    assert _majority_chance(np.array([1, 1, 1])) == 1.0


def test_c2st_recovers_separable_signal():
    """KNOWN-ANSWER: two well-separated Gaussians must yield near-perfect held-
    out accuracy and a significant permutation p-value. A C2ST that always
    returned chance (or shuffled train/test) would fail here."""
    rng = np.random.default_rng(20)
    dim, n = 12, 60
    emb = np.vstack([_gaussian(n, dim, +2.0, rng), _gaussian(n, dim, -2.0, rng)])
    labels = np.array([1] * n + [0] * n)
    authors = [f"t{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    out = run_linear_c2st(
        emb,
        labels,
        authors,
        cv_folds=5,
        n_permutations=100,
        rng=np.random.default_rng(0),
    )
    assert out.accuracy > 0.95, out.accuracy
    assert out.p_value <= 0.02, out.p_value
    # balanced data -> the null centers at 0.5, far below the observed accuracy
    assert abs(np.median(out.null_accuracies) - 0.5) < 0.08
    assert not any(math.isnan(a) for a in out.null_accuracies)


def test_c2st_balanced_null_no_false_positive():
    """MONTE-CARLO NULL: same distribution both sides, balanced. Observed
    accuracy ~ 0.5, the permutation null centers at 0.5, and the p-value is not
    spuriously significant."""
    rng = np.random.default_rng(21)
    dim, n = 16, 60
    emb = _gaussian(2 * n, dim, 0.0, rng)  # identical law for both sides
    labels = np.array([1] * n + [0] * n)
    authors = [f"t{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    out = run_linear_c2st(
        emb,
        labels,
        authors,
        cv_folds=5,
        n_permutations=200,
        rng=np.random.default_rng(1),
    )
    assert abs(out.accuracy - 0.5) < 0.12, out.accuracy
    assert abs(np.median(out.null_accuracies) - 0.5) < 0.06
    assert out.p_value > 0.05, out.p_value  # no false discovery under the null


def test_c2st_imbalance_null_centers_on_majority():
    """THE IMBALANCE PROOF (10 vs 200, no signal):

      * chance_level == majority rate (200/210), not 0.5
      * observed accuracy ~ majority rate (classifier predicts majority)
      * the permutation null CENTERS ON THE MAJORITY RATE, not on 0.5
      * therefore the p-value is NOT falsely significant.

    If the null were (wrongly) centered at 0.5, the 0.95 observed accuracy would
    look wildly significant -- this test would catch that."""
    rng = np.random.default_rng(22)
    dim, n_t, n_b = 16, 10, 200
    emb = _gaussian(n_t + n_b, dim, 0.0, rng)  # no signal, imbalanced
    labels = np.array([1] * n_t + [0] * n_b)
    authors = [f"t{i}" for i in range(n_t)] + [f"b{i}" for i in range(n_b)]
    maj = n_b / (n_t + n_b)
    out = run_linear_c2st(
        emb,
        labels,
        authors,
        cv_folds=5,
        n_permutations=200,
        rng=np.random.default_rng(2),
    )
    assert _majority_chance(labels) == pytest.approx(maj)
    assert abs(out.accuracy - maj) < 0.04, out.accuracy
    null_med = float(np.median(out.null_accuracies))
    assert null_med > 0.85, null_med  # decisively NOT near 0.5
    assert abs(null_med - maj) < 0.04, null_med  # centered on the majority rate
    assert out.p_value > 0.05, out.p_value


def test_c2st_degenerate_one_author_per_side_is_graceful():
    """FAULT INJECTION: all target prompts share one author, all baseline
    another. Only 2 groups -> the degenerate all-one-class training branch is
    exercised. Must not crash or emit NaN."""
    rng = np.random.default_rng(23)
    dim = 8
    emb = np.vstack([_gaussian(6, dim, 1.0, rng), _gaussian(6, dim, -1.0, rng)])
    labels = np.array([1] * 6 + [0] * 6)
    authors = ["only_target"] * 6 + ["only_baseline"] * 6
    out = run_linear_c2st(
        emb,
        labels,
        authors,
        cv_folds=5,
        n_permutations=20,
        rng=np.random.default_rng(3),
    )
    assert 0.0 <= out.accuracy <= 1.0
    assert not math.isnan(out.p_value)
    assert all(0.0 <= a <= 1.0 for a in out.null_accuracies)


# ===========================================================================
# C. LEXICAL PROOFS (marked words)
# ===========================================================================
def _lexical(target: PromptSet, baseline: PromptSet, **overrides) -> object:
    cfg = LexicalConfig(**overrides)
    ctx = PipelineContext(random_seed=0)
    return compute_lexical(target, baseline, cfg, ctx)


def test_lexical_recovers_planted_marked_word():
    """KNOWN-ANSWER + SIGN CONVENTION: a content word planted only in the target
    side must come back with z>0 and land in marked_words_target; a word planted
    only in baseline must have z<0 and land in marked_words_baseline. No NaNs."""
    filler = "we were talking about the weather and the plans for the week"
    target = _pset(
        GROUP_TARGET,
        GROUP_TARGET,
        [f"{filler} zqxwv matters here" for _ in range(40)],
    )
    baseline = _pset(
        GROUP_BASELINE,
        GROUP_BASELINE,
        [f"{filler} qwzxv matters here" for _ in range(40)],
    )
    res = _lexical(target, baseline)
    words = {w.word: w for w in res.all_words}
    assert "zqxwv" in words and "qwzxv" in words
    assert words["zqxwv"].z_score > 0  # over-represented in target
    assert words["qwzxv"].z_score < 0  # over-represented in baseline
    assert "zqxwv" in {w.word for w in res.marked_words_target}
    assert "qwzxv" in {w.word for w in res.marked_words_baseline}
    assert all(not math.isnan(w.z_score) for w in res.all_words)
    assert res.n_significant_words >= 1


# A pool of common English words; sampling both sides from it makes every
# frequent (register) word roughly balanced, yet with enough sampling noise that
# a weak prior lets some of them cross |z|=1.96 -- exactly what calibration must
# suppress. The pool is >40 words so the top-40 register mask stays a proper
# subset that never swallows the rare planted content words.
_COMMON_POOL = list(
    dict.fromkeys(
        [
            "the",
            "of",
            "and",
            "to",
            "in",
            "for",
            "is",
            "on",
            "that",
            "with",
            "this",
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
            "an",
            "my",
            "we",
            "he",
            "she",
            "they",
            "have",
            "has",
            "had",
            "was",
            "were",
            "will",
            "can",
            "could",
            "would",
            "should",
            "about",
            "into",
            "over",
            "under",
            "after",
            "before",
            "then",
            "than",
            "them",
            "our",
            "their",
            "his",
            "her",
            "its",
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "some",
            "many",
            "few",
            "more",
            "most",
            "other",
            "such",
            "only",
            "own",
            "same",
            "so",
            "no",
            "nor",
        ]
    )
)


def test_lexical_auto_calibration_suppresses_register_words():
    """INVARIANT PROOF of the Mickel auto-calibration contract, shown to be
    LOAD-BEARING: under a weak prior some register words cross |z|>=1.96, but at
    the auto-resolved constant C NO register word (stopword / contraction /
    top-English word) clears it -- the whole point of the calibration. The rare
    planted content words are correctly excluded from the register mask. A broken
    calibration (empty register mask, reversed binary search) fails this."""
    rng = np.random.default_rng(5)
    targ, base = [], []
    for _ in range(120):
        targ.append(" ".join(rng.choice(_COMMON_POOL, size=15)) + " zqxwv")
        base.append(" ".join(rng.choice(_COMMON_POOL, size=15)) + " qwzxv")
    # weak prior (large fixed C -> little shrinkage): register words are NOT yet
    # suppressed, so calibration genuinely has work to do (non-vacuous test).
    vocab_w, _dw, z_weak, _cw, _rcw = marked_words_z_scores(
        targ,
        base,
        min_word_count=2,
        reference_corpus="wordfreq:en",
        calibration_constant=5.0,
    )
    reg_w = _register_word_mask(vocab_w, "wordfreq:en")
    assert reg_w.sum() >= 20
    assert float(np.abs(z_weak[reg_w]).max()) >= _RAW_Z_THRESHOLD
    # auto calibration: the contract -> every register word is pulled under 1.96.
    vocab, _delta, z, _cwords, resolved_c = marked_words_z_scores(
        targ,
        base,
        min_word_count=2,
        reference_corpus="wordfreq:en",
        calibration_constant="auto",
    )
    reg = _register_word_mask(vocab, "wordfreq:en")
    assert reg.any()  # there ARE register words in this vocabulary
    assert _AUTO_C_RANGE[0] <= resolved_c <= _AUTO_C_RANGE[1]
    assert float(np.abs(z[reg]).max()) < _RAW_Z_THRESHOLD, dict(
        zip(np.asarray(vocab)[reg].tolist(), z[reg].round(2).tolist(), strict=True)
    )
    # the rare planted content words are NOT register (proper-subset mask)
    varr = np.asarray(vocab)
    assert not bool(reg[varr == "zqxwv"][0])
    assert not bool(reg[varr == "qwzxv"][0])


def test_lexical_null_split_has_no_false_discoveries():
    """MONTE-CARLO NULL: split ONE homogeneous corpus at random into two halves
    (same distribution) and demand ~zero BH-FDR discoveries across several
    splits. A mis-calibrated test (uncorrected z, wrong variance) would leak
    false marked words."""
    rng = np.random.default_rng(9)
    vocab_pool = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
    ]
    corpus = []
    for _ in range(160):
        k = rng.integers(6, 12)
        corpus.append(" ".join(rng.choice(vocab_pool, size=k)))
    total_sig = 0
    for _split in range(4):
        rng.shuffle(corpus)
        half = len(corpus) // 2
        a = _pset("a", GROUP_TARGET, corpus[:half])
        b = _pset("b", GROUP_BASELINE, corpus[half:])
        res = _lexical(a, b)
        assert all(not math.isnan(w.z_score) for w in res.all_words)
        total_sig += res.n_significant_words
    # 4 null splits over a 10-word vocab at FDR 0.05 -> expect ~0 discoveries
    assert total_sig <= 1, total_sig


@pytest.mark.parametrize(
    "case",
    [
        "empty_strings",
        "single_word",
        "all_identical",
        "unicode_emoji",
        "very_long",
        "one_prompt_side",
        "one_author_each",
    ],
)
def test_lexical_adversarial_inputs(case):
    """FAULT INJECTION across the pathological inputs the task names. Each must
    yield a sane result (finite z, p in [0,1], counts consistent) or a clear
    error -- never a crash, NaN, or silent wrong number."""
    if case == "empty_strings":
        t = _pset("t", GROUP_TARGET, ["", "", ""])
        b = _pset("b", GROUP_BASELINE, ["", ""])
    elif case == "single_word":
        t = _pset("t", GROUP_TARGET, ["cat"] * 5)
        b = _pset("b", GROUP_BASELINE, ["dog"] * 5)
    elif case == "all_identical":
        t = _pset("t", GROUP_TARGET, ["hello there world"] * 6)
        b = _pset("b", GROUP_BASELINE, ["hello there world"] * 6)
    elif case == "unicode_emoji":
        t = _pset("t", GROUP_TARGET, ["I love 🌈 pride 🏳️‍🌈 café naïve"] * 5)
        b = _pset("b", GROUP_BASELINE, ["plain ascii question here"] * 5)
    elif case == "very_long":
        t = _pset("t", GROUP_TARGET, [("lorem ipsum dolor " * 20000)])
        b = _pset("b", GROUP_BASELINE, ["short baseline text here"] * 3)
    elif case == "one_prompt_side":
        t = _pset("t", GROUP_TARGET, ["only one prompt on this side ever"])
        b = _pset("b", GROUP_BASELINE, ["baseline one two", "baseline three four"] * 4)
    else:  # one_author_each
        t = _pset(
            "t",
            GROUP_TARGET,
            ["alpha beta", "alpha gamma", "beta gamma"],
            author_ids=["A", "A", "A"],
        )
        b = _pset(
            "b", GROUP_BASELINE, ["delta epsilon", "delta zeta"], author_ids=["B", "B"]
        )
    res = _lexical(t, b)
    assert res.vocabulary_size >= 0
    for w in res.all_words:
        assert not math.isnan(w.z_score) and not math.isinf(w.z_score)
        assert 0.0 <= w.p_value <= 1.0
        assert 0.0 <= w.p_adjusted <= 1.0
        assert w.count_target >= 0 and w.count_baseline >= 0
    assert res.n_significant_words == sum(w.significant for w in res.all_words)
    # degenerate vocabularies (<2 words) must report an empty, not-crashed test
    if case in ("empty_strings",):
        assert res.vocabulary_size == 0
        assert res.n_significant_words == 0
        assert res.min_p_adjusted is None


def test_lexical_scale_4000_prompts(monkeypatch):
    """SCALE: ~4000 real prompts through compute_lexical. Completes quickly, no
    NaN, and reports counts consistent with the token totals."""
    ds = PromptDataset.load(REPO / "data" / "multi_value_aave")
    base_t = ds.prompt_set("value_p100").texts
    base_b = ds.prompt_set("baseline").texts
    # up-sample to ~2000/side; trailing "copy N" makes each raw string distinct
    # without polluting the [a-z]+ vocabulary (digits are not tokens).
    reps = 2000 // len(base_t) + 1
    texts_t = [f"{txt} copy {i}" for i in range(reps) for txt in base_t][:2000]
    texts_b = [f"{txt} copy {i}" for i in range(reps) for txt in base_b][:2000]
    t = _pset("t", GROUP_TARGET, texts_t)
    b = _pset("b", GROUP_BASELINE, texts_b)
    start = time.time()
    res = _lexical(t, b)
    elapsed = time.time() - start
    assert res.n_prompts_target == 2000 and res.n_prompts_baseline == 2000
    assert res.vocabulary_size > 50
    assert res.total_tokens_target > 0 and res.total_tokens_baseline > 0
    assert all(not math.isnan(w.z_score) for w in res.all_words)
    assert res.n_significant_words == sum(w.significant for w in res.all_words)
    assert elapsed < 120, elapsed


# ===========================================================================
# D. SYNTACTIC PROOFS (neurobiber, local model)
# ===========================================================================
def test_syntactic_valid_ranges_and_directional_sign():
    """No NaN anywhere; prevalences in [0,1]; log-odds sign agrees with the raw
    prevalence gap (>0 leans target); n_significant matches the flags; a feature
    absent from both sides gets exactly 0 log-odds (degenerate guard)."""
    ctx = PipelineContext(random_seed=0)
    cfg = SyntacticConfig(batch_size=16)
    # imperative/terse vs hedged/elaborated: a real stylistic contrast
    t = _pset(
        GROUP_TARGET,
        GROUP_TARGET,
        ["I was wondering if perhaps you might gently help me reflect on how I feel."]
        * 30,
    )
    b = _pset(
        GROUP_BASELINE,
        GROUP_BASELINE,
        ["List three facts. Do it now."] * 30,
    )
    res = compute_syntactic(t, b, cfg, ctx)
    assert len(res.feature_contrasts) == 96
    for c in res.feature_contrasts:
        assert not math.isnan(c.log_odds) and not math.isinf(c.log_odds)
        assert not math.isnan(c.z_score)
        assert 0.0 <= c.prevalence_target <= 1.0
        assert 0.0 <= c.prevalence_baseline <= 1.0
        assert 0.0 <= c.p_value <= 1.0
        assert 0.0 <= c.p_adjusted <= 1.0
        # sign of log-odds must track the prevalence direction
        if c.prevalence_target > c.prevalence_baseline:
            assert c.log_odds > 0
        elif c.prevalence_target < c.prevalence_baseline:
            assert c.log_odds < 0
        # features absent on both sides carry no contrast
        if c.count_target == 0 and c.count_baseline == 0:
            assert c.log_odds == 0.0
    assert res.n_significant_features == sum(
        c.significant for c in res.feature_contrasts
    )


@pytest.mark.parametrize("case", ["empty", "identical", "unicode", "one_prompt_side"])
def test_syntactic_adversarial_inputs(case):
    """FAULT INJECTION for the syntactic path: degenerate inputs must not crash
    or produce NaN feature statistics."""
    ctx = PipelineContext(random_seed=0)
    cfg = SyntacticConfig(batch_size=8)
    if case == "empty":
        t = _pset("t", GROUP_TARGET, ["", "", ""])
        b = _pset("b", GROUP_BASELINE, ["", ""])
    elif case == "identical":
        t = _pset("t", GROUP_TARGET, ["same exact sentence here"] * 4)
        b = _pset("b", GROUP_BASELINE, ["same exact sentence here"] * 4)
    elif case == "unicode":
        t = _pset("t", GROUP_TARGET, ["feeling 🌈 café naïve résumé"] * 4)
        b = _pset("b", GROUP_BASELINE, ["plain question mark"] * 4)
    else:  # one_prompt_side
        t = _pset("t", GROUP_TARGET, ["a single lonely prompt on this side"])
        b = _pset("b", GROUP_BASELINE, ["one", "two", "three"])
    res = compute_syntactic(t, b, cfg, ctx)
    assert len(res.feature_contrasts) == 96
    for c in res.feature_contrasts:
        assert not math.isnan(c.log_odds) and not math.isinf(c.log_odds)
        assert not math.isnan(c.z_score)
        assert 0.0 <= c.p_value <= 1.0


# ===========================================================================
# E. USAGE / ATTITUDES PROOFS (author-level Mann-Whitney)
# ===========================================================================
def _usage_sets(target_vals, baseline_vals, scale="llm_freq", prompts_per_author=1):
    """Build target/baseline PromptSets whose `scale` carries the given per-author
    values (each author repeated `prompts_per_author` times)."""

    def side(name, group, vals):
        texts, authors, contexts = [], [], []
        for ai, v in enumerate(vals):
            for _ in range(prompts_per_author):
                texts.append("some prompt text")
                authors.append(f"{name}_au{ai}")
                contexts.append(InteractionContext(**{scale: int(v)}))
        return _pset(name, group, texts, author_ids=authors, contexts=contexts)

    return side("t", GROUP_TARGET, target_vals), side(
        "b", GROUP_BASELINE, baseline_vals
    )


def test_usage_recovers_planted_scale_gap():
    """KNOWN-ANSWER: target authors all report llm_freq=8, baseline all report 1.
    The Mann-Whitney test must be significant after FDR, and the rank-biserial
    must be exactly -1 (target ranks strictly higher; the sign convention in the
    code says target-higher -> r<0)."""
    t, b = _usage_sets([8] * 8, [1] * 8)
    res = compute_usage(t, b, UsageConfig(), PipelineContext(0))
    test = next(s for s in res.scale_tests if s.scale == "llm_freq")
    assert test.n_target_authors == 8 and test.n_baseline_authors == 8
    assert test.mean_target == pytest.approx(8.0)
    assert test.mean_baseline == pytest.approx(1.0)
    assert test.p_value is not None and test.p_value < 0.01
    assert test.rank_biserial == pytest.approx(-1.0)
    assert test.significant is True
    assert res.n_significant_scales >= 1
    # scales nobody answered stay skipped, never faked as significant
    for other in res.scale_tests:
        if other.scale != "llm_freq":
            assert other.n_target_authors == 0
            assert other.significant in (None, False)


def test_usage_null_no_false_positive():
    """MONTE-CARLO-STYLE NULL: identical value distributions on both sides ->
    the scale is not flagged significant."""
    rng = np.random.default_rng(4)
    tv = rng.integers(1, 6, size=12).tolist()
    bv = rng.integers(1, 6, size=12).tolist()
    # force identical multisets so there is genuinely no effect
    bv = list(tv)
    t, b = _usage_sets(tv, bv, scale="satisfaction")
    res = compute_usage(t, b, UsageConfig(), PipelineContext(0))
    test = next(s for s in res.scale_tests if s.scale == "satisfaction")
    assert test.significant is False
    assert test.p_value is not None and test.p_value > 0.05


def test_usage_aggregates_at_author_level_not_prompt_level():
    """A survey answer repeated across an author's many prompts must count ONCE:
    n_*_authors reflects authors, not prompts. Prompt-level counting would fake
    independence and shrink the p-value."""
    # 4 authors/side but 10 prompts each -> author-level n must be 4, not 40.
    t, b = _usage_sets([7, 7, 7, 7], [2, 2, 2, 2], prompts_per_author=10)
    res = compute_usage(t, b, UsageConfig(), PipelineContext(0))
    test = next(s for s in res.scale_tests if s.scale == "llm_freq")
    assert test.n_target_authors == 4
    assert test.n_baseline_authors == 4
    assert res.n_prompts_target == 40 and res.n_prompts_baseline == 40


def test_usage_skips_underpowered_and_unrecorded():
    """FAULT INJECTION: (a) fewer than MIN_AUTHORS_PER_SIDE recorded authors ->
    the test is skipped (p_value None, significant None), never crashed;
    (b) all-zero (unrecorded) values -> no authors counted, still skipped."""
    # (a) only 2 authors/side, below the minimum of 3
    assert MIN_AUTHORS_PER_SIDE == 3
    t, b = _usage_sets([5, 5], [1, 1])
    res = compute_usage(t, b, UsageConfig(), PipelineContext(0))
    test = next(s for s in res.scale_tests if s.scale == "llm_freq")
    assert test.n_target_authors == 2
    assert test.p_value is None and test.significant is None
    # (b) values of 0 mean "not recorded": no author contributes any value
    t0, b0 = _usage_sets([0, 0, 0, 0], [0, 0, 0, 0])
    res0 = compute_usage(t0, b0, UsageConfig(), PipelineContext(0))
    test0 = next(s for s in res0.scale_tests if s.scale == "llm_freq")
    assert test0.n_target_authors == 0 and test0.n_baseline_authors == 0
    assert test0.p_value is None


# ===========================================================================
# F. SCALE: MiniLM linear C2ST end-to-end via compute_distributional
# ===========================================================================
def test_distributional_minilm_scale_end_to_end():
    """SCALE + INTEGRATION: ~4000 prompts embedded with MiniLM and run through
    the real compute_distributional linear-C2ST path (no API keys). Must
    complete in reasonable time with sane, NaN-free values and chance == the
    majority rate."""
    ds = PromptDataset.load(REPO / "data" / "multi_value_aave")
    base_t = ds.prompt_set("value_p100").texts
    base_b = ds.prompt_set("baseline").texts
    reps = 2000 // len(base_t) + 1
    # trailing "copy N" keeps each raw string unique so MiniLM truly embeds ~4000
    texts_t = [f"{txt} copy {i}" for i in range(reps) for txt in base_t][:2000]
    texts_b = [f"{txt} copy {i}" for i in range(reps) for txt in base_b][:2000]
    t = _pset("t", GROUP_TARGET, texts_t)
    b = _pset("b", GROUP_BASELINE, texts_b)
    cfg = DistributionalConfig(
        embedders=[MINILM],
        classifiers=["linear"],
        cv_folds=4,
        n_permutations=20,
    )
    ctx = PipelineContext(random_seed=0)
    start = time.time()
    res = compute_distributional(t, b, cfg, ctx)
    elapsed = time.time() - start
    ctx.cleanup()
    assert len(res.variants) == 1, res.skipped_variants
    v = res.variants[0]
    assert v.classifier == "linear"
    assert 0.0 <= v.accuracy <= 1.0 and not math.isnan(v.accuracy)
    assert v.p_value is not None and 0.0 < v.p_value <= 1.0
    assert v.chance_level == pytest.approx(0.5)  # balanced 2000 vs 2000
    assert len(v.heldout_scores) == 4000
    assert not any(math.isnan(s) for s in v.heldout_scores)
    assert len(v.null_accuracies) == 20
    assert all(0.0 <= a <= 1.0 for a in v.null_accuracies)
    assert elapsed < 180, elapsed
