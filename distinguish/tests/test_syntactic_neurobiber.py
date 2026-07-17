"""Proof-grade tests that the NeuroBiber syntactic extractor applies the model
correctly, and that the syntactic dimension's log-odds math is exact.

Strategy (each test is designed to FAIL if the code were subtly wrong, not just
to check that a float comes back):

1. GROUND TRUTH — an INDEPENDENT reimplementation of the reference library's
   documented prediction algorithm (Blablablab/neurobiber model card / README:
   ``chunk_text`` -> tokenize(max_length=512, truncation, padding) -> sigmoid ->
   ``> 0.5`` -> max-pool over chunks), loaded as a SEPARATE model instance and
   run PER TEXT (batch of one, so no cross-text padding). Our extractor batches
   and pads. Requiring bit-exact agreement proves tokenizer params, threshold
   direction, sigmoid, label ordering, output axis, AND correct attention
   masking of padding all match the reference.
2. SHAPE / RANGE — 96 features per prompt, strictly {0, 1}, integer dtype;
   active-count == row sum.
3. DEGENERATE — empty / whitespace / single-token inputs do not crash and stay
   binary; and (ground-truth for degenerate input) their vectors must match the
   reference library, which returns an all-zero vector for empty input.
4. KNOWN ANSWER — the smoothed log-odds / prevalence / z / per-prompt-count /
   BH-FDR pipeline is verified against fully hand-computed arithmetic on a tiny
   2-vs-2 case, with the neural extractor replaced by a fixed feature matrix so
   the statistics are proven independently of the model.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Model is fully cached locally; force offline so the suite is fast and does not
# depend on network reachability.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from numpy.typing import NDArray
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.common.prompt_set_schema import (
    GROUP_BASELINE,
    GROUP_TARGET,
    PromptRecord,
    PromptSet,
)
from src.common.run_config import SyntacticConfig
from src.syntactic import syntactic_dimension as sd
from src.syntactic.neurobiber_extractor import NeurobiberExtractor

MODEL_NAME = "Blablablab/neurobiber"
N_FEATURES = 96
_MAX_LEN = 512
_THRESHOLD = 0.5

# A handful of deliberately diverse sentences (question, passive, first-person
# with amplifiers, ultra-short, that-clause + prediction modal, quantifiers) so
# the reference vectors are non-trivial and genuinely vary feature-to-feature.
REAL_SENTENCES = [
    "Do you think the weather will be nice tomorrow?",
    "The report was written by the committee last week.",
    "I really loved that amazing and wonderful movie!",
    "Cats sleep.",
    "He said that she would probably come to the party.",
    "There are many quantifiers such as all, some, and several here.",
]

# Degenerate inputs. The reference library normalizes whitespace via
# ``text.strip().split()`` inside chunk_text, so empty / whitespace-only text
# yields NO chunks and therefore an all-zero feature vector.
DEGENERATE_INPUTS = ["", "   ", "\n\t ", "a", "Hi.", "?"]


# --------------------------------------------------------------------------- #
# Independent reference: the library's documented algorithm, verbatim.
# --------------------------------------------------------------------------- #
def _chunk_text(text: str, chunk_size: int = _MAX_LEN) -> list[str]:
    """Reference chunk_text (Blablablab/neurobiber README), verbatim behavior."""
    tokens = text.strip().split()
    if not tokens:
        return []
    return [
        " ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
    ]


def reference_predict(
    texts: list[str],
    tokenizer,
    model,
    device: str,
) -> NDArray[np.int64]:
    """Faithful reimplementation of NeuroBiber's intended prediction path.

    Each text is processed on its OWN (its chunks tokenized together, batch of
    one text). No cross-text padding, deliberately different from our batched
    extractor, so agreement also exercises correct padding/attention masking.
    """
    out = np.zeros((len(texts), N_FEATURES), dtype=np.int64)
    for row, text in enumerate(texts):
        chunks = _chunk_text(text)
        if not chunks:
            # No word tokens -> no chunks -> nothing detected -> all zeros.
            continue
        encoded = tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        preds = (torch.sigmoid(logits) > _THRESHOLD).cpu().numpy().astype(np.int64)
        out[row] = preds.max(axis=0)  # a feature fires if present in ANY chunk
    return out


# --------------------------------------------------------------------------- #
# Fixtures: load our extractor once and an independent reference model once.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def extractor():
    torch.manual_seed(0)
    np.random.seed(0)
    ex = NeurobiberExtractor(MODEL_NAME, batch_size=16)
    yield ex
    ex.cleanup()


@pytest.fixture(scope="module")
def reference(extractor):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(extractor.device)
    model.eval()
    id2label = model.config.id2label
    ordered_labels = [id2label[i] for i in range(len(id2label))]
    return tokenizer, model, ordered_labels


# --------------------------------------------------------------------------- #
# 1. GROUND TRUTH
# --------------------------------------------------------------------------- #
def test_label_order_matches_reference(extractor, reference):
    _, _, ordered_labels = reference
    assert len(extractor.feature_names) == N_FEATURES
    # Our extractor claims its columns are id2label[0..95] in order; verify it.
    assert extractor.feature_names == ordered_labels


def test_ground_truth_binary_vectors_match_reference(extractor, reference):
    """The core proof: our extractor == the library's algorithm, bit for bit."""
    tokenizer, model, _ = reference
    ours = extractor.extract(REAL_SENTENCES)
    ref = reference_predict(REAL_SENTENCES, tokenizer, model, extractor.device)

    assert ours.shape == (len(REAL_SENTENCES), N_FEATURES)
    assert ref.shape == ours.shape
    # Non-degenerate check: the reference is not trivially all-zero/all-one, so a
    # match is meaningful (guards against a broken test that always passes).
    assert 0 < int(ref.sum()) < ref.size

    mismatches = int((ours != ref).sum())
    if mismatches:
        diff_rows = np.where((ours != ref).any(axis=1))[0].tolist()
        details = "; ".join(
            f"sent[{r}] active ours={int(ours[r].sum())} ref={int(ref[r].sum())}"
            for r in diff_rows
        )
        pytest.fail(
            f"Extractor diverges from reference NeuroBiber on {mismatches} of "
            f"{ours.size} cells across rows {diff_rows}: {details}"
        )


# --------------------------------------------------------------------------- #
# 2. SHAPE / RANGE
# --------------------------------------------------------------------------- #
def test_shape_range_and_active_count(extractor):
    m = extractor.extract(REAL_SENTENCES)
    assert m.shape == (len(REAL_SENTENCES), N_FEATURES)
    assert np.issubdtype(m.dtype, np.integer)
    # Strictly binary: no probabilities leaked, no counts > 1.
    assert set(np.unique(m).tolist()).issubset({0, 1})
    active = m.sum(axis=1)
    # active-count is exactly the number of 1-cells in the row.
    manual = np.array([int((row == 1).sum()) for row in m])
    assert np.array_equal(active, manual)
    assert np.all(active >= 0) and np.all(active <= N_FEATURES)


# --------------------------------------------------------------------------- #
# 3. DEGENERATE
# --------------------------------------------------------------------------- #
def test_degenerate_does_not_crash_and_is_binary(extractor):
    d = extractor.extract(DEGENERATE_INPUTS)
    assert d.shape == (len(DEGENERATE_INPUTS), N_FEATURES)
    assert np.issubdtype(d.dtype, np.integer)
    assert set(np.unique(d).tolist()).issubset({0, 1})


def test_degenerate_matches_reference_library(extractor, reference):
    """Ground truth for degenerate input.

    The reference library returns an all-zero vector for empty / whitespace-only
    text (chunk_text yields no chunks). A faithful application of the model must
    do the same: an empty prompt has no words and so no stylistic features.
    """
    tokenizer, model, _ = reference
    ours = extractor.extract(DEGENERATE_INPUTS)
    ref = reference_predict(DEGENERATE_INPUTS, tokenizer, model, extractor.device)

    report = "; ".join(
        f"{text!r}: ours={int(ours[i].sum())} ref={int(ref[i].sum())}"
        for i, text in enumerate(DEGENERATE_INPUTS)
    )
    assert np.array_equal(ours, ref), (
        "Extractor does not reproduce the reference library's handling of "
        f"degenerate input. Per-input active counts -> {report}"
    )


# Realistic multi-line / irregular-whitespace prompts. These PASS
# PromptSet.validate() (non-empty stripped text), so they DO reach the extractor
# in the live pipeline -- unlike the empty/whitespace degenerate inputs above.
INTERNAL_WHITESPACE_PROMPTS = [
    "def add(a, b):\n    return a + b\n\nExplain    what   this does.",
    "Hello   there,   how    are   you?",
]


def test_internal_whitespace_matches_reference(extractor, reference):
    """Realistic prompts with newlines / repeated spaces must still match.

    The reference normalizes whitespace inside chunk_text
    (``text.strip().split()`` then single-space rejoin) BEFORE tokenizing; our
    extractor tokenizes the raw string. For prompts containing runs of spaces,
    tabs, or newlines -- ordinary in multi-line chatbot prompts, code, or lists,
    all of which pass dataset validation -- the two paths must still agree if the
    extractor faithfully applies the model as the library intends.
    """
    tokenizer, model, _ = reference
    ours = extractor.extract(INTERNAL_WHITESPACE_PROMPTS)
    ref = reference_predict(
        INTERNAL_WHITESPACE_PROMPTS, tokenizer, model, extractor.device
    )
    per_prompt = "; ".join(
        f"prompt[{i}] mismatch_cells={int((ours[i] != ref[i]).sum())} "
        f"(ours_active={int(ours[i].sum())}, ref_active={int(ref[i].sum())})"
        for i in range(len(INTERNAL_WHITESPACE_PROMPTS))
    )
    assert np.array_equal(ours, ref), (
        "Extractor diverges from reference NeuroBiber on realistic "
        f"internal-whitespace prompts (these reach the live pipeline): {per_prompt}"
    )


# --------------------------------------------------------------------------- #
# 4. KNOWN ANSWER: log-odds / prevalence / z / FDR on a hand-computed 2-vs-2.
# --------------------------------------------------------------------------- #
_FAKE_FEATURES = ["BIN_A", "BIN_B", "BIN_C", "BIN_D"]
_FAKE_MATRIX = {
    "t-prompt-0": [1, 1, 0, 0],
    "t-prompt-1": [1, 0, 0, 0],
    "b-prompt-0": [0, 0, 1, 0],
    "b-prompt-1": [0, 0, 1, 0],
}


class _FakeExtractor:
    """Drop-in for NeurobiberExtractor returning a fixed, known feature matrix."""

    def __init__(self, model_name: str, batch_size: int = 16):
        self.feature_names = list(_FAKE_FEATURES)

    def extract(self, texts: list[str]) -> NDArray[np.int64]:
        return np.array([_FAKE_MATRIX[t] for t in texts], dtype=np.int64)

    def cleanup(self) -> None:  # pragma: no cover - trivial
        pass


def _make_set(name: str, group: str, texts: list[str]) -> PromptSet:
    return PromptSet(
        name=name,
        group=group,
        prompts=[
            PromptRecord(prompt_id=f"{name}-{i}", author_id=f"{name}-auth", text=t)
            for i, t in enumerate(texts)
        ],
    )


def _hand_log_odds(ct: float, cb: float, nt: int, nb: int, s: float) -> float:
    return math.log(((ct + s) / (nt - ct + s)) / ((cb + s) / (nb - cb + s)))


def _hand_se(ct: float, cb: float, nt: int, nb: int, s: float) -> float:
    return math.sqrt(
        1.0 / (ct + s) + 1.0 / (nt - ct + s) + 1.0 / (cb + s) + 1.0 / (nb - cb + s)
    )


def _two_sided_p(z: float) -> float:
    # Independent of scipy: 2*(1 - Phi(|z|)) == erfc(|z|/sqrt(2)).
    return math.erfc(abs(z) / math.sqrt(2.0))


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Independent Benjamini-Hochberg adjustment (matches scipy's method='bh')."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    ranked = [pvals[i] for i in order]
    adj_sorted = [min(1.0, ranked[k] * m / (k + 1)) for k in range(m)]
    # Enforce monotonic non-decreasing from the largest rank downward.
    for k in range(m - 2, -1, -1):
        adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    out = [0.0] * m
    for k, i in enumerate(order):
        out[i] = adj_sorted[k]
    return out


def test_log_odds_known_answer(monkeypatch):
    monkeypatch.setattr(sd, "NeurobiberExtractor", _FakeExtractor)

    target = _make_set("tgt", GROUP_TARGET, ["t-prompt-0", "t-prompt-1"])
    baseline = _make_set("base", GROUP_BASELINE, ["b-prompt-0", "b-prompt-1"])
    config = SyntacticConfig()  # smoothing_count=0.5, fdr_alpha=0.05
    s, nt, nb = config.smoothing_count, 2, 2

    result = sd.compute_syntactic(target, baseline, config, context=None)

    # Ground-truth counts from the fixed matrix.
    counts = {"BIN_A": (2, 0), "BIN_B": (1, 0), "BIN_C": (0, 2), "BIN_D": (0, 0)}
    by_name = {c.feature_name: c for c in result.feature_contrasts}
    assert set(by_name) == set(counts)
    assert len(result.feature_contrasts) == 4

    # Per-prompt active-feature counts == row sums of the matrix.
    assert result.features_per_prompt_target == [2, 1]
    assert result.features_per_prompt_baseline == [1, 1]
    assert result.n_prompts_target == 2 and result.n_prompts_baseline == 2

    exp_p = {}
    for name, (ct, cb) in counts.items():
        c = by_name[name]
        assert c.count_target == ct and c.count_baseline == cb
        assert c.prevalence_target == pytest.approx(ct / nt)
        assert c.prevalence_baseline == pytest.approx(cb / nb)

        degenerate = (ct == 0 and cb == 0) or (ct == nt and cb == nb)
        if degenerate:
            # Absent-in-both must be zeroed out (no spurious contrast).
            assert c.log_odds == 0.0
            expected_z = 0.0
        else:
            expected_lo = _hand_log_odds(ct, cb, nt, nb, s)
            assert c.log_odds == pytest.approx(expected_lo, rel=1e-9)
            expected_z = expected_lo / _hand_se(ct, cb, nt, nb, s)
        assert c.z_score == pytest.approx(expected_z, rel=1e-9, abs=1e-12)
        assert c.p_value == pytest.approx(_two_sided_p(expected_z), rel=1e-9)
        exp_p[name] = c.p_value

    # Explicit hand anchors (independent of the helper formulas above):
    #   BIN_A: ct=2,cb=0 -> log((2.5/0.5)/(0.5/2.5)) = log(25).
    #   BIN_C: ct=0,cb=2 -> -log(25) (exact mirror).
    assert by_name["BIN_A"].log_odds == pytest.approx(math.log(25.0), rel=1e-9)
    assert by_name["BIN_C"].log_odds == pytest.approx(-math.log(25.0), rel=1e-9)
    assert by_name["BIN_A"].log_odds == pytest.approx(-by_name["BIN_C"].log_odds)
    assert by_name["BIN_D"].p_value == pytest.approx(1.0)

    # BH-FDR adjusted p-values and significance, recomputed independently.
    names_in_order = [c.feature_name for c in result.feature_contrasts]
    p_in_order = [exp_p[n] for n in names_in_order]
    expected_adj = _bh_adjust(p_in_order)
    for c, padj in zip(result.feature_contrasts, expected_adj, strict=True):
        assert c.p_adjusted == pytest.approx(padj, rel=1e-9, abs=1e-12)
        assert c.significant == (padj <= config.fdr_alpha)
    # With n=2 per side nothing is significant; the count must reflect that.
    assert result.n_significant_features == sum(
        1 for c in result.feature_contrasts if c.significant
    )
    assert result.n_significant_features == 0

    # Contrasts are sorted by |log_odds| descending (BIN_D, the zeroed one, last).
    abs_lo = [abs(c.log_odds) for c in result.feature_contrasts]
    assert abs_lo == sorted(abs_lo, reverse=True)
    assert result.feature_contrasts[-1].feature_name == "BIN_D"
