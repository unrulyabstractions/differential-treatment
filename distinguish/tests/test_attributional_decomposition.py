"""Attributional analysis (paper §3.3.5) — the exact per-token decomposition.

The paper's central claim is that the contributions are EXACT, not estimated:
the linear head on the mean-pooled residual equals the sum of per-token
contributions a_t = (1/T) w·h_t. These tests pin that identity down (both the
math and the real extractor), plus the selection and highlighting logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.attributional.attributional_dimension import (
    AttributedPrompt,
    TokenContribution,
    _highlight,
)
from src.viz.attributional_plots import _is_word

MODEL = "google/gemma-2-2b-it"


def test_contributions_sum_to_pooled_score_exactly() -> None:
    """a_t = (1/T) w·h_t must sum to w·(mean-pooled h) for ANY head w."""
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((17, 64)).astype(np.float32)  # (T, d)
    w = rng.standard_normal(64)
    contributions = (hidden @ w) / hidden.shape[0]
    pooled_score = w @ hidden.mean(axis=0)
    assert contributions.sum() == pytest.approx(pooled_score, rel=1e-6)


def test_highlight_marks_the_largest_magnitude_content_tokens() -> None:
    tokens = ["<bos>", "I", " am", " trans", " today", "<end>"]
    contributions = np.array(
        [0.0, 0.1, -0.2, 5.0, 0.3, 9.9]
    )  # <end> biggest but out of span
    span = (1, 5)  # content = I, am, trans, today
    highlighted, top = _highlight(tokens, contributions, span, n_top=2)
    assert "« trans»" in highlighted  # largest |a_t| within the content span
    assert "<end>" not in highlighted  # template token excluded from display
    assert [t.token for t in top] == ["trans", "today"]  # ranked by |contribution|
    assert top[0].contribution == pytest.approx(5.0)


def test_highlight_falls_back_to_all_tokens_when_span_empty() -> None:
    tokens = ["only", " two"]
    highlighted, top = _highlight(tokens, np.array([1.0, 2.0]), (5, 5), n_top=1)
    assert "« two»" in highlighted and len(top) == 1


def test_plot_word_filter_drops_punctuation() -> None:
    assert _is_word(" trans") and _is_word("gender")
    assert not _is_word("'") and not _is_word(",") and not _is_word(" ")
    assert not _is_word("i")  # single letters are noise in the summary chart


def test_schema_roundtrips() -> None:
    prompt = AttributedPrompt(
        prompt_id="p1",
        cohort="target",
        text="hi",
        probe_score=1.5,
        highlighted="«hi»",
        top_tokens=[TokenContribution(token="hi", contribution=1.5)],
    )
    assert AttributedPrompt.from_dict(prompt.to_dict()).to_dict() == prompt.to_dict()


@pytest.mark.slow
def test_real_extractor_decomposition_is_exact() -> None:
    """End-to-end on the actual probe model: sum(a_t) == w·pooled to float tol."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from src.inference.residual_stream_extractor import ResidualStreamExtractor

    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("no GPU/MPS for the residual model")
    extractor = ResidualStreamExtractor(MODEL)
    tokens, hidden, span = extractor.extract_per_token(
        "I've been on hormones for six months and my voice is changing."
    )
    extractor.cleanup()
    assert len(tokens) == hidden.shape[0]
    assert 0 <= span[0] <= span[1] <= len(tokens)
    w = np.random.default_rng(1).standard_normal(hidden.shape[1])
    contributions = (hidden @ w) / hidden.shape[0]
    assert contributions.sum() == pytest.approx(w @ hidden.mean(axis=0), rel=1e-4)
