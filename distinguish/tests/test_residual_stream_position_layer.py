"""PROOF-GRADE ground-truth test for the residual-stream extractor.

Target: ``src/inference/residual_stream_extractor.py``
        (``ResidualStreamExtractor``).

This is the CRITICAL test: it proves the extractor pulls activations at the
RIGHT TOKEN POSITION (the change-of-turn / assistant-turn-start sentinel) and
the RIGHT LAYER (~75 % depth), by comparing against an INDEPENDENT ground truth
rather than merely checking that a float comes back.

Strategy
--------
Model: ``meta-llama/Llama-3.2-1B-Instruct`` (16 layers, cached locally). The
extractor is forced onto CPU/float32 so the forward pass is deterministic and a
tight ``allclose`` is meaningful.

1. TOKEN POSITION.  We independently rebuild the chat-formatted input, tokenize
   it, and assert its tail is *exactly* the Llama assistant-turn-start markup
   ``<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`` — decoding and
   printing every extracted token — then assert the extractor's private position
   logic selects precisely those positions (and NOT the last-k fallback).

2. LAYER.  We independently recompute ``round(0.75 * n_layers)`` and assert the
   extractor chose it, and that ``hidden_states`` has ``n_layers + 1`` entries so
   that index really is block-12's output (12/16 = 75 % depth).

3. GROUND-TRUTH MATCH.  We run the model ourselves with
   ``output_hidden_states=True``, manually index ``hidden_states[layer][0]`` at
   the change-of-turn positions, mean-pool, and assert it EQUALS the extractor's
   output (``allclose``).

4. FAULT INJECTION.  We prove the match above is discriminating: computing at the
   WRONG layer (11 or 13) or over the WRONG positions (user content / all tokens)
   yields a vector that is NOT allclose to the extractor's output. So a subtly
   wrong extractor would be caught, not silently passed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import src.inference.residual_stream_extractor as rse
from src.inference.residual_stream_extractor import ResidualStreamExtractor

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER_FRACTION = 0.75

# Ground-truth assistant-turn-start markup for the Llama-3 chat template. This is
# an INDEPENDENT source of truth (the model's documented turn structure), not
# derived from the extractor's own sentinel trick.
EXPECTED_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

TEXTS = [
    "Explain gravity briefly.",
    "Write a haiku about the ocean.",
    "What is 2 + 2, and why?",
    "def foo(x):\n    return x * 2",
    "Translate 'good morning' into French.",
]


@pytest.fixture(scope="module")
def extractor():
    """Load the extractor once, forced onto CPU/float32 for a clean ground truth."""
    orig = rse.get_torch_device
    rse.get_torch_device = lambda: "cpu"
    try:
        # This suite ground-truths the change-of-turn extraction path explicitly
        # (the default is now all_prompt pooling).
        ext = ResidualStreamExtractor(
            MODEL_NAME,
            layer_fraction=LAYER_FRACTION,
            positions_mode="change_of_turn",
        )
    except Exception as exc:  # pragma: no cover - environment/network issue only
        rse.get_torch_device = orig
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")
    finally:
        rse.get_torch_device = orig
    torch.manual_seed(0)
    assert str(next(ext.model.parameters()).device) == "cpu"
    assert next(ext.model.parameters()).dtype == torch.float32
    yield ext
    ext.cleanup()


# --------------------------------------------------------------------------- #
# Independent re-derivations (do NOT call the extractor's private logic).
# --------------------------------------------------------------------------- #
def _build_input_ids(tok, text: str):
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tok(formatted, add_special_tokens=False, return_tensors="pt").input_ids
    return formatted, input_ids


def _independent_change_of_turn_positions(tok, input_ids) -> list[int]:
    """Positions of the change-of-turn markup, derived from the KNOWN Llama suffix.

    Independent of the extractor: we tokenize the documented assistant-turn-start
    string on its own and require the full sequence to end with exactly those ids.
    """
    suffix_ids = tok(EXPECTED_SUFFIX, add_special_tokens=False).input_ids
    seq = input_ids[0].tolist()
    assert seq[-len(suffix_ids) :] == suffix_ids, (
        "sequence tail does not match the standalone-tokenized change-of-turn "
        f"suffix; tail={seq[-len(suffix_ids) :]} suffix={suffix_ids}"
    )
    seq_len = len(seq)
    return list(range(seq_len - len(suffix_ids), seq_len))


# --------------------------------------------------------------------------- #
# 1. LAYER: ~75 % depth.
# --------------------------------------------------------------------------- #
def test_layer_index_is_75pct_depth(extractor):
    n = extractor.model.config.num_hidden_layers
    expected = max(1, min(n, round(LAYER_FRACTION * n)))

    # Concrete, model-specific ground truth: 16 layers -> layer 12.
    assert n == 16, f"expected Llama-3.2-1B to have 16 layers, got {n}"
    assert expected == 12
    assert extractor.layer_fraction == LAYER_FRACTION
    assert extractor.layer_index == expected == round(0.75 * n)
    assert 0 < extractor.layer_index <= n

    # Confirm hidden_states indexing convention: [0]=embeddings, [i]=block i out,
    # so len == n_layers + 1 and index 12 is exactly 12/16 = 75 % depth.
    _, input_ids = _build_input_ids(extractor.tokenizer, TEXTS[0])
    with torch.inference_mode():
        out = extractor.model(input_ids, output_hidden_states=True)
    assert len(out.hidden_states) == n + 1
    assert extractor.layer_index / n == pytest.approx(0.75)
    print(
        f"[LAYER] n_layers={n} layer_index={extractor.layer_index} "
        f"depth={extractor.layer_index / n:.3f} hidden_states={len(out.hidden_states)}"
    )


# --------------------------------------------------------------------------- #
# 2. TOKEN POSITION: the change-of-turn / assistant-turn-start sentinel.
# --------------------------------------------------------------------------- #
def test_extraction_positions_are_assistant_turn_start(extractor):
    tok = extractor.tokenizer
    suffix_ids = tok(EXPECTED_SUFFIX, add_special_tokens=False).input_ids

    for text in TEXTS:
        formatted, input_ids = _build_input_ids(tok, text)
        seq_len = input_ids.shape[1]

        expected_positions = _independent_change_of_turn_positions(tok, input_ids)
        code_positions = extractor._change_of_turn_positions(formatted, seq_len)

        # (a) The code's positions match the independent change-of-turn derivation.
        assert code_positions == expected_positions, (
            f"code positions {code_positions} != independent {expected_positions}"
        )

        # (b) It is NOT the last-k fallback: full 5-token change-of-turn span.
        assert len(code_positions) == len(suffix_ids) == 5
        assert code_positions == list(range(seq_len - 5, seq_len))

        # (c) Decode the extracted tokens and assert they ARE the assistant-turn-
        #     start sentinel, and that the token just before is user content, not
        #     part of the turn markup.
        decoded_tokens = [tok.decode([input_ids[0, p].item()]) for p in code_positions]
        assert "".join(decoded_tokens) == EXPECTED_SUFFIX
        assert "assistant" in decoded_tokens
        assert decoded_tokens[0] == "<|eot_id|>"  # change-of-turn boundary
        before = tok.decode([input_ids[0, code_positions[0] - 1].item()])
        assert before not in ("<|eot_id|>", "<|start_header_id|>")
        print(
            f"[POS] text={text!r:40} positions={code_positions} "
            f"tokens={decoded_tokens} before={before!r}"
        )


# --------------------------------------------------------------------------- #
# 3. GROUND-TRUTH MATCH: manual hidden-state index == extractor output.
# --------------------------------------------------------------------------- #
def test_extractor_matches_manual_hidden_state(extractor):
    tok = extractor.tokenizer
    n = extractor.model.config.num_hidden_layers
    layer = round(LAYER_FRACTION * n)  # 12, derived independently
    assert layer == extractor.layer_index

    max_diff_overall = 0.0
    for text in TEXTS:
        _, input_ids = _build_input_ids(tok, text)
        positions = _independent_change_of_turn_positions(tok, input_ids)

        with torch.inference_mode():
            out = extractor.model(input_ids, output_hidden_states=True)
        manual = (
            out.hidden_states[layer][0][positions].mean(dim=0).float().cpu().numpy()
        )

        got = extractor.extract([text])[0]

        assert got.shape == manual.shape == (extractor.model.config.hidden_size,)
        assert got.dtype == np.float32
        assert np.all(np.isfinite(got))
        max_diff = float(np.max(np.abs(got - manual)))
        max_diff_overall = max(max_diff_overall, max_diff)
        assert np.allclose(got, manual, atol=1e-4, rtol=1e-4), (
            f"ground-truth mismatch for {text!r}: max|diff|={max_diff:.3e}"
        )
    print(
        f"[MATCH] worst max|extractor - manual| over {len(TEXTS)} texts = "
        f"{max_diff_overall:.3e}"
    )
    assert max_diff_overall < 1e-3


# --------------------------------------------------------------------------- #
# 4a. FAULT INJECTION: wrong LAYER would be caught.
# --------------------------------------------------------------------------- #
def test_wrong_layer_is_distinguishable(extractor):
    tok = extractor.tokenizer
    n = extractor.model.config.num_hidden_layers
    correct = round(LAYER_FRACTION * n)  # 12

    text = TEXTS[0]
    _, input_ids = _build_input_ids(tok, text)
    positions = _independent_change_of_turn_positions(tok, input_ids)
    with torch.inference_mode():
        out = extractor.model(input_ids, output_hidden_states=True)
    got = extractor.extract([text])[0]

    checked = 0
    for wrong in (correct - 1, correct + 1):
        if not (0 < wrong <= n):
            continue
        wrong_vec = (
            out.hidden_states[wrong][0][positions].mean(dim=0).float().cpu().numpy()
        )
        diff = float(np.max(np.abs(got - wrong_vec)))
        # Adjacent layers must be clearly distinct from the correct one, so the
        # allclose(atol=1e-4) match in the ground-truth test genuinely pins the
        # layer rather than passing by coincidence.
        assert not np.allclose(got, wrong_vec, atol=1e-4, rtol=1e-4)
        assert diff > 1e-2, (
            f"layer {wrong} indistinguishable from {correct}: {diff:.3e}"
        )
        print(f"[WRONG-LAYER] layer {wrong} vs {correct}: max|diff|={diff:.3e}")
        checked += 1
    assert checked >= 1


# --------------------------------------------------------------------------- #
# 4b. FAULT INJECTION: wrong POSITION would be caught.
# --------------------------------------------------------------------------- #
def test_wrong_position_is_distinguishable(extractor):
    tok = extractor.tokenizer
    n = extractor.model.config.num_hidden_layers
    layer = round(LAYER_FRACTION * n)

    text = TEXTS[0]
    _, input_ids = _build_input_ids(tok, text)
    seq_len = input_ids.shape[1]
    positions = _independent_change_of_turn_positions(tok, input_ids)
    with torch.inference_mode():
        out = extractor.model(input_ids, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]
    got = extractor.extract([text])[0]

    # Wrong window 1: the user-content tokens strictly BEFORE the change-of-turn.
    user_positions = list(range(0, positions[0]))
    assert user_positions and user_positions[-1] < positions[0]
    user_vec = hidden[user_positions].mean(dim=0).float().cpu().numpy()
    d_user = float(np.max(np.abs(got - user_vec)))

    # Wrong window 2: mean over ALL positions.
    all_vec = hidden[list(range(seq_len))].mean(dim=0).float().cpu().numpy()
    d_all = float(np.max(np.abs(got - all_vec)))

    # Wrong window 3: only the final token, not the whole change-of-turn span.
    last_vec = hidden[[seq_len - 1]].mean(dim=0).float().cpu().numpy()
    d_last = float(np.max(np.abs(got - last_vec)))

    for name, vec, diff in (
        ("user-content", user_vec, d_user),
        ("all-tokens", all_vec, d_all),
        ("last-token", last_vec, d_last),
    ):
        assert not np.allclose(got, vec, atol=1e-4, rtol=1e-4)
        assert diff > 1e-2, f"{name} indistinguishable from change-of-turn: {diff:.3e}"
        print(f"[WRONG-POS] {name}: max|diff|={diff:.3e}")


# --------------------------------------------------------------------------- #
# 5. Output contract.
# --------------------------------------------------------------------------- #
def test_output_shape_and_dtype(extractor):
    out = extractor.extract(TEXTS[:3])
    assert out.shape == (3, extractor.model.config.hidden_size)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Distinct prompts should give distinct embeddings (not a constant vector).
    assert not np.allclose(out[0], out[1])
