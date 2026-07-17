"""Proof-grade fault-injection tests for the markedness/codedness annotator.

These tests PROVE the annotation module (`src/annotation/markedness_codedness_
annotator.py`) and its CLI (`scripts/annotate_prompt_set.py`) recover from faults
and NEVER silently misassign a label, WITHOUT touching a live API: every OpenAI
call is served by a monkeypatched fake client.

The proof strategy is fault injection + planted known-answer + ground-truth
alignment, run as a Monte-Carlo campaign so a subtly-wrong implementation would
fail:

  1. TRANSIENT — a fake reply that is malformed/misaligned ONCE then valid: the
     batch retry must recover with the CORRECT labels (and must actually re-issue
     the API call).
  2. REORDER — the fake returns the indexed objects in shuffled order (correct
     "i", wrong position). Over 300 random permutations (single- and multi-batch)
     labels must re-bind to the right prompt by index, never by position.
  3. DROP / DUP / EXTRA / WRONG-INDEX / None-content / bad-field — over 300 random
     corruptions the batch must raise a clear ValueError after retries are
     exhausted, never an AttributeError/KeyError and never a silent wrong label.
  4. END-TO-END CLI — run the CLI's `main()` over data/synthetic with a fake
     client that plants a distinctive text->label mapping; the written parquet
     must round-trip through PromptDataset.load with markedness in {0,1},
     codedness in [0,1], each label bound to the RIGHT prompt_id, other cohorts
     untouched.

A planted label is a pure, collision-resistant function of the prompt TEXT
(`fake_labels`), so any misbinding (swap, drift, cross-batch off-by-one) produces
an observable mismatch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd
import pytest

import src.annotation.markedness_codedness_annotator as annot
from src.annotation.markedness_codedness_annotator import (
    PromptLabelEstimate,
    annotate_markedness_codedness,
)
from src.common.dataset_tables import PromptDataset

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Planted ground-truth label: a pure, collision-resistant function of the text.
# markedness in {0,1}, codedness in [0,1], rationale UNIQUE per distinct text so
# any misbinding is detectable on at least one field.
# --------------------------------------------------------------------------- #
def fake_labels(text: str) -> tuple[int, float, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    h = int(digest, 16)
    markedness = h % 2
    codedness = round((h % 1000) / 999.0, 4)  # in [0.0, 1.0]
    rationale = f"why::{digest[:20]}"  # unique per distinct text
    return markedness, codedness, rationale


def make_texts(n: int, tag: str) -> list[str]:
    """n distinct prompt texts."""
    return [f"[{tag}] prompt #{k} unique-token-{k}-{tag} sample body" for k in range(n)]


# --------------------------------------------------------------------------- #
# Fake OpenAI client. `annotate_markedness_codedness` calls `OpenAI()` with no
# args, so we monkeypatch the module symbol to a zero-arg factory. The client is
# used exactly as `client.chat.completions.create(model=, messages=)` and the
# reply is read as `response.choices[0].message.content`.
# --------------------------------------------------------------------------- #
class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responder: Responder) -> None:
        self._responder = responder

    def create(self, *, model, messages, **kwargs):
        return self._responder(model, messages)


class _Chat:
    def __init__(self, responder: Responder) -> None:
        self.completions = _Completions(responder)


class FakeOpenAI:
    def __init__(self, responder: Responder) -> None:
        self.chat = _Chat(responder)


class Responder:
    """Serves fake completions. `transform(call_index, items)` returns the reply
    content string (or None) or raises; `items` is the parsed user payload
    (list of {"i", "t"}). Tracks call count to prove retries actually re-call."""

    def __init__(self, transform) -> None:
        self._transform = transform
        self.calls = 0

    def __call__(self, model, messages):
        self.calls += 1
        items = json.loads(messages[1]["content"])
        result = self._transform(self.calls, items)
        if isinstance(result, _Response) or result is None or isinstance(result, str):
            return result if isinstance(result, _Response) else _Response(result)
        raise AssertionError("transform returned an unexpected type")


def install(monkeypatch, transform) -> Responder:
    responder = Responder(transform)
    monkeypatch.setattr(annot, "OpenAI", lambda: FakeOpenAI(responder))
    return responder


def valid_reply(items, order=None, fence=None) -> str:
    """A correct, complete reply echoing each item's index, optionally reordered
    and/or wrapped in code fences."""
    reply = []
    for it in items:
        m, c, why = fake_labels(it["t"])
        reply.append({"i": it["i"], "m": m, "c": c, "why": why})
    if order is not None:
        reply = [reply[j] for j in order]
    body = json.dumps(reply, ensure_ascii=False)
    if fence == "json":
        return "```json\n" + body + "\n```"
    if fence == "plain":
        return "```\n" + body + "\n```"
    return body


def assert_matches_planted(estimates: list[PromptLabelEstimate], texts: list[str]):
    """Every estimate is bound to the text at the SAME position (input order)."""
    assert len(estimates) == len(texts)
    for est, text in zip(estimates, texts, strict=True):
        m, c, why = fake_labels(text)
        assert est.markedness == m, f"markedness misbound for {text!r}"
        assert est.codedness == pytest.approx(c), f"codedness misbound for {text!r}"
        assert est.rationale == why, f"rationale misbound for {text!r}"


# =========================================================================== #
# 0. KNOWN-ANSWER: hand-written batch, exact recovery (belt-and-suspenders).
# =========================================================================== #
def test_known_answer_exact_recovery(monkeypatch):
    texts = make_texts(7, "ka")
    install(monkeypatch, lambda call, items: valid_reply(items))
    estimates = annotate_markedness_codedness(texts, model_name="fake")
    assert_matches_planted(estimates, texts)
    # And in the plain input order, first estimate corresponds to first text.
    m0, _c0, why0 = fake_labels(texts[0])
    assert (estimates[0].markedness, estimates[0].rationale) == (m0, why0)


# =========================================================================== #
# 1. TRANSIENT FAILURE: malformed/misaligned ONCE, then valid -> recovers.
# =========================================================================== #
@pytest.mark.parametrize(
    "bad",
    [
        "not valid json at all {[",  # JSONDecodeError (subclass of ValueError)
        "[]",  # valid JSON but wrong count -> ValueError
        '[{"i": 0, "m": 5, "c": 0.1, "why": "x"}]',  # bad field -> ValueError
        None,  # empty/non-text reply -> ValueError
    ],
)
def test_transient_malformed_then_valid_recovers(monkeypatch, bad):
    texts = make_texts(12, "transient")  # single batch (<=30) => 2 calls total

    def transform(call, items):
        if call == 1:
            return bad
        return valid_reply(items)

    responder = install(monkeypatch, transform)
    estimates = annotate_markedness_codedness(texts, model_name="fake")
    assert_matches_planted(estimates, texts)
    # The retry must have actually RE-ISSUED the API call (not silently reused).
    assert responder.calls == 2


def test_transient_recovers_through_code_fences(monkeypatch):
    """Recovery also works when the valid retry is wrapped in ```json fences."""
    texts = make_texts(9, "fence")

    def transform(call, items):
        if call == 1:
            return "```json\ngarbage\n```"  # fenced but not JSON -> ValueError
        return valid_reply(items, fence="json")

    responder = install(monkeypatch, transform)
    estimates = annotate_markedness_codedness(texts, model_name="fake")
    assert_matches_planted(estimates, texts)
    assert responder.calls == 2


def test_both_fence_styles_parse(monkeypatch):
    for style in ("json", "plain", None):
        texts = make_texts(6, f"f-{style}")
        install(monkeypatch, lambda call, items, s=style: valid_reply(items, fence=s))
        estimates = annotate_markedness_codedness(texts, model_name="fake")
        assert_matches_planted(estimates, texts)


def test_exhausting_retries_raises_valueerror(monkeypatch):
    """Malformed on BOTH attempts -> a clear ValueError, not an infinite loop."""
    texts = make_texts(5, "exhaust")
    responder = install(monkeypatch, lambda call, items: "still not json {[")
    with pytest.raises(ValueError):
        annotate_markedness_codedness(texts, model_name="fake")
    assert responder.calls == annot._MAX_ATTEMPTS  # exactly the bounded retries


# =========================================================================== #
# 1b. API-CALL EXCEPTION boundary (documents the deliberate, codebase-wide
#     design: the create() call sits OUTSIDE the try, so a raised exception is
#     delegated to the OpenAI SDK's own max_retries, not the app loop). The
#     contract we PIN: such an exception surfaces cleanly and is NEVER swallowed
#     into a wrong/empty label. It must NOT be masked, and it must NOT recover to
#     bogus data.
# =========================================================================== #
def test_api_exception_is_not_swallowed(monkeypatch):
    texts = make_texts(8, "raise")
    sentinel = RuntimeError("simulated transient 503 from provider")

    def transform(call, items):
        raise sentinel

    install(monkeypatch, transform)
    with pytest.raises(RuntimeError) as excinfo:
        annotate_markedness_codedness(texts, model_name="fake")
    # The real error surfaces verbatim (not masked into an AttributeError or a
    # silently-empty result).
    assert excinfo.value is sentinel


# =========================================================================== #
# 2. REORDER: shuffled-but-correctly-indexed replies re-bind by index, not
#    position. Monte-Carlo over 300 random permutations, single & multi batch.
# =========================================================================== #
@pytest.mark.parametrize("n", [26, 65])  # 65 forces >1 batch (BATCH_SIZE=30)
def test_reorder_rebinds_by_index_montecarlo(monkeypatch, n):
    texts = make_texts(n, f"reorder{n}")
    state: dict[str, object] = {}

    def transform(call, items):
        # Shuffle the reply order per batch using the trial's seeded rng; each
        # object keeps its own correct "i".
        rng: random.Random = state["rng"]  # type: ignore[assignment]
        order = list(range(len(items)))
        rng.shuffle(order)
        return valid_reply(items, order=order)

    install(monkeypatch, transform)
    n_nonidentity = 0
    for seed in range(300):
        rng = random.Random(seed)
        state["rng"] = rng
        estimates = annotate_markedness_codedness(texts, model_name="fake")
        # Ground truth: labels must be in INPUT order regardless of reply order.
        assert_matches_planted(estimates, texts)
        # Track that we genuinely exercised non-identity orderings.
        probe = list(range(min(len(texts), 30)))
        random.Random(seed).shuffle(probe)
        if probe != sorted(probe):
            n_nonidentity += 1
    assert n_nonidentity > 250  # the campaign really did permute the replies


# =========================================================================== #
# 3. DROP / DUP / EXTRA / WRONG-INDEX / None / bad-field: always a ValueError
#    after retries, never a silent misassignment or a non-ValueError crash.
#    Monte-Carlo over 300 random corruptions.
# =========================================================================== #
def corrupt(items, ctype, rng):
    base = []
    for it in items:
        m, c, why = fake_labels(it["t"])
        base.append({"i": it["i"], "m": m, "c": c, "why": why})
    n = len(base)
    if ctype == "drop":
        del base[rng.randrange(n)]
    elif ctype == "extra":
        base.append({"i": n + rng.randrange(1, 5), "m": 0, "c": 0.0, "why": "x"})
    elif ctype == "dup":  # duplicate one index (len stays n, another goes missing)
        j, k = rng.sample(range(n), 2)
        base[j]["i"] = base[k]["i"]
    elif ctype == "wrong_index":  # in-range count, but an index out of 0..n-1
        base[rng.randrange(n)]["i"] = n + rng.randrange(1, 9)
    elif ctype == "none":
        return None
    elif ctype == "not_list":
        return json.dumps({"i": 0, "m": 0, "c": 0.0})
    elif ctype == "not_json":
        return "definitely ) not [ json"
    elif ctype == "entry_not_dict":
        base[rng.randrange(n)] = [1, 2, 3]
    elif ctype == "bad_m":
        base[rng.randrange(n)]["m"] = rng.choice([2, -1, 0.5, "1", True])
    elif ctype == "bad_c":
        base[rng.randrange(n)]["c"] = rng.choice([1.5, -0.1, "hi", None, True])
    elif ctype == "bad_why":
        base[rng.randrange(n)]["why"] = rng.choice([123, None, ["x"]])
    elif ctype == "index_float":
        base[rng.randrange(n)]["i"] = 0.0  # float, not int
    elif ctype == "index_bool":
        base[rng.randrange(n)]["i"] = True  # bool must be rejected as an index
    return json.dumps(base)


CORRUPTIONS = [
    "drop",
    "extra",
    "dup",
    "wrong_index",
    "none",
    "not_list",
    "not_json",
    "entry_not_dict",
    "bad_m",
    "bad_c",
    "bad_why",
    "index_float",
    "index_bool",
]


def test_corruption_always_raises_valueerror_montecarlo(monkeypatch):
    per_type_seen: dict[str, int] = dict.fromkeys(CORRUPTIONS, 0)
    for trial in range(300):
        rng = random.Random(1000 + trial)
        ctype = CORRUPTIONS[trial % len(CORRUPTIONS)]
        n = rng.randint(2, 12)  # single batch; n>=2 so dup/sample is valid
        texts = make_texts(n, f"corrupt-{trial}")

        # Corrupt on EVERY attempt so retries are exhausted -> must raise.
        def transform(call, items, ct=ctype, r=rng):
            return corrupt(items, ct, r)

        install(monkeypatch, transform)
        try:
            annotate_markedness_codedness(texts, model_name="fake")
        except ValueError:
            per_type_seen[ctype] += 1
        except Exception as exc:
            pytest.fail(
                f"corruption {ctype!r} raised {type(exc).__name__} "
                f"(expected ValueError): {exc}"
            )
        else:
            pytest.fail(
                f"corruption {ctype!r} did NOT raise -> silent misassignment risk"
            )
    # Every corruption family was actually exercised.
    assert all(v > 0 for v in per_type_seen.values()), per_type_seen


def test_each_corruption_family_raises_valueerror(monkeypatch):
    """One deterministic pass per family (guards against a family never firing)."""
    for ctype in CORRUPTIONS:
        rng = random.Random(7)
        texts = make_texts(6, f"fam-{ctype}")
        install(monkeypatch, lambda call, items, ct=ctype, r=rng: corrupt(items, ct, r))
        with pytest.raises(ValueError):
            annotate_markedness_codedness(texts, model_name="fake")


def test_valid_reply_must_not_raise_control(monkeypatch):
    """Control: the SAME harness with an UNcorrupted reply succeeds — proving the
    corruption tests fail for the right reason (the corruption), not the setup."""
    for n in (2, 6, 12):
        texts = make_texts(n, f"control-{n}")
        install(monkeypatch, lambda call, items: valid_reply(items))
        estimates = annotate_markedness_codedness(texts, model_name="fake")
        assert_matches_planted(estimates, texts)


# =========================================================================== #
# 4. END-TO-END CLI over data/synthetic with a fake client. Planted text->label
#    mapping is checked in the written parquet to PROVE prompt_id alignment.
# =========================================================================== #
def load_cli_module():
    path = REPO / "scripts" / "annotate_prompt_set.py"
    spec = importlib.util.spec_from_file_location(
        "annotate_prompt_set_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_end_to_end_alignment(monkeypatch, tmp_path, synthetic):
    cli = load_cli_module()
    out_dir = tmp_path / "synthetic_annotated"

    # Fake client derives every label from the ACTUAL prompt text it receives.
    install(monkeypatch, lambda call, items: valid_reply(items))

    argv = [
        "annotate_prompt_set.py",
        "--dataset",
        str(REPO / "data" / "synthetic"),
        "--cohort",
        "target",
        "--out",
        str(out_dir),
        "--model",
        "fake-model",
    ]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()  # writes out_dir and re-validates via PromptDataset.load internally

    # (a) The written dataset round-trips through the loader.
    reloaded = PromptDataset.load(out_dir)
    written = reloaded.prompts

    original = synthetic.prompts
    target_ids = set(original.loc[original["cohort"] == "target", "prompt_id"])
    assert len(target_ids) == 48  # sanity: forces the multi-batch path (30+18)

    # (b) Every TARGET prompt's labels equal the planted fake_labels(text), bound
    #     to the RIGHT prompt_id, and within the declared ranges.
    text_by_id = dict(zip(original["prompt_id"], original["text"], strict=True))
    for _, row in written[written["cohort"] == "target"].iterrows():
        pid = row["prompt_id"]
        m, c, _why = fake_labels(text_by_id[pid])
        assert row["markedness"] == m, f"markedness misaligned for {pid}"
        assert float(row["codedness"]) == pytest.approx(c), (
            f"codedness misaligned {pid}"
        )
        assert row["markedness"] in (0, 1)
        assert 0.0 <= float(row["codedness"]) <= 1.0

    # (c) NON-target cohorts are left completely untouched.
    for cohort in ("baseline", "target_twin"):
        orig_c = original[original["cohort"] == cohort].set_index("prompt_id")
        new_c = written[written["cohort"] == cohort].set_index("prompt_id")
        pd.testing.assert_series_equal(
            orig_c["markedness"].sort_index(),
            new_c["markedness"].sort_index(),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            orig_c["codedness"].sort_index(),
            new_c["codedness"].sort_index(),
            check_names=False,
        )

    # (d) rationales.json maps each TARGET prompt_id to the planted rationale.
    rationales = json.loads((out_dir / "rationales.json").read_text())
    assert set(rationales) == target_ids
    for pid, why in rationales.items():
        assert why == fake_labels(text_by_id[pid])[2], f"rationale misaligned {pid}"


def test_cli_planted_swap_would_be_detected(monkeypatch, tmp_path, synthetic):
    """Negative control: if the fake client BINDS labels by reply position while
    RETURNING them shuffled, the module's index re-binding still recovers the
    correct per-text labels — proving alignment is enforced by the code, not by
    luck of ordering. (A position-binding implementation would fail test (b).)"""
    cli = load_cli_module()
    out_dir = tmp_path / "shuffled_annotated"

    def transform(call, items):
        order = list(range(len(items)))
        random.Random(len(items)).shuffle(order)  # deterministic non-identity
        return valid_reply(items, order=order)

    install(monkeypatch, transform)
    monkeypatch.setattr(
        "sys.argv",
        [
            "annotate_prompt_set.py",
            "--dataset",
            str(REPO / "data" / "synthetic"),
            "--cohort",
            "target",
            "--out",
            str(out_dir),
            "--model",
            "fake",
        ],
    )
    cli.main()
    reloaded = PromptDataset.load(out_dir)
    original = synthetic.prompts
    text_by_id = dict(zip(original["prompt_id"], original["text"], strict=True))
    written = reloaded.prompts
    for _, row in written[written["cohort"] == "target"].iterrows():
        m, c, _ = fake_labels(text_by_id[row["prompt_id"]])
        assert row["markedness"] == m
        assert float(row["codedness"]) == pytest.approx(c)
