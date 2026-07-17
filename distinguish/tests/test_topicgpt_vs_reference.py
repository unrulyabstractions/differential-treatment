"""Differential test: our TopicGPT reimplementation vs the actual reference.

The reference is Pham et al. TopicGPT (github.com/chtmp223/topicGPT). Its topic
generation, refinement, and assignment/correction are LLM calls, so we cannot do
a numeric diff end-to-end. Instead we pin the PURE-LOGIC cores that surround the
LLM calls to faithful reimplementations of the reference's actual code, feeding a
controlled taxonomy / cosine matrix / reply string and asserting our code makes
the SAME decisions the reference algorithm would:

- generation dedup + count accumulation  vs generation_1.generate_topics
  (case-insensitive find_duplicates, count += 1 on a match, reset-on-new-topic).
- merge pair selection by cosine          vs refinement.topic_pairs
  (all i<j pairs, sort by score desc, take up to num_pair=2 with score > 0.5,
  skipping already-prompted pairs).
- rare-topic pruning                       vs refinement.remove_topics
  (drop a topic whose count < 0.01 * total_count).
- merge count accounting                   vs utils.TopicTree.update_tree
  (merged topic's count == sum of the merged originals' counts).
- indexed-reply parsing / correction trigger vs correction.topic_parser
  (an assignment that is empty or references an out-of-taxonomy topic is a fault
  that must be re-prompted).

The reimplementations below are transcribed from the reference source at
topicgpt_python/{generation_1,refinement,correction}.py and topicgpt_python/
utils.py (TopicTree). Divergences that are deliberate simplifications for the
single-level intent taxonomy (one topic per doc, JSON replies instead of line
regexes, seed protection, in-loop correction retry) are called out inline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.run_config import TopicalConfig
from src.topical import topicgpt_taxonomy as tgt
from src.topical.topic_assignment import _parse_topic_ids
from src.topical.topicgpt_taxonomy import (
    GeneratedTaxonomy,
    GeneratedTopic,
    _apply_merge,
    _generate,
    _prune,
    _similar_pairs,
)


# --------------------------------------------------------------------------- #
# Faithful reimplementations of the reference algorithms (clean-room, from the
# cited reference source).                                                     #
# --------------------------------------------------------------------------- #
def _reference_generate_counts(
    seed_labels: list[str],
    proposals: list[tuple[str, str] | None],
) -> dict[str, int]:
    """Mirror generation_1.generate_topics dedup + count accumulation.

    Reference (topicgpt_python/generation_1.py, lines ~124-136): for each parsed
    topic, `dups = topics_root.find_duplicates(name, lvl)` where find_duplicates
    (utils.TopicTree) matches `node.name.lower() == name.lower()`; on a duplicate
    `dups[0].count += 1`, otherwise `_add_node(...)`. Early stop is exercised
    separately; here it is disabled. Seeds start at count 0 to match our schema
    (the reference seeds start at 1 via from_seed_file — an offset, not logic).
    A doc that yields no topic ("None"/invalid format) is skipped, contributing
    nothing (generation_1.py line 116 `continue`).
    """
    counts: dict[str, int] = {label.lower(): 0 for label in seed_labels}
    for proposal in proposals:
        if proposal is None:
            continue
        label = proposal[0]
        key = label.lower()
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
    return counts


def _reference_topic_pairs(
    similarity: np.ndarray,
    labels: list[str],
    already_prompted: list[list[str]],
    threshold: float = 0.5,
    num_pair: int = 2,
) -> list[tuple[int, int]]:
    """Mirror refinement.topic_pairs pair selection (index form).

    Reference (topicgpt_python/refinement.py, lines 33-53): build every i<j pair
    with its cosine score, `sorted(pairs, key=lambda x: x["score"], reverse=True)`
    (a STABLE sort, so ties keep the i-asc/j-asc enumeration order), then walk the
    sorted pairs selecting up to num_pair whose `score > threshold` and whose
    sorted label pair is not in `all_pairs`.
    """
    pairs = [
        {"index": (i, j), "score": float(similarity[i][j])}
        for i in range(len(similarity))
        for j in range(i + 1, len(similarity))
    ]
    pairs.sort(key=lambda pair: pair["score"], reverse=True)
    selected: list[tuple[int, int]] = []
    for pair in pairs:
        if len(selected) >= num_pair:
            break
        i, j = pair["index"]
        if (
            pair["score"] > threshold
            and sorted([labels[i], labels[j]]) not in already_prompted
        ):
            selected.append((i, j))
            already_prompted.append(sorted([labels[i], labels[j]]))
    return selected


def _reference_remove(counts: list[int], threshold: float = 0.01) -> list[bool]:
    """Mirror refinement.remove_topics: True == removed.

    Reference (topicgpt_python/refinement.py, lines 144-170): `total_count =
    sum(node.count ...)`, `threshold_count = total_count * threshold`, remove a
    node when `node.count < threshold_count`.
    """
    total = sum(counts)
    threshold_count = total * threshold
    return [count < threshold_count for count in counts]


# --------------------------------------------------------------------------- #
# Controlled embedding-store / context stubs (no SBERT, no network).          #
# --------------------------------------------------------------------------- #
class _StubStore:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def get_text_embeddings(self, texts: list[str], spec: str) -> np.ndarray:
        return np.array([self._vectors[text] for text in texts], dtype=np.float64)


class _StubContext:
    def __init__(self, store: _StubStore) -> None:
        self.embedding_store = store


def _cosine_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype=np.float64)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return unit @ unit.T


# --------------------------------------------------------------------------- #
# 1. Generation: dedup + count accumulation == reference generate_topics.      #
# --------------------------------------------------------------------------- #
def test_generation_dedup_and_counts_match_reference(monkeypatch):
    """Our _generate reproduces the reference's find_duplicates/count logic."""
    seed = [("Factual Inquiry", "Asks for objective information.")]
    taxonomy = GeneratedTaxonomy(
        topics=[GeneratedTopic(label, desc) for label, desc in seed],
        seed_labels=[label for label, _ in seed],
    )
    # One proposal per doc; None == "no topic". Case variants exercise the
    # case-insensitive dedup (reference: name.lower() == name.lower()).
    proposals: list[tuple[str, str] | None] = [
        ("Advice Seeking", "Requests guidance."),
        ("factual inquiry", "dup of seed, different case"),
        ("Advice Seeking", "dup"),
        None,
        ("Decision Support", "Weigh options."),
        ("ADVICE SEEKING", "dup again"),
    ]
    docs = [f"doc-{i}" for i in range(len(proposals))]
    by_doc = dict(zip(docs, proposals, strict=True))

    def fake_chat_json(client, model, system, user, validated):
        return by_doc[user]

    monkeypatch.setattr(tgt, "_chat_json", fake_chat_json)
    config = TopicalConfig(topicgpt_early_stop=10_000)  # disable early stop
    _generate(
        client=None, model_name="stub", docs=docs, taxonomy=taxonomy, config=config
    )

    ours = {topic.label.lower(): topic.count for topic in taxonomy.topics}
    reference = _reference_generate_counts([label for label, _ in seed], proposals)
    assert ours == reference
    # Explicit expected values so the oracle itself is pinned.
    assert reference == {
        "factual inquiry": 1,
        "advice seeking": 3,
        "decision support": 1,
    }


def test_generation_early_stop_resets_on_new_topic(monkeypatch):
    """Early stop fires after `early_stop` consecutive no-new-topic docs.

    Mirrors the reference's running_dups counter (reset to 0 whenever a new topic
    is added; generation_1.py line 136). Our threshold is inclusive (>=) with a
    small default (documented deviation from the reference's `> 1000`), so we set
    it explicitly and assert the exact stop point.
    """
    seed = [("Factual Inquiry", "Asks for objective information.")]
    taxonomy = GeneratedTaxonomy(
        topics=[GeneratedTopic(label, desc) for label, desc in seed],
        seed_labels=[label for label, _ in seed],
    )
    # 2 dups, then a NEW topic (resets), then 3 dups -> with early_stop=3 the run
    # must stop on the 3rd post-reset dup; the trailing proposal is never seen.
    proposals: list[tuple[str, str] | None] = [
        ("Factual Inquiry", "dup 1"),
        ("Factual Inquiry", "dup 2"),
        ("Advice Seeking", "NEW -> reset"),
        ("Factual Inquiry", "dup 1 of 3"),
        ("Factual Inquiry", "dup 2 of 3"),
        ("Factual Inquiry", "dup 3 of 3 -> counter reaches early_stop"),
        ("Never Seen", "loop breaks before this doc is prompted"),
    ]
    docs = [f"doc-{i}" for i in range(len(proposals))]
    by_doc = dict(zip(docs, proposals, strict=True))
    seen: list[str] = []

    def fake_chat_json(client, model, system, user, validated):
        seen.append(user)
        return by_doc[user]

    monkeypatch.setattr(tgt, "_chat_json", fake_chat_json)
    config = TopicalConfig(topicgpt_early_stop=3)
    _generate(
        client=None, model_name="stub", docs=docs, taxonomy=taxonomy, config=config
    )

    labels = {topic.label for topic in taxonomy.topics}
    assert "Never Seen" not in labels  # early stop halted the loop
    assert "Advice Seeking" in labels  # the reset topic was added
    assert "doc-6" not in seen  # trailing doc never prompted


# --------------------------------------------------------------------------- #
# 2. Refinement: merge pair selection by cosine == reference topic_pairs.       #
# --------------------------------------------------------------------------- #
def _topics_from_vectors(vectors: list[list[float]]):
    topics = [
        GeneratedTopic(f"T{i}", f"desc {i}", count=1) for i in range(len(vectors))
    ]
    strings = {f"{t.label}: {t.description}": vectors[i] for i, t in enumerate(topics)}
    return topics, _StubContext(_StubStore(strings))


def test_similar_pairs_selection_matches_reference():
    """Same cosine matrix -> same top-2 above-threshold pairs as the reference."""
    # (0,1)=0.9 and (2,3)=0.7 are the only pairs above 0.5; everything else 0.
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, np.sqrt(0.19), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.7, np.sqrt(0.51)],
    ]
    topics, context = _topics_from_vectors(vectors)
    config = TopicalConfig(topicgpt_merge_similarity=0.5)

    ours = [
        (topics.index(a), topics.index(b))
        for a, b in _similar_pairs(topics, config, context, prompted=set())
    ]
    reference = _reference_topic_pairs(
        _cosine_matrix(vectors), [t.label for t in topics], already_prompted=[]
    )
    assert ours == reference == [(0, 1), (2, 3)]


def test_similar_pairs_threshold_excludes_below_half():
    """A pair whose cosine is below 0.5 is never selected (reference: score > 0.5)."""
    # (0,1)=0.4 must be excluded; (2,3)=0.8 is the only survivor.
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.4, np.sqrt(0.84), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, 0.6],
    ]
    topics, context = _topics_from_vectors(vectors)
    config = TopicalConfig(topicgpt_merge_similarity=0.5)
    ours = [
        (topics.index(a), topics.index(b))
        for a, b in _similar_pairs(topics, config, context, prompted=set())
    ]
    reference = _reference_topic_pairs(
        _cosine_matrix(vectors), [t.label for t in topics], already_prompted=[]
    )
    assert ours == reference == [(2, 3)]


def test_similar_pairs_tie_breaks_like_reference_stable_sort():
    """On an exact cosine tie, pick the smaller (i, j) — the reference order.

    Regression pin for the tie-break fix: (0,2) and (0,3) tie at exactly 0.6/norm
    (c and d share the same component multiset, so their unit dot with a is
    bit-identical). The reference's stable sort-by-score keeps (0,2) before (0,3);
    a plain reverse sort on (score, i, j) would wrongly prefer (0,3).
    """
    vectors = [
        [1.0, 0.0, 0.0, 0.0],  # a
        [0.9, np.sqrt(0.19), 0.0, 0.0],  # b: (0,1)=0.9 highest
        [0.6, 0.0, 0.8, 0.0],  # c: (0,2)=0.6
        [0.6, 0.0, 0.0, 0.8],  # d: (0,3)=0.6  (exact tie with (0,2))
    ]
    similarity = _cosine_matrix(vectors)
    assert similarity[0, 2] == similarity[0, 3]  # the tie is bit-exact

    topics, context = _topics_from_vectors(vectors)
    config = TopicalConfig(topicgpt_merge_similarity=0.5)
    ours = [
        (topics.index(a), topics.index(b))
        for a, b in _similar_pairs(topics, config, context, prompted=set())
    ]
    reference = _reference_topic_pairs(
        similarity, [t.label for t in topics], already_prompted=[]
    )
    assert ours == reference == [(0, 1), (0, 2)]


def test_similar_pairs_skips_already_prompted():
    """A pair already prompted is excluded next round (reference: all_pairs)."""
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, np.sqrt(0.19), 0.0, 0.0],  # (0,1)=0.9
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.7, np.sqrt(0.51)],  # (2,3)=0.7
    ]
    topics, context = _topics_from_vectors(vectors)
    config = TopicalConfig(topicgpt_merge_similarity=0.5)
    prompted = {frozenset((tgt._key("T0"), tgt._key("T1")))}
    ours = [
        (topics.index(a), topics.index(b))
        for a, b in _similar_pairs(topics, config, context, prompted=prompted)
    ]
    # (0,1) is suppressed; only (2,3) remains.
    assert ours == [(2, 3)]


# --------------------------------------------------------------------------- #
# 3. Pruning: rare-topic removal boundary == reference remove_topics.          #
# --------------------------------------------------------------------------- #
def test_prune_boundary_matches_reference():
    """count < 0.01 * total is dropped; count == threshold is kept (both)."""
    # total = 1000 -> threshold = 10. counts: 9 (drop), 10 (keep, ==), 11 (keep),
    # 970 (keep). Reference removes strictly-less-than the threshold.
    counts = [9, 10, 11, 970]
    taxonomy = GeneratedTaxonomy(
        topics=[GeneratedTopic(f"T{i}", f"d{i}", count=c) for i, c in enumerate(counts)]
    )
    config = TopicalConfig(topicgpt_prune_fraction=0.01)
    _prune(taxonomy, config, protected=set())  # no protection -> pure reference

    kept = {topic.label for topic in taxonomy.topics}
    removed = _reference_remove(counts, threshold=0.01)
    reference_kept = {f"T{i}" for i, drop in enumerate(removed) if not drop}
    assert kept == reference_kept == {"T1", "T2", "T3"}
    assert set(taxonomy.pruned) == {"T0"}


def test_prune_protects_seeds_our_documented_deviation():
    """We protect seed topics from pruning; the reference does not.

    Justified deviation: the seed topics anchor the intent/style axis that is
    deliberately orthogonal to the survey catalog, so they must survive even when
    rare. This asserts our added behavior, not the reference's.
    """
    taxonomy = GeneratedTaxonomy(
        topics=[
            GeneratedTopic("Seed", "rare seed", count=1),
            GeneratedTopic("Common", "common", count=999),
        ]
    )
    config = TopicalConfig(topicgpt_prune_fraction=0.01)
    _prune(taxonomy, config, protected={tgt._key("Seed")})
    labels = {topic.label for topic in taxonomy.topics}
    assert labels == {"Seed", "Common"}  # rare seed retained
    assert taxonomy.pruned == []
    # Sanity: the reference (no protection) WOULD drop it (1 < 0.01*1000 == 10).
    assert _reference_remove([1, 999]) == [True, False]


# --------------------------------------------------------------------------- #
# 4. Merge count accounting == reference TopicTree.update_tree.                 #
# --------------------------------------------------------------------------- #
def test_apply_merge_sums_counts_like_update_tree():
    """Merged topic count == sum of the merged originals (reference update_tree)."""
    taxonomy = GeneratedTaxonomy(
        topics=[
            GeneratedTopic("Alpha", "a", count=3),
            GeneratedTopic("Beta", "b", count=5),
            GeneratedTopic("Gamma", "c", count=2),
        ]
    )
    _apply_merge(taxonomy, "Merged", "merged desc", ["Alpha", "Beta"], protected=set())
    by_label = {topic.label: topic.count for topic in taxonomy.topics}
    assert by_label == {"Merged": 8, "Gamma": 2}  # 3 + 5, Gamma untouched
    # Merged node takes the position of the first original (Alpha, index 0).
    assert taxonomy.topics[0].label == "Merged"
    assert taxonomy.merges == ["Alpha + Beta -> Merged"]


def test_apply_merge_noops_when_pair_already_consumed():
    """A merge whose originals are no longer both present is a safe no-op.

    Reference update_tree only merges the duplicates it can still find; a second
    merge line naming an already-merged topic finds < 2 nodes and changes nothing.
    """
    taxonomy = GeneratedTaxonomy(topics=[GeneratedTopic("Solo", "s", count=4)])
    _apply_merge(taxonomy, "X", "x", ["Solo", "Ghost"], protected=set())
    assert [t.label for t in taxonomy.topics] == ["Solo"]
    assert taxonomy.topics[0].count == 4
    assert taxonomy.merges == []


# --------------------------------------------------------------------------- #
# 5. Assignment reply parsing / correction trigger == reference topic_parser.  #
# --------------------------------------------------------------------------- #
def test_parse_topic_ids_accepts_valid_indexed_reply():
    """A well-formed [index, topic_id] array parses in prompt order."""
    content = "[[0, 3], [1, 1], [2, 3]]"
    assert _parse_topic_ids(content, expected_count=3, valid_ids={1, 2, 3}) == [3, 1, 3]


def test_parse_topic_ids_reorders_by_index():
    """Entries out of order are placed by their declared index (0..n-1)."""
    content = "[[2, 1], [0, 2], [1, 3]]"
    assert _parse_topic_ids(content, expected_count=3, valid_ids={1, 2, 3}) == [2, 3, 1]


def test_parse_topic_ids_tolerates_code_fence():
    """A ```json fenced reply is unwrapped before parsing (matches our path)."""
    content = "```json\n[[0, 1]]\n```"
    assert _parse_topic_ids(content, expected_count=1, valid_ids={1}) == [1]


def test_parse_topic_ids_hallucinated_id_is_a_fault():
    """An out-of-taxonomy topic id is rejected == reference 'hallucinated'.

    correction.topic_parser flags a response whose topic is not in
    root.get_root_descendants_name(); we reject an id not in valid_ids, which is
    the same correction trigger surfaced as a retriable ValueError.
    """
    with pytest.raises(ValueError, match="Unknown topic_id"):
        _parse_topic_ids("[[0, 9]]", expected_count=1, valid_ids={1, 2, 3})


def test_parse_topic_ids_missing_entry_is_a_fault():
    """A reply that does not cover every index is rejected == reference 'error'."""
    # Right length, but index 1 is never assigned (index 0 is repeated).
    with pytest.raises(ValueError, match="do not cover"):
        _parse_topic_ids("[[0, 1], [0, 2]]", expected_count=2, valid_ids={1, 2, 3})


def test_parse_topic_ids_wrong_count_is_a_fault():
    """Too many/few entries is rejected before per-entry checks."""
    with pytest.raises(ValueError, match="Expected 2 entries"):
        _parse_topic_ids(
            "[[0, 1], [1, 2], [2, 3]]", expected_count=2, valid_ids={1, 2, 3}
        )


def test_parse_topic_ids_non_json_is_a_fault():
    """A non-JSON reply is a retriable fault, not a crash."""
    with pytest.raises(ValueError, match="not JSON"):
        _parse_topic_ids("no json here", expected_count=1, valid_ids={1})
