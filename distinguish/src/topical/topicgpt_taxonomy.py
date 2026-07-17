"""TopicGPT taxonomy generation + refinement (Pham et al., arXiv 2311.01449).

Phase 1 of the "topicgpt:<model>" topical backend: derive an intent/style-level
topic catalog from the pooled prompt corpus. One chat call per (subsampled)
document either matches an existing topic or proposes a new one, starting from
seed topics deliberately ORTHOGONAL to the survey catalog (communicative
intent, not subject matter). Refinement merges embedding-similar near-duplicate
topics via an LLM merge prompt and prunes rare ones. Replies are validated
JSON with one corrective retry — never the upstream line regexes — and calls
never set temperature (gpt-5 models reject overrides).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from openai import OpenAI

from src.common.base_schema import BaseSchema
from src.common.logging_utils import log
from src.common.run_config import TopicalConfig

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

# Seed topics anchor the granularity and style of generated topics (the role of
# TopicGPT's prompt/seed_1.md). They describe HOW/WHY a user asks — properties
# any question about any subject can have — so the generated taxonomy measures
# an axis orthogonal to the subject-matter survey topics.
SEED_TOPICS: list[tuple[str, str]] = [
    (
        "Factual Inquiry",
        "Asks for objective information, facts, or an explanation of how something works.",
    ),
    (
        "Advice Seeking",
        "Requests personalized guidance or recommendations about what the asker should do.",
    ),
    (
        "Emotional Disclosure",
        "Shares feelings, worries, or personal struggles, seeking support, empathy, or validation.",
    ),
    (
        "Decision Support",
        "Lays out options or a dilemma and asks for help weighing or choosing between them.",
    ),
    (
        "Procedural Guidance",
        "Asks for concrete step-by-step instructions for accomplishing a specific task.",
    ),
    (
        "Creative Composition",
        "Asks the assistant to draft, write, or rehearse text on the asker's behalf (e.g. a message to send someone).",
    ),
    (
        "Opinion Elicitation",
        "Asks for the assistant's own judgment, stance, or evaluation of something.",
    ),
    (
        "Self-Understanding",
        "Seeks help interpreting the asker's own identity, feelings, behavior, or experiences.",
    ),
]

_MAX_REFINE_ROUNDS = 10  # upstream refinement loop cap
_PAIRS_PER_ROUND = 2  # upstream num_pair
_MAX_ATTEMPTS = 2  # one corrective retry per call, like topic_assignment

# {topics} is substituted with str.replace, so the JSON braces stay literal.
_GENERATION_TEMPLATE = """\
You maintain a topic taxonomy for user prompts sent to an AI chatbot. Given \
one prompt, decide whether it fits an existing topic or needs a new one.

Current topics:
{topics}

Rules:
- Topics describe the COMMUNICATIVE INTENT OR STYLE of the user's request \
(what the user is trying to get from the assistant), NOT its subject matter.
- Prefer an existing topic; add a new one only when no listed topic fits.
- A new topic must be generalizable to prompts about any subject, capture \
exactly one concept, and have a short Title Case label plus a one-sentence \
description.

Examples:
- "How does estrogen change fat distribution?" fits an existing topic: \
{"label": "Factual Inquiry", "description": "Asks for objective information, \
facts, or an explanation of how something works."}
- "Can you role-play the talk where I tell my boss I'm quitting?" fits no \
listed topic, so add one: {"label": "Rehearsal & Role-Play", "description": \
"Asks the assistant to act out or practice a conversation or scenario with \
the asker."}

Reply with ONLY a JSON object {"label": <label>, "description": \
<description>} — copy an existing topic's label exactly, or give one new \
topic. Reply with JSON null only if the text is empty or unintelligible."""

_MERGE_SYSTEM = """\
You refine a topic taxonomy for user prompts sent to an AI chatbot. The user \
message lists pairs of possibly redundant topics. Merge a pair only when the \
two topics are paraphrases or near-duplicates of each other; keep topics that \
are merely related or differ in communicative intent separate.

Reply with ONLY a JSON array of merges — [] if no merge is warranted. Each \
merge is {"label": <Title Case label>, "description": <one-sentence \
description>, "merge": [<existing label>, <existing label>]}, with existing \
labels copied exactly from the pairs."""


@dataclass
class GeneratedTopic(BaseSchema):
    """One topic of a generated taxonomy; count = Phase-1 proposals/matches."""

    label: str
    description: str
    count: int = 0


@dataclass
class GeneratedTaxonomy(BaseSchema):
    """Corpus-derived topic catalog with its refinement audit trail."""

    topics: list[GeneratedTopic] = field(default_factory=list)
    seed_labels: list[str] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)  # "A + B -> C" records
    pruned: list[str] = field(default_factory=list)  # labels dropped as rare

    def catalog(self) -> str:
        """Indexed catalog lines for assignment prompts (id = position + 1)."""
        return "\n".join(
            f"{topic_id}. {topic.label}: {topic.description}"
            for topic_id, topic in enumerate(self.topics, start=1)
        )


def build_taxonomy(
    texts: list[str],
    model_name: str,
    config: TopicalConfig,
    rng: np.random.Generator,
    context: PipelineContext,
) -> GeneratedTaxonomy:
    """Generate + refine a taxonomy from the pooled interleaved corpus."""
    docs = list(texts)
    if len(docs) > config.topicgpt_max_generation_docs:
        keep = np.sort(
            rng.choice(
                len(docs), size=config.topicgpt_max_generation_docs, replace=False
            )
        )
        docs = [docs[i] for i in keep]  # sorted keeps the interleaved balance
    taxonomy = GeneratedTaxonomy(
        topics=[
            GeneratedTopic(label, description) for label, description in SEED_TOPICS
        ],
        seed_labels=[label for label, _ in SEED_TOPICS],
    )
    client = OpenAI()
    log(f"TopicGPT: generating taxonomy from {len(docs)} pooled docs via {model_name}")
    _generate(client, model_name, docs, taxonomy, config)
    protected = _merge(client, model_name, taxonomy, config, context)
    _prune(taxonomy, config, protected)
    log(
        f"TopicGPT: taxonomy has {len(taxonomy.topics)} topics "
        f"({len(taxonomy.merges)} merges, {len(taxonomy.pruned)} pruned)"
    )
    return taxonomy


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _key(label: str) -> str:
    """Case-insensitive dedup key for topic labels."""
    return _normalize(label).casefold()


def _generate(
    client: OpenAI,
    model_name: str,
    docs: list[str],
    taxonomy: GeneratedTaxonomy,
    config: TopicalConfig,
) -> None:
    """One call per doc: match an existing topic (count += 1) or add a new one."""
    by_key = {_key(topic.label): topic for topic in taxonomy.topics}
    consecutive_known = 0
    for done, doc in enumerate(docs):
        if consecutive_known >= config.topicgpt_early_stop:
            log(
                f"TopicGPT: early stop after {done} docs "
                f"(no new topic in the last {consecutive_known})"
            )
            break
        topic_lines = "\n".join(
            f"- {t.label}: {t.description}" for t in taxonomy.topics
        )
        system = _GENERATION_TEMPLATE.replace("{topics}", topic_lines)
        proposed = _chat_json(client, model_name, system, doc, _validated_topic)
        if proposed is None:  # explicit null: no intent-level topic applies
            consecutive_known += 1
            continue
        label, description = proposed
        known = by_key.get(_key(label))
        if known is None:
            topic = GeneratedTopic(label, description, count=1)
            taxonomy.topics.append(topic)
            by_key[_key(label)] = topic
            consecutive_known = 0
        else:
            known.count += 1
            consecutive_known += 1
        if (done + 1) % 10 == 0:
            log(f"TopicGPT: {done + 1}/{len(docs)} docs, {len(taxonomy.topics)} topics")


def _merge(
    client: OpenAI,
    model_name: str,
    taxonomy: GeneratedTaxonomy,
    config: TopicalConfig,
    context: PipelineContext,
) -> set[str]:
    """Merge near-duplicate topics; returns the prune-protected label keys."""
    protected = {_key(label) for label in taxonomy.seed_labels}
    prompted: set[frozenset[str]] = set()  # each pair is prompted at most once
    for _ in range(_MAX_REFINE_ROUNDS):
        pairs = _similar_pairs(taxonomy.topics, config, context, prompted)
        if not pairs:
            break
        prompted.update(frozenset((_key(a.label), _key(b.label))) for a, b in pairs)
        pairs_block = "\n".join(
            f'- "{a.label}: {a.description}" vs "{b.label}: {b.description}"'
            for a, b in pairs
        )
        keys = {_key(topic.label) for topic in taxonomy.topics}
        merges = _chat_json(
            client,
            model_name,
            _MERGE_SYSTEM,
            pairs_block,
            lambda parsed, keys=keys: _validated_merges(parsed, keys),
        )
        for label, description, originals in merges:
            _apply_merge(taxonomy, label, description, originals, protected)
    return protected


def _similar_pairs(
    topics: list[GeneratedTopic],
    config: TopicalConfig,
    context: PipelineContext,
    prompted: set[frozenset[str]],
) -> list[tuple[GeneratedTopic, GeneratedTopic]]:
    """Up to 2 not-yet-prompted pairs with embedding cosine above the threshold."""
    if len(topics) < 2:
        return []
    strings = [f"{t.label}: {t.description}" for t in topics]
    vectors = context.embedding_store.get_text_embeddings(
        strings, config.embedding_model
    )
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = unit @ unit.T
    candidates = [
        (float(similarity[i, j]), i, j)
        for i in range(len(topics))
        for j in range(i + 1, len(topics))
        if similarity[i, j] > config.topicgpt_merge_similarity
        and frozenset((_key(topics[i].label), _key(topics[j].label))) not in prompted
    ]
    # Highest similarity first, breaking ties by ascending (i, j). This matches
    # the reference's stable sort-by-score (refinement.topic_pairs), which keeps
    # the i<j enumeration order among equal cosines; a plain reverse sort would
    # instead prefer the largest indices on a tie.
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    return [(topics[i], topics[j]) for _, i, j in candidates[:_PAIRS_PER_ROUND]]


def _apply_merge(
    taxonomy: GeneratedTaxonomy,
    label: str,
    description: str,
    originals: list[str],
    protected: set[str],
) -> None:
    """Replace the original topics with the merged one, summing their counts."""
    original_keys = {_key(o) for o in originals}
    matched = [t for t in taxonomy.topics if _key(t.label) in original_keys]
    if len(matched) < 2:
        return  # an earlier merge in the same reply already consumed the pair
    merged_key = _key(label)
    existing = next(
        (
            t
            for t in taxonomy.topics
            if _key(t.label) == merged_key and t not in matched
        ),
        None,
    )
    merged = existing or GeneratedTopic(label, description)
    merged.count += sum(t.count for t in matched)
    if existing is None:
        taxonomy.topics.insert(taxonomy.topics.index(matched[0]), merged)
    taxonomy.topics = [t for t in taxonomy.topics if t not in matched]
    if original_keys & protected:
        protected.add(merged_key)
    taxonomy.merges.append(" + ".join(t.label for t in matched) + f" -> {merged.label}")


def _prune(
    taxonomy: GeneratedTaxonomy, config: TopicalConfig, protected: set[str]
) -> None:
    """Drop rare non-seed topics: count < prune_fraction of the total count."""
    total = sum(topic.count for topic in taxonomy.topics)
    threshold = config.topicgpt_prune_fraction * total
    kept = []
    for topic in taxonomy.topics:
        if _key(topic.label) in protected or topic.count >= threshold:
            kept.append(topic)
        else:
            taxonomy.pruned.append(topic.label)
    taxonomy.topics = kept


def _chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    validated: Callable[[Any], Any],
) -> Any:
    """One validated-JSON chat exchange with a single corrective retry."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: ValueError | None = None
    for _ in range(_MAX_ATTEMPTS):
        # No temperature override: reasoning models (gpt-5 family) reject
        # anything but the default, and the reply is validated anyway.
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content or ""
        try:
            return validated(_json_reply(content))
        except ValueError as error:
            last_error = error
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your reply was invalid ({error}). Reply again with "
                        "ONLY the JSON in the required shape."
                    ),
                },
            ]
    raise last_error


def _json_reply(content: str) -> Any:
    """Parse a chat reply as JSON, tolerating code fences; ValueError if not."""
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"Reply is not JSON: {error}") from error


def _validated_topic(parsed: Any) -> tuple[str, str] | None:
    """(label, description) from a topic object; None for an explicit null."""
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object with 'label' and 'description'")
    label = parsed.get("label")
    description = parsed.get("description")
    if not isinstance(label, str) or not _normalize(label):
        raise ValueError(f"Invalid 'label': {label!r}")
    if not isinstance(description, str) or not _normalize(description):
        raise ValueError(f"Invalid 'description': {description!r}")
    return _normalize(label), _normalize(description)


def _validated_merges(
    parsed: Any, valid_keys: set[str]
) -> list[tuple[str, str, list[str]]]:
    """(label, description, originals) merge tuples; [] means no merge."""
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of merges")
    merges = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed merge entry: {entry!r}")
        label, description = _validated_topic(entry)
        originals = entry.get("merge")
        if (
            not isinstance(originals, list)
            or len(originals) < 2
            or any(
                not isinstance(o, str) or _key(o) not in valid_keys for o in originals
            )
        ):
            raise ValueError(
                f"'merge' must list >= 2 existing topic labels, got {originals!r}"
            )
        merges.append((label, description, [_normalize(o) for o in originals]))
    return merges
