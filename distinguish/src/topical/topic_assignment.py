"""Assign each prompt to one topic of a catalog.

Backend specs:
- "embedding" — offline default; nearest survey-topic description by cosine
  similarity in a shared sentence-embedding space.
- "openai:<model>" — TopicGPT Phase-2 classification against the fixed survey
  catalog (needs OPENAI_API_KEY).
- "topicgpt:<model>" — same LLM classification but against a catalog GENERATED
  from the pooled corpus (Phase 1 in topicgpt_taxonomy.py); passed in via
  `taxonomy`. Callers check availability via assignment_unavailable_reason first.

The OpenAI path is catalog-agnostic: it takes catalog lines + the set of valid
ids, so the survey and generated taxonomies share one validated-JSON, one-retry
assignment loop.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
from openai import OpenAI

from src.common.logging_utils import log
from src.common.run_config import TopicalConfig
from src.inference.embedding_store import embedder_unavailable_reason
from src.topical.survey_topic_catalog import SURVEY_TOPICS
from src.topical.topicgpt_taxonomy import GeneratedTaxonomy

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

EMBEDDING_BACKEND = "embedding"
# Both LLM backends resolve their model from a "<provider>:<model>" spec and are
# gated on OPENAI_API_KEY. "topicgpt" is not an embedding provider, so it is not
# in embedding_store._PROVIDER_KEYS — gate it here instead.
_LLM_PROVIDERS = ("openai", "topicgpt")
_OPENAI_KEY = "OPENAI_API_KEY"

# Phrasing catalog entries as user requests moves them into the same region of
# embedding space as the prompts they must match.
_TOPIC_PROMPT_TEMPLATE = "A user asks an AI chatbot for help with {description}."

_OPENAI_BATCH_SIZE = 40
_OPENAI_MAX_ATTEMPTS = 2  # models occasionally drop an entry from large batches
_OPENAI_SYSTEM_TEMPLATE = (
    "You classify user prompts sent to an AI chatbot into a fixed catalog of "
    "topics:\n{catalog}\n\nThe user message is a JSON array of "
    '{{"i": <index>, "text": <prompt>}} objects. Reply with ONLY a JSON array '
    "containing EXACTLY one entry per prompt: [<index>, <topic_id>], covering "
    "every index exactly once."
)


def assignment_model_name(backend: str, config: TopicalConfig) -> str:
    """The concrete model one backend spec resolves to; rejects unknown specs."""
    if backend == EMBEDDING_BACKEND:
        return config.embedding_model
    provider, _, model_name = backend.partition(":")
    if provider in _LLM_PROVIDERS and model_name:
        return model_name
    raise ValueError(
        f"Unknown assignment backend '{backend}'; expected 'embedding', "
        "'openai:<model>', or 'topicgpt:<model>'"
    )


def is_topicgpt_backend(backend: str) -> bool:
    """True for a "topicgpt:<model>" generated-taxonomy backend spec."""
    return backend.partition(":")[0] == "topicgpt"


def assignment_unavailable_reason(backend: str) -> str:
    """Non-empty reason when a backend cannot run (missing API key)."""
    if is_topicgpt_backend(backend):  # not an embedding provider; gate on OpenAI
        return "" if os.environ.get(_OPENAI_KEY) else f"{_OPENAI_KEY} not set"
    return embedder_unavailable_reason(backend)


def assign_topics(
    texts: list[str],
    backend: str,
    config: TopicalConfig,
    context: PipelineContext,
    taxonomy: GeneratedTaxonomy | None = None,
) -> list[int]:
    """Topic id for every text via one configured backend.

    Survey backends map onto SURVEY_TOPICS (ids 1-15); the topicgpt backend maps
    onto `taxonomy`'s generated topics (ids 1..K, position order).
    """
    model_name = assignment_model_name(backend, config)  # validates the spec
    if backend == EMBEDDING_BACKEND:
        return _assign_by_embedding(texts, config, context)
    if taxonomy is None:
        catalog = "\n".join(
            f"{topic.topic_id}. [{topic.domain}] {topic.description}"
            for topic in SURVEY_TOPICS
        )
        valid_ids = {topic.topic_id for topic in SURVEY_TOPICS}
    else:
        catalog = taxonomy.catalog()
        valid_ids = set(range(1, len(taxonomy.topics) + 1))
    return _assign_by_openai(texts, model_name, catalog, valid_ids)


def _assign_by_embedding(
    texts: list[str], config: TopicalConfig, context: PipelineContext
) -> list[int]:
    """Nearest survey topic by cosine similarity of shared embeddings."""
    topic_texts = [
        _TOPIC_PROMPT_TEMPLATE.format(description=topic.description)
        for topic in SURVEY_TOPICS
    ]
    prompt_vectors = context.embedding_store.get_text_embeddings(
        texts, config.embedding_model
    )
    topic_vectors = context.embedding_store.get_text_embeddings(
        topic_texts, config.embedding_model
    )
    prompt_unit = prompt_vectors / np.linalg.norm(prompt_vectors, axis=1, keepdims=True)
    topic_unit = topic_vectors / np.linalg.norm(topic_vectors, axis=1, keepdims=True)
    nearest = np.argmax(prompt_unit @ topic_unit.T, axis=1)
    return [SURVEY_TOPICS[index].topic_id for index in nearest]


def _assign_by_openai(
    texts: list[str], model_name: str, catalog: str, valid_ids: set[int]
) -> list[int]:
    """Batched LLM classification against a catalog (TopicGPT-style)."""
    log(f"Assigning {len(texts)} prompts to topics via OpenAI {model_name}")
    client = OpenAI()
    system_prompt = _OPENAI_SYSTEM_TEMPLATE.format(catalog=catalog)
    assigned: list[int] = []
    for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
        batch = texts[start : start + _OPENAI_BATCH_SIZE]
        assigned.extend(
            _assign_batch(client, model_name, system_prompt, batch, valid_ids)
        )
    return assigned


def _assign_batch(
    client: OpenAI,
    model: str,
    system_prompt: str,
    batch: list[str],
    valid_ids: set[int],
) -> list[int]:
    """Assign one batch with indexed replies, retrying once on a bad reply."""
    payload = json.dumps(
        [{"i": i, "text": t} for i, t in enumerate(batch)], ensure_ascii=False
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    last_error: ValueError | None = None
    for _ in range(_OPENAI_MAX_ATTEMPTS):
        # No temperature override: reasoning models (gpt-5 family) reject
        # anything but the default, and the reply is validated anyway.
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content or ""
        try:
            return _parse_topic_ids(content, len(batch), valid_ids)
        except ValueError as error:
            last_error = error
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your reply was invalid ({error}). Reply again with ONLY "
                        f"the JSON array: exactly {len(batch)} entries, one per "
                        f"index 0..{len(batch) - 1}, each [index, topic_id]."
                    ),
                },
            ]
    raise last_error


def _parse_topic_ids(
    content: str, expected_count: int, valid_ids: set[int]
) -> list[int]:
    """Validate the model's indexed JSON reply; fail loudly on malformed output."""
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        entries = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"Reply is not JSON: {error}") from error
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(
            f"Expected {expected_count} entries, got "
            f"{len(entries) if isinstance(entries, list) else type(entries).__name__}"
        )
    by_index: dict[int, int] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Malformed entry: {entry!r}")
        index, topic_id = entry
        if topic_id not in valid_ids:
            raise ValueError(f"Unknown topic_id {topic_id!r}")
        by_index[int(index)] = int(topic_id)
    if sorted(by_index) != list(range(expected_count)):
        raise ValueError(f"Reply indices do not cover 0..{expected_count - 1} exactly")
    return [by_index[i] for i in range(expected_count)]
