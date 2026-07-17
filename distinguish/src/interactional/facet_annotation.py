"""Annotate every prompt with one option per interactional facet.

Backend specs: "embedding" (offline default; nearest option description by
cosine similarity in a shared sentence-embedding space) or "openai:<model>"
(batched LLM annotation; needs OPENAI_API_KEY — callers check availability
via embedder_unavailable_reason before invoking).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from openai import OpenAI

from src.common.base_schema import BaseSchema
from src.common.logging_utils import log
from src.common.run_config import InteractionalConfig
from src.inference.embedding_store import split_embedder_spec
from src.interactional.interaction_facets import FACET_NAMES, INTERACTION_FACETS

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

EMBEDDING_BACKEND = "embedding"

# Phrasing option descriptions as user messages moves them into the same region
# of embedding space as the prompts they must match.
_OPTION_PROMPT_TEMPLATE = "A user message to an AI chatbot that {description}."

_OPENAI_BATCH_SIZE = 40
_OPENAI_MAX_ATTEMPTS = 2  # models occasionally drop an entry from large batches
_OPENAI_SYSTEM_TEMPLATE = (
    "You annotate user prompts sent to an AI chatbot along three fixed facets. "
    "For each facet pick the single best-fitting option:\n{catalog}\n\n"
    'The user message is a JSON array of {{"i": <index>, "text": <prompt>}} '
    "objects. Reply with ONLY a JSON array containing EXACTLY one entry per "
    "prompt: [<index>, <speech_act>, <disclosure_depth>, <anthropomorphization>], "
    "covering every index exactly once."
)

_OPTION_INDEX: dict[str, dict[str, int]] = {
    facet: {opt.option: i for i, opt in enumerate(options)}
    for facet, options in INTERACTION_FACETS.items()
}


@dataclass
class FacetAnnotations(BaseSchema):
    """Per facet, the chosen option index for every annotated prompt."""

    speech_act: list[int] = field(default_factory=list)
    disclosure_depth: list[int] = field(default_factory=list)
    anthropomorphization: list[int] = field(default_factory=list)

    def for_facet(self, facet: str) -> list[int]:
        return getattr(self, facet)

    def split(self, unpool) -> tuple[FacetAnnotations, FacetAnnotations]:
        """Undo a pooled annotation into the two original sides."""
        first: dict[str, list[int]] = {}
        second: dict[str, list[int]] = {}
        for facet in FACET_NAMES:
            first[facet], second[facet] = unpool(self.for_facet(facet))
        return FacetAnnotations(**first), FacetAnnotations(**second)


def annotation_model_name(backend: str, config: InteractionalConfig) -> str:
    """The concrete model one backend spec resolves to; rejects unknown specs."""
    if backend == EMBEDDING_BACKEND:
        return config.embedding_model
    provider, model_name = split_embedder_spec(backend)
    if provider == "openai":
        return model_name
    raise ValueError(
        f"Unknown annotation backend '{backend}'; "
        "expected 'embedding' or 'openai:<model>'"
    )


def annotate_facets(
    texts: list[str],
    backend: str,
    config: InteractionalConfig,
    context: PipelineContext,
) -> FacetAnnotations:
    """Option index per facet for every text, via one configured backend spec."""
    model_name = annotation_model_name(backend, config)  # validates the spec
    if backend == EMBEDDING_BACKEND:
        return _annotate_by_embedding(texts, config, context)
    return _annotate_by_openai(texts, model_name)


def _annotate_by_embedding(
    texts: list[str], config: InteractionalConfig, context: PipelineContext
) -> FacetAnnotations:
    """Nearest facet option by cosine similarity of shared embeddings."""
    prompt_vectors = context.embedding_store.get_text_embeddings(
        texts, config.embedding_model
    )
    prompt_unit = prompt_vectors / np.linalg.norm(prompt_vectors, axis=1, keepdims=True)
    chosen: dict[str, list[int]] = {}
    for facet, options in INTERACTION_FACETS.items():
        option_texts = [
            _OPTION_PROMPT_TEMPLATE.format(description=opt.description)
            for opt in options
        ]
        option_vectors = context.embedding_store.get_text_embeddings(
            option_texts, config.embedding_model
        )
        option_unit = option_vectors / np.linalg.norm(
            option_vectors, axis=1, keepdims=True
        )
        nearest = np.argmax(prompt_unit @ option_unit.T, axis=1)
        chosen[facet] = [int(index) for index in nearest]
    return FacetAnnotations(**chosen)


def _annotate_by_openai(texts: list[str], model_name: str) -> FacetAnnotations:
    """Batched LLM annotation against the facet catalog."""
    log(f"Annotating {len(texts)} prompts on facets via OpenAI {model_name}")
    client = OpenAI()
    catalog = "\n".join(
        f"{facet}:\n"
        + "\n".join(f"  - {opt.option}: {opt.description}" for opt in options)
        for facet, options in INTERACTION_FACETS.items()
    )
    system_prompt = _OPENAI_SYSTEM_TEMPLATE.format(catalog=catalog)
    triples: list[list[int]] = []
    for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
        batch = texts[start : start + _OPENAI_BATCH_SIZE]
        triples.extend(_annotate_batch(client, model_name, system_prompt, batch))
    per_facet = list(zip(*triples, strict=True))
    return FacetAnnotations(
        **{facet: list(per_facet[i]) for i, facet in enumerate(FACET_NAMES)}
    )


def _annotate_batch(
    client: OpenAI, model: str, system_prompt: str, batch: list[str]
) -> list[list[int]]:
    """Annotate one batch, retrying once — models sometimes drop an entry."""
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
        content = response.choices[0].message.content
        try:
            return _parse_option_triples(content, len(batch))
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
                        "index 0..{last}, each [index, speech_act, "
                        "disclosure_depth, anthropomorphization]."
                    ).replace("{last}", str(len(batch) - 1)),
                },
            ]
    raise last_error


def _parse_option_triples(content: str, expected_count: int) -> list[list[int]]:
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
    by_index: dict[int, list[int]] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 1 + len(FACET_NAMES):
            raise ValueError(f"Malformed facet annotation entry: {entry!r}")
        index, *options = entry
        triple = []
        for facet, option_name in zip(FACET_NAMES, options, strict=True):
            if option_name not in _OPTION_INDEX[facet]:
                raise ValueError(f"Unknown option '{option_name}' for '{facet}'")
            triple.append(_OPTION_INDEX[facet][option_name])
        by_index[int(index)] = triple
    if sorted(by_index) != list(range(expected_count)):
        raise ValueError(f"Reply indices do not cover 0..{expected_count - 1} exactly")
    return [by_index[i] for i in range(expected_count)]
