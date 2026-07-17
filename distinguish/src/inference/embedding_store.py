"""Process-wide embedding cache shared across sections and explorations.

Providers are addressed by spec string: "openai:<model>", "cohere:<model>",
"residual:<causal-lm>" (whole-prompt mean-pooled hidden state at 75% depth),
anything else = a sentence-transformers model name. Exploration reruns hit the
per-text cache, so filtered subsets cost nothing new.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

from src.inference.cohere_embedding_client import embed_texts_cohere
from src.inference.embedding_runner import EmbeddingRunner
from src.inference.openai_embedding_client import embed_texts_openai
from src.inference.residual_stream_extractor import ResidualStreamExtractor

_PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "cohere": "COHERE_API_KEY"}
_RESIDUAL_PREFIX = "residual:"


def split_embedder_spec(spec: str) -> tuple[str, str]:
    """ "openai:text-embedding-3-small" -> ("openai", ...); else sentence-transformers."""
    if spec.startswith(_RESIDUAL_PREFIX):
        return "residual", spec[len(_RESIDUAL_PREFIX) :]
    for provider in _PROVIDER_KEYS:
        prefix = f"{provider}:"
        if spec.startswith(prefix):
            return provider, spec[len(prefix) :]
    return "sentence-transformers", spec


def embedder_unavailable_reason(spec: str) -> str:
    """Non-empty reason when a provider spec cannot run (missing API key)."""
    provider, _ = split_embedder_spec(spec)
    key = _PROVIDER_KEYS.get(provider)
    if key and not os.environ.get(key):
        return f"{key} not set"
    return ""


class EmbeddingStore:
    """Memoizing facade over the embedding backends."""

    def __init__(self) -> None:
        self._runners: dict[str, EmbeddingRunner] = {}
        self._residual: dict[str, ResidualStreamExtractor] = {}
        self._cache: dict[tuple[str, str], NDArray[np.float32]] = {}

    def get_text_embeddings(self, texts: list[str], spec: str) -> NDArray[np.float32]:
        """Embeddings for `texts` under a provider spec, cached per (spec, text)."""
        missing = [t for t in texts if (spec, t) not in self._cache]
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            vectors = self._compute(spec, unique_missing)
            for text, vector in zip(unique_missing, vectors, strict=True):
                self._cache[(spec, text)] = vector
        return np.stack([self._cache[(spec, t)] for t in texts])

    def _compute(self, spec: str, texts: list[str]) -> NDArray[np.float32]:
        provider, model_name = split_embedder_spec(spec)
        if provider == "openai":
            return embed_texts_openai(texts, model_name)
        if provider == "cohere":
            return embed_texts_cohere(texts, model_name)
        if provider == "residual":
            if model_name not in self._residual:
                self._residual[model_name] = ResidualStreamExtractor(model_name)
            return self._residual[model_name].extract(texts)
        if model_name not in self._runners:
            self._runners[model_name] = EmbeddingRunner(model_name)
        return self._runners[model_name].embed(texts)

    def cleanup(self) -> None:
        """Release all loaded embedding models."""
        for runner in self._runners.values():
            runner.cleanup()
        self._runners.clear()
        for extractor in self._residual.values():
            extractor.cleanup()
        self._residual.clear()
