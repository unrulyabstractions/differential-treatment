"""Shared runtime resources handed to every dimension's compute function."""

from __future__ import annotations

import numpy as np

from src.inference.embedding_store import EmbeddingStore


class PipelineContext:
    """Per-run resources: the shared embedding cache and seeded RNG factory."""

    def __init__(self, random_seed: int):
        self.random_seed = random_seed
        self.embedding_store = EmbeddingStore()

    def make_rng(self, salt: int = 0) -> np.random.Generator:
        """Fresh deterministic RNG; salt keeps dimensions independent."""
        return np.random.default_rng(self.random_seed + salt)

    def cleanup(self) -> None:
        self.embedding_store.cleanup()
