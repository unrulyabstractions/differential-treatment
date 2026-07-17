"""Cohere embed-v4 client (semantic-dimension backend; needs COHERE_API_KEY)."""

from __future__ import annotations

import numpy as np
from cohere import ClientV2
from numpy.typing import NDArray

from src.common.logging_utils import log

_BATCH_SIZE = 96  # Cohere embed API limit


def embed_texts_cohere(texts: list[str], model_name: str) -> NDArray[np.float32]:
    """Embed texts with a Cohere embedding model, preserving order."""
    log(f"Embedding {len(texts)} texts with Cohere model: {model_name}")
    client = ClientV2()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embed(
            texts=batch,
            model=model_name,
            input_type="search_document",
            embedding_types=["float"],
        )
        vectors.extend(response.embeddings.float_)
    return np.asarray(vectors, dtype=np.float32)
