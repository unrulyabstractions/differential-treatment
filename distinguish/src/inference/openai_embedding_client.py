"""OpenAI text-embedding client (optional semantic-dimension backend).

Requires OPENAI_API_KEY in the environment; callers should only enable this
backend when the key is available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from src.common.logging_utils import log

_BATCH_SIZE = 256


def embed_texts_openai(texts: list[str], model_name: str) -> NDArray[np.float32]:
    """Embed texts with an OpenAI embedding model, preserving order."""
    log(f"Embedding {len(texts)} texts with OpenAI model: {model_name}")
    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)
