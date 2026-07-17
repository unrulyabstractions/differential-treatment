"""Sentence-transformers embedding runner.

Ported from queering-nlp-bias (src/inference/embedding_runner.py), trimmed to
what this framework needs: batch text -> vector encoding.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.common.logging_utils import log


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr at the file descriptor level.

    sentence-transformers prints loading noise directly to the file
    descriptors, so ordinary sys.stdout redirection is not enough.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


class EmbeddingRunner:
    """Runner for computing text embeddings using sentence-transformers."""

    def __init__(self, model_name: str):
        log(f"Loading embedding model: {model_name}")
        with suppress_stdout_stderr():
            self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Compute embeddings of shape (len(texts), embedding_dim)."""
        return self.model.encode(texts, convert_to_numpy=True)

    def cleanup(self) -> None:
        """Release model memory."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
