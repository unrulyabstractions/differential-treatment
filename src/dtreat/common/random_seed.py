"""Stable per-item seed derivation (mock backend, response sampling,
permutation streams): every (prompt, sample, ...) combination gets its own
reproducible RNG regardless of execution order."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(*parts: str | int) -> int:
    """Derive a stable 32-bit seed from arbitrary string/int parts.

    Used to give every (prompt, sample_index, ...) combination its own
    reproducible RNG stream regardless of execution order.
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")


def rng_for(*parts: str | int) -> np.random.Generator:
    """Get a numpy Generator seeded deterministically from parts."""
    return np.random.default_rng(derive_seed(*parts))
