"""Thin wrapper over MMD-Fuse (Biggs, Schrab & Gretton 2023, `mmdfuse` package).

Keeps the jax surface confined to one module: callers hand in numpy embedding
matrices and get back a plain BaseSchema outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import random
from mmdfuse import mmdfuse
from numpy.typing import NDArray

from src.common.base_schema import BaseSchema


@dataclass
class MmdFuseOutcome(BaseSchema):
    """Verdict of one MMD-Fuse two-sample test."""

    rejected: bool
    p_value: float


def run_mmd_fuse(
    embeddings_target: NDArray[np.floating],
    embeddings_baseline: NDArray[np.floating],
    seed: int,
    alpha: float = 0.05,
) -> MmdFuseOutcome:
    """Test H0: both embedding clouds are drawn from the same distribution.

    `alpha` governs the reject verdict; pass the pipeline's significance level so
    `rejected` and the caller's p-value threshold agree. Embeddings are promoted
    to float32 before jax sees them: residual-stream extractors run in float16 on
    MPS, and half precision distorts the kernel bandwidth medians MMD-Fuse
    computes internally.
    """
    matrix_target = jnp.asarray(np.asarray(embeddings_target, dtype=np.float32))
    matrix_baseline = jnp.asarray(np.asarray(embeddings_baseline, dtype=np.float32))
    verdict, p_value = mmdfuse(
        matrix_target,
        matrix_baseline,
        random.PRNGKey(seed),
        alpha=alpha,
        return_p_val=True,
    )
    return MmdFuseOutcome(rejected=bool(verdict), p_value=float(p_value))
