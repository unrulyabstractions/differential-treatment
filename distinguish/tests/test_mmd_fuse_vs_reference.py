"""Differential test: our ``run_mmd_fuse`` wrapper vs a DIRECT ``mmdfuse()`` call.

The strongest fidelity check is not "our p-value looks reasonable" but "our
wrapper returns exactly what the reference ``mmdfuse`` returns on the same inputs,
key, and alpha". ``src.semantic.mmd_fuse_runner.run_mmd_fuse`` is meant to be a
thin, faithful wrapper over the pip-installed MMD-Fuse reference (Biggs, Schrab &
Gretton, "MMD-FUSE" 2023; github.com/antoninschrab/mmdfuse). Here we pin the
wrapper to a byte-identical head-to-head against ``mmdfuse.mmdfuse`` itself:

  * IDENTITY  wrapper ``(rejected, p_value)`` == direct
              ``mmdfuse(X, Y, PRNGKey(seed), alpha, return_p_val=True)`` for
              reject / null / unequal-sample-size regimes. Because we compare
              against the reference run with *its own defaults*, this also proves
              the wrapper does not silently distort the kernel list, bandwidth
              count, permutation count, or lambda -- any such override would make
              the p-values disagree.
  * DTYPE     the wrapper's explicit float32 promotion does not change the result
              vs the reference's own dtype handling (float32 in is a literal
              no-op; float64 in is downcast to float32 by jax under the pipeline's
              default ``jax_enable_x64=False``).
  * ALPHA     the passed alpha reaches the verdict: ``rejected == (p_value <=
              alpha)`` and equals the direct call's verdict at that same alpha.
  * DEFAULTS  the reference defaults the wrapper *silently relies on*
              (``kernels``, ``lambda_multiplier``, ``number_bandwidths``,
              ``number_permutations``) are still the values MMD-Fuse ships, which
              are the paper's recommended configuration. If upstream ever changes
              them, this guard fires.

The reference is imported directly (it is pip-installed), so this is a genuine
differential test, not a reimplementation. Run:
``uv run pytest tests/test_mmd_fuse_vs_reference.py -q``.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from mmdfuse import mmdfuse
from numpy.typing import NDArray

from src.semantic.mmd_fuse_runner import MmdFuseOutcome, run_mmd_fuse

_DATA_SEED = 20260707


def _cloud(
    rng: np.random.Generator, n: int, d: int, loc: float = 0.0, scale: float = 1.0
) -> NDArray[np.float32]:
    """A float32 embedding cloud, matching what the pipeline hands the runner."""
    return (rng.standard_normal((n, d)) * scale + loc).astype(np.float32)


def _direct(
    x: NDArray[np.floating], y: NDArray[np.floating], seed: int, alpha: float
) -> tuple[bool, float]:
    """Call the reference exactly as the wrapper does, with reference defaults."""
    verdict, p_value = mmdfuse(
        jnp.asarray(x),
        jnp.asarray(y),
        random.PRNGKey(seed),
        alpha=alpha,
        return_p_val=True,
    )
    return bool(verdict), float(p_value)


def _scenarios() -> list[
    tuple[str, NDArray[np.float32], NDArray[np.float32], int, float]
]:
    """(name, X, Y, seed, alpha) covering reject, null, and swap regimes.

    The reference internally swaps X and Y when ``Y.shape[0] > X.shape[0]``; the
    two unequal-size cases exercise both orientations so any wrapper mishandling
    of that branch would surface as a p-value mismatch.
    """
    rng = np.random.default_rng(_DATA_SEED)
    return [
        (
            "reject_mean_shift",
            _cloud(rng, 40, 8, 0.0),
            _cloud(rng, 40, 8, 1.2),
            123,
            0.05,
        ),
        ("null_same_dist", _cloud(rng, 45, 6), _cloud(rng, 45, 6), 2024, 0.05),
        (
            "unequal_baseline_larger",
            _cloud(rng, 30, 7),
            _cloud(rng, 52, 7, 0.6),
            77,
            0.10,
        ),
        (
            "unequal_target_larger",
            _cloud(rng, 55, 5, 0.5),
            _cloud(rng, 33, 5),
            88,
            0.20,
        ),
    ]


_SCENARIOS = _scenarios()


@pytest.mark.parametrize(
    ("name", "x", "y", "seed", "alpha"),
    _SCENARIOS,
    ids=[s[0] for s in _SCENARIOS],
)
def test_wrapper_equals_direct_mmdfuse(
    name: str,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    seed: int,
    alpha: float,
) -> None:
    """Wrapper output is byte-identical to a direct ``mmdfuse`` call.

    The wrapper only overrides ``alpha`` and ``return_p_val``; everything else
    (kernels, lambda, bandwidth/permutation counts) is inherited from the
    reference defaults, which is what ``_direct`` also uses. Exact equality holds
    because both paths run the identical jitted computation on the identical
    float32 arrays with the identical PRNGKey.
    """
    out = run_mmd_fuse(x, y, seed, alpha)
    ref_reject, ref_p = _direct(x, y, seed, alpha)

    assert isinstance(out, MmdFuseOutcome)
    assert isinstance(out.p_value, float) and isinstance(out.rejected, bool)
    assert out.p_value == ref_p, (
        f"[{name}] wrapper p={out.p_value!r} != direct p={ref_p!r}"
    )
    assert out.rejected == ref_reject, (
        f"[{name}] wrapper reject={out.rejected} != direct reject={ref_reject}"
    )
    # The passed alpha reaches the reject decision, on both sides identically.
    assert out.rejected == (out.p_value <= alpha), (
        f"[{name}] verdict {out.rejected} does not track p={out.p_value:.6f} "
        f"<= alpha={alpha}"
    )
    assert ref_reject == (ref_p <= alpha)


def test_null_and_reject_scenarios_are_both_exercised() -> None:
    """Guard: the identity sweep must contain at least one reject AND one non-reject.

    Otherwise the ``rejected == (p <= alpha)`` equality above could pass vacuously
    (e.g. everything rejected). This pins the fixtures to keep both signs live.
    """
    verdicts = {
        name: run_mmd_fuse(x, y, seed, alpha).rejected
        for name, x, y, seed, alpha in _SCENARIOS
    }
    assert any(verdicts.values()), f"no scenario rejects: {verdicts}"
    assert not all(verdicts.values()), f"no scenario fails to reject: {verdicts}"


def test_float32_promotion_matches_reference_dtype_handling() -> None:
    """The wrapper's float32 promotion equals the reference's own dtype handling.

    Two ways the reference could see the data, both of which the promotion must
    reproduce exactly:
      (i)  the same values already as float32 -- promotion is a literal no-op;
      (ii) the raw float64 array -- under the pipeline default
           ``jax_enable_x64=False`` jax downcasts float64 -> float32, so the
           wrapper's explicit ``astype(float32)`` changes nothing.
    (Half precision is the case the promotion actually guards; that it is
    load-bearing there is proven in test_mmd_fuse_calibration.py.)
    """
    rng = np.random.default_rng(_DATA_SEED + 1)
    x64 = rng.standard_normal((40, 8)) + 0.4  # float64, mean-shifted
    y64 = rng.standard_normal((36, 8))  # float64
    seed, alpha = 505, 0.05

    wrap = run_mmd_fuse(x64, y64, seed, alpha)

    # (i) reference handed the identical values already as float32.
    ref_reject32, ref_p32 = _direct(
        x64.astype(np.float32), y64.astype(np.float32), seed, alpha
    )
    assert wrap.p_value == ref_p32
    assert wrap.rejected == ref_reject32

    # (ii) reference handed the raw float64 array; jax downcasts it itself.
    assert not jax.config.read("jax_enable_x64"), (
        "test assumes the pipeline default jax_enable_x64=False; with x64 on, an "
        "explicit float32 promotion is a real (and intended) change, not a no-op"
    )
    ref_reject64, ref_p64 = _direct(x64, y64, seed, alpha)
    assert wrap.p_value == ref_p64, (
        f"promotion changed the result vs jax's own float64->float32 downcast: "
        f"{wrap.p_value!r} != {ref_p64!r}"
    )
    assert wrap.rejected == ref_reject64


def test_wrapper_relies_only_on_faithful_reference_defaults() -> None:
    """Pin the reference defaults the wrapper silently inherits.

    The wrapper passes only ``X, Y, key, alpha, return_p_val`` and lets
    ``kernels``, ``lambda_multiplier``, ``number_bandwidths`` and
    ``number_permutations`` fall through to ``mmdfuse``'s defaults -- which are
    the MMD-FUSE paper's configuration (Laplace+Gaussian collection, 10
    bandwidths per kernel, 2000 permutations, lambda multiplier 1). If a future
    upstream release changes any of these, the wrapper's behaviour would shift
    without any change on our side; this guard fires so that is never silent.
    """
    defaults = {
        name: param.default
        for name, param in inspect.signature(mmdfuse).parameters.items()
    }
    assert defaults["kernels"] == ("laplace", "gaussian")
    assert defaults["lambda_multiplier"] == 1
    assert defaults["number_bandwidths"] == 10
    assert defaults["number_permutations"] == 2000
    assert defaults["alpha"] == 0.05
    assert defaults["return_p_val"] is False
