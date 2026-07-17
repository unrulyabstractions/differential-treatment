"""PROOF-GRADE calibration & correctness tests for MMD-Fuse.

Target: ``src.semantic.mmd_fuse_runner.run_mmd_fuse`` (the pipeline's only
semantic two-sample test, paper section 3.3.3). These tests are built to FAIL
if the wrapper were subtly wrong, not merely to check "it returns a float":

  * NULL          Monte-Carlo type-I-error calibration over ~300 seeds:
                  the permutation p-values must be ~Uniform(0,1) (KS) and the
                  false-positive rate at alpha must track alpha (binomial).
  * POWER         a mean shift and a covariance shift must be rejected w.h.p.
  * ALPHA THREADING  the ``rejected`` verdict must follow the *passed* alpha,
                  i.e. ``rejected == (p_value <= alpha)`` for every alpha -- the
                  regression guard for the just-fixed hardcoded-0.05 bug.
  * FLOAT16       feeding float16 embeddings must give the *identical* result to
                  the same values as float32 (the internal promotion works) and
                  never produce NaN; a contrast against the un-promoted path
                  proves the promotion is load-bearing.
  * DETERMINISM   same seed -> byte-identical verdict and p-value.

Everything is seeded; trial counts are modest but large enough for real signal.
Run: ``uv run pytest tests/test_mmd_fuse_calibration.py -q``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from mmdfuse import mmdfuse
from scipy import stats

from src.semantic.mmd_fuse_runner import MmdFuseOutcome, run_mmd_fuse

# --- shared, fully reproducible sampling knobs -----------------------------
_NULL_TRIALS = 300  # Monte-Carlo seeds for the null calibration
_NULL_N = 25  # samples per cloud (kept fixed so jax JIT-caches the shape)
_NULL_D = 6  # embedding dimensionality
_MASTER_SEED = 20260707


def _gaussian(rng: np.random.Generator, n: int, d: int, loc=0.0, scale=1.0):
    """A float32 embedding cloud, matching what the pipeline hands the runner."""
    return (rng.standard_normal((n, d)) * scale + loc).astype(np.float32)


def _null_pvalues() -> np.ndarray:
    """Monte-Carlo null: both clouds i.i.d. N(0, I_d), independent per trial.

    Under H0 a valid permutation test yields p-values that are ~Uniform(0,1).
    A single deterministic master seed makes the whole array reproducible so the
    KS/binomial verdicts below are stable, not flaky.
    """
    rng = np.random.default_rng(_MASTER_SEED)
    pvals = np.empty(_NULL_TRIALS)
    for i in range(_NULL_TRIALS):
        x = _gaussian(rng, _NULL_N, _NULL_D)
        y = _gaussian(rng, _NULL_N, _NULL_D)
        seed = int(rng.integers(2**31))
        pvals[i] = run_mmd_fuse(x, y, seed=seed).p_value
    return pvals


# ---------------------------------------------------------------------------
# (1) NULL  -- type-I-error calibration
# ---------------------------------------------------------------------------
def test_null_pvalues_are_uniform_ks() -> None:
    """Under identical distributions the p-values must be ~Uniform(0,1).

    A miscalibrated test (wrong permutation count, bad bandwidth, off-by-one in
    the rank p-value) would skew this distribution and the KS test would reject
    uniformity. Threshold is loose (only gross miscalibration fails), yet the
    shape check is real: mean pinned near 0.5 and full [0,1] support.
    """
    pvals = _null_pvalues()

    assert np.all((pvals >= 0.0) & (pvals <= 1.0)), "p-values escaped [0, 1]"
    # Uniform mean is 0.5; std of the sample mean over 300 draws ~= 0.017, so a
    # [0.42, 0.58] window is ~5 sigma -- catches any systematic bias, not noise.
    assert 0.42 < pvals.mean() < 0.58, f"null p-value mean {pvals.mean():.3f} biased"
    # Genuine spread across the unit interval (not clumped at one value).
    assert pvals.min() < 0.1 and pvals.max() > 0.9, "null p-values lack full support"

    ks = stats.kstest(pvals, "uniform")
    assert ks.pvalue > 0.005, (
        f"null p-values not Uniform(0,1): KS stat={ks.statistic:.3f}, "
        f"KS p={ks.pvalue:.4g}"
    )


def test_null_false_positive_rate_tracks_alpha() -> None:
    """FPR(alpha) must be ~= alpha and, crucially, controlled at <= alpha.

    For each alpha, the number of rejections over the null trials is Binomial(N,
    alpha) iff the test is calibrated. A two-sided binomial test guards both
    over-rejection (broken type-I control -> real bug) and gross under-coverage.
    """
    pvals = _null_pvalues()
    for alpha in (0.05, 0.10, 0.20):
        n_reject = int((pvals <= alpha).sum())
        fpr = n_reject / _NULL_TRIALS
        bt = stats.binomtest(n_reject, _NULL_TRIALS, alpha)
        assert bt.pvalue > 0.001, (
            f"FPR at alpha={alpha} is {fpr:.3f} ({n_reject}/{_NULL_TRIALS}); "
            f"binomial calibration p={bt.pvalue:.4g}"
        )
        # Hard type-I-error ceiling: never rejects far more often than alpha.
        assert fpr <= alpha + 0.05, f"FPR {fpr:.3f} exceeds alpha={alpha} ceiling"


# ---------------------------------------------------------------------------
# (2) POWER  -- must detect real differences
# ---------------------------------------------------------------------------
def test_power_mean_shift_is_rejected() -> None:
    """A clear mean shift between the clouds must be rejected almost always."""
    rng = np.random.default_rng(101)
    trials, rejects = 30, 0
    for _ in range(trials):
        x = _gaussian(rng, 30, _NULL_D, loc=0.0)
        y = _gaussian(rng, 30, _NULL_D, loc=1.2)
        rejects += run_mmd_fuse(x, y, seed=int(rng.integers(2**31))).rejected
    power = rejects / trials
    assert power >= 0.9, f"mean-shift power only {power:.2f} (expected ~1.0)"


def test_power_covariance_shift_is_rejected() -> None:
    """A scale/covariance shift (same mean) must also be rejected w.h.p.

    Guards against a test that only ever notices first moments -- MMD-Fuse's
    kernels should catch a variance difference too.
    """
    rng = np.random.default_rng(202)
    trials, rejects = 30, 0
    for _ in range(trials):
        x = _gaussian(rng, 30, _NULL_D, scale=1.0)
        y = _gaussian(rng, 30, _NULL_D, scale=2.5)
        rejects += run_mmd_fuse(x, y, seed=int(rng.integers(2**31))).rejected
    power = rejects / trials
    assert power >= 0.9, f"covariance-shift power only {power:.2f} (expected ~1.0)"


# ---------------------------------------------------------------------------
# (3) ALPHA THREADING  -- regression guard for the hardcoded-0.05 bug
# ---------------------------------------------------------------------------
def test_rejected_follows_passed_alpha_not_hardcoded() -> None:
    """``rejected`` must equal ``p_value <= alpha`` for the alpha actually passed.

    The just-fixed bug compared against a hardcoded 0.05 regardless of the caller
    alpha. To PROVE alpha is honored we build clouds spanning a range of p-values
    and, for each, sweep an alpha grid straddling that p-value. We further assert
    the sweep actually exercised at least one (dataset, alpha) pair whose correct
    verdict DIFFERS from the hardcoded-0.05 verdict -- so a reintroduced hardcode
    would necessarily flip a checked assertion.
    """
    rng = np.random.default_rng(303)
    alphas = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    # Varied separations => a spread of p-values across (0, 1).
    shifts = (0.0, 0.0, 0.3, 0.4, 0.5, 0.7, 0.9)

    diverged_from_hardcoded = False
    for shift in shifts:
        x = _gaussian(rng, 22, 5, loc=0.0)
        y = _gaussian(rng, 22, 5, loc=shift)
        seed = int(rng.integers(2**31))
        # p_value is a property of the data+seed, independent of alpha.
        p = run_mmd_fuse(x, y, seed=seed, alpha=0.05).p_value
        for alpha in alphas:
            out = run_mmd_fuse(x, y, seed=seed, alpha=alpha)
            assert out.p_value == pytest.approx(p), (
                "p_value must not depend on alpha "
                f"(alpha={alpha}: {out.p_value} vs {p})"
            )
            assert out.rejected == (p <= alpha), (
                f"rejected={out.rejected} but p={p:.4f}, alpha={alpha}: "
                "verdict does not track the passed alpha"
            )
            if (p <= alpha) != (p <= 0.05):
                diverged_from_hardcoded = True

    assert diverged_from_hardcoded, (
        "test did not exercise an alpha whose verdict differs from hardcoded "
        "0.05; it cannot distinguish the bug -- widen the p-value spread"
    )


# ---------------------------------------------------------------------------
# (4) FLOAT16  -- the internal promotion to float32 works and is load-bearing
# ---------------------------------------------------------------------------
def test_float16_input_matches_float32_and_no_nan() -> None:
    """float16 embeddings must yield the IDENTICAL result to the same float32.

    Residual-stream extractors run in float16 on MPS. The runner promotes to
    float32 before jax; so run_mmd_fuse(x16) must exactly equal
    run_mmd_fuse(x16.astype(float32)). We use float16 values as the common
    reference (not float32 downcast to float16) so equality is exact, then also
    show the un-promoted path diverges -- proving the promotion is not a no-op.
    """
    rng = np.random.default_rng(404)
    # Residual-stream-scale magnitudes stress half precision.
    x32 = (rng.standard_normal((30, 8)) * 40 + 60).astype(np.float32)
    y32 = (rng.standard_normal((30, 8)) * 40 + 75).astype(np.float32)
    x16, y16 = x32.astype(np.float16), y32.astype(np.float16)

    # Exact reference: the float16 values, promoted to float32 by the caller.
    ref = run_mmd_fuse(x16.astype(np.float32), y16.astype(np.float32), seed=7)
    # Under test: raw float16 in, promoted internally.
    got = run_mmd_fuse(x16, y16, seed=7)

    assert not np.isnan(got.p_value), "float16 input produced a NaN p-value"
    assert got.p_value == ref.p_value, (
        f"float16 promotion changed the p-value: {got.p_value} != {ref.p_value}"
    )
    assert got.rejected == ref.rejected
    assert isinstance(got, MmdFuseOutcome)

    # Load-bearing check: WITHOUT promotion (raw float16 all the way to mmdfuse)
    # the kernel-bandwidth medians are distorted, so the p-value differs. If this
    # ever stopped differing, the equality above would be a vacuous tautology.
    x16_j = jnp.asarray(x16)  # jax keeps float16 dtype
    assert x16_j.dtype == jnp.float16
    _v_unpromoted, p_unpromoted = mmdfuse(
        jnp.asarray(x16),
        jnp.asarray(y16),
        random.PRNGKey(7),
        return_p_val=True,
    )
    assert float(p_unpromoted) != ref.p_value, (
        "un-promoted float16 path matches float32; the float16 test is vacuous"
    )


# ---------------------------------------------------------------------------
# (5) DETERMINISM  -- same seed => identical outcome
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 13, 4242])
def test_same_seed_is_deterministic(seed: int) -> None:
    """Two calls with the same seed on the same data must be byte-identical."""
    rng = np.random.default_rng(505 + seed)
    x = _gaussian(rng, 28, 7, loc=0.0)
    y = _gaussian(rng, 28, 7, loc=0.4)
    a = run_mmd_fuse(x, y, seed=seed)
    b = run_mmd_fuse(x, y, seed=seed)
    assert a.p_value == b.p_value, "identical seed produced different p-values"
    assert a.rejected == b.rejected, "identical seed produced different verdicts"


def test_seed_drives_the_permutation_resample() -> None:
    """The seed must be threaded into MMD-Fuse's permutation RNG, not ignored.

    On non-saturated (null) data the permutation p-value is genuinely random in
    the seed, so distinct seeds must yield distinct p-values. (A saturated strong
    signal instead pins every seed at the floor 1/(B+1); that is correct but
    would make this probe vacuous, so we deliberately use null data here.) If the
    seed were dropped, all p-values would collapse to one value and this fails.
    """
    rng = np.random.default_rng(606)
    x = _gaussian(rng, 30, 6, loc=0.0)
    y = _gaussian(rng, 30, 6, loc=0.0)  # pure null -> p-values spread, not floored
    pvals = {run_mmd_fuse(x, y, seed=s).p_value for s in (1, 2, 3, 4, 5, 6)}
    assert len(pvals) >= 2, (
        f"seed appears ignored: {len(pvals)} distinct p-value(s) across 6 seeds"
    )
