"""Calibration-justification plots for the Mickel et al. marked-words prior.

Three diagnostics, written into ``lexical/calibration/`` when the section's
``calibration_plots`` flag is set:

- **Null z QQ** — under author-level label permutation (no real group signal),
  a well-calibrated statistic yields z-scores on the N(0,1) diagonal. Fixed and
  mickel priors are overlaid so over-dispersion (heavy tails = false positives)
  is visible.
- **Constant sweep** — how many words survive as the calibration constant C
  (hence the prior strength) sweeps a log range, under both significance rules.
  The chosen operating point is marked.
- **Rank plot** — the top-|z| words, colored register (in the MotS calibration
  word set) vs signature, so the reader sees that calibration clears register
  words out of the top ranks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

from src.common.stats_utils import permute_labels_by_author
from src.lexical.lexical_dimension import LexicalResult
from src.lexical.marked_words_analyzer import (
    marked_words_z_scores,
    significance_mask,
)
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_COLOR,
    BASELINE_SET_COLOR,
    CATEGORICAL_SLOTS,
    INK_MUTED,
    INK_SECONDARY,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_MICKEL_COLOR = TARGET_COLOR
_FIXED_COLOR = CATEGORICAL_SLOTS[7]  # orange; never a set-identity color here
_N_PERMUTATIONS = 20
_SWEEP_POINTS = 24
_SWEEP_LO, _SWEEP_HI = 5e-3, 1.0  # calibration-constant sweep range (log x)
_RANK_WORDS = 30
_PERM_SEED = 0


def plot_lexical_calibration(result: LexicalResult, out_dir: Path) -> list[Path]:
    """Render the three calibration-justification figures into `out_dir`."""
    apply_plot_style()
    if not result.all_words:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_null_qq(result, out_dir / "lexical_null_qq.png"),
        _plot_constant_sweep(result, out_dir / "lexical_constant_sweep.png"),
        _plot_rank(result, out_dir / "lexical_rank.png"),
    ]


def _split_by_label(
    texts: list[str], labels: np.ndarray
) -> tuple[list[str], list[str]]:
    target = [t for t, y in zip(texts, labels, strict=True) if y == 1]
    baseline = [t for t, y in zip(texts, labels, strict=True) if y == 0]
    return target, baseline


def _mode_z_scores(
    texts_target: list[str], texts_baseline: list[str], result: LexicalResult, mode: str
) -> np.ndarray:
    """z-scores over the shared vocabulary under one prior mode."""
    _, _, z_scores, _, _ = marked_words_z_scores(
        texts_target,
        texts_baseline,
        min_word_count=result.min_word_count,
        reference_corpus=result.reference_corpus,
        prior_calibration=mode,
        reference_prior_weight=result.reference_prior_weight,
        calibration_constant=result.calibration_constant,
        english_prior_weight=result.english_prior_weight,
        prior_strength=result.prior_strength,
    )
    return z_scores


def _plot_null_qq(result: LexicalResult, path: Path) -> Path:
    """Permuted-null z-scores vs N(0,1) quantiles; fixed and mickel overlaid."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    style_axes(ax, grid_axis="both")
    headline(
        ax,
        "Null z calibration",
        f"{_N_PERMUTATIONS} author-permuted pairs · N(0,1) reference",
    )
    ax.set_xlabel("theoretical N(0,1) quantile")
    ax.set_ylabel("permuted-null z quantile")

    texts, labels = result._pooled_texts, np.array(result._pooled_labels)
    author_ids = result._pooled_author_ids
    can_permute = bool(texts) and len(set(author_ids)) > 2
    if can_permute:
        rng = np.random.default_rng(_PERM_SEED)
        pooled: dict[str, list[np.ndarray]] = {"mickel": [], "fixed": []}
        for _ in range(_N_PERMUTATIONS):
            permuted = permute_labels_by_author(author_ids, labels, rng)
            if permuted.sum() == 0 or permuted.sum() == len(permuted):
                continue
            t_texts, b_texts = _split_by_label(texts, permuted)
            for mode in ("mickel", "fixed"):
                z = _mode_z_scores(t_texts, b_texts, result, mode)
                if z.size:
                    pooled[mode].append(z)
        for mode, color, label in (
            ("mickel", _MICKEL_COLOR, "mickel (calibrated)"),
            ("fixed", _FIXED_COLOR, "fixed prior"),
        ):
            if not pooled[mode]:
                continue
            z_sorted = np.sort(np.concatenate(pooled[mode]))
            quantiles = norm.ppf((np.arange(z_sorted.size) + 0.5) / z_sorted.size)
            ax.plot(quantiles, z_sorted, color=color, linewidth=2.0, label=label)

    lim = 4.2
    ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    if can_permute:
        legend_below(
            ax,
            [
                Line2D([], [], color=_MICKEL_COLOR, linewidth=2.0, label="mickel"),
                Line2D([], [], color=_FIXED_COLOR, linewidth=2.0, label="fixed"),
                Line2D(
                    [],
                    [],
                    color=INK_MUTED,
                    linewidth=1.0,
                    linestyle="--",
                    label="N(0,1)",
                ),
            ],
            ncols=3,
        )
    else:
        ax.text(
            0,
            0,
            "too few authors to permute",
            ha="center",
            va="center",
            color=INK_MUTED,
            fontsize=10,
        )
    return save_figure(fig, path)


def _plot_constant_sweep(result: LexicalResult, path: Path) -> Path:
    """n_significant vs the calibration constant C (log x), both sig rules."""
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    style_axes(ax, grid_axis="y")
    headline(ax, "Significant words vs prior strength", "mickel prior · both sig rules")
    ax.set_xlabel("calibration constant C  (smaller → stronger prior)")
    ax.set_ylabel("significant words")
    ax.set_xscale("log")

    t_texts, b_texts = _split_by_label(
        result._pooled_texts, np.array(result._pooled_labels)
    )
    have_texts = bool(t_texts) and bool(b_texts)
    if have_texts:
        constants = np.geomspace(_SWEEP_LO, _SWEEP_HI, _SWEEP_POINTS)
        counts = {"bh_fdr": [], "raw_z": []}
        for c in constants:
            _, _, z, _, _ = marked_words_z_scores(
                t_texts,
                b_texts,
                min_word_count=result.min_word_count,
                reference_corpus=result.reference_corpus,
                prior_calibration="mickel",
                reference_prior_weight=result.reference_prior_weight,
                calibration_constant=float(c),
            )
            for rule in ("bh_fdr", "raw_z"):
                _, _, rejected = significance_mask(z, rule, result.fdr_alpha)
                counts[rule].append(int(rejected.sum()))
        for rule, color, label in (
            ("bh_fdr", TARGET_COLOR, "BH-FDR"),
            ("raw_z", _FIXED_COLOR, "raw |z| ≥ 1.96"),
        ):
            ax.plot(
                constants,
                counts[rule],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=label,
            )
        ax.axvline(
            result.calibration_constant, color=INK_MUTED, linewidth=1.0, linestyle="--"
        )
        ax.annotate(
            f"C = {result.calibration_constant:g}",
            xy=(result.calibration_constant, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.set_ylim(bottom=0)
        legend_below(
            ax,
            [
                Line2D([], [], color=TARGET_COLOR, linewidth=2.0, label="BH-FDR"),
                Line2D(
                    [], [], color=_FIXED_COLOR, linewidth=2.0, label="raw |z| ≥ 1.96"
                ),
            ],
        )
    else:
        ax.text(
            0.5,
            0.5,
            "no texts available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=INK_MUTED,
            fontsize=10,
        )
    return save_figure(fig, path)


def _plot_rank(result: LexicalResult, path: Path) -> Path:
    """Top-|z| words, colored register (calibration set) vs signature."""
    words = result.all_words[:_RANK_WORDS]
    register = set(result.calibration_words)
    fig_height = max(3.4, 0.30 * len(words) + 1.9)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    style_axes(ax, grid_axis="x")
    n_register = sum(1 for w in words if w.word in register)
    headline(
        ax,
        "Top marked words by |z|",
        f"{n_register} of {len(words)} are register (calibration set)",
    )
    ax.set_xlabel("|log-odds z|")

    y_positions = np.arange(len(words))[::-1]
    abs_z = np.array([abs(w.z_score) for w in words])

    def _color(word) -> str:
        if word.word in register:
            return INK_MUTED
        return TARGET_COLOR if word.z_score > 0 else BASELINE_SET_COLOR

    ax.barh(
        y_positions,
        abs_z,
        height=BAR_THICKNESS,
        color=[_color(w) for w in words],
    )
    ax.axvline(1.96, color=BASELINE_COLOR, linewidth=1.0, linestyle="--")
    ax.annotate(
        "|z| = 1.96",
        xy=(1.96, len(words) - 0.5),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_yticks(y_positions, labels=[w.word for w in words])
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
    ax.set_ylim(-0.7, len(words) - 0.3)
    legend_below(
        ax,
        [
            Patch(facecolor=TARGET_COLOR, label=f"signature · {result.target_label}"),
            Patch(
                facecolor=BASELINE_SET_COLOR,
                label=f"signature · {result.baseline_label}",
            ),
            Patch(facecolor=INK_MUTED, label="register (calibration set)"),
        ],
        ncols=3,
    )
    return save_figure(fig, path)
