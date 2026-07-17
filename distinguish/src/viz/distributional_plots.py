"""Distributional plots: per-variant C2ST nulls, ROC overlay, fold accuracies.

Each variant owns one CATEGORICAL_SLOTS color, assigned in the fixed order
the variants appear in the result (linear variants in config order, then
modernbert) and reused across the ROC and fold charts so color follows the
variant, never the chart. Chance is a dashed INK_MUTED hairline everywhere,
the observed statistic a solid TARGET_COLOR line; label text stays in ink
tokens, never series color.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import auc, roc_curve

from src.distributional.distributional_dimension import (
    C2stVariantResult,
    DistributionalResult,
)
from src.viz.plot_style import (
    CATEGORICAL_SLOTS,
    INK_MUTED,
    LINE_WIDTH,
    NULL_FILL,
    SURFACE,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_DASHED = (0, (4, 3))
_MAX_NULL_COLUMNS = 3
_PANEL_WIDTH = 4.6  # inches per null-histogram panel


def _variant_color(index: int) -> str:
    return CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)]


def plot_distributional(result: DistributionalResult, out_dir: Path) -> list[Path]:
    """Null-histogram grid, ROC overlay, and fold-accuracy dot rows as PNGs."""
    apply_plot_style()
    paths = []
    linear_variants = [v for v in result.variants if v.classifier == "linear"]
    if linear_variants:
        paths.append(_plot_null_histograms(linear_variants, out_dir))
    if result.variants:
        paths.append(_plot_roc_overlay(result.variants, out_dir))
        paths.append(_plot_fold_accuracies(result.variants, out_dir))
    return paths


def _plot_null_histograms(variants: list[C2stVariantResult], out_dir: Path) -> Path:
    """One permutation-null panel per linear variant, shared reference legend."""
    n_panels = len(variants)
    n_cols = min(n_panels, _MAX_NULL_COLUMNS)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(_PANEL_WIDTH * n_cols, 3.6 * n_rows), squeeze=False
    )
    flat = list(axes.ravel())
    for ax, variant in zip(flat, variants, strict=False):
        _draw_null_panel(ax, variant)
    for ax in flat[n_panels:]:
        ax.set_visible(False)

    # Headroom for suptitle/legend shrinks with row count in figure fractions.
    fig.suptitle(
        "Classifier two-sample test",
        y=1 + 0.20 / n_rows,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.legend(
        handles=[
            Patch(facecolor=NULL_FILL, label="permutation null"),
            Line2D(
                [],
                [],
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=_DASHED,
                label="chance",
            ),
            Line2D([], [], color=TARGET_COLOR, linewidth=LINE_WIDTH, label="observed"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.02, 1 + 0.02 / n_rows),
        ncols=3,
    )
    return save_figure(fig, out_dir / "distributional_c2st_null.png")


def _draw_null_panel(ax: plt.Axes, variant: C2stVariantResult) -> None:
    null_accuracies = np.asarray(variant.null_accuracies, dtype=float)
    style_axes(ax)
    lo = min(variant.chance_level, null_accuracies.min(), variant.accuracy) - 0.06
    hi = max(variant.chance_level, null_accuracies.max(), variant.accuracy) + 0.06
    ax.hist(
        null_accuracies,
        bins=np.linspace(lo, hi, 25),
        color=NULL_FILL,
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    ax.set_xlim(lo, hi)
    ax.axvline(variant.chance_level, color=INK_MUTED, linewidth=1.0, linestyle=_DASHED)
    ax.axvline(variant.accuracy, color=TARGET_COLOR, linewidth=LINE_WIDTH)
    # Every panel in this grid is linear, so the title is the embedder alone.
    headline(
        ax,
        variant.representation.split("/")[-1],
        f"accuracy {variant.accuracy:.2f} · p = {variant.p_value:.3g}",
    )
    ax.set_xlabel("held-out accuracy")
    ax.set_ylabel("permutations")


def _plot_roc_overlay(variants: list[C2stVariantResult], out_dir: Path) -> Path:
    """All variants' held-out ROC curves in one square panel, one slot each."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    style_axes(ax, grid_axis="both")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=_DASHED)

    handles = []
    auc_values = []
    for index, variant in enumerate(variants):
        fpr, tpr, _ = roc_curve(variant.heldout_labels, variant.heldout_scores)
        auc_value = auc(fpr, tpr)
        auc_values.append(auc_value)
        color = _variant_color(index)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=LINE_WIDTH,
            solid_joinstyle="round",
            solid_capstyle="round",
            # Earlier slots draw on top so coincident curves hide the later
            # variant, never the primary one; the legend still names both.
            zorder=3 + len(variants) - index,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=LINE_WIDTH,
                label=f"{variant.short_label} · AUC {auc_value:.2f}",
            )
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    stat = (
        f"best AUC = {max(auc_values):.2f}"
        if len(auc_values) > 1
        else (f"AUC = {auc_values[0]:.2f}")
    )
    headline(ax, "Held-out ROC", stat)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    legend_below(ax, handles, ncols=1)
    return save_figure(fig, out_dir / "distributional_roc_curve.png")


def _stacked_y_positions(values: list[float], row_y: float) -> list[float]:
    """Row y per dot, with exact ties fanned vertically so none fully hide."""
    totals = Counter(values)
    seen: Counter[float] = Counter()
    positions = []
    for value in values:
        offset = seen[value] - (totals[value] - 1) / 2
        seen[value] += 1
        positions.append(row_y + 0.15 * offset)
    return positions


def _plot_fold_accuracies(variants: list[C2stVariantResult], out_dir: Path) -> Path:
    """Dot rows of per-fold accuracies; one row per variant, slot colors."""
    n_rows = len(variants)
    fig, ax = plt.subplots(figsize=(7.4, 1.7 + 0.62 * n_rows))
    style_axes(ax, grid_axis="x")
    ax.axvline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=_DASHED, zorder=1)

    y_ticks, y_labels = [], []
    for index, variant in enumerate(variants):
        y = n_rows - 1 - index  # first variant on the top row
        ax.plot(
            variant.fold_accuracies,
            _stacked_y_positions(variant.fold_accuracies, y),
            "o",
            markersize=9,
            color=_variant_color(index),
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        y_ticks.append(y)
        y_labels.append(variant.short_label)

    all_accuracies = [a for v in variants for a in v.fold_accuracies]
    lo = min(0.5, min(all_accuracies)) - 0.06
    hi = min(max([*all_accuracies, 0.5]) + 0.06, 1.02)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_yticks(y_ticks, y_labels)
    ax.text(  # tiny ink-token tag on the chance reference line
        0.5 - 0.008 * (hi - lo),
        n_rows - 0.32,
        "chance",
        ha="right",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )

    n_folds = max(len(v.fold_accuracies) for v in variants)
    headline(ax, "Per-fold accuracy", f"{n_folds} author-grouped folds")
    ax.set_xlabel("held-out accuracy")
    return save_figure(fig, out_dir / "distributional_fold_accuracies.png")
