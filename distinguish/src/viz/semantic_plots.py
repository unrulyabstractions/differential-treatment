"""Semantic-dimension figures: PCA scatter panels + cosine-similarity histograms.

Both figures show one panel per successful embedding variant, gridded at up
to three panels per row. Target is always blue, baseline always aqua;
between-set similarity mass is muted ink. Headings stay minimal — variant
label plus p-value per panel, one short figure title — with method detail
left to semantic.json. Skipped variants appear as one muted note line.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.semantic.semantic_dimension import SemanticResult, SemanticVariantResult
from src.viz.plot_style import (
    BASELINE_SET_COLOR,
    INK_MUTED,
    LINE_WIDTH,
    SURFACE,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    save_figure,
    style_axes,
)

_MARKER_AREA = 38.0  # points^2, comfortably above the 8px-area minimum
_EDGE_WIDTH = 1.2  # surface-color ring keeps overlapping points separable
_PANEL_WIDTH = 4.6  # inches per variant panel
_PCA_PANEL_HEIGHT = 4.4
_HIST_PANEL_HEIGHT = 3.9
_MAX_COLUMNS = 3
# Vertical chrome in fixed inches so 1-row and 3-row grids look alike.
_TITLE_GAP = 0.62  # figure title above the top edge
_LEGEND_GAP = 0.07  # legend between the title and the top row
_TOP_MARGIN = 0.55  # space above the top row for each panel's headline
_BOTTOM_MARGIN = 0.5  # space under the bottom row for x-axis labels
_NOTE_GAP = 0.14  # skipped-variants note below the figure


def plot_semantic(result: SemanticResult, out_dir: Path) -> list[Path]:
    """Render the PCA-projection and similarity-histogram figures."""
    apply_plot_style()
    return [
        _plot_pca_projections(result, out_dir),
        _plot_similarity_histograms(result, out_dir),
    ]


def _plot_pca_projections(result: SemanticResult, out_dir: Path) -> Path:
    """One joint-PCA scatter panel per successful embedding variant."""
    fig, panels = _panel_grid(len(result.variants), _PCA_PANEL_HEIGHT, hspace=0.30)
    for ax, variant in zip(panels, result.variants, strict=True):
        _draw_projection_panel(ax, variant, result)
    # One shared legend: set identity is constant across panels.
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=label)
        for color, label in [
            (TARGET_COLOR, result.target_label),
            (BASELINE_SET_COLOR, result.baseline_label),
        ]
    ]
    _finish_figure(fig, "Semantic embeddings (MMD-Fuse)", handles, 2, result)
    return save_figure(fig, out_dir / "semantic_pca_projections.png")


def _plot_similarity_histograms(result: SemanticResult, out_dir: Path) -> Path:
    """Overlaid step histograms of pairwise cosine similarity per variant."""
    fig, panels = _panel_grid(len(result.variants), _HIST_PANEL_HEIGHT, hspace=0.35)
    for index, (ax, variant) in enumerate(zip(panels, result.variants, strict=True)):
        _draw_similarity_panel(ax, variant, show_ylabel=index % _MAX_COLUMNS == 0)
    handles = [
        Line2D([], [], color=color, linewidth=LINE_WIDTH, label=label)
        for color, label in [
            (TARGET_COLOR, f"within {result.target_label}"),
            (BASELINE_SET_COLOR, f"within {result.baseline_label}"),
            (INK_MUTED, "between sets"),
        ]
    ]
    _finish_figure(fig, "Pairwise cosine similarity", handles, 3, result)
    return save_figure(fig, out_dir / "semantic_similarity_histograms.png")


def _panel_grid(
    n_panels: int, panel_height: float, hspace: float
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Grid of axes, up to _MAX_COLUMNS wide; surplus axes are hidden."""
    n_columns = min(max(n_panels, 1), _MAX_COLUMNS)
    n_rows = max(1, math.ceil(n_panels / _MAX_COLUMNS))
    height = panel_height * n_rows
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(_PANEL_WIDTH * n_columns, height), squeeze=False
    )
    fig.subplots_adjust(
        top=1 - _TOP_MARGIN / height,
        bottom=_BOTTOM_MARGIN / height,
        wspace=0.32,
        hspace=hspace,
    )
    panels = list(axes.ravel())
    for ax in panels[n_panels:]:
        ax.set_visible(False)
    return fig, panels[:n_panels]


def _finish_figure(
    fig: plt.Figure,
    title: str,
    handles: list[Line2D],
    legend_columns: int,
    result: SemanticResult,
) -> None:
    """Title + shared legend above the grid, muted skip/empty notes."""
    height = fig.get_size_inches()[1]
    fig.suptitle(title, y=1 + _TITLE_GAP / height, fontweight="bold")
    if result.variants:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1 + _LEGEND_GAP / height),
            ncols=legend_columns,
        )
    else:
        fig.text(0.5, 0.5, "no embedding variants ran", ha="center", color=INK_MUTED)
    if result.skipped_variants:
        fig.text(
            0.5,
            -_NOTE_GAP / height,
            "skipped: " + " · ".join(result.skipped_variants),
            ha="center",
            va="top",
            fontsize=9,
            color=INK_MUTED,
        )


def _display_label(variant: str) -> str:
    """Compact panel title: model paths reduced to their last component."""
    prefix, _, model = variant.partition(":")
    return f"{prefix}:{model.split('/')[-1]}"


def _draw_projection_panel(
    ax: plt.Axes, variant: SemanticVariantResult, result: SemanticResult
) -> None:
    projection = variant.projection
    for x, y, color in [
        (projection.target_x, projection.target_y, TARGET_COLOR),
        (projection.baseline_x, projection.baseline_y, BASELINE_SET_COLOR),
    ]:
        ax.scatter(
            x,
            y,
            s=_MARKER_AREA,
            color=color,
            edgecolors=SURFACE,
            linewidths=_EDGE_WIDTH,
            zorder=3,
        )
    style_axes(ax, grid_axis="both")
    ax.set_xlabel(f"PC1 ({projection.variance_ratio_1:.0%} var)")
    ax.set_ylabel(f"PC2 ({projection.variance_ratio_2:.0%} var)")
    headline(ax, _display_label(variant.variant), f"p = {variant.p_value:.4g}")


def _draw_similarity_panel(
    ax: plt.Axes, variant: SemanticVariantResult, show_ylabel: bool
) -> None:
    similarity = variant.similarity
    pooled = similarity.within_target + similarity.within_baseline + similarity.between
    # Bin count follows the smallest group so sparse smoke-sized inputs stay
    # smooth while full 500-pair runs keep their resolution.
    smallest = min(
        len(similarity.within_target),
        len(similarity.within_baseline),
        len(similarity.between),
    )
    n_bins = int(np.clip(round(np.sqrt(2 * smallest)), 10, 24))
    bins = np.linspace(min(pooled), max(pooled), n_bins + 1)
    # Between-set mass first so the two within-set outlines sit on top of it.
    for values, color in [
        (similarity.between, INK_MUTED),
        (similarity.within_baseline, BASELINE_SET_COLOR),
        (similarity.within_target, TARGET_COLOR),
    ]:
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=LINE_WIDTH,
        )
    style_axes(ax, grid_axis="y")
    ax.set_xlabel("cosine similarity")
    if show_ylabel:
        ax.set_ylabel("density")
    headline(ax, _display_label(variant.variant), f"p = {variant.p_value:.4g}")
