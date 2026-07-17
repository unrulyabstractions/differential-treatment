"""Shared matplotlib styling for all pipeline plots.

Palette and mark specs follow the dataviz design method: recessive hairline
grid, thin marks, ink-token text (never series-colored text), fixed slot order
(target = blue, baseline = aqua). Plots are static light-mode PNGs.

Titles stay minimal: `headline(fig_or_ax, title, stat)` renders a short title
plus one small statistic line — method details belong in the JSON, not prose
subtitles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

matplotlib.use("Agg")

# Surfaces and ink tokens
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"

# Categorical slots in fixed order: slot 1 = target set, slot 2 = baseline set
TARGET_COLOR = "#2a78d6"  # blue
BASELINE_SET_COLOR = "#1baf7a"  # aqua
CATEGORICAL_SLOTS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

# Diverging pair for signed statistics (never used for set identity)
DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#e34948"
NEUTRAL_MIDPOINT = "#f0efec"
NULL_FILL = "#d9d8d2"  # permutation-null histograms and other non-data mass

BAR_THICKNESS = 0.62  # fraction of the band; thin marks, never slot-filling
LINE_WIDTH = 2.0
SAVE_DPI = 300


def apply_plot_style() -> None:
    """Set global rcParams; call once before building any figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": BASELINE_COLOR,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def headline(ax: plt.Axes, title: str, stat: str = "") -> None:
    """Minimal chart heading: short bold title + one small statistic line.

    Keep `title` to a few words (what is shown) and `stat` to one compact
    figure (e.g. "21/96 significant · p < 0.0001"). Anything longer belongs in
    the dimension's JSON, not on the chart.
    """
    pad = 26 if stat else 10
    ax.set_title(title, loc="left", pad=pad)
    if stat:
        ax.annotate(
            stat,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color=INK_SECONDARY,
        )


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Recessive chart chrome: baseline-only spines, hairline grid behind marks."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_COLOR)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend_below(ax: plt.Axes, handles: list, ncols: int = 2) -> None:
    """Legend anchored below the axes in points, clear of the x-axis label."""
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        bbox_transform=offset_copy(ax.transAxes, ax.figure, x=0, y=-34, units="points"),
        ncols=ncols,
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as a crisp PNG and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
