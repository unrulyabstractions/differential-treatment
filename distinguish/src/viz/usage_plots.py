"""Usage & attitudes plots: scale dot-pairs and per-domain paired bars.

Both charts draw author-mean survey levels: a dumbbell row per scale
(frequency 1-8 and attitude 1-5 panels kept on separate axes so the ordinal
ranges are never mixed) and a small-multiple domain profile.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.common.dataset_annotations import DOMAINS
from src.usage.usage_attitudes import DOMAIN_SCALES, UsageResult
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_COLOR,
    BASELINE_SET_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_SCALE_NAMES = {
    "llm_freq": "Chatbot use (domain)",
    "professional_freq": "Professional help",
    "general_freq": "Chatbot use (general)",
    "aversion": "Aversion",
    "satisfaction": "Satisfaction",
}
_FREQUENCY_PANEL = (
    ["llm_freq", "professional_freq", "general_freq"],
    8,
    "very rarely",
    ">15/day",
)
_ATTITUDE_PANEL = (["aversion", "satisfaction"], 5, "very negative", "very positive")
_MARKER_SIZE = 9
_PAIR_GAP = 0.05  # air between paired bars, in band units


def plot_usage(result: UsageResult, out_dir: Path) -> list[Path]:
    """Write the two usage PNGs directly into out_dir."""
    apply_plot_style()
    return [_plot_scale_means(result, out_dir), _plot_domain_profile(result, out_dir)]


def _dot_pair_row(ax, y: float, test, filled: bool) -> None:
    """One dumbbell: hairline connector plus a dot per set."""
    near_coincident = (
        test.mean_target is not None
        and test.mean_baseline is not None
        and abs(test.mean_target - test.mean_baseline) < 0.12
    )
    if test.mean_target is not None and test.mean_baseline is not None:
        ax.plot(
            [test.mean_target, test.mean_baseline],
            [y, y],
            color=BASELINE_COLOR,
            linewidth=1.2,
            zorder=2,
        )
    for offset_sign, (mean, color) in enumerate(
        (
            (test.mean_target, TARGET_COLOR),
            (test.mean_baseline, BASELINE_SET_COLOR),
        )
    ):
        if mean is None:
            continue
        # Coincident means would occlude each other; dodge vertically a touch.
        dodged_y = y + (0.13 - 0.26 * offset_sign) if near_coincident else y
        ax.plot(
            [mean],
            [dodged_y],
            marker="o",
            markersize=_MARKER_SIZE,
            markerfacecolor=color if filled else SURFACE,
            markeredgecolor=SURFACE if filled else color,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=3,
        )


def _dot_pair_panel(ax, result: UsageResult, scales, top: int, extremes) -> None:
    """Dumbbell rows for one ordinal range; extremes annotated under the axis."""
    tests = {test.scale: test for test in result.scale_tests}
    for row, scale in enumerate(scales):
        y = len(scales) - 1 - row
        test = tests[scale]
        if test.mean_target is None and test.mean_baseline is None:
            ax.text(
                (1 + top) / 2,
                y,
                "no recorded values",
                color=INK_MUTED,
                fontsize=9,
                ha="center",
                va="center",
            )
            continue
        _dot_pair_row(ax, y, test, filled=bool(test.significant))
    ax.set_yticks(range(len(scales) - 1, -1, -1))
    ax.set_yticklabels([_SCALE_NAMES[s] for s in scales], color=INK_SECONDARY)
    ax.set_ylim(-0.6, len(scales) - 0.4)
    ax.set_xlim(0.7, top + 0.3)
    ax.set_xticks(range(1, top + 1))
    for x, text in zip((1, top), extremes, strict=True):
        ax.annotate(
            text,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
        )
    style_axes(ax, grid_axis="x")


def _plot_scale_means(result: UsageResult, out_dir: Path) -> Path:
    """Dot-pair chart: one dumbbell row per survey scale."""
    freq_scales, freq_top, *freq_extremes = _FREQUENCY_PANEL
    att_scales, att_top, *att_extremes = _ATTITUDE_PANEL
    fig, (ax_freq, ax_att) = plt.subplots(
        2, 1, figsize=(7.4, 4.9), height_ratios=[3, 2]
    )
    fig.subplots_adjust(hspace=0.42)
    _dot_pair_panel(ax_freq, result, freq_scales, freq_top, freq_extremes)
    _dot_pair_panel(ax_att, result, att_scales, att_top, att_extremes)
    headline(ax_freq, "Usage & attitudes", "author-mean survey scales")
    dot = {"marker": "o", "linestyle": "none", "markersize": 8}
    legend_below(
        ax_att,
        [
            Line2D([], [], color=TARGET_COLOR, label=result.target_label, **dot),
            Line2D(
                [], [], color=BASELINE_SET_COLOR, label=result.baseline_label, **dot
            ),
            Line2D(
                [],
                [],
                markerfacecolor=INK_SECONDARY,
                markeredgecolor=SURFACE,
                label=f"significant (FDR {result.fdr_alpha:g})",
                **dot,
            ),
            Line2D(
                [],
                [],
                markerfacecolor=SURFACE,
                markeredgecolor=INK_SECONDARY,
                markeredgewidth=1.4,
                label="not significant",
                **dot,
            ),
        ],
        ncols=2,
    )
    return save_figure(fig, out_dir / "usage_scale_means.png")


def _plot_domain_profile(result: UsageResult, out_dir: Path) -> Path:
    """Small multiples: paired domain bars, one panel per context scale."""
    cells = {(row.scale, row.domain): row for row in result.domain_means}
    domains = [d for d in DOMAINS if any(d == domain for _, domain in cells)]
    fig, axes = plt.subplots(1, len(DOMAIN_SCALES), figsize=(10.8, 3.1))
    positions = np.arange(len(domains))
    bar_width = BAR_THICKNESS / 2
    offset = (bar_width + _PAIR_GAP) / 2
    for ax, scale in zip(axes, DOMAIN_SCALES, strict=True):
        rows = [cells[(scale, domain)] for domain in domains]
        target_means = [row.mean_target or 0.0 for row in rows]
        baseline_means = [row.mean_baseline or 0.0 for row in rows]
        ax.bar(positions - offset, target_means, width=bar_width, color=TARGET_COLOR)
        ax.bar(
            positions + offset,
            baseline_means,
            width=bar_width,
            color=BASELINE_SET_COLOR,
        )
        top = 8 if scale in _FREQUENCY_PANEL[0] else 5
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, 2 if top == 8 else 1))
        ax.set_xticks(positions)
        ax.set_xticklabels(domains, color=INK_SECONDARY)
        ax.set_xlim(-0.6, len(domains) - 0.4)
        ax.set_xlabel(_SCALE_NAMES[scale], fontsize=10)
        style_axes(ax)
    axes[0].set_ylabel("mean level")
    headline(axes[0], "By domain", "")
    legend_below(
        axes[0],
        [
            Patch(color=TARGET_COLOR, label=result.target_label),
            Patch(color=BASELINE_SET_COLOR, label=result.baseline_label),
        ],
        ncols=2,
    )
    return save_figure(fig, out_dir / "usage_domain_profile.png")
