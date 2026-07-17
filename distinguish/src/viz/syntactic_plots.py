"""Syntactic (NeuroBiber) plots: log-odds bars, prevalence scatter, count hist."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from src.syntactic.syntactic_dimension import (
    FeatureContrast,
    SyntacticResult,
    readable_feature_name,
)
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_COLOR,
    BASELINE_SET_COLOR,
    INK_SECONDARY,
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

# Not-significant marks are washed to 0.45 opacity everywhere (matches the
# lexical dimension), so significance reads as a single consistent channel.
_NONSIGNIFICANT_ALPHA = 0.45
_SCATTER_LABELED_FEATURES = 6  # direct labels for the largest |log-odds|
_HIST_FILL_ALPHA = 0.18  # light fill under the overlaid step outlines


def plot_syntactic(result: SyntacticResult, out_dir: Path) -> list[Path]:
    """Log-odds bars, full-inventory prevalence scatter, per-group count hist."""
    apply_plot_style()
    return [
        _plot_log_odds_bars(result, out_dir / "syntactic_feature_log_odds.png"),
        _plot_prevalence_scatter(result, out_dir / "syntactic_prevalence_scatter.png"),
        _plot_feature_count_hist(result, out_dir / "syntactic_feature_count_hist.png"),
    ]


def _lean_color(contrast: FeatureContrast) -> str:
    return TARGET_COLOR if contrast.log_odds > 0 else BASELINE_SET_COLOR


def _plot_log_odds_bars(result: SyntacticResult, path: Path) -> Path:
    """Diverging bar chart of the strongest feature contrasts."""
    # Contrasts arrive sorted by |log_odds|; re-sort the top slice by signed
    # value so target-leaning bars cascade at the top, baseline at the bottom.
    top = sorted(
        result.feature_contrasts[: result.top_features_reported],
        key=lambda c: c.log_odds,
    )
    n = len(top)

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.34 * n + 1.8)))
    # Adaptive limits: pad sides that carry bars generously so the "X% vs Y%"
    # tip labels stay inside; a one-sided result doesn't waste half the chart.
    lo_min = min((c.log_odds for c in top), default=-1.0)
    lo_max = max((c.log_odds for c in top), default=1.0)
    span = (max(0.0, lo_max) - min(0.0, lo_min)) or 1.0
    left_pad = 0.24 * span if lo_min < 0 else 0.03 * span
    right_pad = 0.24 * span if lo_max > 0 else 0.03 * span
    ax.set_xlim(min(0.0, lo_min) - left_pad, max(0.0, lo_max) + right_pad)

    for i, contrast in enumerate(top):
        leans_target = contrast.log_odds > 0
        ax.barh(
            i,
            contrast.log_odds,
            height=BAR_THICKNESS,
            color=_lean_color(contrast),
            alpha=1.0 if contrast.significant else _NONSIGNIFICANT_ALPHA,
        )
        ax.annotate(
            f"{contrast.prevalence_target:.0%} vs {contrast.prevalence_baseline:.0%}",
            xy=(contrast.log_odds, i),
            xytext=(4 if leans_target else -4, 0),
            textcoords="offset points",
            ha="left" if leans_target else "right",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.axvline(0, color=BASELINE_COLOR, linewidth=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([readable_feature_name(c.feature_name) for c in top])
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("log-odds ratio")
    style_axes(ax, grid_axis="x")

    headline(
        ax,
        "Syntactic style features",
        f"{result.n_significant_features}/{len(result.feature_contrasts)} "
        f"significant · FDR {result.fdr_alpha:g}",
    )
    # Not-significant bars keep their lean color washed to 0.45; the legend
    # marks that state with NULL_FILL, the same "not significant" token the
    # prevalence scatter uses — no stray hue enters the palette.
    legend_below(
        ax,
        handles=[
            Patch(facecolor=TARGET_COLOR, label=result.target_label),
            Patch(facecolor=BASELINE_SET_COLOR, label=result.baseline_label),
            Patch(facecolor=NULL_FILL, label="not significant"),
        ],
        ncols=3,
    )
    return save_figure(fig, path)


# Approximate label geometry in data units (axes are square, data span ~1):
# used only to dodge label-label collisions, so rough estimates suffice.
_CHAR_W = 0.014
_LABEL_H = 0.028
_LABEL_GAP = 0.022
_Box = tuple[float, float, float, float]


def _label_candidates(
    x: float, y: float, w: float
) -> list[tuple[_Box, tuple[int, int], str, str]]:
    """Candidate (box, offset-points, ha, va) placements around a point."""
    g, h = _LABEL_GAP, _LABEL_H
    dy = 0.7 * g  # vertical clearance of the diagonal slots
    right = ((x + g, y - h / 2, x + g + w, y + h / 2), (8, 0), "left", "center")
    right_up = ((x + g, y + dy, x + g + w, y + dy + h), (8, 7), "left", "bottom")
    right_dn = ((x + g, y - dy - h, x + g + w, y - dy), (8, -7), "left", "top")
    left = ((x - g - w, y - h / 2, x - g, y + h / 2), (-8, 0), "right", "center")
    left_up = ((x - g - w, y + dy, x - g, y + dy + h), (-8, 7), "right", "bottom")
    left_dn = ((x - g - w, y - dy - h, x - g, y - dy), (-8, -7), "right", "top")
    above = ((x - w / 2, y + g, x + w / 2, y + g + h), (0, 8), "center", "bottom")
    below = ((x - w / 2, y - g - h, x + w / 2, y - g), (0, -8), "center", "top")
    if x > 0.85:
        return [left, left_dn, left_up, below, above, right]
    if y > 0.85:
        return [right, right_dn, right_up, left, below, above]
    return [right, left, above, below, right_up, right_dn]


def _boxes_clash(a: _Box, b: _Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _direct_label_features(
    ax: plt.Axes, contrasts: list[FeatureContrast], n_labels: int
) -> None:
    """Label the top-|log-odds| points, merging co-located ones, dodging collisions.

    Significant features are prioritised (they carry the plot's message); a label
    with no collision-free slot is DROPPED rather than overprinted, so a cramped
    near-origin cluster never garbles into an illegible blob.
    """
    ranked = sorted(contrasts, key=lambda c: (not c.significant,))  # sig first, stable
    groups: dict[tuple[float, float], list[str]] = {}
    for contrast in ranked[:n_labels]:
        key = (
            round(contrast.prevalence_baseline, 2),
            round(contrast.prevalence_target, 2),
        )
        groups.setdefault(key, []).append(readable_feature_name(contrast.feature_name))

    # Salient (filled) markers are obstacles, so no label sweeps across one.
    point_r = 0.013
    salient = {
        (round(c.prevalence_baseline, 2), round(c.prevalence_target, 2))
        for c in contrasts
        if c.significant
    } | set(groups)
    points: list[_Box] = [
        (x - point_r, y - point_r, x + point_r, y + point_r) for x, y in salient
    ]

    labels: list[_Box] = []
    for (x, y), names in sorted(groups.items(), key=lambda kv: -kv[0][1]):
        text = " · ".join(sorted(names))
        candidates = [
            c
            for c in _label_candidates(x, y, len(text) * _CHAR_W)
            if all(-0.03 <= v <= 1.05 for v in c[0])
            and not any(_boxes_clash(c[0], box) for box in labels)
        ]
        # Prefer a slot clear of markers too. If nothing is even clear of other
        # LABELS, drop this one — an overprinted label is worse than a missing
        # one (the point itself still shows).
        chosen = next(
            (
                c
                for c in candidates
                if not any(_boxes_clash(c[0], box) for box in points)
            ),
            candidates[0] if candidates else None,
        )
        if chosen is None:
            continue
        box, offset, ha, va = chosen
        labels.append(box)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9,
            color=INK_SECONDARY,
            zorder=4,
        )


def _plot_prevalence_scatter(result: SyntacticResult, path: Path) -> Path:
    """Per-feature prevalence in target (y) vs baseline (x), all features."""
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, linewidth=1.0, zorder=1)

    significant = [c for c in result.feature_contrasts if c.significant]
    nonsignificant = [c for c in result.feature_contrasts if not c.significant]
    ax.scatter(
        [c.prevalence_baseline for c in nonsignificant],
        [c.prevalence_target for c in nonsignificant],
        s=34,
        color=NULL_FILL,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        [c.prevalence_baseline for c in significant],
        [c.prevalence_target for c in significant],
        s=52,
        c=[_lean_color(c) for c in significant],
        edgecolors=SURFACE,
        linewidths=1.0,
        zorder=3,
    )

    # Direct labels for the features with the largest |log-odds| (contrasts
    # arrive sorted by |log_odds| descending).
    _direct_label_features(ax, result.feature_contrasts, _SCATTER_LABELED_FEATURES)

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(f"prevalence in {result.baseline_label}")
    ax.set_ylabel(f"prevalence in {result.target_label}")
    style_axes(ax, grid_axis="both")

    headline(ax, "Feature prevalence", "each point = one feature")
    legend_below(
        ax,
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=TARGET_COLOR,
                markeredgecolor=SURFACE,
                label=f"leans {result.target_label}",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=BASELINE_SET_COLOR,
                markeredgecolor=SURFACE,
                label=f"leans {result.baseline_label}",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=NULL_FILL,
                markeredgecolor="none",
                label="not significant",
            ),
        ],
        ncols=3,
    )
    return save_figure(fig, path)


def _overlaid_step_hist(
    ax: plt.Axes,
    target_values: list[float],
    baseline_values: list[float],
    bins,
) -> None:
    """Two density step histograms sharing one bin edge set (light fill + line)."""
    for values, color in (
        (baseline_values, BASELINE_SET_COLOR),
        (target_values, TARGET_COLOR),
    ):
        if not values:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=_HIST_FILL_ALPHA,
            zorder=2,
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=LINE_WIDTH,
            zorder=3,
        )


def _plot_feature_count_hist(result: SyntacticResult, path: Path) -> Path:
    """Per-group distributions: active features per prompt + per-feature prevalence.

    Left: how many NeuroBiber features each prompt activates (stylistic richness
    per prompt). Right: how prevalent the 96 features are across each group's
    prompts — a rightward shift means a broader, richer active-feature repertoire.
    """
    fig, (ax_count, ax_prev) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    counts_target = result.features_per_prompt_target
    counts_baseline = result.features_per_prompt_baseline
    all_counts = np.array(counts_target + counts_baseline, dtype=float)
    if all_counts.size and all_counts.max() > all_counts.min():
        count_bins = np.histogram_bin_edges(all_counts, bins="auto")
    else:  # degenerate (empty or single value): a minimal integer-width grid
        center = all_counts[0] if all_counts.size else 0.0
        count_bins = np.linspace(center - 0.5, center + 0.5, 3)
    _overlaid_step_hist(ax_count, counts_target, counts_baseline, count_bins)
    ax_count.set_xlabel("active features per prompt")
    ax_count.set_ylabel("density")
    style_axes(ax_count, grid_axis="y")
    headline(ax_count, "Feature counts per prompt", "active NeuroBiber features")

    prev_target = [c.prevalence_target for c in result.feature_contrasts]
    prev_baseline = [c.prevalence_baseline for c in result.feature_contrasts]
    prev_bins = np.linspace(0.0, 1.0, 21)
    _overlaid_step_hist(ax_prev, prev_target, prev_baseline, prev_bins)
    ax_prev.set_xlim(0.0, 1.0)
    ax_prev.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_prev.set_xlabel("prevalence across prompts")
    ax_prev.set_ylabel("density")
    style_axes(ax_prev, grid_axis="y")
    headline(
        ax_prev,
        "Per-feature prevalence",
        f"across {len(result.feature_contrasts)} features",
    )

    fig.subplots_adjust(bottom=0.24, wspace=0.24)
    fig.legend(
        handles=[
            Patch(facecolor=TARGET_COLOR, label=result.target_label),
            Patch(facecolor=BASELINE_SET_COLOR, label=result.baseline_label),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=2,
    )
    return save_figure(fig, path)
