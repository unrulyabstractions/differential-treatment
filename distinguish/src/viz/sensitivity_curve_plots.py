"""Sensitivity curve: per-module detection evidence vs AAVE dialect density p.

Multi-VALUE is the project's exact-label sensitivity dial (docs/ITERATION4_PLAN.md,
"Validation design" point 6). Comparisons whose target cohort is named
``value_p<NN>`` set an ordered density axis ``p = NN / 100`` (``value_p05`` ->
0.05, ``value_p100`` -> 1.00). For each such comparison and each section
(dimension), the section's variant verdicts are reduced to one detection-evidence
value ``-log10(max(p_value, 1e-4))`` — taking the STRONGEST (most significant)
variant — and one line per section is drawn across the density axis.

Intended reading: surface-form modules (syntactic, lexical) rise with p, while
embedding-based modules (semantic, distributional) stay flat/low — they are
blind to meaning-preserving dialect. Using the strongest variant per module is
deliberately charitable to the embedding methods, so a flat line is a strong
statement (their best shot still misses the signal).

A run with no ``value_p*`` comparisons yields no curve (returns ``[]``), so wiring
this into every dataset run is a no-op for non-Multi-VALUE datasets.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.pipeline.run_summary import ComparisonSummary, DatasetRunSummary
from src.viz.plot_style import (
    CATEGORICAL_SLOTS,
    INK_MUTED,
    INK_SECONDARY,
    LINE_WIDTH,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_P_FLOOR = 1e-4
_ALPHA = 0.05
_VALUE_COHORT_RE = re.compile(r"value_p(\d+)")
# How a section's variant verdicts collapse to one evidence value per density.
# "best" = the strongest (most significant) variant; noted on the plot.
_AGGREGATE = "best"


def _evidence(p_value: float) -> float:
    """Detection evidence against 'same distribution': -log10 p, floored."""
    return float(-np.log10(max(p_value, _P_FLOOR)))


def _density_points(
    summary: DatasetRunSummary,
) -> list[tuple[float, ComparisonSummary]]:
    """Ordered (density p, comparison) pairs for every value_p<NN> target."""
    points: list[tuple[float, ComparisonSummary]] = []
    for comparison in summary.comparisons:
        match = _VALUE_COHORT_RE.fullmatch(comparison.target.name)
        if match is None:
            continue
        points.append((int(match.group(1)) / 100.0, comparison))
    points.sort(key=lambda item: item[0])
    return points


def _section_evidence(comparison: ComparisonSummary, section: str) -> float:
    """Aggregate one section's variant verdicts into a single evidence value."""
    values = [
        _evidence(verdict.p_value)
        for verdict in comparison.verdicts
        if verdict.dimension == section and verdict.p_value is not None
    ]
    if not values:
        return float("nan")
    return max(values) if _AGGREGATE == "best" else float(np.median(values))


def plot_sensitivity_curve(summary: DatasetRunSummary, run_dir: Path) -> list[Path]:
    """One line per section: detection evidence vs AAVE density p.

    Returns ``[sensitivity_curve.png]`` when the run has ``value_p*`` comparisons,
    else ``[]`` (a no-op for non-Multi-VALUE datasets).
    """
    points = _density_points(summary)
    if not points:
        return []

    apply_plot_style()
    densities = [p for p, _ in points]
    comparisons = [comparison for _, comparison in points]

    # Sections in fixed config order (colors stay stable across runs), then any
    # extra dimensions that appear in verdicts (e.g. usage) after them.
    seen = {
        verdict.dimension
        for comparison in comparisons
        for verdict in comparison.verdicts
        if verdict.p_value is not None
    }
    order = list(summary.dimensions_run) + sorted(seen - set(summary.dimensions_run))

    x = np.arange(len(densities))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    threshold = -np.log10(_ALPHA)

    handles: list[Line2D] = []
    max_evidence = threshold
    for index, section in enumerate(order):
        y = np.array([_section_evidence(c, section) for c in comparisons])
        if np.all(np.isnan(y)):
            continue
        color = CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=LINE_WIDTH,
            color=color,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                markersize=5,
                linewidth=LINE_WIDTH,
                label=section,
            )
        )
        finite = y[np.isfinite(y)]
        if finite.size:
            max_evidence = max(max_evidence, float(finite.max()))

    ax.axhline(threshold, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.text(
        x[-1],
        threshold,
        f"α = {_ALPHA:g} ",  # noqa: RUF001
        color=INK_MUTED,
        fontsize=9,
        va="bottom",
        ha="right",
    )
    ax.text(
        0.015,
        0.965,
        "line = strongest variant per module",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=9,
        va="top",
        ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in densities])
    ax.set_xlim(-0.25, len(densities) - 0.75)
    ax.set_ylim(0.0, max_evidence + 0.5)
    ax.set_xlabel("AAVE rule density  p")
    ax.set_ylabel("detection evidence  (−log₁₀ p)")  # noqa: RUF001

    headline(ax, "Dialect sensitivity", "detection vs AAVE density")
    legend_below(ax, handles, ncols=min(len(handles), 3))
    style_axes(ax, grid_axis="y")
    return [save_figure(fig, run_dir / "sensitivity_curve.png")]
