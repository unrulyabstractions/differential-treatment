"""Comparison-level overview: evidence strength per test across all sections."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.run_summary import ComparisonSummary
from src.viz.plot_style import (
    BAR_THICKNESS,
    INK_MUTED,
    INK_SECONDARY,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    save_figure,
    style_axes,
)

_P_FLOOR = 1e-4  # permutation/MMD p-values are bounded away from 0 by design
_ALPHA = 0.05


def plot_summary(summary: ComparisonSummary, out_dir: Path) -> list[Path]:
    """Horizontal -log10(p) bars, one per test, with the significance threshold."""
    apply_plot_style()
    rows = [v for v in summary.verdicts if v.p_value is not None]
    if not rows:
        return []

    labels = [
        f"{v.dimension} · {v.variant}" if v.variant else v.dimension for v in rows
    ]
    evidence = [-np.log10(max(v.p_value, _P_FLOOR)) for v in rows]
    # Full-opacity bars for significant tests, washed out for the rest; one hue —
    # the entity here is "evidence", not set identity.
    alphas = [1.0 if v.significant else 0.35 for v in rows]

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.8))
    positions = np.arange(len(rows))[::-1]
    for pos, value, alpha in zip(positions, evidence, alphas, strict=True):
        ax.barh(pos, value, height=BAR_THICKNESS, color=TARGET_COLOR, alpha=alpha)
    ax.set_ylim(-0.6, len(rows))
    threshold = -np.log10(_ALPHA)
    ax.axvline(threshold, color=INK_MUTED, linewidth=1.0)
    ax.text(
        threshold,
        len(rows) - 0.35,
        " α = 0.05",  # noqa: RUF001
        color=INK_MUTED,
        fontsize=9,
        va="bottom",
    )

    for pos, value, verdict in zip(positions, evidence, rows, strict=True):
        ax.text(
            value + 0.05,
            pos,
            _format_p(verdict.p_value),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("evidence against 'same distribution'  (−log₁₀ p)")  # noqa: RUF001
    expected = f" · expected {summary.expectation}" if summary.expectation else ""
    headline(
        ax,
        f"{summary.target.display_name} vs {summary.baseline.display_name}",
        f"{summary.n_significant}/{summary.n_tests} tests significant{expected}",
    )
    style_axes(ax, grid_axis="x")
    return [save_figure(fig, out_dir / "summary_overview.png")]


def _format_p(p_value: float) -> str:
    return f"p < {_P_FLOOR:g}" if p_value <= _P_FLOOR else f"p = {p_value:.3g}"
