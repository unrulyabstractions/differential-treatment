"""Run-root comparison matrix: every test's evidence, one bar per comparison.

Juxtaposes the manifest comparisons (e.g. target_vs_baseline vs the
target_vs_twin null pair; generalizes to N comparisons such as future
sensitivity sweeps). Comparison identity uses CATEGORICAL_SLOTS in fixed
manifest order; washed-out bars are non-significant, dots mark tests that ran.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.common.dimension_result import DimensionVerdict
from src.pipeline.run_summary import DatasetRunSummary
from src.viz.plot_style import (
    CATEGORICAL_SLOTS,
    INK_MUTED,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_P_FLOOR = 1e-4
_ALPHA = 0.05
_BAND = 0.76  # vertical fraction of each test row shared by the comparison bars


def plot_comparison_matrix(summary: DatasetRunSummary, run_dir: Path) -> list[Path]:
    """Grouped -log10(p) bars: tests on y, one slot-colored bar per comparison."""
    apply_plot_style()
    tests: list[tuple[str, str, str]] = []  # (dimension, test, variant), first-seen
    keyed_verdicts: list[dict[tuple[str, str, str], DimensionVerdict]] = []
    for comparison in summary.comparisons:
        keyed = {}
        for verdict in comparison.verdicts:
            if verdict.p_value is None:
                continue
            key = (verdict.dimension, verdict.test_name, verdict.variant)
            keyed[key] = verdict
            if key not in tests:
                tests.append(key)
        keyed_verdicts.append(keyed)
    if not tests:
        return []

    n_comparisons = len(summary.comparisons)
    bar_height = _BAND / n_comparisons
    fig, ax = plt.subplots(figsize=(8.5, 0.26 * n_comparisons * len(tests) + 2.2))
    positions = np.arange(len(tests))[::-1]
    for index, keyed in enumerate(keyed_verdicts):
        color = CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)]
        offsets = positions + _BAND / 2 - (index + 0.5) * bar_height
        for key, y in zip(tests, offsets, strict=True):
            verdict = keyed.get(key)
            if verdict is None:
                continue
            value = -np.log10(max(verdict.p_value, _P_FLOOR))
            alpha = 1.0 if verdict.significant else 0.35
            ax.barh(y, value, height=bar_height * 0.9, color=color, alpha=alpha)
            ax.plot(value, y, marker="o", markersize=2.5, color=color, alpha=alpha)

    ax.set_ylim(-0.7, len(tests) - 1 + 0.7)
    threshold = -np.log10(_ALPHA)
    ax.axvline(threshold, color=INK_MUTED, linewidth=1.0)
    ax.text(
        threshold,
        len(tests) - 1 + 0.55,
        f" α = {_ALPHA:g}",  # noqa: RUF001
        color=INK_MUTED,
        fontsize=9,
        va="top",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([_test_label(key) for key in tests], fontsize=9)
    ax.set_xlabel("evidence against 'same distribution'  (−log₁₀ p)")  # noqa: RUF001
    handles = [
        Patch(
            facecolor=CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)],
            label=comparison.name,
        )
        for index, comparison in enumerate(summary.comparisons)
    ]
    legend_below(ax, handles, ncols=min(n_comparisons, 3))
    headline(
        ax,
        "Evidence by comparison",
        f"{n_comparisons} comparisons · {len(tests)} tests · α = {_ALPHA:g}",  # noqa: RUF001
    )
    style_axes(ax, grid_axis="x")
    return [save_figure(fig, run_dir / "comparison_matrix.png")]


def _test_label(key: tuple[str, str, str]) -> str:
    dimension, _test_name, variant = key
    return f"{dimension} · {variant}" if variant else dimension
