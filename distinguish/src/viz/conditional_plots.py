"""Aggregate (marginal) vs conditional distinguishability, per conditioning variable.

For every (dimension, test, variant) the marginal evidence against "same
distribution" is drawn beside the conditional evidence (within-stratum, Fisher
combined). Reading the pair answers the paper's question: does the difference
survive holding the content variable fixed (coded style) or vanish (topic choice)?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.conditional.conditional_analysis import ConditionalResult
from src.viz.plot_style import (
    BAR_THICKNESS,
    INK_MUTED,
    INK_SECONDARY,
    NULL_FILL,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_P_FLOOR = 1e-4
_ALPHA = 0.05
# Conditional-bar hue by what conditioning did to the distinguishability.
_INTERP_COLOR = {
    "survives": TARGET_COLOR,  # coded signal beyond the variable
    "revealed": "#8250df",  # Simpson: within-stratum separation hidden when pooled
    "topic-choice": NULL_FILL,  # marginal difference explained by the variable
    "inconclusive": NULL_FILL,
}


def _evidence(p_value: float | None) -> float:
    return 0.0 if p_value is None else -np.log10(max(p_value, _P_FLOOR))


def plot_conditional_distinguishability(
    results: list[ConditionalResult], out_dir: Path
) -> list[Path]:
    """One marginal-vs-conditional evidence chart per conditioning variable."""
    apply_plot_style()
    by_variable: dict[str, list[ConditionalResult]] = {}
    for result in results:
        by_variable.setdefault(result.conditioning_variable, []).append(result)

    paths: list[Path] = []
    for variable, group in by_variable.items():
        path = _chart(
            variable,
            group,
            out_dir / f"conditional_{variable}.png",
        )
        if path is not None:
            paths.append(path)
    return paths


def _chart(variable: str, group: list[ConditionalResult], path: Path) -> Path | None:
    labels, marginal, conditional, interps = [], [], [], []
    for result in group:
        for verdict in result.conditional_verdicts:
            suffix = f" · {verdict.variant}" if verdict.variant else ""
            labels.append(f"{result.section}{suffix}")
            marginal.append(_evidence(verdict.marginal_p))
            conditional.append(_evidence(verdict.conditional_p))
            interps.append(verdict.interpretation)
    if not labels:
        return None

    positions = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * max(len(labels), 3) + 1.6))
    half = BAR_THICKNESS / 2
    for pos, m, c, interp in zip(
        positions, marginal, conditional, interps, strict=True
    ):
        ax.barh(pos + half / 1.4, m, height=half, color=INK_MUTED, alpha=0.55)
        ax.barh(
            pos - half / 1.4,
            c,
            height=half,
            color=_INTERP_COLOR.get(interp, NULL_FILL),
        )
    ax.axvline(-np.log10(_ALPHA), color=INK_MUTED, linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.7, len(labels))
    ax.set_xlabel("evidence  (−log₁₀ p)")  # noqa: RUF001
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
    headline(
        ax,
        f"Marginal vs conditional · {variable}",
        "does distinguishability survive holding content fixed?",
    )
    style_axes(ax, grid_axis="x")
    legend_below(
        ax,
        handles=[
            Patch(facecolor=INK_MUTED, alpha=0.55, label="marginal (pooled)"),
            Patch(facecolor=TARGET_COLOR, label="conditional · survives"),
            Patch(facecolor="#8250df", label="conditional · revealed"),
            Patch(facecolor=NULL_FILL, label="conditional · topic-choice"),
        ],
        ncols=4,
    )
    return save_figure(fig, path)
