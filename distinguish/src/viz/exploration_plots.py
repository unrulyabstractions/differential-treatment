"""Evidence charts for section explorations (implicit breakdown, slices)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.section_explorations import (
    ExplorationRow,
    SectionExplorations,
    slice_facet_and_value,
)
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

_P_FLOOR = 1e-4
_ALPHA = 0.05


def plot_explorations(result: SectionExplorations, out_dir: Path) -> list[Path]:
    """Implicit-breakdown chart plus omnibus and per-facet slice charts."""
    apply_plot_style()
    charts: list[Path | None] = []
    if result.implicit_rows:
        charts.append(
            _evidence_chart(
                result.implicit_rows,
                out_dir / "implicit" / f"{result.section}_implicit_evidence.png",
                "Implicitness breakdown",
                lambda row: row.exploration,
            )
        )
    if result.slice_rows:
        slices_dir = out_dir / "slices"
        charts.append(
            _evidence_chart(
                result.slice_rows,
                slices_dir / f"{result.section}_slices_evidence.png",
                "Identity slices",
                lambda row: "/".join(slice_facet_and_value(row)),
            )
        )
        facets = list(
            dict.fromkeys(slice_facet_and_value(r)[0] for r in result.slice_rows)
        )
        for facet in facets:
            rows = [
                r for r in result.slice_rows if slice_facet_and_value(r)[0] == facet
            ]
            charts.append(
                _evidence_chart(
                    rows,
                    slices_dir / f"{result.section}_{facet}_evidence.png",
                    f"Slices · {facet}",
                    lambda row: slice_facet_and_value(row)[1],
                )
            )
    return [path for path in charts if path is not None]


def _evidence_chart(
    rows: list[ExplorationRow],
    path: Path,
    title: str,
    row_label: Callable[[ExplorationRow], str],
) -> Path | None:
    """Horizontal -log10(p) bars, one per rerun x test variant."""
    labels, evidence, alphas, sizes = [], [], [], []
    for row in rows:
        for verdict in row.verdicts:
            if verdict.p_value is None:
                continue
            suffix = f" · {verdict.variant}" if verdict.variant else ""
            labels.append(f"{row_label(row)}{suffix}")
            evidence.append(-np.log10(max(verdict.p_value, _P_FLOOR)))
            alphas.append(1.0 if verdict.significant else 0.35)
            sizes.append(f"n={row.n_prompts_target}/{row.n_prompts_baseline}")
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * max(len(labels), 3) + 1.6))
    positions = np.arange(len(labels))[::-1]
    for pos, value, alpha in zip(positions, evidence, alphas, strict=True):
        ax.barh(pos, value, height=BAR_THICKNESS, color=TARGET_COLOR, alpha=alpha)
    ax.set_ylim(-0.6, len(labels))
    threshold = -np.log10(_ALPHA)
    ax.axvline(threshold, color=INK_MUTED, linewidth=1.0)
    for pos, value, size in zip(positions, evidence, sizes, strict=True):
        ax.text(value + 0.04, pos, size, va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("evidence  (−log₁₀ p)")  # noqa: RUF001
    headline(ax, title, f"filtered reruns · α = {_ALPHA:g}")  # noqa: RUF001
    style_axes(ax, grid_axis="x")
    return save_figure(fig, path)
