"""Interactional dimension plot: facet option shares, one panel per backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from src.interactional.interactional_dimension import (
    InteractionalBackendResult,
    InteractionalResult,
)
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_SET_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_BAR_HEIGHT = BAR_THICKNESS / 2  # the two paired bars share each option band
_PAIR_GAP = 0.06  # air between the paired bars, in band units
_FACET_GAP = 0.95  # extra space opening each facet block; holds its header
_FACET_LABELS = {
    "speech_act": "speech act",
    "disclosure_depth": "disclosure depth",
    "anthropomorphization": "anthropomorphization",
}


def plot_interactional(result: InteractionalResult, out_dir: Path) -> list[Path]:
    """Paired option-share bars grouped by facet, one panel column per backend."""
    apply_plot_style()
    backends = result.backend_results
    if not backends:
        return []
    fig, axes = plt.subplots(
        1,
        len(backends),
        figsize=(2.3 + 3.5 * len(backends), 6.8),
        sharey=True,
        squeeze=False,
    )
    band_centers, headers = _band_layout(backends[0].share_rows)
    xmax = max(
        max(row.share_target, row.share_baseline)
        for backend in backends
        for row in backend.share_rows
    )
    for ax, backend in zip(axes[0], backends, strict=True):
        _backend_panel(ax, backend, band_centers, headers, xmax)

    # Shared y decorations live on the leftmost panel only.
    first = axes[0][0]
    first.set_yticks(band_centers)
    first.set_yticklabels(
        [row.option.replace("_", " ") for row in backends[0].share_rows],
        color=INK_SECONDARY,
    )
    for facet, header_y in headers:
        first.text(
            -0.012,
            header_y,
            _FACET_LABELS[facet],
            color=INK_MUTED,
            fontsize=9,
            ha="right",
            va="center",
            transform=first.get_yaxis_transform(),
        )
    legend_below(
        first,
        handles=[
            Patch(facecolor=TARGET_COLOR, label=result.target_label),
            Patch(facecolor=BASELINE_SET_COLOR, label=result.baseline_label),
        ],
    )
    return [save_figure(fig, out_dir / "interactional_facet_shares.png")]


def _band_layout(rows: list) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Descending band centers, with a header slot opening each facet block."""
    centers: list[float] = []
    headers: list[tuple[str, float]] = []
    y = 0.0
    previous_facet = ""
    for row in rows:
        if row.facet != previous_facet:
            y -= _FACET_GAP
            headers.append((row.facet, y + _FACET_GAP * 0.55))
            previous_facet = row.facet
        centers.append(y)
        y -= 1.0
    return np.asarray(centers), headers


def _backend_panel(
    ax: plt.Axes,
    backend: InteractionalBackendResult,
    band_centers: np.ndarray,
    headers: list[tuple[str, float]],
    xmax: float,
) -> None:
    """One backend's paired bars, per-facet stats, and extreme-pair labels."""
    offset = (_BAR_HEIGHT + _PAIR_GAP) / 2
    shares_target = np.array([row.share_target for row in backend.share_rows])
    shares_baseline = np.array([row.share_baseline for row in backend.share_rows])
    ax.barh(
        band_centers + offset, shares_target, height=_BAR_HEIGHT, color=TARGET_COLOR
    )
    ax.barh(
        band_centers - offset,
        shares_baseline,
        height=_BAR_HEIGHT,
        color=BASELINE_SET_COLOR,
    )
    # Shared scale across backends; room for labels, never far past 100%.
    ax.set_xlim(0, min(1.05, xmax * 1.15))

    # Each facet block carries its own test statistic on the right.
    tests_by_facet = {test.facet: test for test in backend.facet_tests}
    for facet, header_y in headers:
        test = tests_by_facet[facet]
        ax.text(
            1.0,
            header_y,
            f"JSD {test.jsd_bits:.3f} · p = {test.p_value:.3g}",
            color=INK_SECONDARY,
            fontsize=8.5,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )

    # Selective direct labels: only the most divergent option pair gets values.
    extreme = int(np.argmax(np.abs(shares_target - shares_baseline)))
    label_pad = ax.get_xlim()[1] * 0.015
    for value, label_y in (
        (shares_target[extreme], band_centers[extreme] + offset),
        (shares_baseline[extreme], band_centers[extreme] - offset),
    ):
        ax.text(
            value + label_pad,
            label_y,
            f"{value:.0%}",
            color=INK_SECONDARY,
            fontsize=8,
            ha="left",
            va="center",
        )

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("share of prompts")
    style_axes(ax, grid_axis="x")
    n_significant = sum(test.significant for test in backend.facet_tests)
    headline(
        ax,
        backend.backend,
        f"{n_significant}/{len(backend.facet_tests)} facets significant",
    )
