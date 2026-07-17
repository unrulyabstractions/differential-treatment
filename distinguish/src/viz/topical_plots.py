"""Topical dimension plots: topic and domain shares, one panel per backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from src.topical.topical_dimension import TopicalBackendResult, TopicalResult
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

_BAR_HEIGHT = BAR_THICKNESS / 2  # the two paired bars share each category band
_PAIR_GAP = 0.06  # air between the paired bars, in band units
_DOMAIN_GAP = 0.9  # extra space before each domain block; holds its header
_DOMAIN_HEADERS = {
    "MH": "MH — mental health",
    "GSH": "GSH — gender & sexual health",
    "REL": "REL — relationships",
}
_DOMAIN_TICK_LABELS = {
    "MH": "mental\nhealth",
    "GSH": "gender &\nsexual health",
    "REL": "relationships",
}


def plot_topical(result: TopicalResult, out_dir: Path) -> list[Path]:
    """Topic-share and domain-share charts, written directly into out_dir.

    Survey backends (fixed 15-topic catalog) and generated topicgpt backends
    use incompatible topic axes, so each catalog family gets its own figure.
    """
    apply_plot_style()
    survey = [b for b in result.backend_results if b.taxonomy is None]
    generated = [b for b in result.backend_results if b.taxonomy is not None]
    paths: list[Path] = []
    if survey:
        paths.append(_plot_topic_shares(result, survey, out_dir))
        paths.append(_plot_domain_shares(result, survey, out_dir))
    if generated:
        paths.append(_plot_topicgpt_shares(result, generated, out_dir))
    return paths


def _set_handles(result: TopicalResult) -> list[Patch]:
    """Legend swatches in the fixed slot order: target blue, baseline aqua."""
    return [
        Patch(facecolor=TARGET_COLOR, label=result.target_label),
        Patch(facecolor=BASELINE_SET_COLOR, label=result.baseline_label),
    ]


def _band_layout(
    rows: list,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Descending band centers, with a header slot opening each domain block."""
    centers: list[float] = []
    headers: list[tuple[str, float]] = []
    y = 0.0
    previous_domain = ""
    for row in rows:
        if row.domain != previous_domain:
            y -= _DOMAIN_GAP
            headers.append((row.domain, y + _DOMAIN_GAP * 0.62))
            previous_domain = row.domain
        centers.append(y)
        y -= 1.0
    return np.asarray(centers), headers


def _paired_barh(
    ax: plt.Axes,
    band_centers: np.ndarray,
    shares_target: np.ndarray,
    shares_baseline: np.ndarray,
) -> float:
    """Draw one panel's paired horizontal bars; returns the pair offset."""
    offset = (_BAR_HEIGHT + _PAIR_GAP) / 2
    ax.barh(
        band_centers + offset, shares_target, height=_BAR_HEIGHT, color=TARGET_COLOR
    )
    ax.barh(
        band_centers - offset,
        shares_baseline,
        height=_BAR_HEIGHT,
        color=BASELINE_SET_COLOR,
    )
    return offset


def _label_extreme_pair(
    ax: plt.Axes,
    band_centers: np.ndarray,
    offset: float,
    shares_target: np.ndarray,
    shares_baseline: np.ndarray,
) -> None:
    """Selective direct labels: only the most divergent category gets values."""
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


def _plot_topic_shares(
    result: TopicalResult, backends: list[TopicalBackendResult], out_dir: Path
) -> Path:
    """Paired per-topic bars grouped by domain, one panel column per backend."""
    fig, axes = plt.subplots(
        1,
        len(backends),
        figsize=(2.4 + 3.4 * len(backends), 8.6),
        sharey=True,
        squeeze=False,
    )
    band_centers, headers = _band_layout(backends[0].topic_rows)
    xmax = max(
        max(row.proportion_target, row.proportion_baseline)
        for backend in backends
        for row in backend.topic_rows
    )
    for ax, backend in zip(axes[0], backends, strict=True):
        shares_target = np.array([r.proportion_target for r in backend.topic_rows])
        shares_baseline = np.array([r.proportion_baseline for r in backend.topic_rows])
        offset = _paired_barh(ax, band_centers, shares_target, shares_baseline)
        ax.set_xlim(0, xmax * 1.18)  # shared scale keeps backends comparable
        _label_extreme_pair(ax, band_centers, offset, shares_target, shares_baseline)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("share of prompts")
        style_axes(ax, grid_axis="x")
        headline(
            ax,
            backend.backend,
            f"topic JSD {backend.topic_jsd:.2f} bits · p = {backend.p_value:.3g}",
        )

    # Shared y decorations live on the leftmost panel only.
    first = axes[0][0]
    first.set_yticks(band_centers)
    first.set_yticklabels(
        [row.short_name for row in backends[0].topic_rows], color=INK_SECONDARY
    )
    for domain, header_y in headers:
        first.text(
            -0.012,
            header_y,
            _DOMAIN_HEADERS[domain],
            color=INK_MUTED,
            fontsize=8.5,
            ha="right",
            va="center",
            transform=first.get_yaxis_transform(),
        )
    legend_below(first, _set_handles(result))
    return save_figure(fig, out_dir / "topical_topic_shares.png")


def _plot_domain_shares(
    result: TopicalResult, backends: list[TopicalBackendResult], out_dir: Path
) -> Path:
    """Compact grouped domain-share bars, one panel column per backend."""
    fig, axes = plt.subplots(
        1,
        len(backends),
        figsize=(1.2 + 3.3 * len(backends), 3.6),
        sharey=True,
        squeeze=False,
    )
    ymax = max(
        max(row.proportion_target, row.proportion_baseline)
        for backend in backends
        for row in backend.domain_rows
    )
    width = BAR_THICKNESS / 2
    offset = (width + _PAIR_GAP * width) / 2
    for ax, backend in zip(axes[0], backends, strict=True):
        _grouped_domain_bars(ax, backend, width, offset)
        ax.set_ylim(0, ymax * 1.2)  # shared scale keeps backends comparable
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        style_axes(ax, grid_axis="y")
        headline(ax, backend.backend, f"domain JSD {backend.domain_jsd:.2f} bits")
    axes[0][0].set_ylabel("share of prompts")
    legend_below(axes[0][0], _set_handles(result))
    return save_figure(fig, out_dir / "topical_domain_shares.png")


def _grouped_domain_bars(
    ax: plt.Axes, backend: TopicalBackendResult, width: float, offset: float
) -> None:
    """One backend's three paired domain bars with named ticks."""
    positions = np.arange(len(backend.domain_rows), dtype=float)
    shares_target = [row.proportion_target for row in backend.domain_rows]
    shares_baseline = [row.proportion_baseline for row in backend.domain_rows]
    ax.bar(positions - offset, shares_target, width=width, color=TARGET_COLOR)
    ax.bar(positions + offset, shares_baseline, width=width, color=BASELINE_SET_COLOR)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_DOMAIN_TICK_LABELS[row.domain] for row in backend.domain_rows],
        color=INK_SECONDARY,
        fontsize=9,
    )


def _plot_topicgpt_shares(
    result: TopicalResult, backends: list[TopicalBackendResult], out_dir: Path
) -> Path:
    """Paired shares over each backend's GENERATED taxonomy (no domain groups).

    Each generated catalog is its own set of topics, so panels do not share a
    y-axis; every panel labels its own generated topics top-to-bottom.
    """
    n_rows = max(len(backend.topic_rows) for backend in backends)
    fig, axes = plt.subplots(
        1,
        len(backends),
        figsize=(3.0 + 3.8 * len(backends), 1.6 + 0.42 * n_rows),
        squeeze=False,
    )
    xmax = max(
        max(row.proportion_target, row.proportion_baseline)
        for backend in backends
        for row in backend.topic_rows
    )
    for ax, backend in zip(axes[0], backends, strict=True):
        rows = backend.topic_rows
        centers = -np.arange(len(rows), dtype=float)
        shares_target = np.array([r.proportion_target for r in rows])
        shares_baseline = np.array([r.proportion_baseline for r in rows])
        offset = _paired_barh(ax, centers, shares_target, shares_baseline)
        ax.set_xlim(0, xmax * 1.18)
        _label_extreme_pair(ax, centers, offset, shares_target, shares_baseline)
        ax.set_yticks(centers)
        ax.set_yticklabels([r.short_name for r in rows], color=INK_SECONDARY)
        ax.set_ylim(centers.min() - 0.8, 0.8)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("share of prompts")
        style_axes(ax, grid_axis="x")
        n_topics = len(backend.taxonomy.topics) if backend.taxonomy else len(rows)
        headline(
            ax,
            backend.backend,
            f"{n_topics} generated topics · JSD {backend.topic_jsd:.2f} bits · "
            f"p = {backend.p_value:.3g}",
        )
    legend_below(axes[0][0], _set_handles(result))
    return save_figure(fig, out_dir / "topical_topicgpt_shares.png")
