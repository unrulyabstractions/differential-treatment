"""Lexical dimension plots: marked-word bars, volcano, clouds, calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from wordcloud import WordCloud

from src.lexical.lexical_dimension import LexicalResult
from src.lexical.marked_words_analyzer import MarkedWord
from src.viz.lexical_calibration_plots import plot_lexical_calibration
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_COLOR,
    BASELINE_SET_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    NULL_FILL,
    SURFACE,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    legend_below,
    save_figure,
    style_axes,
)

_CLOUD_MAX_WORDS = 60
_CLOUD_WIDTH = 900
_CLOUD_HEIGHT = 640

_SIGNIFICANT_WORDS_PER_SIDE = 12
_FALLBACK_WORDS_PER_SIDE = 8
_VOLCANO_LABELED_WORDS = 6
_TIE_ABSORB_RANKS = 4  # ranks past the labeled cut that may join a tied cluster
_WASHED_ALPHA = 0.45  # fallback bars: nothing passed FDR, so the ink is washed
# Approximate label geometry (points) at figsize (7.2, 5.0) and 9pt text, for
# the volcano's collision checks — coarse is fine, labels only need daylight.
_AXES_WIDTH_PT = 400.0
_AXES_HEIGHT_PT = 270.0
_CHAR_WIDTH_PT = 5.0
_P_FLOOR = 1e-300  # keeps -log10 finite if a p-value underflows to 0


def plot_lexical(result: LexicalResult, out_dir: Path) -> list[Path]:
    """Write the marked-words bars, volcano, clouds, and calibration figures."""
    apply_plot_style()
    paths = [
        _plot_marked_words(result, out_dir / "lexical_marked_words.png"),
        _plot_volcano(result, out_dir / "lexical_volcano.png"),
        _plot_wordclouds(result, out_dir / "lexical_wordclouds.png"),
    ]
    if result.calibration_plots:
        paths.extend(plot_lexical_calibration(result, out_dir / "calibration"))
    return paths


def _stat_line(result: LexicalResult) -> str:
    return (
        f"{result.n_significant_words} of {result.vocabulary_size} "
        f"significant · FDR {result.fdr_alpha:g}"
    )


def _select_bar_words(result: LexicalResult) -> tuple[list[MarkedWord], str]:
    """Words to draw and the fallback mode: "" (passed), "raw_z", or "topz".

    When nothing survives the configured rule, prefer the raw-|z|>=1.96 words
    (MotS's own, more powerful rule) over an uninformative top-|z| slice, so a
    strict-correction "0 significant" still shows the exploratory signal.
    """
    significant_target = [w for w in result.marked_words_target if w.significant]
    significant_baseline = [w for w in result.marked_words_baseline if w.significant]
    if significant_target or significant_baseline:
        chosen = (
            significant_target[:_SIGNIFICANT_WORDS_PER_SIDE]
            + significant_baseline[:_SIGNIFICANT_WORDS_PER_SIDE]
        )
        return chosen, ""
    raw_target = [w for w in result.marked_words_target if w.significant_raw_z]
    raw_baseline = [w for w in result.marked_words_baseline if w.significant_raw_z]
    if raw_target or raw_baseline:
        chosen = (
            raw_target[:_SIGNIFICANT_WORDS_PER_SIDE]
            + raw_baseline[:_SIGNIFICANT_WORDS_PER_SIDE]
        )
        return chosen, "raw_z"
    chosen = (
        result.marked_words_target[:_FALLBACK_WORDS_PER_SIDE]
        + result.marked_words_baseline[:_FALLBACK_WORDS_PER_SIDE]
    )
    return chosen, "topz"


def _plot_marked_words(result: LexicalResult, path: Path) -> Path:
    """Diverging bars: z > 0 toward the target set, z < 0 toward the baseline."""
    words, fallback = _select_bar_words(result)
    words = sorted(words, key=lambda w: w.z_score, reverse=True)
    if fallback == "raw_z" and words:
        stat = f"0 pass FDR · {result.n_significant_raw_z} pass raw |z| ≥ 1.96"
    elif fallback == "topz" and words:
        stat = "0 significant · top words by |z|"
    else:
        stat = _stat_line(result)
    bar_alpha = _WASHED_ALPHA if fallback else 1.0

    fig_height = max(3.2, 0.32 * len(words) + 1.9)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    style_axes(ax, grid_axis="x")
    headline(ax, "Marked words", stat)
    ax.set_xlabel("log-odds z")

    if words:
        y_positions = np.arange(len(words))[::-1]
        z_values = np.array([w.z_score for w in words])
        colors = [TARGET_COLOR if z > 0 else BASELINE_SET_COLOR for z in z_values]
        ax.barh(
            y_positions, z_values, height=BAR_THICKNESS, color=colors, alpha=bar_alpha
        )
        ax.axvline(0, color=BASELINE_COLOR, linewidth=1.0)
        x_max = float(np.abs(z_values).max()) * 1.15
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-0.7, len(words) - 0.3)
        ax.set_yticks(y_positions, labels=[w.word for w in words])
        ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
        legend_below(
            ax,
            [
                Patch(
                    facecolor=TARGET_COLOR,
                    alpha=bar_alpha,
                    label=result.target_label,
                ),
                Patch(
                    facecolor=BASELINE_SET_COLOR,
                    alpha=bar_alpha,
                    label=result.baseline_label,
                ),
            ],
        )
    else:
        _empty_note(ax, result)

    return save_figure(fig, path)


def _plot_volcano(result: LexicalResult, path: Path) -> Path:
    """Full-vocabulary scatter: log-odds z vs evidence, FDR survivors in color."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    style_axes(ax, grid_axis="y")
    headline(ax, "Marked-word volcano", _stat_line(result))
    ax.set_xlabel("log-odds z")
    ax.set_ylabel(r"$-\log_{10}$ adjusted p")

    words = result.all_words
    if not words:
        _empty_note(ax, result)
        return save_figure(fig, path)

    z_values = np.array([w.z_score for w in words])
    evidence = -np.log10(np.maximum([w.p_adjusted for w in words], _P_FLOOR))
    sig = np.array([w.significant for w in words])
    # Null mass first, FDR survivors on top in the leaning set's color.
    ax.scatter(
        z_values[~sig],
        evidence[~sig],
        s=34,
        c=NULL_FILL,
        edgecolors=SURFACE,
        linewidths=0.7,
        zorder=2,
    )
    ax.scatter(
        z_values[sig],
        evidence[sig],
        s=34,
        c=[TARGET_COLOR if z > 0 else BASELINE_SET_COLOR for z in z_values[sig]],
        edgecolors=SURFACE,
        linewidths=0.7,
        zorder=3,
    )

    x_max = float(np.abs(z_values).max()) * 1.18
    ax.set_xlim(-x_max, x_max)
    threshold = -np.log10(0.05)
    ax.set_ylim(0, max(float(evidence.max()) * 1.12, threshold * 1.6))
    ax.axvline(0, color=BASELINE_COLOR, linewidth=1.0)
    ax.axhline(threshold, color=INK_MUTED, linewidth=1.0)
    ax.annotate(
        "p = 0.05",
        xy=(x_max, threshold),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_MUTED,
    )

    _label_extreme_words(ax, words)

    legend_below(
        ax,
        [
            _dot(TARGET_COLOR, result.target_label),
            _dot(BASELINE_SET_COLOR, result.baseline_label),
            _dot(NULL_FILL, "not significant"),
        ],
        ncols=3,
    )
    return save_figure(fig, path)


def _label_extreme_words(ax: plt.Axes, words: list[MarkedWord]) -> None:
    """Direct-label the most extreme words, one label per point cluster.

    BH ties stack words on exactly the same (z, p) spot, so co-located words
    share a comma-joined label instead of colliding or drifting onto a
    neighboring point.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_eps, y_eps = 0.03 * (x_hi - x_lo), 0.025 * (y_hi - y_lo)
    # (x, y, member words); the first _VOLCANO_LABELED_WORDS may open a new
    # cluster, the next few ranks may only join one (keeps tied dots honest).
    clusters: list[tuple[float, float, list[str]]] = []
    for rank, word in enumerate(words[: _VOLCANO_LABELED_WORDS + _TIE_ABSORB_RANKS]):
        x = word.z_score
        y = -np.log10(max(word.p_adjusted, _P_FLOOR))
        home = next(
            (c for c in clusters if abs(x - c[0]) < x_eps and abs(y - c[1]) < y_eps),
            None,
        )
        if home is not None:
            home[2].append(word.word)
        elif rank < _VOLCANO_LABELED_WORDS:
            clusters.append((x, y, [word.word]))

    placed: list[tuple[float, float, float, float]] = []  # (x0, x1, y0, y1) in pt
    for x, y, members in clusters:
        text = ", ".join(members[:2])
        if len(members) > 2:
            text += f" +{len(members) - 2}"
        # Label outward from zero, but flip inward near the axes edge so the
        # text never spills out of the frame.
        x_frac = (x - x_lo) / (x_hi - x_lo)
        outward = x > 0
        if x_frac > 0.85 or x_frac < 0.15:
            outward = not outward
        anchor_y = (y - y_lo) / (y_hi - y_lo) * _AXES_HEIGHT_PT

        def _x_box(
            flip: bool, out: bool = outward, frac: float = x_frac, txt: str = text
        ) -> tuple[float, float, bool]:
            side_out = out if not flip else not out
            anchor_x = frac * _AXES_WIDTH_PT + (6 if side_out else -6)
            if side_out:
                return anchor_x, anchor_x + _CHAR_WIDTH_PT * len(txt), side_out
            return anchor_x - _CHAR_WIDTH_PT * len(txt), anchor_x, side_out

        # Above the point by default; on collision try below/further out, then
        # the same slots mirrored to the other side of the point. Slots whose
        # text box would leave the axes are skipped; the last candidate that
        # fits inside the axes is the fallback if everything collides.
        x0, x1, side_out = _x_box(flip=False)
        dy, va = 4, "bottom"
        y0, y1 = anchor_y + 4, anchor_y + 13
        for flip in (False, True):
            cand_x0, cand_x1, cand_side = _x_box(flip)
            found = False
            for slot_dy, slot_va in (
                (4, "bottom"),
                (-6, "top"),
                (16, "bottom"),
                (-18, "top"),
            ):
                box_y0, box_y1 = (
                    (anchor_y + slot_dy - 9, anchor_y + slot_dy)
                    if slot_va == "top"
                    else (anchor_y + slot_dy, anchor_y + slot_dy + 9)
                )
                if box_y0 < 0 or box_y1 > _AXES_HEIGHT_PT:
                    continue
                if cand_x0 < 0 or cand_x1 > _AXES_WIDTH_PT:
                    continue
                if not any(
                    cand_x0 < px1 and px0 < cand_x1 and box_y0 < py1 and py0 < box_y1
                    for px0, px1, py0, py1 in placed
                ):
                    x0, x1, side_out = cand_x0, cand_x1, cand_side
                    dy, va, y0, y1 = slot_dy, slot_va, box_y0, box_y1
                    found = True
                    break
            if found:
                break
        # No collision-free slot anywhere: drop this label rather than overprint
        # a neighbour (the dot still shows). An unreadable stack is worse than a
        # missing secondary label.
        if not found:
            continue
        outward = side_out
        placed.append((x0, x1, y0, y1))
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(6 if outward else -6, dy),
            textcoords="offset points",
            ha="left" if outward else "right",
            va=va,
            fontsize=9,
            color=INK_SECONDARY,
        )


def _cloud_words(
    words: list[MarkedWord], want_target: bool
) -> tuple[dict[str, float], bool]:
    """Word -> |z| weights for one side; significant words, else top |z| fallback.

    `want_target` picks z > 0 (marked for the target set) vs z < 0 (baseline).
    """
    side = [w for w in words if (w.z_score > 0) == want_target and w.z_score != 0]
    significant = [w for w in side if w.significant]
    chosen = significant or side[:_CLOUD_MAX_WORDS]
    return {w.word: abs(w.z_score) for w in chosen}, not significant


def _shade_color_func(base_hex: str):
    """wordcloud color_func: shades of one hue, deeper for heavier words."""
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.ones(3)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rng = random_state or np.random
        tint = rng.uniform(0.08, 0.5)
        return mcolors.to_hex(base * (1.0 - tint) + white * tint)

    return color_func


def _draw_cloud(
    ax: plt.Axes, weights: dict[str, float], base_hex: str, label: str, fallback: bool
) -> None:
    ax.axis("off")
    ax.set_title(label, loc="center", fontsize=12, color=INK_SECONDARY, pad=6)
    if not weights:
        ax.text(
            0.5,
            0.5,
            "no marked words",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=INK_MUTED,
            fontsize=10,
        )
        return
    cloud = WordCloud(
        width=_CLOUD_WIDTH,
        height=_CLOUD_HEIGHT,
        background_color=SURFACE,
        color_func=_shade_color_func(base_hex),
        prefer_horizontal=0.95,
        max_words=_CLOUD_MAX_WORDS,
        relative_scaling=0.55,
        random_state=0,
    ).generate_from_frequencies(weights)
    ax.imshow(cloud.to_array(), interpolation="bilinear")
    if fallback:
        ax.text(
            0.5,
            -0.04,
            "top |z| (none passed significance)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=INK_MUTED,
            fontsize=9,
        )


def _plot_wordclouds(result: LexicalResult, path: Path) -> Path:
    """Two clouds sized by |z|: target words (blue) and baseline words (aqua)."""
    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.suptitle("Marked-word clouds", x=0.02, ha="left", fontweight="bold")
    target_weights, t_fallback = _cloud_words(result.all_words, want_target=True)
    baseline_weights, b_fallback = _cloud_words(result.all_words, want_target=False)
    _draw_cloud(ax_t, target_weights, TARGET_COLOR, result.target_label, t_fallback)
    _draw_cloud(
        ax_b, baseline_weights, BASELINE_SET_COLOR, result.baseline_label, b_fallback
    )
    fig.subplots_adjust(top=0.86, wspace=0.06)
    return save_figure(fig, path)


def _dot(color: str, label: str) -> Line2D:
    return Line2D(
        [], [], linestyle="", marker="o", markersize=7, color=color, label=label
    )


def _empty_note(ax: plt.Axes, result: LexicalResult) -> None:
    ax.text(
        0.5,
        0.5,
        f"vocabulary empty (min word count {result.min_word_count})",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK_MUTED,
        fontsize=10,
    )
    ax.set_xlim(-1, 1)
