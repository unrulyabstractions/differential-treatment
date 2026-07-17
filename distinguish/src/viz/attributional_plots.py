"""Attributional plots: the tokens that drove the probe (bar chart) and a concept
card showing the actual activating/contrasting prompt sets Opus read, with the
strongest-contributing tokens bold-highlighted inline (paper §3.3.5)."""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker

from src.attributional.attributional_dimension import (
    AttributedPrompt,
    AttributionalResult,
)
from src.viz.plot_style import (
    BAR_THICKNESS,
    BASELINE_SET_COLOR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    TARGET_COLOR,
    apply_plot_style,
    headline,
    save_figure,
    style_axes,
)

_TOP_TOKENS = 14
_CARD_EXAMPLES = 5  # prompts shown per side on the concept card
_CARD_WRAP = 92  # chars per wrapped line on the card


def _is_word(token: str) -> bool:
    """Keep alphabetic word tokens; drop bare punctuation/whitespace for the plot."""
    stripped = token.strip()
    return len(stripped) >= 2 and any(char.isalpha() for char in stripped)


def _short_label(label: str, cap: int = 28) -> str:
    """Axis-friendly group label: drop the '(y=…)' suffix and cap length."""
    text = label.split(" (y=")[0].strip()
    return text[: cap - 1] + "…" if len(text) > cap else text


def _segments(highlighted: str) -> list[tuple[str, bool]]:
    """Split a «marked» highlighted string into (word, is_strong_token) pairs."""
    out: list[tuple[str, bool]] = []
    for part in re.split(r"(«[^»]*»)", highlighted):
        if not part:
            continue
        marked = part.startswith("«") and part.endswith("»")
        for word in (part[1:-1] if marked else part).split():
            out.append((word, marked))
    return out


def _prompt_block(prompt: AttributedPrompt, color: str) -> VPacker:
    """One prompt as wrapped rows; strong tokens bold+solid, rest dimmed."""
    lines: list[list[TextArea]] = []
    row: list[TextArea] = []
    row_len = 0
    for word, marked in _segments(prompt.highlighted):
        if row and row_len + len(word) + 1 > _CARD_WRAP:
            lines.append(row)
            row, row_len = [], 0
            if len(lines) >= 2:  # cap each prompt at 2 lines
                break
        prefix = "•  " if not lines and not row else ""
        row.append(
            TextArea(
                f"{prefix}{word} ",
                textprops={
                    "color": color,
                    "fontsize": 8.5,
                    "weight": "bold" if marked else "normal",
                    "alpha": 1.0 if marked else 0.6,
                },
            )
        )
        row_len += len(word) + 1
    if row and len(lines) < 2:
        lines.append(row)
    rows = [HPacker(children=r, align="baseline", pad=0, sep=0) for r in lines]
    return VPacker(children=rows or [TextArea(" ")], align="left", pad=0, sep=1)


def _concept_card(result: AttributionalResult, out_dir: Path) -> Path:
    """Card: the named concept + the prompt sets Opus read, tokens bold inline."""
    apply_plot_style()
    tgt = _short_label(result.target_label) or "target"
    base = _short_label(result.baseline_label) or "baseline"
    children: list = [
        TextArea(
            textwrap.fill(f"Concept: {result.concept or '(unnamed)'}", 68),
            textprops={"color": INK_PRIMARY, "fontsize": 15, "weight": "bold"},
        )
    ]
    if result.concept_explanation:
        children.append(
            TextArea(
                textwrap.fill(result.concept_explanation, 96),
                textprops={"color": INK_SECONDARY, "fontsize": 9},
            )
        )
    children.append(
        TextArea(
            f"held-out separability {result.probe_heldout_accuracy:.2f}   ·   "
            "strongest-contributing tokens shown bold",
            textprops={"color": INK_MUTED, "fontsize": 8.5},
        )
    )
    children.append(
        TextArea(
            f"▸ prompts the probe scores toward  {tgt}",
            textprops={"color": TARGET_COLOR, "fontsize": 11, "weight": "bold"},
        )
    )
    children += [
        _prompt_block(p, TARGET_COLOR) for p in result.activating[:_CARD_EXAMPLES]
    ]
    children.append(
        TextArea(
            f"▸ prompts the probe scores toward  {base}",
            textprops={"color": BASELINE_SET_COLOR, "fontsize": 11, "weight": "bold"},
        )
    )
    children += [
        _prompt_block(p, BASELINE_SET_COLOR)
        for p in result.contrasting[:_CARD_EXAMPLES]
    ]
    box = VPacker(children=children, align="left", pad=4, sep=8)

    # Size the figure to the estimated rendered-line count so the tight-bbox save
    # crops to content (the off-axis otherwise spans the whole figure).
    shown = result.activating[:_CARD_EXAMPLES] + result.contrasting[:_CARD_EXAMPLES]
    est_lines = (
        2  # title
        + (
            len(result.concept_explanation) // 96 + 2
            if result.concept_explanation
            else 1
        )
        + 2  # stat + two side headers
        + sum(min(2, len(p.highlighted) // _CARD_WRAP + 1) for p in shown)
    )
    fig, ax = plt.subplots(figsize=(10.5, 0.24 * est_lines + 0.9))
    ax.axis("off")
    ax.add_artist(
        AnnotationBbox(
            box,
            (0.0, 1.0),
            xycoords="axes fraction",
            box_alignment=(0, 1),
            frameon=False,
        )
    )
    return save_figure(fig, out_dir / "attributional_concept.png")


def plot_attributional(result: AttributionalResult, out_dir: Path) -> list[Path]:
    """Diverging bar of the word tokens contributing most toward each side."""
    apply_plot_style()
    # Aggregate contribution per word token across the two example sets. Bare
    # punctuation tokens still carry probe weight but are dropped here for
    # legibility — the exact attribution lives in the JSON, untouched.
    toward_target: dict[str, float] = defaultdict(float)
    toward_baseline: dict[str, float] = defaultdict(float)
    for prompt in result.activating:
        for tok in prompt.top_tokens:
            if tok.contribution > 0 and _is_word(tok.token):
                toward_target[tok.token.lower()] += tok.contribution
    for prompt in result.contrasting:
        for tok in prompt.top_tokens:
            if tok.contribution < 0 and _is_word(tok.token):
                toward_baseline[tok.token.lower()] += tok.contribution

    top_t = sorted(toward_target.items(), key=lambda kv: -kv[1])[: _TOP_TOKENS // 2]
    top_b = sorted(toward_baseline.items(), key=lambda kv: kv[1])[: _TOP_TOKENS // 2]
    rows = list(reversed(top_b)) + list(reversed(top_t))
    if not rows:  # no word tokens for the bar chart — still emit the concept card
        return [_concept_card(result, out_dir)]

    labels = [w for w, _ in rows]
    values = [v for _, v in rows]
    colors = [TARGET_COLOR if v > 0 else BASELINE_SET_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 1.9))
    positions = np.arange(len(rows))
    ax.barh(positions, values, height=BAR_THICKNESS, color=colors)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    tgt = _short_label(result.target_label) or "target"
    base = _short_label(result.baseline_label) or "baseline"
    ax.set_xlabel(f"summed token contribution  (→ {tgt} · ← {base})")
    style_axes(ax, grid_axis="x")
    concept = result.concept or "(concept naming unavailable)"
    title = textwrap.fill(f"Discriminating concept: {concept}", width=64)
    # Lead with the HELD-OUT separability — the in-sample probe accuracy is ~1.0
    # for any high-dim probe and would overstate a near-null contrast.
    subtitle = (
        f"held-out separability {result.probe_heldout_accuracy:.2f}"
        f"  ·  in-sample {result.probe_accuracy:.2f}"
    )
    headline(ax, title, subtitle)
    tokens_png = save_figure(fig, out_dir / "attributional_tokens.png")
    return [tokens_png, _concept_card(result, out_dir)]
