"""Registry wiring each dimension name to its compute and plot functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.distributional.distributional_dimension import compute_distributional
from src.interactional.interactional_dimension import compute_interactional
from src.lexical.lexical_dimension import compute_lexical
from src.semantic.semantic_dimension import compute_semantic
from src.syntactic.syntactic_dimension import compute_syntactic
from src.topical.topical_dimension import compute_topical
from src.viz.distributional_plots import plot_distributional
from src.viz.interactional_plots import plot_interactional
from src.viz.lexical_plots import plot_lexical
from src.viz.semantic_plots import plot_semantic
from src.viz.syntactic_plots import plot_syntactic
from src.viz.topical_plots import plot_topical


@dataclass
class DimensionSpec:
    """One dimension's entry points; config_attr names its PipelineConfig field."""

    name: str
    compute: Callable
    plot: Callable
    config_attr: str


DIMENSION_SPECS: dict[str, DimensionSpec] = {
    spec.name: spec
    for spec in [
        DimensionSpec("lexical", compute_lexical, plot_lexical, "lexical"),
        DimensionSpec("syntactic", compute_syntactic, plot_syntactic, "syntactic"),
        DimensionSpec("semantic", compute_semantic, plot_semantic, "semantic"),
        DimensionSpec(
            "distributional",
            compute_distributional,
            plot_distributional,
            "distributional",
        ),
        DimensionSpec("topical", compute_topical, plot_topical, "topical"),
        DimensionSpec(
            "interactional",
            compute_interactional,
            plot_interactional,
            "interactional",
        ),
    ]
}
