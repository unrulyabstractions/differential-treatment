"""Explorations: how each section's statistics move under filtered reruns.

Two families per section and comparison:
- implicit/: breakdown over the y annotations — reruns at codedness
  thresholds plus markedness splits (implicit-only / marked-only), per H1/H2;
- slices/{facet}/{name}/: author subsets (gender, race, age, ...) per config.

Reruns use a lightened section config (local embedder, linear classifier,
reduced permutations) — the full variant space belongs to the main run. With
`full_outputs` on, every rerun writes the section's complete JSON and plots
into its own directory (e.g. the marked WORDS per slice stay inspectable).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.common.base_schema import BaseSchema
from src.common.dataset_tables import ComparisonSpec, PromptDataset
from src.common.dimension_result import DimensionVerdict
from src.common.file_io import ensure_dir, save_json
from src.common.logging_utils import log
from src.common.run_config import (
    DEFAULT_TEXT_EMBEDDING_MODEL,
    AuthorSliceSpec,
    ExplorationsConfig,
    PipelineConfig,
)
from src.pipeline.pipeline_context import PipelineContext

# Dataset convention "0 = unrecorded"; "*" in an in/not_in value list matches it.
_UNRECORDED_VALUES = frozenset({"", "0"})


@dataclass
class ExplorationRow(BaseSchema):
    """One filtered rerun of a section."""

    exploration: str  # "codedness>=0.5", "marked_only", "slice:gender/women", ...
    n_prompts_target: int
    n_prompts_baseline: int
    verdicts: list[DimensionVerdict] = field(default_factory=list)


@dataclass
class SectionExplorations(BaseSchema):
    """All exploration rows for one section under one comparison."""

    section: str
    comparison: str
    implicit_rows: list[ExplorationRow] = field(default_factory=list)
    slice_rows: list[ExplorationRow] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # name (reason)


def slice_facet_and_value(row: ExplorationRow) -> tuple[str, str]:
    """Split a slice row's "slice:{facet}/{value}" name into its parts."""
    facet, _, value = row.exploration.removeprefix("slice:").partition("/")
    return facet, value


def implicit_row_directory(name: str) -> str:
    """Directory-safe implicit row name: 'codedness>=0.5' -> 'codedness_ge_0.5'."""
    return name.replace(">=", "_ge_")


def light_section_config(section: str, config: PipelineConfig):
    """Cheap rerun settings; the main run owns the full variant space."""
    n_perm = config.explorations.n_permutations
    base = getattr(config, section)
    overrides = {
        "semantic": {
            "text_embedders": [DEFAULT_TEXT_EMBEDDING_MODEL],
            "residual_models": [],
        },
        "distributional": {
            "embedders": [DEFAULT_TEXT_EMBEDDING_MODEL],
            "classifiers": ["linear"],
            "n_permutations": n_perm,
        },
        "topical": {"assignment_backends": ["embedding"], "n_permutations": n_perm},
        "interactional": {
            "annotation_backends": ["embedding"],
            "n_permutations": n_perm,
        },
        # Calibration-justification plots belong once at the top-level section,
        # not repeated (with their permutation nulls) in every filtered rerun.
        "lexical": {"calibration_plots": False},
    }.get(section, {})
    return dataclasses.replace(base, **overrides)


def run_section_explorations(
    section: str,
    compute,
    light_config,
    dataset: PromptDataset,
    comparison: ComparisonSpec,
    config: ExplorationsConfig,
    context: PipelineContext,
    plot: Callable | None = None,
    section_dir: Path | None = None,
) -> SectionExplorations:
    """Run every configured filtered rerun for one section.

    With `config.full_outputs` and a `plot` function plus `section_dir`, each
    rerun also writes the full section JSON and plots into its own directory
    (implicit/{row}/ or slices/{facet}/{name}/ under `section_dir`).
    """
    result = SectionExplorations(section=section, comparison=comparison.name)
    write_full = config.full_outputs and plot is not None and section_dir is not None

    def rerun(name: str, target, baseline, into: list[ExplorationRow], subdir: str):
        n_t, n_b = len(target.prompts), len(baseline.prompts)
        if n_t < config.min_prompts_per_side or n_b < config.min_prompts_per_side:
            result.skipped.append(
                f"{name} ({n_t}/{n_b} prompts < {config.min_prompts_per_side})"
            )
            return
        log(f"    exploration {name}: {n_t} vs {n_b} prompts")
        outcome = compute(target, baseline, light_config, context)
        into.append(
            ExplorationRow(
                exploration=name,
                n_prompts_target=n_t,
                n_prompts_baseline=n_b,
                verdicts=outcome.to_verdicts(),
            )
        )
        if write_full:
            row_dir = ensure_dir(section_dir / subdir)
            save_json(outcome.to_dict(), row_dir / f"{section}.json")
            plot(outcome, row_dir)

    t_cohort, b_cohort = comparison.target_cohort, comparison.baseline_cohort
    if config.run_implicit_breakdown:
        for threshold in config.codedness_thresholds:
            name = f"codedness>={threshold:g}"
            rerun(
                name,
                dataset.prompt_set(
                    t_cohort, prompt_mask=dataset.prompts["codedness"] >= threshold
                ),
                dataset.prompt_set(b_cohort),
                result.implicit_rows,
                f"implicit/{implicit_row_directory(name)}",
            )
        if config.include_markedness_splits:
            for name, markedness in (("implicit_only", 0), ("marked_only", 1)):
                mask = dataset.prompts["markedness"] == markedness
                rerun(
                    name,
                    dataset.prompt_set(t_cohort, prompt_mask=mask),
                    dataset.prompt_set(b_cohort, prompt_mask=mask),
                    result.implicit_rows,
                    f"implicit/{name}",
                )
    if config.run_slices:
        for spec in config.slices:
            mask = _slice_mask(dataset, spec)
            baseline = (
                dataset.prompt_set(b_cohort, author_mask=mask)
                if spec.apply_to == "both"
                else dataset.prompt_set(b_cohort)
            )
            rerun(
                f"slice:{spec.facet}/{spec.name}",
                dataset.prompt_set(t_cohort, author_mask=mask),
                baseline,
                result.slice_rows,
                f"slices/{spec.facet}/{spec.name}",
            )
    return result


def _slice_mask(dataset: PromptDataset, spec: AuthorSliceSpec):
    """Boolean author-table mask for one slice spec."""
    column = dataset.authors[spec.z_field]
    if spec.op == "eq":
        return column.astype(str) == spec.value
    if spec.op in ("in", "not_in"):
        accepted = {v.strip() for v in spec.value.split(",")}
        if "*" in accepted:
            accepted = (accepted - {"*"}) | _UNRECORDED_VALUES
        member = column.astype(str).str.strip().isin(accepted)
        return member if spec.op == "in" else ~member
    # Case-insensitive substring containment tolerates label variants
    # ("white" matching "White/Caucasian"). not_contains excludes authors whose
    # multi-select is unrecorded (empty) rather than counting [] as "not X".
    needle = spec.value.strip().lower()
    contains = column.map(
        lambda values: any(needle in str(v).strip().lower() for v in list(values))
    )
    recorded = column.map(lambda values: len(list(values)) > 0)
    if spec.op == "contains":
        return contains
    if spec.op == "not_contains":
        return recorded & ~contains
    raise ValueError(f"Unknown slice op '{spec.op}' in slice '{spec.name}'")
