"""Runs the COMPLETE analysis for one dataset and writes runs/{dataset}/.

Layout:
    runs/{dataset}/
    ├── summary.json / config.json
    ├── comparison_matrix.png            # every test, one bar per comparison
    ├── {comparison}/                    # one dir per manifest comparison
    │   ├── summary_overview.png
    │   └── {section}/                   # section.json + plots
    │       ├── implicit/                # codedness sweep + markedness splits
    │       │   └── {row}/               # full section output (full_outputs)
    │       └── slices/{facet}/{name}/   # per-slice full section output
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from src.attributional.attributional_dimension import compute_attributional
from src.common.dataset_tables import ComparisonSpec, PromptDataset
from src.common.dimension_result import DimensionVerdict
from src.common.file_io import ensure_dir, save_json
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet
from src.common.run_config import ExplorationsConfig, PipelineConfig
from src.conditional.conditional_analysis import ConditionalResult, compute_conditional
from src.pipeline.dimension_registry import DIMENSION_SPECS
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.run_summary import (
    ComparisonSummary,
    DatasetRunSummary,
    DimensionTiming,
    PromptSetSummary,
)
from src.pipeline.section_explorations import (
    light_section_config,
    run_section_explorations,
)
from src.usage.usage_attitudes import compute_usage
from src.viz.attributional_plots import plot_attributional
from src.viz.conditional_plots import plot_conditional_distinguishability
from src.viz.exploration_plots import plot_explorations
from src.viz.run_comparison_plots import plot_comparison_matrix
from src.viz.sensitivity_curve_plots import plot_sensitivity_curve
from src.viz.summary_plots import plot_summary
from src.viz.usage_plots import plot_usage


def run_dataset(
    dataset_dir: Path,
    config: PipelineConfig,
    out_root: Path = Path("runs"),
) -> Path:
    """Run every configured section for every manifest comparison."""
    config.validate()
    dataset = PromptDataset.load(Path(dataset_dir))
    run_name = config.run_name or dataset.manifest.name
    run_dir = ensure_dir(Path(out_root) / run_name)
    save_json(config.to_dict(), run_dir / "config.json")
    context = PipelineContext(config.random_seed)

    summary = DatasetRunSummary(
        run_name=run_name,
        dataset_name=dataset.manifest.name,
        dataset_path=str(dataset_dir),
        created_at=datetime.now().isoformat(timespec="seconds"),
        dimensions_run=list(config.dimensions),
    )
    for comparison in dataset.manifest.comparisons:
        summary.comparisons.append(
            _run_comparison(dataset, comparison, config, context, run_dir, summary)
        )
    save_json(summary.to_dict(), run_dir / "summary.json")
    plot_comparison_matrix(summary, run_dir)
    # Multi-VALUE sensitivity curve (detection vs AAVE density); no-op (writes
    # nothing) for datasets without value_p* comparisons.
    plot_sensitivity_curve(summary, run_dir)
    context.cleanup()
    for comparison_summary in summary.comparisons:
        log(
            f"{comparison_summary.name}: {comparison_summary.n_significant}/"
            f"{comparison_summary.n_tests} tests significant"
            f" (expected {comparison_summary.expectation or 'n/a'})"
        )
    if summary.skipped_variants:
        log(f"Skipped variants: {', '.join(sorted(set(summary.skipped_variants)))}")
    return run_dir


def _run_comparison(
    dataset: PromptDataset,
    comparison: ComparisonSpec,
    config: PipelineConfig,
    context: PipelineContext,
    run_dir: Path,
    summary: DatasetRunSummary,
) -> ComparisonSummary:
    target = dataset.prompt_set(comparison.target_cohort)
    baseline = dataset.prompt_set(comparison.baseline_cohort)
    comparison_dir = ensure_dir(run_dir / comparison.name)
    log(f"=== comparison {comparison.name}: {target.label} vs {baseline.label} ===")

    sections = [
        (
            name,
            DIMENSION_SPECS[name].compute,
            DIMENSION_SPECS[name].plot,
            getattr(config, DIMENSION_SPECS[name].config_attr),
        )
        for name in config.dimensions
    ]
    if config.include_usage_attitudes:
        sections.append(("usage", compute_usage, plot_usage, config.usage))
    if config.include_attributional:
        sections.append(
            ("attributional", compute_attributional, plot_attributional, config.attributional)
        )

    verdicts = []
    conditional_results: list[ConditionalResult] = []
    for name, compute, plot, section_config in sections:
        log(f"--- {comparison.name}/{name} ---")
        started = time.time()
        section_dir = ensure_dir(comparison_dir / name)
        result = compute(target, baseline, section_config, context)
        save_json(result.to_dict(), section_dir / f"{name}.json")
        plot(result, section_dir)
        verdicts.extend(result.to_verdicts())
        summary.skipped_variants.extend(getattr(result, "skipped_variants", []))

        # Explorations (implicit / slices / conditional) apply to the standard
        # distinguishability dimensions, not usage or the attributional probe.
        if comparison.explorations and name in config.dimensions:
            explorations = run_section_explorations(
                name,
                compute,
                light_section_config(name, config),
                dataset,
                comparison,
                config.explorations,
                context,
                plot=plot,
                section_dir=section_dir,
            )
            if explorations.implicit_rows:
                save_json(
                    explorations.to_dict(),
                    section_dir / "implicit" / "implicit.json",
                )
            if explorations.slice_rows:
                save_json(
                    explorations.to_dict(), section_dir / "slices" / "slices.json"
                )
            plot_explorations(explorations, section_dir)

            if config.explorations.run_conditional:
                _run_conditional(
                    name,
                    compute,
                    light_section_config(name, config),
                    dataset,
                    comparison,
                    config.explorations,
                    context,
                    result.to_verdicts(),
                    section_dir,
                    conditional_results,
                )
        summary.timings.append(
            DimensionTiming(comparison.name, name, round(time.time() - started, 2))
        )

    comparison_summary = _summarize(comparison, target, baseline, verdicts)
    plot_summary(comparison_summary, comparison_dir)
    if conditional_results:
        plot_conditional_distinguishability(conditional_results, comparison_dir)
    return comparison_summary


def _run_conditional(
    section: str,
    compute,
    section_config,
    dataset: PromptDataset,
    comparison: ComparisonSpec,
    explorations: ExplorationsConfig,
    context: PipelineContext,
    marginal_verdicts: list[DimensionVerdict],
    section_dir: Path,
    sink: list[ConditionalResult],
) -> None:
    """Conditional distinguishability per content variable; save under conditional/."""
    for variable in explorations.conditioning_variables:
        conditional = compute_conditional(
            dataset,
            comparison.target_cohort,
            comparison.baseline_cohort,
            variable,
            section,
            compute,
            section_config,
            context,
            marginal_verdicts,
            explorations.min_prompts_per_side,
        )
        if not conditional.conditional_verdicts:
            continue
        save_json(
            conditional.to_dict(),
            section_dir / "conditional" / variable / "conditional.json",
        )
        sink.append(conditional)


def _summarize(
    comparison: ComparisonSpec,
    target: PromptSet,
    baseline: PromptSet,
    verdicts,
) -> ComparisonSummary:
    decided = [v for v in verdicts if v.significant is not None]
    return ComparisonSummary(
        name=comparison.name,
        expectation=comparison.expectation,
        target=_set_summary(target),
        baseline=_set_summary(baseline),
        n_tests=len(decided),
        n_significant=sum(bool(v.significant) for v in decided),
        overall_distinguishable=any(bool(v.significant) for v in decided),
        verdicts=verdicts,
    )


def _set_summary(prompt_set: PromptSet) -> PromptSetSummary:
    return PromptSetSummary(
        name=prompt_set.name,
        display_name=prompt_set.label,
        group=prompt_set.group,
        n_prompts=len(prompt_set.prompts),
        n_authors=len(set(prompt_set.author_ids)),
    )
