"""`dtreat compare-runs` — compare differential treatment across group pairs.

The pipeline is group-agnostic; this report puts multiple completed runs
(e.g. lgbtq-vs-cishet, women-vs-men, over40-vs-young in the same deployment
domain) side by side: how legible is each pair's input, how much treatment
difference shows in behavior, and along which axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dtreat.common.base_schema import BaseSchema
from dtreat.common.console_logging import log
from dtreat.common.file_io import save_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.stages.prompt_collection.prompt_set_schemas import PromptStageArtifact
from dtreat.stages.prompt_distinguishability.distinguish_bridge_stage import (
    load_input_report_if_present,
)
from dtreat.stages.treatment_analysis.analysis_report_schemas import AnalysisReport


@dataclass
class TopAxisSummary(BaseSchema):
    """One of a run's most informative significant axes."""

    axis_id: str
    question: str
    delta: float
    info_bits: float


@dataclass
class RunComparisonEntry(BaseSchema):
    """One group pair's headline numbers."""

    run_name: str
    target_community: str
    baseline_community: str
    prompts_per_side: int
    input_c2st: float | None = None
    input_tests_significant: int = 0
    input_tests_total: int = 0
    significant_axes: int = 0
    total_axes: int = 0
    d_pi_bits: float | None = None
    behavior_c2st: float | None = None
    signal_usage: float | None = None
    refusal_gap: float | None = None
    top_axes: list[TopAxisSummary] = field(default_factory=list)


@dataclass
class CrossRunComparison(BaseSchema):
    """Side-by-side treatment picture across group pairs."""

    entries: list[RunComparisonEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_cross_run_comparison(run_dirs: list[Path]) -> CrossRunComparison:
    comparison = CrossRunComparison()
    for run_dir in run_dirs:
        paths = RunDirectoryPaths(run_dir)
        if not paths.analysis_report_path.exists():
            comparison.notes.append(f"{run_dir}: no analysis report — skipped")
            continue
        report = AnalysisReport.from_json(paths.analysis_report_path)
        prompt_artifact = PromptStageArtifact.from_json(paths.prompt_sets_path)
        input_report = load_input_report_if_present(paths)

        significant = sorted(
            report.significant_axes(), key=lambda a: -a.info_bits
        )
        entry = RunComparisonEntry(
            run_name=run_dir.name,
            target_community=report.target_community,
            baseline_community=report.baseline_community,
            prompts_per_side=len(prompt_artifact.target_set.prompts),
            significant_axes=len(significant),
            total_axes=len(report.axes),
            d_pi_bits=report.d_pi_bits_significant_axes,
            behavior_c2st=report.c2st.accuracy if report.c2st else None,
            signal_usage=report.input_output.signal_usage if report.input_output else None,
            refusal_gap=(
                report.refusals.target_rate - report.refusals.baseline_rate
                if report.refusals
                else None
            ),
            top_axes=[
                TopAxisSummary(
                    axis_id=axis.axis_id,
                    question=axis.question,
                    delta=axis.delta,
                    info_bits=axis.info_bits,
                )
                for axis in significant[:3]
            ],
        )
        if input_report:
            entry.input_c2st = input_report.best_c2st_accuracy
            entry.input_tests_significant = input_report.n_significant
            entry.input_tests_total = input_report.n_tests
        comparison.entries.append(entry)
    return comparison


def render_cross_run_markdown(comparison: CrossRunComparison) -> str:
    lines = [
        "# Cross-group differential-treatment comparison",
        "",
        "| pair | n/side | input C2ST | input sig | output sig axes | D_π (bits) | signal usage |",
        "|------|-------:|-----------:|----------:|----------------:|-----------:|-------------:|",
    ]
    for entry in comparison.entries:
        lines.append(
            f"| {entry.target_community} vs {entry.baseline_community} "
            f"| {entry.prompts_per_side} "
            f"| {_fmt(entry.input_c2st)} "
            f"| {entry.input_tests_significant}/{entry.input_tests_total} "
            f"| {entry.significant_axes}/{entry.total_axes} "
            f"| {_fmt(entry.d_pi_bits)} "
            f"| {_fmt_pct(entry.signal_usage)} |"
        )
    for entry in comparison.entries:
        lines += ["", f"## {entry.target_community} vs {entry.baseline_community} — top axes", ""]
        if not entry.top_axes:
            lines.append("(no significant axes)")
        for axis in entry.top_axes:
            lines.append(
                f"- `{axis.axis_id}` (Δ = {axis.delta:+.2f}, I = {axis.info_bits:.2f} bits): "
                f"{axis.question}"
            )
    for note in comparison.notes:
        lines += ["", f"> {note}"]
    lines.append("")
    return "\n".join(lines)


def run_cross_run_comparison(run_dirs: list[Path], out_dir: Path) -> CrossRunComparison:
    """Build, write, and log the comparison."""
    comparison = build_cross_run_comparison(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(comparison.to_dict(), out_dir / "cross_run_comparison.json")
    (out_dir / "cross_run_comparison.md").write_text(
        render_cross_run_markdown(comparison)
    )
    for entry in comparison.entries:
        log(
            f"  {entry.target_community:>10} vs {entry.baseline_community:<12} "
            f"input C2ST {_fmt(entry.input_c2st)}  sig axes "
            f"{entry.significant_axes}/{entry.total_axes}  usage {_fmt_pct(entry.signal_usage)}"
        )
    log(f"  wrote {out_dir / 'cross_run_comparison.md'}")
    return comparison


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "—"
