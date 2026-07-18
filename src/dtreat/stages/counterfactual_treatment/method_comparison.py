"""`dtreat methods-compare` — the three methods side by side, per axis.

The pipeline's three modular arms answer the same question different ways:

1. naturalistic   — distribution comparison over real prompts (run-all)
2. counterfactual — paired voice-swapped twins, content held fixed
3. discovery      — where each axis CAME from (a-priori helper conditions vs
                    response-grounded bias enumeration)

This module joins their artifacts into one per-axis table plus a method-level
summary, so designs can be compared on identical axes and responses.
"""

from __future__ import annotations

from dtreat.common.console_logging import log
from dtreat.common.file_io import load_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.treatment_analysis.analysis_report_schemas import AnalysisReport

from .twin_schemas import CounterfactualReport


def write_method_comparison(paths: RunDirectoryPaths) -> str:
    """Render + write the per-axis three-method table; returns the markdown."""
    if not paths.analysis_report_path.exists():
        raise FileNotFoundError(
            "No analysis report yet — run `dtreat analyze` before methods-compare"
        )
    analysis = AnalysisReport.from_json(paths.analysis_report_path)
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    sources = {axis.axis_id: axis.source for axis in hypothesis_set.axes}

    counterfactual = None
    if paths.counterfactual_report_path.exists():
        counterfactual = CounterfactualReport.from_dict(
            load_json(paths.counterfactual_report_path)
        )
    cf_by_axis = {axis.axis_id: axis for axis in counterfactual.axes} if counterfactual else {}

    lines = [
        "# Method comparison: naturalistic vs counterfactual vs discovery",
        "",
        f"Pair: **{analysis.target_community}** vs **{analysis.baseline_community}**",
        "",
        "| axis | discovered by | naturalistic Δ | sig | counterfactual Δ (paired) | sig | judge κ |",
        "|------|---------------|---------------:|:---:|--------------------------:|:---:|--------:|",
    ]
    both_significant, either_significant = [], []
    for axis in sorted(analysis.axes, key=lambda a: -a.info_bits):
        cf = cf_by_axis.get(axis.axis_id)
        nat_sig = "✓" if axis.significant else "·"
        cf_sig = ("✓" if cf.significant else "·") if cf else "—"
        if axis.significant and cf and cf.significant:
            both_significant.append(axis.axis_id)
        if axis.significant or (cf and cf.significant):
            either_significant.append(axis.axis_id)
        lines.append(
            f"| `{axis.axis_id}` | {sources.get(axis.axis_id, 'helper')} "
            f"| {axis.delta:+.2f} | {nat_sig} "
            f"| {f'{cf.delta:+.2f}' if cf else '—'} | {cf_sig} "
            f"| {f'{axis.judge_kappa:.2f}' if axis.judge_kappa is not None else '—'} |"
        )

    lines += ["", "## Method-level summary", ""]
    lines.append(
        f"- naturalistic: {len(analysis.significant_axes())}/{len(analysis.axes)} "
        "axes significant (unpaired, distribution-matched prompts)"
    )
    if counterfactual:
        lines.append(
            f"- counterfactual: {len(counterfactual.significant_axes())}/"
            f"{len(counterfactual.axes)} axes significant over "
            f"{counterfactual.n_pairs} voice-swapped pairs "
            f"({counterfactual.n_twins_flagged} twins failed content validation)"
        )
        if counterfactual.naturalistic_correlation is not None:
            lines.append(
                f"- cross-design agreement: Pearson r = "
                f"{counterfactual.naturalistic_correlation:.2f} between paired and "
                f"unpaired per-axis effects; significant under BOTH designs: "
                f"{', '.join(f'`{a}`' for a in both_significant) or 'none'}"
            )
    else:
        lines.append("- counterfactual: not run (`dtreat counterfactual`)")
    condition_counts: dict[str, int] = {}
    for source in sources.values():
        condition_counts[source] = condition_counts.get(source, 0) + 1
    lines.append(
        "- discovery sources: "
        + ", ".join(f"{name} ({count})" for name, count in sorted(condition_counts.items()))
    )
    lines.append("")
    markdown = "\n".join(lines)
    paths.method_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    paths.method_comparison_path.write_text(markdown)
    log(f"  wrote {paths.method_comparison_path}")
    return markdown
