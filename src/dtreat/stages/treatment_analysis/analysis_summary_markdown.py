"""Human-readable markdown rendering of the analysis report."""

from __future__ import annotations

from .analysis_report_schemas import AnalysisReport, AxisResult


def render_analysis_summary(report: AnalysisReport) -> str:
    """The auditable summary a practitioner reads first."""
    lines = [
        "# Differential treatment analysis",
        "",
        f"Comparing **{report.target_community}** (target) vs "
        f"**{report.baseline_community}** (baseline).",
        "",
        _headline(report),
        "",
        "## Axes of treatment",
        "",
        "| axis | question | z_target | z_baseline | Δ | p | q | significant | I (bits) |",
        "|------|----------|---------:|-----------:|--:|--:|--:|:-----------:|---------:|",
    ]
    for axis in _sorted_axes(report):
        lines.append(_axis_row(axis))
    lines += [
        "",
        f"Permutation test: {report.n_permutations} permutations at the "
        f"{report.permutation_unit} level; BH-FDR alpha = {report.fdr_alpha}.",
        "",
        "## Distribution-level measures",
        "",
    ]
    if report.d_pi_bits_significant_axes is not None:
        lines.append(
            f"- **D_pi (significant axes)** = {report.d_pi_bits_significant_axes:.2f} bits "
            f"(profiles smoothed with epsilon = {report.epsilon})"
        )
    if report.d_pi_bits_all_axes is not None:
        lines.append(f"- D_pi (all axes) = {report.d_pi_bits_all_axes:.2f} bits")
    if report.c2st:
        c2st = report.c2st
        verdict = "separable" if c2st.above_chance else "not distinguishable from chance"
        lines.append(
            f"- **C2ST accuracy** = {c2st.accuracy:.3f} "
            f"[{c2st.accuracy_ci_low:.3f}, {c2st.accuracy_ci_high:.3f}] vs majority "
            f"baseline {c2st.majority_baseline:.3f} -> behavior {verdict} "
            f"(train {c2st.n_train} / test {c2st.n_test}"
            + (
                f", {c2st.n_dropped_incomplete} dropped incomplete"
                if c2st.n_dropped_incomplete
                else ""
            )
            + ")"
        )
    if report.refusals:
        refusals = report.refusals
        lines += [
            "",
            "## Refusals",
            "",
            f"- {report.target_community}: {refusals.target_refusals}/{refusals.target_total} "
            f"({refusals.target_rate:.1%})",
            f"- {report.baseline_community}: {refusals.baseline_refusals}/"
            f"{refusals.baseline_total} ({refusals.baseline_rate:.1%})",
            f"- Fisher exact p = {refusals.fisher_p_value:.3f}",
        ]
    insufficient = [axis.axis_id for axis in report.axes if axis.insufficient_data]
    if insufficient:
        lines += [
            "",
            f"Axes with insufficient data (excluded): {', '.join(insufficient)}",
        ]
    lines.append("")
    return "\n".join(lines)


def _headline(report: AnalysisReport) -> str:
    n_significant = len(report.significant_axes())
    if n_significant == 0:
        return (
            "**No significant differential treatment detected** on the "
            f"{len(report.axes)} hypothesized axes at FDR {report.fdr_alpha}."
        )
    top = max(report.significant_axes(), key=lambda axis: axis.info_bits)
    return (
        f"**{n_significant} of {len(report.axes)} axes show significant differential "
        f"treatment** (FDR {report.fdr_alpha}). Most informative: `{top.axis_id}` "
        f"(Δ = {top.delta:+.2f}, I = {top.info_bits:.2f} bits)."
    )


def _sorted_axes(report: AnalysisReport) -> list[AxisResult]:
    """Significant first (by information, descending), then the rest by p."""
    return sorted(
        report.axes,
        key=lambda axis: (not axis.significant, -axis.info_bits, axis.p_value),
    )


def _axis_row(axis: AxisResult) -> str:
    marker = "**yes**" if axis.significant else ("n/a" if axis.insufficient_data else "no")
    return (
        f"| `{axis.axis_id}` | {axis.question} | {axis.rate_target:.2f} "
        f"| {axis.rate_baseline:.2f} | {axis.delta:+.2f} | {axis.p_value:.3f} "
        f"| {axis.q_value:.3f} | {marker} | {axis.info_bits:.2f} |"
    )
