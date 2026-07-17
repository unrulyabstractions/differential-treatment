"""Stage 5 — comparing distributions (paper §4.5).

Turns scored responses into the differential-treatment picture: per-axis rate
gaps (Eq 9–10), permutation significance with BH-FDR filtering (§4.5.1),
mutual-information ranking (§4.5.2, Eq 13–14), treatment-profile divergence
D_pi (Eq 11–12), and the classifier two-sample test (§4.5.3).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact

from dtreat.common.console_logging import log, log_kv
from dtreat.common.discrete_information import (
    community_axis_information,
    kl_divergence_bits,
    normalize_profile,
)
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.file_io import load_jsonl, save_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.prompt_distinguishability.distinguish_bridge_stage import (
    load_input_report_if_present,
)
from dtreat.stages.prompt_distinguishability.distinguish_report_schemas import (
    InputOutputComparison,
)
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord
from dtreat.stages.response_scoring.scored_response_schemas import ScoredResponse

from .analysis_report_schemas import (
    AnalysisReport,
    AxisResult,
    PromptBehaviorRates,
    RefusalAnalysis,
)
from .analysis_summary_markdown import render_analysis_summary
from .classifier_two_sample_test import run_c2st
from .permutation_significance import benjamini_hochberg, permutation_p_values


def run_treatment_analysis(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> AnalysisReport:
    """Execute stage 5 and write the report + human-readable summary."""
    log("Stage 5: comparing behavior distributions")
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    scored = [
        ScoredResponse.from_dict(record)
        for record in load_jsonl(paths.scored_responses_path)
    ]
    target_name = config.target_community.name
    axis_ids = hypothesis_set.axis_ids()
    questions = {axis.axis_id: axis.question for axis in hypothesis_set.axes}

    if not scored:
        raise ValueError("No scored responses found; run `dtreat score` first")

    clusters = build_permutation_clusters(scored, axis_ids, config.permutation_unit)
    sums, counts, is_target = build_cluster_matrices(clusters, axis_ids, target_name)

    p_values, deltas = permutation_p_values(
        sums, counts, is_target, config.n_permutations, config.seed
    )
    q_values, significant = benjamini_hochberg(p_values, config.fdr_alpha)

    axes = _axis_results(
        axis_ids, questions, sums, counts, is_target, deltas, p_values, q_values, significant
    )
    report = AnalysisReport(
        target_community=target_name,
        baseline_community=config.baseline_community.name,
        axes=axes,
        d_pi_bits_significant_axes=_profile_divergence(
            [a for a in axes if a.significant], config.epsilon
        ),
        d_pi_bits_all_axes=_profile_divergence(
            [a for a in axes if not a.insufficient_data], config.epsilon
        ),
        c2st=_c2st_from_scored(scored, axis_ids, target_name, config),
        refusals=_refusal_analysis(paths, target_name),
        input_output=None,
        prompt_rates=_prompt_rates(scored, axis_ids),
        n_permutations=config.n_permutations,
        permutation_unit=config.permutation_unit,
        fdr_alpha=config.fdr_alpha,
        epsilon=config.epsilon,
        seed=config.seed,
    )

    report.input_output = _input_output_comparison(paths, report)
    save_json(report.to_dict(), paths.analysis_report_path)
    paths.analysis_summary_path.parent.mkdir(parents=True, exist_ok=True)
    paths.analysis_summary_path.write_text(render_analysis_summary(report))

    log_kv(
        {
            "significant axes": f"{len(report.significant_axes())}/{len(axes)}",
            "D_pi (significant axes)": _fmt_bits(report.d_pi_bits_significant_axes),
            "C2ST accuracy": f"{report.c2st.accuracy:.3f}" if report.c2st else "n/a",
        }
    )
    log(f"  wrote {paths.analysis_report_path}")
    log(f"  wrote {paths.analysis_summary_path}")
    return report


# ── data shaping ─────────────────────────────────────────────────────────


def build_permutation_clusters(
    scored: list[ScoredResponse], axis_ids: list[str], permutation_unit: str
) -> list[tuple[str, str, list[ScoredResponse]]]:
    """Group scored responses into exchangeable units for permutation."""
    if permutation_unit == "response":
        return [(s.response_id, s.community, [s]) for s in scored]
    grouped: dict[str, list[ScoredResponse]] = {}
    for record in scored:
        grouped.setdefault(record.prompt_id, []).append(record)
    return [
        (prompt_id, records[0].community, records)
        for prompt_id, records in sorted(grouped.items())
    ]


def build_cluster_matrices(
    clusters: list[tuple[str, str, list[ScoredResponse]]],
    axis_ids: list[str],
    target_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_axes, n_clusters = len(axis_ids), len(clusters)
    sums = np.zeros((n_axes, n_clusters))
    counts = np.zeros((n_axes, n_clusters))
    is_target = np.zeros(n_clusters, dtype=bool)
    for cluster_index, (_cluster_id, community, records) in enumerate(clusters):
        is_target[cluster_index] = community == target_name
        for axis_index, axis_id in enumerate(axis_ids):
            verdicts = [r.verdicts[axis_id] for r in records if axis_id in r.verdicts]
            sums[axis_index, cluster_index] = sum(verdicts)
            counts[axis_index, cluster_index] = len(verdicts)
    return sums, counts, is_target


def _axis_results(
    axis_ids, questions, sums, counts, is_target, deltas, p_values, q_values, significant
) -> list[AxisResult]:
    results = []
    for j, axis_id in enumerate(axis_ids):
        n_target = int(counts[j, is_target].sum())
        n_baseline = int(counts[j, ~is_target].sum())
        insufficient = n_target == 0 or n_baseline == 0
        rate_target = float(sums[j, is_target].sum() / n_target) if n_target else 0.0
        rate_baseline = float(sums[j, ~is_target].sum() / n_baseline) if n_baseline else 0.0
        weight_target = n_target / (n_target + n_baseline) if (n_target + n_baseline) else 0.5
        results.append(
            AxisResult(
                axis_id=axis_id,
                question=questions[axis_id],
                n_target=n_target,
                n_baseline=n_baseline,
                rate_target=rate_target,
                rate_baseline=rate_baseline,
                delta=0.0 if insufficient else float(deltas[j]),
                p_value=float(p_values[j]),
                q_value=float(q_values[j]),
                significant=bool(significant[j]) and not insufficient,
                info_bits=0.0
                if insufficient
                else community_axis_information(rate_target, rate_baseline, weight_target),
                insufficient_data=insufficient,
            )
        )
    return results


# ── summary statistics ───────────────────────────────────────────────────


def _profile_divergence(axes: list[AxisResult], epsilon: float) -> float | None:
    """D_pi over the given axes (Eq 11–12); None if fewer than 2 axes."""
    if len(axes) < 2:
        return None
    target_profile = normalize_profile(
        np.array([axis.rate_target for axis in axes]), epsilon
    )
    baseline_profile = normalize_profile(
        np.array([axis.rate_baseline for axis in axes]), epsilon
    )
    return kl_divergence_bits(target_profile, baseline_profile)


def _c2st_from_scored(scored, axis_ids, target_name, config):
    complete = [s for s in scored if all(axis in s.verdicts for axis in axis_ids)]
    dropped = len(scored) - len(complete)
    if not complete:
        return None
    features = np.array(
        [[1.0 if s.verdicts[axis] else 0.0 for axis in axis_ids] for s in complete]
    )
    labels = np.array([s.community == target_name for s in complete])
    return run_c2st(features, labels, config.c2st_test_fraction, config.seed, dropped)


def _refusal_analysis(paths: RunDirectoryPaths, target_name: str) -> RefusalAnalysis | None:
    """Refusal rates come from stage 3 (refused responses are never judged)."""
    try:
        responses = [
            ResponseRecord.from_dict(r) for r in load_jsonl(paths.responses_path)
        ]
    except FileNotFoundError:
        return None
    target = [r for r in responses if r.community == target_name]
    baseline = [r for r in responses if r.community != target_name]
    if not target or not baseline:
        return None
    target_refused = sum(r.refused for r in target)
    baseline_refused = sum(r.refused for r in baseline)
    table = [
        [target_refused, len(target) - target_refused],
        [baseline_refused, len(baseline) - baseline_refused],
    ]
    _odds, p_value = fisher_exact(table)
    return RefusalAnalysis(
        target_refusals=target_refused,
        target_total=len(target),
        baseline_refusals=baseline_refused,
        baseline_total=len(baseline),
        target_rate=target_refused / len(target),
        baseline_rate=baseline_refused / len(baseline),
        fisher_p_value=float(p_value),
    )


def _prompt_rates(scored: list[ScoredResponse], axis_ids: list[str]) -> list[PromptBehaviorRates]:
    grouped: dict[str, list[ScoredResponse]] = {}
    for record in scored:
        grouped.setdefault(record.prompt_id, []).append(record)
    rates = []
    for prompt_id, records in sorted(grouped.items()):
        axis_rates = {}
        for axis_id in axis_ids:
            verdicts = [r.verdicts[axis_id] for r in records if axis_id in r.verdicts]
            if verdicts:
                axis_rates[axis_id] = sum(verdicts) / len(verdicts)
        rates.append(
            PromptBehaviorRates(
                prompt_id=prompt_id,
                community=records[0].community,
                n_responses=len(records),
                rates=axis_rates,
            )
        )
    return rates


def _input_output_comparison(
    paths: RunDirectoryPaths, report: AnalysisReport
) -> InputOutputComparison | None:
    """Compare input-side prompt legibility with output-side treatment
    (present only when the distinguish bridge has run for this run)."""
    input_report = load_input_report_if_present(paths)
    if input_report is None:
        return None
    input_acc = input_report.best_c2st_accuracy
    output_acc = report.c2st.accuracy if report.c2st else None
    signal_usage = None
    if input_acc is not None and output_acc is not None and input_acc > 0.5:
        signal_usage = max(0.0, (output_acc - 0.5) / (input_acc - 0.5))
    comparison = InputOutputComparison(
        input_c2st_accuracy=input_acc,
        input_n_significant=input_report.n_significant,
        input_n_tests=input_report.n_tests,
        output_c2st_accuracy=output_acc,
        output_significant_axes=len(report.significant_axes()),
        output_total_axes=len(report.axes),
        output_d_pi_bits=report.d_pi_bits_significant_axes,
        signal_usage=signal_usage,
    )
    output_evidence = bool(report.c2st and report.c2st.above_chance) or bool(
        report.significant_axes()
    )
    comparison.interpretation = _interpret_input_output(comparison, output_evidence)
    return comparison


def _interpret_input_output(
    comparison: InputOutputComparison, output_evidence: bool
) -> str:
    """Honest tiering: 'the model acts on the signal' is only claimed when the
    output side is itself statistically significant, not from point estimates."""
    input_acc, output_acc = comparison.input_c2st_accuracy, comparison.output_c2st_accuracy
    if input_acc is None or input_acc <= 0.55:
        return (
            "The prompt sets themselves are barely separable, so any treatment "
            "difference cannot be attributed to community legibility of the inputs."
        )
    if comparison.signal_usage is None or output_acc is None:
        return "Input prompts are community-legible; output separability was not measurable."
    usage_pct = f"{comparison.signal_usage:.0%}"
    if output_evidence:
        return (
            f"Prompts are community-legible (input C2ST {input_acc:.2f}) and the model "
            f"acts on it: behavior separability {output_acc:.2f} carries ~{usage_pct} of "
            f"the input signal, with {comparison.output_significant_axes} significant "
            "treatment axes."
        )
    if comparison.signal_usage < 0.25:
        return (
            f"Prompts are community-legible (input C2ST {input_acc:.2f}) but the "
            f"model's behavior is close to indistinguishable (output C2ST "
            f"{output_acc:.2f}): it carries ~{usage_pct} of the input signal into "
            "behavior — little visible differential treatment on the tested axes."
        )
    return (
        f"Prompts are community-legible (input C2ST {input_acc:.2f}); output point "
        f"estimates suggest ~{usage_pct} of that signal may carry into behavior "
        f"(output C2ST {output_acc:.2f}), but neither axis-level nor "
        "distribution-level differences are statistically significant at this "
        "sample size — collect more prompts before drawing conclusions."
    )


def _fmt_bits(value: float | None) -> str:
    return f"{value:.2f} bits" if value is not None else "n/a"
