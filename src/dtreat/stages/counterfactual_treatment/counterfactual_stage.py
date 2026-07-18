"""Stage 6 (optional) — counterfactual voice-swap analysis.

Complements the naturalistic pipeline: instead of comparing distributions of
different real prompts, each real prompt is re-voiced as the other community
and the target's responses to the pair are compared — isolating the causal
effect of community voice with request content held fixed (adapting the
name-counterfactual design of arXiv:2410.19803 to implicit voice).

Requires stages 1-4 (prompts, axes, original responses+scores) to exist.
"""

from __future__ import annotations

import numpy as np

from dtreat.common.console_logging import log, log_kv
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.file_io import load_jsonl, save_json, save_jsonl
from dtreat.common.random_seed import derive_seed
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.llm.parallel_chat_execution import ChatJob, execute_chat_jobs
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.prompt_collection.prompt_set_schemas import PromptStageArtifact
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord
from dtreat.stages.response_scoring.response_scoring_stage import (
    aggregate_panel_verdict,
    judge_all_responses,
)
from dtreat.stages.response_scoring.scored_response_schemas import ScoredResponse
from dtreat.stages.treatment_analysis.analysis_report_schemas import AnalysisReport
from dtreat.stages.treatment_analysis.permutation_significance import benjamini_hochberg

from .paired_significance import sign_flip_p_values
from .twin_generation import generate_twins, validate_twins
from .twin_schemas import CounterfactualAxisResult, CounterfactualReport, TwinPair


def run_counterfactual(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> CounterfactualReport:
    """Execute the counterfactual arm and write its report."""
    log("Stage 6: counterfactual voice-swap analysis")
    artifact = PromptStageArtifact.from_json(paths.prompt_sets_path)
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)

    twins = _load_or_generate_twins(config, paths, artifact)
    usable = [twin for twin in twins if twin.content_preserved]
    log(f"  {len(usable)}/{len(twins)} twins usable (content preserved)")

    twin_scored = _collect_and_score_twins(config, paths, hypothesis_set, usable)
    original_scored = {
        record["response_id"]: ScoredResponse.from_dict(record)
        for record in load_jsonl(paths.scored_responses_path)
    }

    report = _paired_analysis(
        config, artifact, hypothesis_set, usable, twin_scored, original_scored
    )
    report.n_twins_flagged = len(twins) - len(usable)
    _join_naturalistic(paths, report)

    save_json(report.to_dict(), paths.counterfactual_report_path)
    log_kv(
        {
            "pairs analyzed": report.n_pairs,
            "significant axes (paired)": f"{len(report.significant_axes())}/{len(report.axes)}",
            "corr with naturalistic Δ": (
                f"{report.naturalistic_correlation:.2f}"
                if report.naturalistic_correlation is not None
                else "n/a"
            ),
        }
    )
    log(f"  wrote {paths.counterfactual_report_path}")
    return report


def _load_or_generate_twins(
    config: ExperimentConfig,
    paths: RunDirectoryPaths,
    artifact: PromptStageArtifact,
) -> list[TwinPair]:
    if paths.twins_path.exists():
        return [TwinPair.from_dict(r) for r in load_jsonl(paths.twins_path)]
    rewriter = ChatClient(
        config.rewriter_model,
        role_label="rewriter",
        cache_dir=paths.llm_cache_dir,
        trace_path=paths.llm_trace_path,
    )
    twins = generate_twins(config, artifact, rewriter, config.max_workers)
    flagged = validate_twins(config, twins, rewriter, config.max_workers)
    if flagged:
        log(f"  [note] {flagged} twins failed content-preservation validation")
    save_jsonl([twin.to_dict() for twin in twins], paths.twins_path)
    return twins


def _collect_and_score_twins(
    config: ExperimentConfig,
    paths: RunDirectoryPaths,
    hypothesis_set: HypothesisSet,
    twins: list[TwinPair],
) -> dict[str, dict[str, dict[str, bool]]]:
    """Sample + judge twin responses.

    Returns {pair_id: {response_key: aggregated verdicts}}.
    """
    target_client = ChatClient(
        config.target_model,
        role_label="target-twin",
        cache_dir=paths.llm_cache_dir,
        trace_path=paths.llm_trace_path,
    )
    jobs = []
    for twin in twins:
        for sample_index in range(config.samples_per_prompt):
            jobs.append(
                ChatJob(
                    job_id=f"{twin.pair_id}~s{sample_index}",
                    request=target_client.build_request(
                        [ChatMessage("user", twin.twin_text)],
                        temperature=config.temperature,
                        max_tokens=config.max_response_tokens,
                        seed=derive_seed(config.seed, "twin-response", twin.pair_id, sample_index),
                    ),
                )
            )
    results, failures = execute_chat_jobs(
        target_client, jobs, max_workers=config.max_workers, description="twin responses"
    )
    if failures:
        log(f"  [warn] {len(failures)} twin response calls failed")
    save_jsonl(
        [
            {"response_id": job_id, "text": result.text, "refused": result.refused}
            for job_id, result in sorted(results.items())
        ],
        paths.twin_responses_path,
    )

    # Judge twin responses with the configured panel on the same axes
    pseudo_records = [
        ResponseRecord(
            response_id=response_id,
            prompt_id=response_id.split("~")[0],
            community="twin",
            instruction_id="",
            sample_index=0,
            seed=0,
            model=config.target_model,
            text=result.text,
            refused=result.refused,
        )
        for response_id, result in sorted(results.items())
        if result.text.strip() and not result.refused
    ]
    per_judge: dict[str, dict[str, dict[str, bool | None]]] = {}
    judge_models = config.judge_panel()
    for judge_model in judge_models:
        judge_client = ChatClient(
            judge_model,
            role_label=f"judge-twin:{judge_model}",
            cache_dir=paths.llm_cache_dir,
            trace_path=paths.llm_trace_path,
        )
        verdict_maps, _replies, _failures, _calls = judge_all_responses(
            judge_client, config, hypothesis_set, pseudo_records
        )
        per_judge[judge_model] = verdict_maps

    axis_ids = hypothesis_set.axis_ids()
    scored: dict[str, dict[str, dict[str, bool]]] = {}
    scored_rows = []
    for record in pseudo_records:
        aggregated = {}
        for axis_id in axis_ids:
            panel = [
                per_judge[judge].get(record.response_id, {}).get(axis_id)
                for judge in judge_models
            ]
            verdict = aggregate_panel_verdict(panel, config.judge_aggregation)
            if verdict is not None:
                aggregated[axis_id] = verdict
        pair_id = record.prompt_id
        scored.setdefault(pair_id, {})[record.response_id] = aggregated
        scored_rows.append({"response_id": record.response_id, "verdicts": aggregated})
    save_jsonl(scored_rows, paths.twin_scored_path)
    return scored


def _paired_analysis(
    config: ExperimentConfig,
    artifact: PromptStageArtifact,
    hypothesis_set: HypothesisSet,
    twins: list[TwinPair],
    twin_scored: dict[str, dict[str, dict[str, bool]]],
    original_scored: dict[str, ScoredResponse],
) -> CounterfactualReport:
    axis_ids = hypothesis_set.axis_ids()
    questions = {axis.axis_id: axis.question for axis in hypothesis_set.axes}
    target_name = config.target_community.name

    # originals grouped by prompt
    originals_by_prompt: dict[str, list[ScoredResponse]] = {}
    for record in original_scored.values():
        originals_by_prompt.setdefault(record.prompt_id, []).append(record)

    diffs = np.full((len(axis_ids), len(twins)), np.nan)
    direction = np.zeros(len(twins), dtype=bool)  # True = original was target-voiced
    for i, twin in enumerate(twins):
        original_records = originals_by_prompt.get(twin.original_prompt_id, [])
        twin_verdict_maps = list(twin_scored.get(twin.pair_id, {}).values())
        if not original_records or not twin_verdict_maps:
            continue
        original_is_target = twin.original_community == target_name
        direction[i] = original_is_target
        for j, axis_id in enumerate(axis_ids):
            original_verdicts = [
                r.verdicts[axis_id] for r in original_records if axis_id in r.verdicts
            ]
            twin_verdicts = [
                verdicts[axis_id] for verdicts in twin_verdict_maps if axis_id in verdicts
            ]
            if not original_verdicts or not twin_verdicts:
                continue
            original_rate = sum(original_verdicts) / len(original_verdicts)
            twin_rate = sum(twin_verdicts) / len(twin_verdicts)
            # orient every diff as (target voice) − (baseline voice)
            diffs[j, i] = (
                original_rate - twin_rate if original_is_target else twin_rate - original_rate
            )

    p_values, observed = sign_flip_p_values(diffs, config.n_permutations, config.seed)
    q_values, significant = benjamini_hochberg(p_values, config.fdr_alpha)

    axes = []
    for j, axis_id in enumerate(axis_ids):
        valid = ~np.isnan(diffs[j])
        n_pairs = int(valid.sum())
        if n_pairs == 0:
            continue
        on_target = diffs[j, valid & direction]
        on_baseline = diffs[j, valid & ~direction]
        # reconstruct voice-conditional rates from oriented diffs is lossy;
        # report the mean rates directly instead
        target_voice_rates, baseline_voice_rates = _voice_rates(
            axis_id, twins, twin_scored, originals_by_prompt, target_name
        )
        axes.append(
            CounterfactualAxisResult(
                axis_id=axis_id,
                question=questions[axis_id],
                n_pairs=n_pairs,
                rate_target_voice=target_voice_rates,
                rate_baseline_voice=baseline_voice_rates,
                delta=float(observed[j]),
                p_value=float(p_values[j]),
                q_value=float(q_values[j]),
                significant=bool(significant[j]),
                delta_on_target_content=(
                    float(np.mean(on_target)) if on_target.size else None
                ),
                delta_on_baseline_content=(
                    float(np.mean(on_baseline)) if on_baseline.size else None
                ),
            )
        )

    return CounterfactualReport(
        target_community=target_name,
        baseline_community=config.baseline_community.name,
        n_pairs=int(np.sum(~np.isnan(diffs).all(axis=0))),
        axes=axes,
        n_permutations=config.n_permutations,
        fdr_alpha=config.fdr_alpha,
        seed=config.seed,
    )


def _voice_rates(
    axis_id: str,
    twins: list[TwinPair],
    twin_scored: dict[str, dict[str, dict[str, bool]]],
    originals_by_prompt: dict[str, list[ScoredResponse]],
    target_name: str,
) -> tuple[float, float]:
    """Mean verdict rate under each voice, pooling originals and twins."""
    target_voice: list[bool] = []
    baseline_voice: list[bool] = []
    for twin in twins:
        original_verdicts = [
            r.verdicts[axis_id]
            for r in originals_by_prompt.get(twin.original_prompt_id, [])
            if axis_id in r.verdicts
        ]
        twin_verdicts = [
            verdicts[axis_id]
            for verdicts in twin_scored.get(twin.pair_id, {}).values()
            if axis_id in verdicts
        ]
        if twin.original_community == target_name:
            target_voice.extend(original_verdicts)
            baseline_voice.extend(twin_verdicts)
        else:
            baseline_voice.extend(original_verdicts)
            target_voice.extend(twin_verdicts)
    rate_target = sum(target_voice) / len(target_voice) if target_voice else 0.0
    rate_baseline = sum(baseline_voice) / len(baseline_voice) if baseline_voice else 0.0
    return round(rate_target, 4), round(rate_baseline, 4)


def _join_naturalistic(paths: RunDirectoryPaths, report: CounterfactualReport) -> None:
    """Attach naturalistic deltas + the cross-design correlation."""
    if not paths.analysis_report_path.exists():
        return
    analysis = AnalysisReport.from_json(paths.analysis_report_path)
    naturalistic = {axis.axis_id: axis.delta for axis in analysis.axes}
    pairs = []
    for axis in report.axes:
        if axis.axis_id in naturalistic:
            axis.naturalistic_delta = naturalistic[axis.axis_id]
            pairs.append((axis.delta, axis.naturalistic_delta))
    if len(pairs) >= 3:
        counterfactual_deltas, naturalistic_deltas = zip(*pairs, strict=True)
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(counterfactual_deltas, naturalistic_deltas)[0, 1])
        report.naturalistic_correlation = None if np.isnan(corr) else round(corr, 4)
