"""`dtreat judge-study` — does the verdict depend on who judges?

Scores the run's responses with EVERY study judge (any mix of providers and
model generations), then reports:
- pairwise Cohen's kappa matrix across all judges
- per-judge per-axis YES rates (systematic strictness differences)
- per-judge downstream statistics: treating each judge alone as the panel,
  which axes come out significant? (judge-dependence of conclusions)
- cost and parse health per judge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from dtreat.common.base_schema import BaseSchema
from dtreat.common.console_logging import log
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.file_io import load_jsonl, save_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.llm.chat_client import ChatClient
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord
from dtreat.stages.response_scoring.judge_agreement_metrics import cohen_kappa
from dtreat.stages.response_scoring.response_scoring_stage import judge_all_responses
from dtreat.stages.response_scoring.scored_response_schemas import ScoredResponse
from dtreat.stages.treatment_analysis.permutation_significance import (
    benjamini_hochberg,
    permutation_p_values,
)
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    build_cluster_matrices,
    build_permutation_clusters,
)


@dataclass
class JudgeStudyEntry(BaseSchema):
    """One judge's behavior over the shared responses."""

    judge_model: str
    n_scored: int = 0
    n_unparsed: int = 0
    yes_rate_by_axis: dict[str, float] = field(default_factory=dict)
    significant_axes: list[str] = field(default_factory=list)
    delta_by_axis: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0


@dataclass
class JudgePairKappa(BaseSchema):
    judge_a: str
    judge_b: str
    kappa: float | None
    raw_agreement: float
    n_pairs: int


@dataclass
class JudgeStudyReport(BaseSchema):
    """Cross-judge comparison over identical responses."""

    judge_models: list[str] = field(default_factory=list)
    entries: list[JudgeStudyEntry] = field(default_factory=list)
    kappa_matrix: list[JudgePairKappa] = field(default_factory=list)
    axes_significant_under_all: list[str] = field(default_factory=list)
    axes_significant_under_any: list[str] = field(default_factory=list)


def run_judge_study(
    config: ExperimentConfig,
    paths: RunDirectoryPaths,
    judge_models: list[str],
) -> JudgeStudyReport:
    """Score with every judge and write `stage4_scores/judge_study.json`."""
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    responses = [
        ResponseRecord.from_dict(record)
        for record in load_jsonl(paths.responses_path)
        if not record.get("refused") and record.get("text", "").strip()
    ]
    axis_ids = hypothesis_set.axis_ids()
    log(f"Judge study: {len(judge_models)} judges × {len(responses)} responses × {len(axis_ids)} axes")

    verdicts_by_judge: dict[str, dict[str, dict[str, bool | None]]] = {}
    report = JudgeStudyReport(judge_models=judge_models)
    for judge_model in judge_models:
        client = ChatClient(
            judge_model,
            role_label=f"judge-study:{judge_model}",
            cache_dir=paths.llm_cache_dir,
            trace_path=paths.llm_trace_path,
        )
        verdict_maps, _replies, failures, _calls = judge_all_responses(
            client, config, hypothesis_set, responses
        )
        verdicts_by_judge[judge_model] = verdict_maps
        entry = _study_entry(
            judge_model, verdict_maps, responses, axis_ids, config, client.stats.cost_usd
        )
        entry.n_unparsed += sum(
            1 for record in responses if record.response_id not in verdict_maps
        ) * len(axis_ids)
        if failures:
            log(f"  [warn] {judge_model}: {len(failures)} failed calls")
        report.entries.append(entry)
        log(
            f"  {judge_model}: {entry.n_scored} scored, {entry.n_unparsed} unparsed, "
            f"significant axes: {entry.significant_axes or 'none'}, ${entry.cost_usd:.4f}"
        )

    report.kappa_matrix = _kappa_matrix(verdicts_by_judge, responses, axis_ids)
    significant_sets = [set(entry.significant_axes) for entry in report.entries]
    if significant_sets:
        report.axes_significant_under_all = sorted(set.intersection(*significant_sets))
        report.axes_significant_under_any = sorted(set.union(*significant_sets))
    save_json(report.to_dict(), paths.judge_study_path)
    log(f"  wrote {paths.judge_study_path}")
    return report


def _study_entry(
    judge_model: str,
    verdict_maps: dict[str, dict[str, bool | None]],
    responses: list[ResponseRecord],
    axis_ids: list[str],
    config: ExperimentConfig,
    cost_usd: float,
) -> JudgeStudyEntry:
    """Single-judge downstream statistics on the shared responses."""
    scored = []
    n_unparsed = 0
    for record in responses:
        verdict_map = verdict_maps.get(record.response_id, {})
        verdicts = {axis: v for axis, v in verdict_map.items() if v is not None}
        n_unparsed += len(axis_ids) - len(verdicts)
        scored.append(
            ScoredResponse(
                response_id=record.response_id,
                prompt_id=record.prompt_id,
                community=record.community,
                instruction_id=record.instruction_id,
                refused=record.refused,
                verdicts=verdicts,
            )
        )

    clusters = build_permutation_clusters(scored, axis_ids, config.permutation_unit)
    sums, counts, is_target = build_cluster_matrices(
        clusters, axis_ids, config.target_community.name
    )
    p_values, deltas = permutation_p_values(
        sums, counts, is_target, config.n_permutations, config.seed
    )
    _q_values, significant = benjamini_hochberg(p_values, config.fdr_alpha)

    yes_rates = {}
    for j, axis_id in enumerate(axis_ids):
        total = counts[j].sum()
        yes_rates[axis_id] = round(float(sums[j].sum() / total), 4) if total else 0.0

    return JudgeStudyEntry(
        judge_model=judge_model,
        n_scored=len(scored),
        n_unparsed=n_unparsed,
        yes_rate_by_axis=yes_rates,
        significant_axes=[axis_ids[j] for j in range(len(axis_ids)) if significant[j]],
        delta_by_axis={
            axis_ids[j]: round(float(deltas[j]), 4)
            for j in range(len(axis_ids))
            if deltas[j] == deltas[j]  # skip NaN
        },
        cost_usd=cost_usd,
    )


def _kappa_matrix(
    verdicts_by_judge: dict[str, dict[str, dict[str, bool | None]]],
    responses: list[ResponseRecord],
    axis_ids: list[str],
) -> list[JudgePairKappa]:
    matrix = []
    for judge_a, judge_b in combinations(verdicts_by_judge, 2):
        paired_a: list[bool] = []
        paired_b: list[bool] = []
        for record in responses:
            map_a = verdicts_by_judge[judge_a].get(record.response_id, {})
            map_b = verdicts_by_judge[judge_b].get(record.response_id, {})
            for axis_id in axis_ids:
                verdict_a, verdict_b = map_a.get(axis_id), map_b.get(axis_id)
                if verdict_a is not None and verdict_b is not None:
                    paired_a.append(verdict_a)
                    paired_b.append(verdict_b)
        raw = (
            sum(a == b for a, b in zip(paired_a, paired_b, strict=True)) / len(paired_a)
            if paired_a
            else 0.0
        )
        matrix.append(
            JudgePairKappa(
                judge_a=judge_a,
                judge_b=judge_b,
                kappa=cohen_kappa(paired_a, paired_b),
                raw_agreement=raw,
                n_pairs=len(paired_a),
            )
        )
    return matrix
