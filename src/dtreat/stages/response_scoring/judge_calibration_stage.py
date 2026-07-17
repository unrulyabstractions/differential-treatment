"""`dtreat calibrate-judge` — validate the judge panel (paper §5.3).

Three lenses, each optional depending on what data exists:
1. Inter-judge agreement (needs >= 2 judges in stage 4): pairwise Cohen's
   kappa and per-axis Fleiss' kappa over the panel.
2. Self-consistency: re-judge a sample with a different seed and measure
   per-axis verdict flip rates (0 for deterministic judges).
3. Gold labels (optional file {response_id: {axis_id: bool}}): per-judge
   accuracy against human annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from dtreat.common.base_schema import BaseSchema
from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import load_json, load_jsonl, save_json
from dtreat.common.random_seed import derive_seed
from dtreat.llm.chat_client import ChatClient
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord

from .judge_agreement_metrics import cohen_kappa, fleiss_kappa
from .response_scoring_stage import judge_all_responses
from .scored_response_schemas import ScoredResponse


@dataclass
class JudgePairAgreement(BaseSchema):
    """Cohen's kappa between two judges, pooled over axes and per axis."""

    judge_a: str
    judge_b: str
    n_paired_verdicts: int
    raw_agreement: float
    kappa_overall: float | None
    kappa_by_axis: dict[str, float] = field(default_factory=dict)


@dataclass
class AxisPanelAgreement(BaseSchema):
    """Fleiss' kappa for the full panel on one axis."""

    axis_id: str
    n_items: int
    fleiss: float | None


@dataclass
class JudgeConsistency(BaseSchema):
    """Verdict flip rate for one judge when re-judged under a new seed."""

    judge_model: str
    n_rejudged: int
    flip_rate_overall: float
    flip_rate_by_axis: dict[str, float] = field(default_factory=dict)


@dataclass
class JudgeGoldAccuracy(BaseSchema):
    """One judge's accuracy against gold labels."""

    judge_model: str
    n_labels: int
    accuracy: float
    accuracy_by_axis: dict[str, float] = field(default_factory=dict)


@dataclass
class JudgeCalibrationReport(BaseSchema):
    """Everything measured about the judge panel's reliability."""

    judge_models: list[str] = field(default_factory=list)
    pair_agreements: list[JudgePairAgreement] = field(default_factory=list)
    axis_panel_agreements: list[AxisPanelAgreement] = field(default_factory=list)
    consistency: list[JudgeConsistency] = field(default_factory=list)
    gold_accuracy: list[JudgeGoldAccuracy] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_judge_calibration(
    config: ExperimentConfig,
    paths: RunDirectoryPaths,
    consistency_sample: int = 20,
    gold_labels_file: str | None = None,
) -> JudgeCalibrationReport:
    """Execute calibration and write `stage4_scores/judge_calibration.json`."""
    log("Judge calibration")
    scored = [
        ScoredResponse.from_dict(record)
        for record in load_jsonl(paths.scored_responses_path)
    ]
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    axis_ids = hypothesis_set.axis_ids()
    judge_models = config.judge_panel()

    report = JudgeCalibrationReport(judge_models=judge_models)
    if len(judge_models) >= 2:
        report.pair_agreements = _pairwise_agreement(scored, judge_models, axis_ids)
        report.axis_panel_agreements = _panel_agreement(scored, judge_models, axis_ids)
    else:
        report.notes.append(
            "Single-judge run: inter-judge agreement unavailable. Add judge_models "
            "to the config for panel calibration."
        )
    if consistency_sample > 0:
        report.consistency = _self_consistency(
            config, paths, hypothesis_set, scored, consistency_sample
        )
    if gold_labels_file:
        report.gold_accuracy = _gold_accuracy(
            scored, judge_models, load_json(Path(gold_labels_file))
        )

    save_json(report.to_dict(), paths.judge_calibration_path)
    _log_summary(report)
    log(f"  wrote {paths.judge_calibration_path}")
    return report


def _pairwise_agreement(scored, judge_models, axis_ids) -> list[JudgePairAgreement]:
    agreements = []
    for judge_a, judge_b in combinations(judge_models, 2):
        pooled_a: list[bool] = []
        pooled_b: list[bool] = []
        by_axis: dict[str, tuple[list[bool], list[bool]]] = {
            axis: ([], []) for axis in axis_ids
        }
        for record in scored:
            verdicts_a = record.verdicts_by_judge.get(judge_a, {})
            verdicts_b = record.verdicts_by_judge.get(judge_b, {})
            for axis in axis_ids:
                if axis in verdicts_a and axis in verdicts_b:
                    pooled_a.append(bool(verdicts_a[axis]))
                    pooled_b.append(bool(verdicts_b[axis]))
                    by_axis[axis][0].append(bool(verdicts_a[axis]))
                    by_axis[axis][1].append(bool(verdicts_b[axis]))
        raw = (
            sum(a == b for a, b in zip(pooled_a, pooled_b, strict=True)) / len(pooled_a)
            if pooled_a
            else 0.0
        )
        kappa_by_axis = {}
        for axis, (list_a, list_b) in by_axis.items():
            kappa = cohen_kappa(list_a, list_b)
            if kappa is not None:
                kappa_by_axis[axis] = round(kappa, 4)
        agreements.append(
            JudgePairAgreement(
                judge_a=judge_a,
                judge_b=judge_b,
                n_paired_verdicts=len(pooled_a),
                raw_agreement=raw,
                kappa_overall=cohen_kappa(pooled_a, pooled_b),
                kappa_by_axis=kappa_by_axis,
            )
        )
    return agreements


def _panel_agreement(scored, judge_models, axis_ids) -> list[AxisPanelAgreement]:
    panel_size = len(judge_models)
    results = []
    for axis in axis_ids:
        yes_counts = []
        for record in scored:
            verdicts = [
                record.verdicts_by_judge.get(judge, {}).get(axis)
                for judge in judge_models
            ]
            if all(v is not None for v in verdicts):
                yes_counts.append(sum(bool(v) for v in verdicts))
        results.append(
            AxisPanelAgreement(
                axis_id=axis,
                n_items=len(yes_counts),
                fleiss=fleiss_kappa(yes_counts, panel_size),
            )
        )
    return results


def _self_consistency(
    config, paths, hypothesis_set, scored, sample_size
) -> list[JudgeConsistency]:
    """Re-judge a sample under a shifted seed and count verdict flips."""
    responses = {
        record["response_id"]: ResponseRecord.from_dict(record)
        for record in load_jsonl(paths.responses_path)
    }
    sample = sorted(scored, key=lambda s: s.response_id)[:sample_size]
    sample_records = [responses[s.response_id] for s in sample if s.response_id in responses]

    consistency_config = ExperimentConfig.from_dict(config.to_dict())
    consistency_config.seed = derive_seed(config.seed, "judge-consistency")

    results = []
    for judge_model in config.judge_panel():
        client = ChatClient(
            judge_model,
            role_label=f"judge-consistency:{judge_model}",
            cache_dir=paths.llm_cache_dir,
            trace_path=paths.llm_trace_path,
        )
        rejudged, _replies, _failures, _calls = judge_all_responses(
            client, consistency_config, hypothesis_set, sample_records
        )
        flips: dict[str, list[bool]] = {}
        for scored_record in sample:
            original = scored_record.verdicts_by_judge.get(judge_model, {})
            redo = rejudged.get(scored_record.response_id, {})
            for axis, original_verdict in original.items():
                redo_verdict = redo.get(axis)
                if redo_verdict is not None:
                    flips.setdefault(axis, []).append(
                        bool(original_verdict) != bool(redo_verdict)
                    )
        all_flips = [flip for axis_flips in flips.values() for flip in axis_flips]
        results.append(
            JudgeConsistency(
                judge_model=judge_model,
                n_rejudged=len(sample_records),
                flip_rate_overall=(sum(all_flips) / len(all_flips)) if all_flips else 0.0,
                flip_rate_by_axis={
                    axis: round(sum(axis_flips) / len(axis_flips), 4)
                    for axis, axis_flips in flips.items()
                    if axis_flips
                },
            )
        )
    return results


def _gold_accuracy(scored, judge_models, gold_labels: dict) -> list[JudgeGoldAccuracy]:
    results = []
    for judge_model in judge_models:
        correct_by_axis: dict[str, list[bool]] = {}
        for record in scored:
            gold = gold_labels.get(record.response_id, {})
            verdicts = record.verdicts_by_judge.get(judge_model, {})
            for axis, gold_verdict in gold.items():
                if axis in verdicts:
                    correct_by_axis.setdefault(axis, []).append(
                        bool(verdicts[axis]) == bool(gold_verdict)
                    )
        all_correct = [c for axis_list in correct_by_axis.values() for c in axis_list]
        if not all_correct:
            continue
        results.append(
            JudgeGoldAccuracy(
                judge_model=judge_model,
                n_labels=len(all_correct),
                accuracy=sum(all_correct) / len(all_correct),
                accuracy_by_axis={
                    axis: round(sum(axis_list) / len(axis_list), 4)
                    for axis, axis_list in correct_by_axis.items()
                },
            )
        )
    return results


def _log_summary(report: JudgeCalibrationReport) -> None:
    for pair in report.pair_agreements:
        kappa_text = "n/a" if pair.kappa_overall is None else f"{pair.kappa_overall:.3f}"
        log_kv(
            {
                f"{pair.judge_a} vs {pair.judge_b}": (
                    f"raw {pair.raw_agreement:.3f}, kappa {kappa_text} "
                    f"(n={pair.n_paired_verdicts})"
                )
            }
        )
    for consistency in report.consistency:
        log_kv(
            {
                f"consistency {consistency.judge_model}": (
                    f"flip rate {consistency.flip_rate_overall:.3f} "
                    f"over {consistency.n_rejudged} responses"
                )
            }
        )
    for note in report.notes:
        log(f"  [Note] {note}")
