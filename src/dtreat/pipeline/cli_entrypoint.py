"""`dtreat` CLI: run any pipeline stage separately, or everything, plus
diagnostics (status / inspect / validate / trace / estimate-cost) and the
debug server (`dtreat serve`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from dtreat.common.console_logging import log, log_stage
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.file_io import save_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.diagnostics.artifact_inspection import inspect_path
from dtreat.diagnostics.cost_estimation import print_cost_estimate
from dtreat.diagnostics.cross_run_comparison import run_cross_run_comparison
from dtreat.diagnostics.llm_trace_reporting import print_trace_report
from dtreat.diagnostics.run_validation import print_run_status, validate_run
from dtreat.server.debug_server_app import serve_debug_ui
from dtreat.stages.hypothesis_generation.helper_condition_study import (
    STANDARD_CONDITIONS,
    run_helper_conditions,
    summarize_downstream,
)
from dtreat.stages.prompt_collection.prompt_collection_stage import run_prompt_collection
from dtreat.stages.prompt_distinguishability.distinguish_bridge_stage import (
    run_prompt_distinguishability,
)
from dtreat.stages.response_collection.response_collection_stage import (
    run_response_collection,
)
from dtreat.stages.response_scoring.judge_calibration_stage import run_judge_calibration
from dtreat.stages.response_scoring.judge_model_study import run_judge_study
from dtreat.stages.response_scoring.response_scoring_stage import run_response_scoring
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    run_treatment_analysis,
)

from .stage_registry import PIPELINE_STAGES, STAGES_BY_NAME


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        log(f"ERROR: {error}")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtreat",
        description="Differential-treatment discovery pipeline (see paper/ for the method)",
    )
    subparsers = parser.add_subparsers(dest="command")

    for stage in PIPELINE_STAGES:
        stage_parser = subparsers.add_parser(
            stage.name, help=f"{stage.title} ({stage.paper_section})"
        )
        _add_config_arguments(stage_parser)
        if stage.name == "responses":
            stage_parser.add_argument(
                "--limit", type=int, default=None,
                help="cap prompts per community (cheap trial runs)",
            )
        stage_parser.set_defaults(handler=_run_single_stage, stage_name=stage.name)

    run_all = subparsers.add_parser("run-all", help="run all five stages in order")
    _add_config_arguments(run_all)
    run_all.set_defaults(handler=_run_all_stages)

    status = subparsers.add_parser("status", help="which stage artifacts exist for a run")
    _add_run_dir_argument(status)
    status.set_defaults(handler=lambda args: print_run_status(_paths_from_args(args)))

    validate = subparsers.add_parser("validate", help="cross-stage consistency checks")
    _add_run_dir_argument(validate)
    validate.set_defaults(handler=lambda args: validate_run(_paths_from_args(args)))

    inspect = subparsers.add_parser("inspect", help="summarize any pipeline artifact")
    inspect.add_argument("artifact", help="path to an artifact file or a run directory")
    inspect.set_defaults(handler=lambda args: inspect_path(Path(args.artifact)))

    trace = subparsers.add_parser("trace", help="summarize/filter the LLM call trace")
    _add_run_dir_argument(trace)
    trace.add_argument("--grep", default=None, help="only records containing this substring")
    trace.add_argument("--errors", action="store_true", help="only errored/refused calls")
    trace.set_defaults(
        handler=lambda args: print_trace_report(
            _paths_from_args(args), grep=args.grep, errors_only=args.errors
        )
    )

    estimate = subparsers.add_parser(
        "estimate-cost", help="pre-run LLM call/token/cost estimate"
    )
    _add_config_arguments(estimate)
    estimate.set_defaults(
        handler=lambda args: print_cost_estimate(_config_from_args(args))
    )

    calibrate = subparsers.add_parser(
        "calibrate-judge", help="judge-panel agreement, consistency, gold accuracy"
    )
    _add_config_arguments(calibrate)
    calibrate.add_argument(
        "--consistency-sample", type=int, default=20,
        help="responses to re-judge for self-consistency (0 disables)",
    )
    calibrate.add_argument(
        "--gold", default=None,
        help="gold labels JSON: {response_id: {axis_id: true/false}}",
    )
    calibrate.set_defaults(handler=_calibrate_judge)

    distinguish = subparsers.add_parser(
        "distinguish", help="input-side prompt distinguishability (distinguish/ bridge)"
    )
    _add_config_arguments(distinguish)
    distinguish.set_defaults(handler=_run_distinguish)

    helper_study = subparsers.add_parser(
        "helper-study",
        help="compare hypothesis-generation conditions (zero_context/literature/grounded/seeded/two_stage)",
    )
    _add_config_arguments(helper_study)
    helper_study.add_argument(
        "--conditions", nargs="+", default=None,
        help=f"conditions to run (default: {' '.join(STANDARD_CONDITIONS)})",
    )
    helper_study.set_defaults(handler=_helper_study)

    judge_study = subparsers.add_parser(
        "judge-study", help="score with many judges; kappa matrix + per-judge conclusions"
    )
    _add_config_arguments(judge_study)
    judge_study.add_argument(
        "--judges", nargs="+", required=True,
        help="judge model specs to compare (e.g. gpt-4o-mini claude-haiku-4-5 gemini-3.5-flash)",
    )
    judge_study.set_defaults(handler=_judge_study)

    compare_runs = subparsers.add_parser(
        "compare-runs", help="cross-group comparison table over completed runs"
    )
    compare_runs.add_argument("--runs", nargs="+", required=True, help="run directories")
    compare_runs.add_argument(
        "--out", default="out/comparisons", help="output directory for the report"
    )
    compare_runs.set_defaults(
        handler=lambda args: (
            run_cross_run_comparison([Path(r) for r in args.runs], Path(args.out)),
            0,
        )[1]
    )

    serve = subparsers.add_parser("serve", help="debug/visualization server + UI")
    serve.add_argument("--runs-root", default="out/runs", help="directory containing runs")
    serve.add_argument("--port", type=int, default=8321)
    serve.set_defaults(handler=_serve)
    return parser


def _add_config_arguments(stage_parser: argparse.ArgumentParser) -> None:
    stage_parser.add_argument("-c", "--config", required=True, help="experiment config JSON")
    stage_parser.add_argument(
        "--run-dir", default=None,
        help="run directory (default: out/runs/<run_name> from config)",
    )


def _add_run_dir_argument(stage_parser: argparse.ArgumentParser) -> None:
    stage_parser.add_argument("--run-dir", required=True, help="run directory")


def _config_from_args(args) -> ExperimentConfig:
    return ExperimentConfig.from_config_file(args.config)


def _paths_from_args(args) -> RunDirectoryPaths:
    return RunDirectoryPaths(args.run_dir)


def _resolve(args) -> tuple[ExperimentConfig, RunDirectoryPaths]:
    config = _config_from_args(args)
    paths = (
        RunDirectoryPaths(args.run_dir)
        if args.run_dir
        else RunDirectoryPaths.for_run_name(config.run_name)
    )
    # Snapshot the config into the run dir for provenance on every invocation
    save_json(config.to_dict(), paths.config_snapshot_path)
    return config, paths


def _run_single_stage(args) -> int:
    config, paths = _resolve(args)
    stage = STAGES_BY_NAME[args.stage_name]
    if stage.name == "responses" and getattr(args, "limit", None) is not None:
        stage.runner(config, paths, limit_prompts=args.limit)
    else:
        stage.runner(config, paths)
    return 0


def _run_all_stages(args) -> int:
    config, paths = _resolve(args)
    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        log_stage(index, len(PIPELINE_STAGES), f"{stage.title} ({stage.paper_section})")
        stage.runner(config, paths)
    log("\nPipeline complete.")
    log(f"  report:  {paths.analysis_report_path}")
    log(f"  summary: {paths.analysis_summary_path}")
    return 0


def _calibrate_judge(args) -> int:
    config, paths = _resolve(args)
    run_judge_calibration(
        config, paths,
        consistency_sample=args.consistency_sample,
        gold_labels_file=args.gold,
    )
    return 0


def _run_distinguish(args) -> int:
    config, paths = _resolve(args)
    run_prompt_distinguishability(config, paths)
    return 0


def _helper_study(args) -> int:
    """Full study: conditions -> union axes -> shared responses/scoring ->
    union analysis -> per-condition downstream comparison."""
    config, paths = _resolve(args)
    conditions = args.conditions or STANDARD_CONDITIONS
    if not paths.prompt_sets_path.exists():
        run_prompt_collection(config, paths)
    _archive_replaced_artifacts(paths)
    # Stages 2 and 3 are parallel DAG branches (both need only stage 1);
    # responses run first so behavior-grounded generation can observe them.
    run_response_collection(config, paths)
    report, union = run_helper_conditions(config, paths, conditions)
    run_response_scoring(config, paths)
    analysis = run_treatment_analysis(config, paths)
    report.downstream = summarize_downstream(report, union, analysis.axes)
    save_json(report.to_dict(), paths.helper_study_path)
    log("\nHelper-study downstream (per condition over shared responses):")
    for entry in report.downstream:
        log(
            f"  {entry.condition:<13} axes={entry.n_axes:<3} significant={entry.n_significant} "
            f"info={entry.total_info_bits:.3f} bits  mean|Δ|={entry.mean_abs_delta:.3f}"
        )
    log(f"  full report: {paths.helper_study_path}")
    return 0


def _archive_replaced_artifacts(paths: RunDirectoryPaths) -> None:
    """helper-study replaces the hypothesis set and downstream artifacts;
    snapshot what exists first so prior results are never silently destroyed."""
    to_archive = [
        paths.hypothesis_set_path,
        paths.scored_responses_path,
        paths.scoring_manifest_path,
        paths.analysis_report_path,
        paths.analysis_summary_path,
    ]
    existing = [p for p in to_archive if p.exists()]
    if not existing:
        return
    archive_dir = paths.run_dir / "archive" / time.strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, archive_dir / path.name)
    log(f"  archived {len(existing)} prior artifacts to {archive_dir}")


def _judge_study(args) -> int:
    config, paths = _resolve(args)
    run_judge_study(config, paths, args.judges)
    return 0


def _serve(args) -> int:
    serve_debug_ui(Path(args.runs_root), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
