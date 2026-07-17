"""`dtreat` CLI: run any pipeline stage separately, or everything, plus
diagnostics (status / inspect / validate / trace / estimate-cost) and the
debug server (`dtreat serve`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dtreat.common.console_logging import log, log_stage
from dtreat.common.file_io import save_json
from dtreat.diagnostics.artifact_inspection import inspect_path
from dtreat.diagnostics.cost_estimation import print_cost_estimate
from dtreat.diagnostics.llm_trace_reporting import print_trace_report
from dtreat.diagnostics.run_validation import print_run_status, validate_run
from dtreat.server.debug_server_app import serve_debug_ui

from .experiment_config import ExperimentConfig
from .run_directory_paths import RunDirectoryPaths
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


def _serve(args) -> int:
    serve_debug_ui(Path(args.runs_root), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
