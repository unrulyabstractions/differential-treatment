"""`dtreat validate` / `dtreat status` — cross-stage consistency checks.

Catches the silent-corruption failure modes: responses referencing unknown
prompts, scores referencing unknown responses or axes, count shortfalls,
quarantined failures, and unparsed verdicts.
"""

from __future__ import annotations

from dtreat.common.console_logging import log, log_header
from dtreat.common.file_io import load_json, load_jsonl
from dtreat.common.run_directory_paths import RunDirectoryPaths


def print_run_status(paths: RunDirectoryPaths) -> int:
    """Which stage artifacts exist, at a glance."""
    log_header(f"Run status: {paths.run_dir}")
    for stage_name, artifact_path in paths.stage_artifact_paths().items():
        marker = "✓" if artifact_path.exists() else "·"
        log(f"  {marker} {stage_name:<11} {artifact_path.relative_to(paths.run_dir)}")
    if paths.llm_trace_path.exists():
        log(f"  ✓ {'trace':<11} {paths.llm_trace_path.relative_to(paths.run_dir)}")
    quarantine_dir = paths.run_dir / "quarantine"
    if quarantine_dir.exists():
        for quarantine_file in sorted(quarantine_dir.glob("*.jsonl")):
            count = len(load_jsonl(quarantine_file))
            log(f"  ! quarantine  {quarantine_file.name}: {count} failures")
    return 0


def collect_validation(paths: RunDirectoryPaths) -> tuple[list[str], list[str]]:
    """Run all cross-stage checks, returning (problems, warnings) as data
    (used by both the CLI and the debug server)."""
    problems: list[str] = []
    warnings: list[str] = []
    prompts = _validate_prompts(paths, problems)
    axis_ids = _validate_hypotheses(paths, problems)
    responses = _validate_responses(paths, prompts, problems, warnings)
    _validate_scores(paths, responses, axis_ids, problems, warnings)
    _check_quarantine(paths, warnings)
    return problems, warnings


def validate_run(paths: RunDirectoryPaths) -> int:
    """Full cross-stage consistency check; exit code 1 if problems found."""
    log_header(f"Validating run: {paths.run_dir}")
    problems, warnings = collect_validation(paths)

    for warning in warnings:
        log(f"  [warn] {warning}")
    for problem in problems:
        log(f"  [PROBLEM] {problem}")
    if not problems and not warnings:
        log("  all checks passed")
    log(f"\n{len(problems)} problems, {len(warnings)} warnings")
    return 1 if problems else 0


def _validate_prompts(paths, problems) -> dict[str, dict]:
    if not paths.prompt_sets_path.exists():
        return {}
    data = load_json(paths.prompt_sets_path)
    prompts = {}
    for side in ("target_set", "baseline_set"):
        for prompt in data[side]["prompts"]:
            if prompt["prompt_id"] in prompts:
                problems.append(f"duplicate prompt_id across sets: {prompt['prompt_id']}")
            prompts[prompt["prompt_id"]] = {**prompt, "community": data[side]["community"]}
    if not data["comparability"]["passed"]:
        problems.append("stage 1 comparability check FAILED in stored artifact")
    return prompts


def _validate_hypotheses(paths, problems) -> list[str]:
    if not paths.hypothesis_set_path.exists():
        return []
    data = load_json(paths.hypothesis_set_path)
    axis_ids = [axis["axis_id"] for axis in data["axes"]]
    if len(axis_ids) != len(set(axis_ids)):
        problems.append("duplicate axis ids in hypothesis set")
    if not axis_ids:
        problems.append("hypothesis set has no axes")
    return axis_ids


def _validate_responses(paths, prompts, problems, warnings) -> dict[str, dict]:
    if not paths.responses_path.exists():
        return {}
    records = {record["response_id"]: record for record in load_jsonl(paths.responses_path)}
    for record in records.values():
        if prompts and record["prompt_id"] not in prompts:
            problems.append(
                f"response {record['response_id']} references unknown prompt "
                f"{record['prompt_id']}"
            )
        elif prompts and record["community"] != prompts[record["prompt_id"]]["community"]:
            problems.append(
                f"response {record['response_id']} community mismatch vs prompt set"
            )
    if paths.collection_manifest_path.exists():
        manifest = load_json(paths.collection_manifest_path)
        if manifest["collected_responses"] < manifest["expected_responses"]:
            warnings.append(
                f"stage 3 shortfall: {manifest['collected_responses']}/"
                f"{manifest['expected_responses']} responses collected"
            )
    return records


def _validate_scores(paths, responses, axis_ids, problems, warnings) -> None:
    if not paths.scored_responses_path.exists():
        return
    scored = load_jsonl(paths.scored_responses_path)
    unparsed_total = 0
    for record in scored:
        if responses and record["response_id"] not in responses:
            problems.append(
                f"scored record references unknown response {record['response_id']}"
            )
        for axis_id in record.get("verdicts", {}):
            if axis_ids and axis_id not in axis_ids:
                problems.append(
                    f"scored record {record['response_id']} has verdict for unknown "
                    f"axis {axis_id}"
                )
        unparsed_total += len(record.get("unparsed_axes", []))
    judgeable = [
        r for r in responses.values() if not r.get("refused") and r.get("text", "").strip()
    ]
    if responses and len(scored) < len(judgeable):
        warnings.append(f"stage 4 shortfall: {len(scored)}/{len(judgeable)} judged")
    if unparsed_total:
        warnings.append(f"{unparsed_total} unparsed judge verdicts (excluded from stats)")


def _check_quarantine(paths, warnings) -> None:
    quarantine_dir = paths.run_dir / "quarantine"
    if quarantine_dir.exists():
        for quarantine_file in sorted(quarantine_dir.glob("*.jsonl")):
            count = len(load_jsonl(quarantine_file))
            if count:
                warnings.append(f"{quarantine_file.name}: {count} quarantined failures")
