"""Data assembly behind the debug server's REST routes.

Pure functions from a runs root + run name to JSON-ready dicts; the FastAPI
app in debug_server_app.py stays a thin routing layer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from dtreat.common.file_io import load_json, load_jsonl
from dtreat.diagnostics.run_validation import collect_validation
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths
from dtreat.stages.response_scoring.scored_response_schemas import ScoredResponse
from dtreat.stages.treatment_analysis.permutation_significance import (
    permutation_p_values,
)
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    build_cluster_matrices,
    build_permutation_clusters,
)


def paths_for(runs_root: Path, run_name: str) -> RunDirectoryPaths:
    run_dir = (runs_root / run_name).resolve()
    if not str(run_dir).startswith(str(runs_root.resolve())):
        raise ValueError("run name escapes runs root")
    return RunDirectoryPaths(run_dir)


def list_runs(runs_root: Path) -> list[dict]:
    """All runs with per-stage artifact existence."""
    runs = []
    if not runs_root.exists():
        return runs
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        paths = RunDirectoryPaths(run_dir)
        stages = {
            name: path.exists() for name, path in paths.stage_artifact_paths().items()
        }
        runs.append(
            {
                "run_name": run_dir.name,
                "stages": stages,
                "has_trace": paths.llm_trace_path.exists(),
                "modified": max(
                    (p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()),
                    default=0,
                ),
            }
        )
    return runs


def run_overview(paths: RunDirectoryPaths) -> dict:
    problems, warnings = collect_validation(paths)
    overview: dict = {
        "config": load_json(paths.config_snapshot_path, default={}),
        "stages": {n: p.exists() for n, p in paths.stage_artifact_paths().items()},
        "validation": {"problems": problems, "warnings": warnings},
    }
    if paths.analysis_report_path.exists():
        report = load_json(paths.analysis_report_path)
        overview["headline"] = {
            "significant_axes": sum(a["significant"] for a in report["axes"]),
            "total_axes": len(report["axes"]),
            "d_pi_bits": report.get("d_pi_bits_significant_axes"),
            "c2st_accuracy": report["c2st"]["accuracy"] if report.get("c2st") else None,
        }
    return overview


def stage1_data(paths: RunDirectoryPaths) -> dict:
    return load_json(paths.prompt_sets_path)


def stage2_data(paths: RunDirectoryPaths) -> dict:
    return load_json(paths.hypothesis_set_path)


def stage3_data(
    paths: RunDirectoryPaths, community: str | None, search: str | None, limit: int, offset: int
) -> dict:
    records = load_jsonl(paths.responses_path)
    if community:
        records = [r for r in records if r["community"] == community]
    if search:
        needle = search.lower()
        records = [r for r in records if needle in r["text"].lower()]
    lengths_by_community: dict[str, list[int]] = {}
    for record in load_jsonl(paths.responses_path):
        lengths_by_community.setdefault(record["community"], []).append(
            len(record.get("text", ""))
        )
    manifest = load_json(paths.collection_manifest_path, default={})
    return {
        "manifest": manifest,
        "total_matching": len(records),
        "records": records[offset : offset + limit],
        "lengths_by_community": lengths_by_community,
    }


def stage4_data(paths: RunDirectoryPaths, limit: int, offset: int) -> dict:
    records = load_jsonl(paths.scored_responses_path)
    manifest = load_json(paths.scoring_manifest_path, default={})
    axis_ids = manifest.get("axis_ids") or sorted(
        {axis for record in records for axis in record.get("verdicts", {})}
    )
    per_axis_rates: dict[str, dict[str, float]] = {}
    for axis_id in axis_ids:
        by_community: dict[str, list[bool]] = {}
        for record in records:
            if axis_id in record.get("verdicts", {}):
                by_community.setdefault(record["community"], []).append(
                    record["verdicts"][axis_id]
                )
        per_axis_rates[axis_id] = {
            community: sum(verdicts) / len(verdicts)
            for community, verdicts in by_community.items()
        }
    # prompt × axis heatmap rates
    by_prompt: dict[str, list[dict]] = {}
    for record in records:
        by_prompt.setdefault(record["prompt_id"], []).append(record)
    heatmap = []
    for prompt_id, prompt_records in sorted(by_prompt.items()):
        row = {"prompt_id": prompt_id, "community": prompt_records[0]["community"]}
        for axis_id in axis_ids:
            verdicts = [
                r["verdicts"][axis_id] for r in prompt_records if axis_id in r.get("verdicts", {})
            ]
            row[axis_id] = sum(verdicts) / len(verdicts) if verdicts else None
        heatmap.append(row)
    return {
        "manifest": manifest,
        "axis_ids": axis_ids,
        "per_axis_rates": per_axis_rates,
        "heatmap": heatmap,
        "total": len(records),
        "records": records[offset : offset + limit],
    }


def stage5_data(paths: RunDirectoryPaths) -> dict:
    report = load_json(paths.analysis_report_path)
    summary_md = (
        paths.analysis_summary_path.read_text()
        if paths.analysis_summary_path.exists()
        else ""
    )
    return {"report": report, "summary_markdown": summary_md}


def permutation_null_for_axis(
    paths: RunDirectoryPaths, axis_id: str, n_permutations: int = 500
) -> dict:
    """Recompute the permutation null for one axis (debug visualization)."""
    config = load_json(paths.config_snapshot_path, default={})
    scored = [ScoredResponse.from_dict(r) for r in load_jsonl(paths.scored_responses_path)]
    hypothesis_set = load_json(paths.hypothesis_set_path)
    axis_ids = [axis["axis_id"] for axis in hypothesis_set["axes"]]
    if axis_id not in axis_ids:
        raise ValueError(f"unknown axis {axis_id}")
    target_name = config.get("target_community", {}).get("name") or next(
        (s.community for s in scored), ""
    )
    clusters = build_permutation_clusters(scored, axis_ids, config.get("permutation_unit", "prompt"))
    sums, counts, is_target = build_cluster_matrices(clusters, axis_ids, target_name)
    axis_index = axis_ids.index(axis_id)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    observed_p, observed_delta = permutation_p_values(
        sums[axis_index : axis_index + 1],
        counts[axis_index : axis_index + 1],
        is_target,
        n_permutations,
        int(config.get("seed", 0)),
    )
    null_deltas = []
    for _ in range(n_permutations):
        permuted = rng.permutation(is_target)
        t_count = counts[axis_index, permuted].sum()
        b_count = counts[axis_index, ~permuted].sum()
        if t_count == 0 or b_count == 0:
            continue
        null_deltas.append(
            float(
                sums[axis_index, permuted].sum() / t_count
                - sums[axis_index, ~permuted].sum() / b_count
            )
        )
    return {
        "axis_id": axis_id,
        "observed_delta": float(observed_delta[0]),
        "p_value": float(observed_p[0]),
        "null_deltas": null_deltas,
    }


def trace_data(
    paths: RunDirectoryPaths, grep: str | None, errors_only: bool, limit: int
) -> dict:
    if not paths.llm_trace_path.exists():
        return {"aggregates": {}, "records": [], "total": 0}
    records = load_jsonl(paths.llm_trace_path)
    aggregates: dict[str, dict] = {}
    for record in records:
        key = f"{record['role_label']} ({record['model']})"
        stats = aggregates.setdefault(
            key,
            {"calls": 0, "cached": 0, "errors": 0, "refused": 0, "cost_usd": 0.0,
             "input_tokens": 0, "output_tokens": 0},
        )
        stats["calls"] += 1
        stats["cached"] += int(record.get("cached", False))
        stats["errors"] += int(bool(record.get("error")))
        stats["refused"] += int(record.get("refused", False))
        stats["cost_usd"] += record.get("cost_usd", 0.0)
        stats["input_tokens"] += record.get("input_tokens", 0)
        stats["output_tokens"] += record.get("output_tokens", 0)
    finish_reasons = dict(Counter(r.get("finish_reason", "?") for r in records))
    matching = records
    if errors_only:
        matching = [r for r in matching if r.get("error") or r.get("refused")]
    if grep:
        needle = grep.lower()
        matching = [r for r in matching if needle in str(r).lower()]
    return {
        "aggregates": aggregates,
        "finish_reasons": finish_reasons,
        "total": len(matching),
        "records": matching[-limit:],
    }
