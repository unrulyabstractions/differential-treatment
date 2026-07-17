"""`dtreat trace` — summarize and filter the run's LLM call trace."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from dtreat.common.console_logging import log, log_header
from dtreat.common.file_io import load_jsonl
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths


def print_trace_report(
    paths: RunDirectoryPaths,
    grep: str | None = None,
    errors_only: bool = False,
) -> int:
    """Aggregate stats per role/model, then matching records if filtered."""
    if not paths.llm_trace_path.exists():
        log(f"No trace at {paths.llm_trace_path}")
        return 1
    records = load_jsonl(paths.llm_trace_path)
    log_header(f"LLM trace: {len(records)} calls")

    per_role: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0, "cached": 0, "errors": 0, "refused": 0,
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "latency": [],
        }
    )
    for record in records:
        stats = per_role[f"{record['role_label']} ({record['model']})"]
        stats["calls"] += 1
        stats["cached"] += int(record.get("cached", False))
        stats["errors"] += int(bool(record.get("error")))
        stats["refused"] += int(record.get("refused", False))
        stats["input_tokens"] += record.get("input_tokens", 0)
        stats["output_tokens"] += record.get("output_tokens", 0)
        stats["cost"] += record.get("cost_usd", 0.0)
        stats["latency"].append(record.get("latency_ms", 0))

    for role, stats in sorted(per_role.items()):
        latencies = sorted(stats["latency"])
        p50 = latencies[len(latencies) // 2] if latencies else 0
        log(
            f"  {role}: {stats['calls']} calls ({stats['cached']} cached, "
            f"{stats['errors']} errors, {stats['refused']} refused), "
            f"{stats['input_tokens']}→{stats['output_tokens']} tokens, "
            f"${stats['cost']:.4f}, p50 latency {p50}ms"
        )

    finish_reasons = Counter(record.get("finish_reason", "?") for record in records)
    log(f"  finish reasons: {dict(finish_reasons)}")

    matching = records
    if errors_only:
        matching = [r for r in matching if r.get("error") or r.get("refused")]
    if grep:
        matching = [r for r in matching if grep.lower() in json.dumps(r).lower()]
    if errors_only or grep:
        log(f"\n{len(matching)} matching records:")
        for record in matching[:50]:
            log(f"  {json.dumps(record, ensure_ascii=False)[:220]}")
        if len(matching) > 50:
            log(f"  ... and {len(matching) - 50} more")
    return 0
