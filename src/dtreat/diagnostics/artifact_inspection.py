"""`dtreat inspect` — quick human summaries of any pipeline artifact.

Detects the artifact type by filename and prints what a debugging human wants
first: counts, distributions, and a few concrete examples.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dtreat.common.console_logging import log, log_header
from dtreat.common.file_io import load_json, load_jsonl


def inspect_path(path: Path) -> int:
    """Inspect one artifact file, or every known artifact in a run directory."""
    if path.is_dir():
        found = False
        for candidate in sorted(path.rglob("*.json*")):
            if candidate.name in _INSPECTORS or candidate.suffix == ".jsonl":
                inspect_path(candidate)
                found = True
        if not found:
            log(f"No known artifacts under {path}")
        return 0

    inspector = _INSPECTORS.get(path.name, _inspect_generic)
    log_header(str(path))
    inspector(path)
    return 0


def _inspect_prompt_sets(path: Path) -> None:
    data = load_json(path)
    for side in ("target_set", "baseline_set"):
        prompts = data[side]["prompts"]
        instructions = Counter(p["instruction_id"] for p in prompts)
        log(f"{side}: community={data[side]['community']!r}, {len(prompts)} prompts")
        log(f"  instructions: {dict(instructions)}")
        if prompts:
            log(f"  example: {prompts[0]['text'][:110]!r}")
    comparability = data["comparability"]
    log(
        f"comparability: TV={comparability['total_variation_distance']:.3f} "
        f"(max {comparability['max_allowed_tv_distance']}), "
        f"chi2 p={comparability['chi2_p_value']:.3f}, "
        f"passed={comparability['passed']}"
    )


def _inspect_hypotheses(path: Path) -> None:
    data = load_json(path)
    log(f"helper model: {data['helper_model']}")
    for axis in data["axes"]:
        log(f"  [{axis['source']}] {axis['axis_id']}: {axis['question']}")


def _inspect_responses(path: Path) -> None:
    records = load_jsonl(path)
    by_community = Counter(r["community"] for r in records)
    refusals = sum(r.get("refused", False) for r in records)
    lengths = [len(r["text"]) for r in records if r.get("text")]
    log(f"{len(records)} responses; by community: {dict(by_community)}; refusals: {refusals}")
    if lengths:
        log(
            f"text length: min {min(lengths)}, "
            f"mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}"
        )
    if records:
        log(f"example ({records[0]['response_id']}): {records[0]['text'][:110]!r}")


def _inspect_scored(path: Path) -> None:
    records = load_jsonl(path)
    log(f"{len(records)} scored responses")
    if not records:
        return
    axis_ids = sorted({axis for r in records for axis in r.get("verdicts", {})})
    for axis_id in axis_ids:
        by_community: dict[str, list[bool]] = {}
        for record in records:
            if axis_id in record.get("verdicts", {}):
                by_community.setdefault(record["community"], []).append(
                    record["verdicts"][axis_id]
                )
        rates = ", ".join(
            f"{community}: {sum(v) / len(v):.2f} (n={len(v)})"
            for community, v in sorted(by_community.items())
        )
        log(f"  {axis_id}: {rates}")
    unparsed = sum(len(r.get("unparsed_axes", [])) for r in records)
    if unparsed:
        log(f"  [Warning] {unparsed} unparsed verdicts across records")


def _inspect_analysis(path: Path) -> None:
    data = load_json(path)
    significant = [a for a in data["axes"] if a["significant"]]
    log(
        f"{len(significant)}/{len(data['axes'])} axes significant; "
        f"D_pi(sig)={data.get('d_pi_bits_significant_axes')}; "
        f"C2ST={data['c2st']['accuracy'] if data.get('c2st') else 'n/a'}"
    )
    for axis in data["axes"]:
        flag = "*" if axis["significant"] else " "
        log(
            f" {flag} {axis['axis_id']}: Δ={axis['delta']:+.2f} "
            f"p={axis['p_value']:.3f} q={axis['q_value']:.3f} I={axis['info_bits']:.2f}"
        )


def _inspect_generic(path: Path) -> None:
    if path.suffix == ".jsonl":
        records = load_jsonl(path)
        log(f"JSONL with {len(records)} records")
        if records:
            log(f"first record keys: {sorted(records[0].keys())}")
    else:
        data = load_json(path)
        keys = sorted(data.keys()) if isinstance(data, dict) else f"list[{len(data)}]"
        log(f"JSON; top-level: {keys}")


_INSPECTORS = {
    "prompt_sets.json": _inspect_prompt_sets,
    "hypothesis_set.json": _inspect_hypotheses,
    "responses.jsonl": _inspect_responses,
    "scored_responses.jsonl": _inspect_scored,
    "analysis_report.json": _inspect_analysis,
}
