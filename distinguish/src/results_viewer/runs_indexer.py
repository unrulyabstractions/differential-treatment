"""Index of runs/ for the results viewer: datasets, headline stats, file lists.

The index stays lean on purpose — headline numbers come from each run's
summary.json, while the UI fetches full summaries and plots directly through the
server's /runs/ route. Files are run-dir-relative paths so the client can group
them by comparison/section/subgroup without any server-side tree schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.base_schema import BaseSchema
from src.common.file_io import load_json

_SERVED_SUFFIXES = {".png", ".json"}
# Sidebar order: fixture first, then validation sets easy->hard, then the dial.
_PREFERRED_ORDER = [
    "synthetic",
    "twitteraae",
    "reddit_l2",
    "pan17_variety",
    "blog_authorship",
    "prism",
    "thoughttrace",
    "wildchat",
    "multi_value_aave",
]


@dataclass
class ComparisonHeadline(BaseSchema):
    """One comparison's headline numbers for the sidebar/overview cards."""

    name: str
    expectation: str
    n_tests: int
    n_significant: int
    overall_distinguishable: bool


@dataclass
class DatasetRunEntry(BaseSchema):
    """One runs/{dataset} directory as the viewer sees it."""

    name: str
    created_at: str
    dimensions_run: list[str] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)
    comparisons: list[ComparisonHeadline] = field(default_factory=list)
    files: list[str] = field(default_factory=list)  # run-dir-relative, sorted


@dataclass
class RunsIndex(BaseSchema):
    """Everything the viewer needs to render the sidebar and route requests."""

    runs_root: str
    datasets: list[DatasetRunEntry] = field(default_factory=list)


def _sidebar_order(run_dir: Path) -> tuple[int, str]:
    try:
        return (_PREFERRED_ORDER.index(run_dir.name), run_dir.name)
    except ValueError:
        return (len(_PREFERRED_ORDER), run_dir.name)


def build_runs_index(runs_root: Path) -> RunsIndex:
    """Scan `runs_root` for completed run dirs (those with a summary.json)."""
    index = RunsIndex(runs_root=str(runs_root))
    if not runs_root.is_dir():
        return index
    run_dirs = [d for d in runs_root.iterdir() if (d / "summary.json").is_file()]
    for run_dir in sorted(run_dirs, key=_sidebar_order):
        summary = load_json(run_dir / "summary.json")
        files = sorted(
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix in _SERVED_SUFFIXES
        )
        index.datasets.append(
            DatasetRunEntry(
                name=run_dir.name,
                created_at=str(summary.get("created_at", "")),
                dimensions_run=list(summary.get("dimensions_run", [])),
                skipped_variants=sorted(set(summary.get("skipped_variants", []))),
                comparisons=[
                    ComparisonHeadline(
                        name=str(comparison.get("name", "")),
                        expectation=str(comparison.get("expectation", "")),
                        n_tests=int(comparison.get("n_tests", 0)),
                        n_significant=int(comparison.get("n_significant", 0)),
                        overall_distinguishable=bool(
                            comparison.get("overall_distinguishable", False)
                        ),
                    )
                    for comparison in summary.get("comparisons", [])
                ],
                files=files,
            )
        )
    return index
