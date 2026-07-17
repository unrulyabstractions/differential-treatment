"""Run the COMPLETE distinguishability analysis for one dataset.

Usage:
    uv run python scripts/run_dataset_pipeline.py --dataset data/synthetic \\
        [--config my_patch.json ...] [--run-name NAME] [--seed 0] [--out-root runs]

Runs every comparison in the dataset's manifest with every configured variant
(all embedders, classifiers, and assignment backends — API-backed variants are
skipped and reported when their key is missing) plus the codedness sweep and
identity-slice explorations. Output: runs/{dataset}/{comparison}/{section}/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config_loading import load_pipeline_config  # noqa: E402
from src.pipeline.dataset_runner import run_dataset  # noqa: E402
from src.pipeline.run_summary import DatasetRunSummary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="config patch file(s), composed in order (see configs/config.json)",
    )
    parser.add_argument(
        "--run-name", default="", help="run dir name (default: dataset name)"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def print_summary(summary: DatasetRunSummary) -> None:
    for comparison in summary.comparisons:
        print(
            f"\n=== {comparison.name} "
            f"({comparison.target.display_name} vs {comparison.baseline.display_name}; "
            f"expected {comparison.expectation or 'n/a'}) ==="
        )
        header = f"{'section':<15} {'variant':<38} {'statistic':<30} {'value':>8} {'p':>10}  sig"
        print(header)
        print("-" * len(header))
        for v in comparison.verdicts:
            p_text = "-" if v.p_value is None else f"{v.p_value:.4g}"
            sig = "-" if v.significant is None else ("YES" if v.significant else "no")
            print(
                f"{v.dimension:<15} {v.variant:<38.38} {v.statistic_name:<30.30} "
                f"{v.statistic_value:>8.3g} {p_text:>10}  {sig}"
            )
        print(f"{comparison.n_significant}/{comparison.n_tests} tests significant")
    if summary.skipped_variants:
        print(f"\nSkipped variants: {', '.join(sorted(set(summary.skipped_variants)))}")


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if args.run_name:
        config.run_name = args.run_name
    if args.seed is not None:
        config.random_seed = args.seed
    run_dir = run_dataset(args.dataset, config, args.out_root)
    print_summary(DatasetRunSummary.from_json(run_dir / "summary.json"))
    print(f"\nArtifacts in: {run_dir}")


if __name__ == "__main__":
    main()
