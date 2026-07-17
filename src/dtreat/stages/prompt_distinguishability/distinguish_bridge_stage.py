"""Bridge to the vendored prompt-distinguishability pipeline (`distinguish/`).

Exports the stage-1 prompt sets into the distinguish dataset format (parquet
tables + manifest), runs its full analysis as a subprocess in its own uv
project, and parses the summary back into an InputDistinguishabilityReport —
the input-side counterpart to stage 5's output-side measurements.

Mapping notes: each prompt gets its own author (we have no author identities),
markedness = 0 (no explicit disclosure, by construction of the method),
codedness = 0.5 (unknown), and our instruction_id becomes the `domain` content
stratum so the distinguish pipeline's conditional analysis holds the ask fixed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import load_json, save_json
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths
from dtreat.stages.prompt_collection.prompt_set_schemas import PromptStageArtifact

from .distinguish_report_schemas import (
    InputDimensionVerdict,
    InputDistinguishabilityReport,
)

DISTINGUISH_ROOT = Path(__file__).parents[4] / "distinguish"


def export_distinguish_dataset(
    artifact: PromptStageArtifact, dataset_dir: Path
) -> Path:
    """Write our prompt sets as a distinguish-format dataset directory."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # topic_id is an integer taxonomy index in the distinguish schema; give
    # each instruction id a stable index. Usage-context ints use their
    # missing sentinel 0 (we have no survey context for these prompts).
    all_instruction_ids = sorted(
        {
            prompt.instruction_id
            for prompt in artifact.target_set.prompts + artifact.baseline_set.prompts
        }
    )
    topic_index = {instruction_id: i + 1 for i, instruction_id in enumerate(all_instruction_ids)}

    prompt_rows = []
    author_rows = []
    for cohort, prompt_set, lgbtq in (
        ("target", artifact.target_set, 1),
        ("baseline", artifact.baseline_set, 0),
    ):
        for prompt in prompt_set.prompts:
            author_id = f"author_{prompt.prompt_id}"
            prompt_rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "author_id": author_id,
                    "cohort": cohort,
                    "text": prompt.text,
                    "lgbtq": lgbtq,
                    "markedness": 0,
                    "codedness": 0.5,
                    "topic_id": topic_index[prompt.instruction_id],
                    "domain": prompt.instruction_id,
                    "provenance": "differential-treatment stage 1",
                    "adoption": 0,
                    "general_freq": 0,
                    "llm_freq": 0,
                    "professional_freq": 0,
                    "aversion": 0,
                    "satisfaction": 0,
                }
            )
            author_rows.append(
                {
                    "author_id": author_id,
                    "cohort": cohort,
                    "transgender": "",
                    "gender": "",
                    "orientation": "",
                    "pronouns": "",
                    "race": "",
                    "age": "",
                    "disability": "",
                    "education": "",
                    "income": "",
                }
            )

    pd.DataFrame(prompt_rows).to_parquet(dataset_dir / "prompts.parquet")
    pd.DataFrame(author_rows).to_parquet(dataset_dir / "authors.parquet")
    manifest = {
        "name": "dtreat_bridge",
        "description": (
            "Prompt sets exported from a differential-treatment run for "
            "input-side distinguishability analysis. Stays local."
        ),
        "cohorts": [
            {
                "name": "target",
                "group": "target",
                "display_name": f"Target ({artifact.target_set.community})",
                "description": "Target community prompts from stage 1.",
            },
            {
                "name": "baseline",
                "group": "baseline",
                "display_name": f"Baseline ({artifact.baseline_set.community})",
                "description": "Baseline community prompts from stage 1.",
            },
        ],
        "comparisons": [
            {
                "name": "target_vs_baseline",
                "target_cohort": "target",
                "baseline_cohort": "baseline",
                "expectation": "",
                "explorations": False,
            }
        ],
    }
    save_json(manifest, dataset_dir / "dataset.json", readable_text=False)
    return dataset_dir


def run_distinguish_pipeline(
    dataset_dir: Path, out_root: Path, seed: int, timeout_s: int = 3600
) -> Path:
    """Run the vendored pipeline in its own uv project; returns the run dir."""
    if not DISTINGUISH_ROOT.exists():
        raise FileNotFoundError(
            f"Vendored distinguish/ project not found at {DISTINGUISH_ROOT}"
        )
    command = [
        "uv", "run", "python", "scripts/run_dataset_pipeline.py",
        "--dataset", str(dataset_dir.resolve()),
        "--run-name", "dtreat_bridge",
        "--seed", str(seed),
        "--out-root", str(out_root.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=DISTINGUISH_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    run_dir = out_root / "dtreat_bridge"
    if completed.returncode != 0 or not (run_dir / "summary.json").exists():
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-25:])
        raise RuntimeError(
            f"distinguish pipeline failed (rc={completed.returncode}). Last output:\n{tail}"
        )
    return run_dir


def parse_distinguish_summary(run_dir: Path) -> InputDistinguishabilityReport:
    """DatasetRunSummary json -> our InputDistinguishabilityReport."""
    summary = load_json(run_dir / "summary.json")
    comparison = summary["comparisons"][0]
    verdicts = [
        InputDimensionVerdict(
            dimension=v["dimension"],
            variant=v["variant"],
            statistic_name=v["statistic_name"],
            statistic_value=float(v["statistic_value"]),
            p_value=v.get("p_value"),
            significant=v.get("significant"),
        )
        for v in comparison["verdicts"]
    ]
    best_acc, best_variant = None, ""
    for verdict in verdicts:
        is_accuracy = (
            verdict.dimension == "distributional"
            and "acc" in verdict.statistic_name.lower()
        )
        if is_accuracy and (best_acc is None or verdict.statistic_value > best_acc):
            best_acc, best_variant = verdict.statistic_value, verdict.variant
    return InputDistinguishabilityReport(
        run_dir=str(run_dir),
        n_tests=comparison["n_tests"],
        n_significant=comparison["n_significant"],
        overall_distinguishable=bool(comparison["overall_distinguishable"]),
        best_c2st_accuracy=best_acc,
        best_c2st_variant=best_variant,
        verdicts=verdicts,
        skipped_variants=list(summary.get("skipped_variants", [])),
    )


def run_prompt_distinguishability(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> InputDistinguishabilityReport:
    """Execute the bridge and write `stage1_prompts/input_distinguishability.json`."""
    log("Stage 1b: input-side prompt distinguishability (distinguish/ bridge)")
    artifact = PromptStageArtifact.from_json(paths.prompt_sets_path)
    dataset_dir = export_distinguish_dataset(
        artifact, paths.run_dir / "stage1_prompts" / "distinguish_dataset"
    )
    run_dir = run_distinguish_pipeline(
        dataset_dir, paths.run_dir / "stage1_prompts" / "distinguish_runs", config.seed
    )
    report = parse_distinguish_summary(run_dir)
    save_json(report.to_dict(), paths.input_distinguishability_path)
    log_kv(
        {
            "tests significant": f"{report.n_significant}/{report.n_tests}",
            "best input C2ST": report.best_c2st_accuracy,
            "skipped variants": len(report.skipped_variants),
        }
    )
    log(f"  wrote {paths.input_distinguishability_path}")
    return report


def load_input_report_if_present(
    paths: RunDirectoryPaths,
) -> InputDistinguishabilityReport | None:
    if not paths.input_distinguishability_path.exists():
        return None
    return InputDistinguishabilityReport.from_json(paths.input_distinguishability_path)
