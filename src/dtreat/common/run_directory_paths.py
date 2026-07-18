"""Canonical layout of a run directory.

Every stage reads its inputs and writes its outputs through this class, so
artifacts live in predictable places and any stage can be run separately
against hand-authored upstream files.

    out/runs/<run_name>/
        experiment_config.json     — config snapshot for provenance
        llm_cache/                 — response cache (stage resumability)
        llm_trace.jsonl            — every LLM call across all stages
        stage1_prompts/prompt_sets.json
        stage2_hypotheses/hypothesis_set.json
        stage3_responses/responses.jsonl, collection_manifest.json
        stage4_scores/scored_responses.jsonl, scoring_manifest.json
        stage5_analysis/analysis_report.json, analysis_summary.md
        quarantine/<stage>_failures.jsonl
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_RUNS_ROOT = Path("out") / "runs"


class RunDirectoryPaths:
    """Path bookkeeping for one run (pure derivation, no I/O beyond mkdir)."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)

    @classmethod
    def for_run_name(cls, run_name: str, runs_root: Path | str = DEFAULT_RUNS_ROOT):
        return cls(Path(runs_root) / run_name)

    # ── run-wide ─────────────────────────────────────────────────────────

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "experiment_config.json"

    @property
    def llm_cache_dir(self) -> Path:
        return self.run_dir / "llm_cache"

    @property
    def llm_trace_path(self) -> Path:
        return self.run_dir / "llm_trace.jsonl"

    def quarantine_path(self, stage_name: str) -> Path:
        return self.run_dir / "quarantine" / f"{stage_name}_failures.jsonl"

    # ── stage 1: prompt collection ───────────────────────────────────────

    @property
    def prompt_sets_path(self) -> Path:
        return self.run_dir / "stage1_prompts" / "prompt_sets.json"

    @property
    def input_distinguishability_path(self) -> Path:
        return self.run_dir / "stage1_prompts" / "input_distinguishability.json"

    # ── stage 2: hypothesis generation ───────────────────────────────────

    @property
    def hypothesis_set_path(self) -> Path:
        return self.run_dir / "stage2_hypotheses" / "hypothesis_set.json"

    @property
    def helper_study_path(self) -> Path:
        return self.run_dir / "stage2_hypotheses" / "helper_study.json"

    def helper_condition_path(self, condition_name: str) -> Path:
        return self.run_dir / "stage2_hypotheses" / "conditions" / f"{condition_name}.json"

    @property
    def judge_study_path(self) -> Path:
        return self.run_dir / "stage4_scores" / "judge_study.json"

    # ── stage 3: response collection ─────────────────────────────────────

    @property
    def responses_path(self) -> Path:
        return self.run_dir / "stage3_responses" / "responses.jsonl"

    @property
    def collection_manifest_path(self) -> Path:
        return self.run_dir / "stage3_responses" / "collection_manifest.json"

    # ── stage 4: response scoring ────────────────────────────────────────

    @property
    def scored_responses_path(self) -> Path:
        return self.run_dir / "stage4_scores" / "scored_responses.jsonl"

    @property
    def scoring_manifest_path(self) -> Path:
        return self.run_dir / "stage4_scores" / "scoring_manifest.json"

    @property
    def judge_calibration_path(self) -> Path:
        return self.run_dir / "stage4_scores" / "judge_calibration.json"

    # ── stage 5: analysis ────────────────────────────────────────────────

    @property
    def analysis_report_path(self) -> Path:
        return self.run_dir / "stage5_analysis" / "analysis_report.json"

    @property
    def analysis_summary_path(self) -> Path:
        return self.run_dir / "stage5_analysis" / "analysis_summary.md"

    # ── helpers ──────────────────────────────────────────────────────────

    def stage_artifact_paths(self) -> dict[str, Path]:
        """Primary artifact per stage, in pipeline order (used by status/validate)."""
        return {
            "prompts": self.prompt_sets_path,
            "hypotheses": self.hypothesis_set_path,
            "responses": self.responses_path,
            "score": self.scored_responses_path,
            "analyze": self.analysis_report_path,
        }
