"""Shared fixtures: temp run directories with generated case-study data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_case_study_prompts import build_prompt_files  # noqa: E402

from dtreat.common.experiment_config import CommunitySpec, ExperimentConfig  # noqa: E402
from dtreat.common.file_io import save_json  # noqa: E402
from dtreat.common.run_directory_paths import RunDirectoryPaths  # noqa: E402


@pytest.fixture
def case_study_prompt_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write the case-study prompt sets into a temp dir."""
    target_file, baseline_file = build_prompt_files()
    target_path = tmp_path / "lgbtq_fitness.json"
    baseline_path = tmp_path / "cishet_fitness.json"
    save_json(target_file, target_path)
    save_json(baseline_file, baseline_path)
    return target_path, baseline_path


@pytest.fixture
def make_mock_config(case_study_prompt_files, tmp_path):
    """Factory for a small, fast mock experiment config + its run paths."""
    target_path, baseline_path = case_study_prompt_files

    def _make(
        run_name: str = "test_run",
        target_model: str = "mock:target:biased",
        judge_model: str = "mock:judge",
        samples_per_prompt: int = 3,
        n_permutations: int = 400,
        **overrides,
    ) -> tuple[ExperimentConfig, RunDirectoryPaths]:
        config = ExperimentConfig(
            run_name=run_name,
            deployment_context="A fitness and nutrition advice assistant.",
            target_community=CommunitySpec(name="lgbtq", prompt_file=str(target_path)),
            baseline_community=CommunitySpec(name="cishet", prompt_file=str(baseline_path)),
            helper_model="mock:helper",
            target_model=target_model,
            judge_model=judge_model,
            max_axes=5,
            samples_per_prompt=samples_per_prompt,
            n_permutations=n_permutations,
            seed=7,
            **overrides,
        )
        config.validate()
        return config, RunDirectoryPaths(tmp_path / "runs" / run_name)

    return _make
