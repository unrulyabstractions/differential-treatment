"""L0 realism: full pipeline against mock LLMs with planted ground truth.

The biased mock target plants known per-community behavior rates, so the run
must recover the gap signs and significance; the null profile must stay quiet.
"""

import pytest

from dtreat.common.file_io import load_jsonl
from dtreat.stages.hypothesis_generation.hypothesis_generation_stage import (
    run_hypothesis_generation,
)
from dtreat.stages.prompt_collection.prompt_collection_stage import run_prompt_collection
from dtreat.stages.response_collection.response_collection_stage import (
    run_response_collection,
)
from dtreat.stages.response_scoring.response_scoring_stage import run_response_scoring
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    run_treatment_analysis,
)


def _run_pipeline(config, paths):
    run_prompt_collection(config, paths)
    run_hypothesis_generation(config, paths)
    run_response_collection(config, paths)
    run_response_scoring(config, paths)
    return run_treatment_analysis(config, paths)


@pytest.fixture(scope="module")
def biased_report_and_paths(tmp_path_factory, request):
    """Run the biased pipeline once for the module (it is the slow part)."""
    # Build fixtures manually because module-scoped fixtures cannot use
    # the function-scoped conftest factories.
    from generate_case_study_prompts import build_prompt_files

    from dtreat.common.experiment_config import CommunitySpec, ExperimentConfig
    from dtreat.common.file_io import save_json
    from dtreat.common.run_directory_paths import RunDirectoryPaths

    tmp_path = tmp_path_factory.mktemp("biased_e2e")
    target_file, baseline_file = build_prompt_files()
    save_json(target_file, tmp_path / "target.json")
    save_json(baseline_file, tmp_path / "baseline.json")
    config = ExperimentConfig(
        run_name="e2e_biased",
        deployment_context="A fitness and nutrition advice assistant.",
        target_community=CommunitySpec("lgbtq", str(tmp_path / "target.json")),
        baseline_community=CommunitySpec("cishet", str(tmp_path / "baseline.json")),
        target_model="mock:target:biased",
        samples_per_prompt=4,
        n_permutations=500,
        seed=7,
    )
    paths = RunDirectoryPaths(tmp_path / "run")
    report = _run_pipeline(config, paths)
    return report, config, paths


class TestBiasedRecovery:
    def test_planted_axes_recovered_with_correct_signs(self, biased_report_and_paths):
        report, _config, _paths = biased_report_and_paths
        by_id = {axis.axis_id: axis for axis in report.axes}
        # Strong planted gaps must be significant with the planted sign
        assert by_id["gives_number"].significant and by_id["gives_number"].delta < 0
        assert by_id["reconsider_goal"].significant and by_id["reconsider_goal"].delta > 0
        # The planted-null axis must NOT be flagged
        assert not by_id["mention_sleep"].significant

    def test_c2st_beats_chance(self, biased_report_and_paths):
        report, _config, _paths = biased_report_and_paths
        assert report.c2st is not None
        assert report.c2st.above_chance
        assert report.c2st.accuracy > 0.6

    def test_d_pi_positive(self, biased_report_and_paths):
        report, _config, _paths = biased_report_and_paths
        assert report.d_pi_bits_significant_axes is not None
        assert report.d_pi_bits_significant_axes > 0.1

    def test_artifacts_exist_and_are_consistent(self, biased_report_and_paths):
        _report, config, paths = biased_report_and_paths
        for artifact_path in paths.stage_artifact_paths().values():
            assert artifact_path.exists()
        responses = load_jsonl(paths.responses_path)
        scored = load_jsonl(paths.scored_responses_path)
        assert len(responses) == 48 * config.samples_per_prompt
        assert len(scored) == len(responses)  # mock never refuses

    def test_stage3_is_resumable(self, biased_report_and_paths):
        _report, config, paths = biased_report_and_paths
        before = load_jsonl(paths.responses_path)
        records = run_response_collection(config, paths)  # re-run: all cached/skipped
        after = [record.to_dict() for record in records]
        assert len(after) == len(before)
        assert {r["response_id"] for r in after} == {r["response_id"] for r in before}


class TestNullStaysQuiet:
    def test_no_significant_axes_on_null_profile(self, make_mock_config):
        config, paths = make_mock_config(
            run_name="e2e_null", target_model="mock:target:null",
            samples_per_prompt=3, n_permutations=400,
        )
        report = _run_pipeline(config, paths)
        assert report.significant_axes() == []
        assert report.c2st is None or not report.c2st.above_chance


class TestNoisyJudge:
    def test_strong_axes_survive_judge_noise(self, make_mock_config):
        config, paths = make_mock_config(
            run_name="e2e_noisy", judge_model="mock:judge:noisy",
            samples_per_prompt=3, n_permutations=400,
        )
        report = _run_pipeline(config, paths)
        by_id = {axis.axis_id: axis for axis in report.axes}
        assert by_id["gives_number"].significant
        assert by_id["reconsider_goal"].significant


class TestPerAxisJudgeMode:
    def test_per_axis_mode_agrees_on_strong_axes(self, make_mock_config):
        config, paths = make_mock_config(
            run_name="e2e_per_axis", judge_mode="per_axis",
            samples_per_prompt=2, n_permutations=300,
        )
        report = _run_pipeline(config, paths)
        by_id = {axis.axis_id: axis for axis in report.axes}
        assert by_id["gives_number"].significant and by_id["gives_number"].delta < 0
