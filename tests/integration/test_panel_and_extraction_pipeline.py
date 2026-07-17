"""L0 integration: extraction + frequency matching + judge panel end-to-end,
plus judge calibration on the panel run."""

from dtreat.stages.hypothesis_generation.hypothesis_generation_stage import (
    run_hypothesis_generation,
)
from dtreat.stages.prompt_collection.prompt_collection_stage import run_prompt_collection
from dtreat.stages.response_collection.response_collection_stage import (
    run_response_collection,
)
from dtreat.stages.response_scoring.judge_calibration_stage import run_judge_calibration
from dtreat.stages.response_scoring.response_scoring_stage import run_response_scoring
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    run_treatment_analysis,
)


class TestExtractionAndMatching:
    def test_extracted_and_matched_sets_are_exactly_comparable(self, make_mock_config):
        config, paths = make_mock_config(
            run_name="e2e_extract",
            annotate_instructions="extract",
            annotator_model="mock:annotator",
            match_instruction_frequencies=True,
            samples_per_prompt=2,
            n_permutations=200,
        )
        artifact = run_prompt_collection(config, paths)
        assert artifact.comparability.total_variation_distance == 0.0
        assert artifact.comparability.passed
        assert len(artifact.target_set.prompts) == len(artifact.baseline_set.prompts)
        assert artifact.matching.enabled
        assert artifact.matching.total_dropped() > 0  # mock annotator noise drops some
        assert all(
            prompt.instruction_source == "extracted"
            for prompt in artifact.target_set.prompts
        )


class TestJudgePanelPipeline:
    def test_panel_run_recovers_bias_and_calibrates(self, make_mock_config):
        config, paths = make_mock_config(
            run_name="e2e_panel",
            judge_model="mock:judge",
            samples_per_prompt=3,
            n_permutations=400,
            judge_models=["mock:judge:noisy"],
            judge_aggregation="majority",
        )
        run_prompt_collection(config, paths)
        run_hypothesis_generation(config, paths)
        run_response_collection(config, paths)
        scored = run_response_scoring(config, paths)
        report = run_treatment_analysis(config, paths)

        # panel verdicts stored per judge
        assert all(len(record.verdicts_by_judge) == 2 for record in scored)
        by_id = {axis.axis_id: axis for axis in report.axes}
        assert by_id["gives_number"].significant and by_id["gives_number"].delta < 0
        assert by_id["reconsider_goal"].significant

        calibration = run_judge_calibration(config, paths, consistency_sample=8)
        pair = calibration.pair_agreements[0]
        # perfect judge vs 5%-flip judge: raw agreement ~0.95, kappa high but < 1
        assert 0.90 <= pair.raw_agreement <= 0.99
        assert pair.kappa_overall is not None and 0.75 <= pair.kappa_overall < 1.0
        flips = {c.judge_model: c.flip_rate_overall for c in calibration.consistency}
        assert flips["mock:judge"] == 0.0  # deterministic judge never flips
        assert 0.0 <= flips["mock:judge:noisy"] <= 0.25
