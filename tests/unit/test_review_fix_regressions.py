"""Regression tests for defects found by the adversarial code review."""


import numpy as np
import pytest

from dtreat.common.file_io import load_json, save_json
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.server.run_data_api import paths_for
from dtreat.stages.prompt_collection.instruction_frequency_matching import (
    match_instruction_frequencies,
)
from dtreat.stages.prompt_collection.prompt_set_schemas import (
    CommunityPromptFile,
    PromptRecord,
)
from dtreat.stages.treatment_analysis.classifier_two_sample_test import run_c2st


class TestLoadJsonDoesNotCorruptValidContent:
    """Repair heuristics must never touch parseable files (was: unconditional
    comma regexes rewrote string content, corrupting cached LLM replies)."""

    def test_commas_inside_strings_survive(self, tmp_path):
        payload = {
            "reply": 'CSV row: a,,b and list [1, 2, ] and "tuple (x, ) here, }"',
            "code": "items = [1, 2, ]\nrow = 'a,,b'",
        }
        path = tmp_path / "cached.json"
        save_json(payload, path, readable_text=False)
        assert load_json(path) == payload

    def test_repair_still_works_for_broken_files(self, tmp_path, capsys):
        path = tmp_path / "broken.json"
        path.write_text('{"a": 1,, "b": 2}')  # genuinely malformed
        assert load_json(path) == {"a": 1, "b": 2}
        assert "Repaired" in capsys.readouterr().out


class TestScoringResumability:
    """A response whose judge calls all failed must be re-judged on re-run,
    not permanently frozen as an empty 'scored' record."""

    def test_empty_records_are_not_treated_as_scored(self, tmp_path):
        from dtreat.common.experiment_config import CommunitySpec, ExperimentConfig
        from dtreat.common.file_io import save_jsonl
        from dtreat.stages.response_scoring.response_scoring_stage import (
            run_response_scoring,
        )

        paths = RunDirectoryPaths(tmp_path / "run")
        # upstream artifacts: 1 hypothesis axis, 1 response
        save_json(
            {
                "deployment_context": "d",
                "helper_model": "mock:helper",
                "axes": [{"axis_id": "gives_number",
                          "question": "Does the response give a concrete number the user can act on?"}],
            },
            paths.hypothesis_set_path,
        )
        save_jsonl(
            [{
                "response_id": "p0~s0", "prompt_id": "p0", "community": "lgbtq",
                "instruction_id": "i", "sample_index": 0, "seed": 1,
                "model": "mock:target:biased",
                "text": "Aim for roughly 300 calories over maintenance.",
            }],
            paths.responses_path,
        )
        # a poisoned prior artifact: empty verdicts (all judges failed before)
        save_jsonl(
            [{
                "response_id": "p0~s0", "prompt_id": "p0", "community": "lgbtq",
                "instruction_id": "i", "refused": False,
                "verdicts": {}, "unparsed_axes": ["gives_number"],
                "verdicts_by_judge": {}, "raw_judge_replies": {},
            }],
            paths.scored_responses_path,
        )
        config = ExperimentConfig(
            run_name="r", deployment_context="d",
            target_community=CommunitySpec("lgbtq", "unused"),
            baseline_community=CommunitySpec("cishet", "unused"),
            judge_model="mock:judge",
        )
        scored = run_response_scoring(config, paths)
        assert scored[0].verdicts.get("gives_number") is True  # re-judged


class TestFrequencyMatchingGuards:
    def test_disjoint_sets_empty_and_report_drops(self):
        def prompt_set(community, instruction_ids):
            return CommunityPromptFile(
                community=community, domain="d",
                prompts=[
                    PromptRecord(f"{community}_{i}", "text", instruction_id)
                    for i, instruction_id in enumerate(instruction_ids)
                ],
            )
        target = prompt_set("t", ["a", "a", "b"])
        baseline = prompt_set("b", ["c", "d", "d"])
        report = match_instruction_frequencies(target, baseline, seed=0)
        assert target.prompts == [] and baseline.prompts == []
        assert report.total_dropped() == 6
        # the stage guard raising on < 2 kept prompts is exercised in the
        # stage; here we assert the matcher reports honestly


class TestPathTraversalGuard:
    def test_prefix_sibling_dir_is_rejected(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        (tmp_path / "runs-evil").mkdir()
        with pytest.raises(ValueError):
            paths_for(runs_root, "../runs-evil")

    def test_normal_run_name_accepted(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        assert paths_for(runs_root, "my_run").run_dir.name == "my_run"


class TestC2stDegenerateSplit:
    def test_tiny_minority_class_returns_none(self):
        features = np.random.default_rng(0).random((12, 3))
        labels = np.array([True] * 11 + [False])  # 1-member minority class
        assert run_c2st(features, labels, 0.3, seed=0, n_dropped_incomplete=0) is None
