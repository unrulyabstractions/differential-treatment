"""Unit tests: agreement metrics, panel aggregation, instruction matching,
and helper-axis rubric plumbing."""

import numpy as np
import pytest

from dtreat.stages.hypothesis_generation.axis_parsing import parse_helper_axes
from dtreat.stages.prompt_collection.instruction_frequency_matching import (
    match_instruction_frequencies,
)
from dtreat.stages.prompt_collection.prompt_set_schemas import (
    CommunityPromptFile,
    PromptRecord,
)
from dtreat.stages.response_scoring.judge_agreement_metrics import (
    cohen_kappa,
    fleiss_kappa,
)
from dtreat.stages.response_scoring.response_scoring_stage import (
    aggregate_panel_verdict,
)


class TestCohenKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0

    def test_textbook_value(self):
        # 2x2 table: a=20 yes/yes, d=15 no/no, b=5, c=10 -> po=0.7,
        # pe = 0.6*0.5 + 0.4*0.5 = 0.5 -> kappa = 0.4
        verdicts_a = [True] * 25 + [False] * 25
        verdicts_b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
        assert cohen_kappa(verdicts_a, verdicts_b) == pytest.approx(0.4)

    def test_constant_judges(self):
        assert cohen_kappa([True, True], [True, True]) == 1.0
        assert cohen_kappa([True, True, True], [True, True, False]) == 0.0

    def test_too_few_items(self):
        assert cohen_kappa([True], [True]) is None


class TestFleissKappa:
    def test_perfect_panel(self):
        # 3 raters, all unanimous but mixed categories across items
        assert fleiss_kappa([3, 0, 3, 0], raters_per_item=3) == pytest.approx(1.0)

    def test_chance_level_near_zero(self):
        rng = np.random.default_rng(0)
        yes_counts = rng.binomial(3, 0.5, size=400).tolist()
        kappa = fleiss_kappa(yes_counts, raters_per_item=3)
        assert abs(kappa) < 0.1

    def test_undefined_cases(self):
        assert fleiss_kappa([1], raters_per_item=3) is None
        assert fleiss_kappa([1, 0], raters_per_item=1) is None

    def test_rejects_invalid_counts(self):
        with pytest.raises(ValueError):
            fleiss_kappa([4, 0], raters_per_item=3)


class TestPanelAggregation:
    def test_majority(self):
        assert aggregate_panel_verdict([True, True, False], "majority") is True
        assert aggregate_panel_verdict([False, False, True], "majority") is False
        assert aggregate_panel_verdict([True, False], "majority") is None  # tie
        assert aggregate_panel_verdict([True, None], "majority") is True

    def test_unanimous(self):
        assert aggregate_panel_verdict([True, True], "unanimous") is True
        assert aggregate_panel_verdict([False, False], "unanimous") is False
        assert aggregate_panel_verdict([True, False], "unanimous") is None

    def test_any(self):
        assert aggregate_panel_verdict([False, True], "any") is True
        assert aggregate_panel_verdict([False, False], "any") is False

    def test_all_unparsed(self):
        assert aggregate_panel_verdict([None, None], "majority") is None

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError):
            aggregate_panel_verdict([True], "median")


def _prompt_set(community, counts):
    prompts = []
    for instruction_id, count in counts.items():
        for index in range(count):
            prompts.append(
                PromptRecord(
                    prompt_id=f"{community}_{instruction_id}_{index}",
                    text=f"text {index}",
                    instruction_id=instruction_id,
                )
            )
    return CommunityPromptFile(community=community, domain="d", prompts=prompts)


class TestFrequencyMatching:
    def test_matches_to_min_counts_and_records_drops(self):
        target = _prompt_set("t", {"a": 5, "b": 2, "only_t": 3})
        baseline = _prompt_set("b", {"a": 3, "b": 4, "only_b": 1})
        report = match_instruction_frequencies(target, baseline, seed=0)
        target_counts = {p.instruction_id for p in target.prompts}
        assert len([p for p in target.prompts if p.instruction_id == "a"]) == 3
        assert len([p for p in baseline.prompts if p.instruction_id == "a"]) == 3
        assert len([p for p in target.prompts if p.instruction_id == "b"]) == 2
        assert "only_t" not in target_counts
        # dropped: 2 from t/a, 2 from b/b... plus one-sided 3 + 1
        assert report.total_dropped() == 2 + 2 + 3 + 1
        assert len(target.prompts) == len(baseline.prompts) == 5

    def test_deterministic_given_seed(self):
        target_1 = _prompt_set("t", {"a": 6})
        baseline_1 = _prompt_set("b", {"a": 3})
        match_instruction_frequencies(target_1, baseline_1, seed=42)
        kept_first = [p.prompt_id for p in target_1.prompts]
        target_2 = _prompt_set("t", {"a": 6})
        baseline_2 = _prompt_set("b", {"a": 3})
        match_instruction_frequencies(target_2, baseline_2, seed=42)
        assert kept_first == [p.prompt_id for p in target_2.prompts]


class TestUnionNearDuplicates:
    def test_inflected_duplicates_merge(self):
        from dtreat.stages.hypothesis_generation.helper_condition_study import (
            _near_duplicate,
        )
        from dtreat.stages.hypothesis_generation.hypothesis_schemas import (
            HypothesisAxis,
        )
        a = HypothesisAxis("x", "Does the response include a recommendation of supplements?")
        b = HypothesisAxis("y", "Does the response recommend supplements?")
        assert _near_duplicate(a, b)
        c = HypothesisAxis("z", "Does the response warn against injury risks?")
        assert not _near_duplicate(a, c)


class TestRubricParsing:
    def test_rubric_carried_through(self):
        reply = (
            '[{"axis_id": "a", "question": "Does it?", '
            '"rationale": "r", "rubric": "Answer YES only if explicit."}]'
        )
        axes = parse_helper_axes(reply)
        assert axes[0].rubric == "Answer YES only if explicit."
