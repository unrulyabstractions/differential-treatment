"""Unit tests: BaseSchema roundtrips, JSONL I/O, JSON extraction, judge
protocol, hypothesis parsing, comparability check, mock determinism."""

import typing
from dataclasses import dataclass, field

import numpy as np

from dtreat.common.base_schema import BaseSchema
from dtreat.common.file_io import load_jsonl, save_jsonl
from dtreat.common.json_text_extraction import (
    extract_first_json_array,
    extract_first_json_object,
)
from dtreat.common.judge_protocol import (
    build_per_axis_judge_prompt,
    build_per_response_judge_prompt,
    parse_per_axis_verdict,
    parse_per_response_verdicts,
)
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.stages.hypothesis_generation.axis_parsing import (
    dedupe_axes,
    parse_helper_axes,
)
from dtreat.stages.prompt_collection.instruction_comparability import (
    check_instruction_comparability,
)
from dtreat.stages.prompt_collection.prompt_set_schemas import (
    CommunityPromptFile,
    PromptRecord,
)


@dataclass
class _Inner(BaseSchema):
    value: float = 0.0


@dataclass
class _Outer(BaseSchema):
    name: str = ""
    inner: _Inner = field(default_factory=_Inner)
    items: list[int] = field(default_factory=list)
    table: dict[str, float] = field(default_factory=dict)


class TestBaseSchema:
    def test_roundtrip(self):
        outer = _Outer(name="x", inner=_Inner(1.5), items=[1, 2], table={"a": 0.25})
        rebuilt = _Outer.from_dict(outer.to_dict())
        assert rebuilt == outer

    def test_deterministic_ids(self):
        a = _Outer(name="x", inner=_Inner(1.5))
        b = _Outer(name="x", inner=_Inner(1.5))
        assert a.get_id() == b.get_id()
        assert a.get_id() != _Outer(name="y", inner=_Inner(1.5)).get_id()

    def test_numpy_values_canonicalize(self):
        outer = _Outer(name="np", inner=_Inner(float(np.float64(0.3))), items=[int(np.int64(2))])
        data = outer.to_dict()
        assert data["inner"]["value"] == 0.3
        assert data["items"] == [2]


class TestJsonl:
    def test_roundtrip_and_corrupt_line(self, tmp_path):
        path = tmp_path / "records.jsonl"
        save_jsonl([{"a": 1}, {"a": 2}], path)
        with open(path, "a") as f:
            f.write("{corrupt\n")
        assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


class TestJsonExtraction:
    def test_array_in_code_fence(self):
        reply = 'Here you go:\n```json\n[{"axis_id": "a", "question": "Q?"}]\n```\nHope it helps!'
        assert extract_first_json_array(reply) == [{"axis_id": "a", "question": "Q?"}]

    def test_object_after_thinking_block(self):
        reply = '<think>hmm {"draft": 1}</think>{"gives_number": "YES"}'
        assert extract_first_json_object(reply) == {"gives_number": "YES"}

    def test_braces_inside_strings(self):
        reply = 'prefix {"key": "value with } brace"} suffix'
        assert extract_first_json_object(reply) == {"key": "value with } brace"}

    def test_garbage_returns_none(self):
        assert extract_first_json_array("no json here") is None
        assert extract_first_json_object("{truncated") is None


class TestJudgeProtocol:
    AXES: typing.ClassVar = [
        ("gives_number", "Does it give a number?"),
        ("warn_fat", "Does it warn?"),
    ]

    def test_per_response_roundtrip(self):
        prompt = build_per_response_judge_prompt(self.AXES, "Some response.")
        assert "=== RESPONSE START ===" in prompt
        verdicts = parse_per_response_verdicts(
            '{"gives_number": "YES", "warn_fat": "no"}', ["gives_number", "warn_fat"]
        )
        assert verdicts == {"gives_number": True, "warn_fat": False}

    def test_missing_axis_maps_to_none(self):
        verdicts = parse_per_response_verdicts('{"gives_number": "YES"}', ["gives_number", "warn_fat"])
        assert verdicts["warn_fat"] is None

    def test_per_axis_verdicts(self):
        assert parse_per_axis_verdict("YES.") is True
        assert parse_per_axis_verdict("no") is False
        assert parse_per_axis_verdict("maybe?") is None
        prompt = build_per_axis_judge_prompt("warn_fat", "Does it warn?", "text")
        assert "ONLY YES or NO" in prompt


class TestHypothesisParsing:
    def test_parse_and_normalize(self):
        reply = '[{"axis_id": "Gives Number!", "question": "Does it?", "rationale": "r"}]'
        axes = parse_helper_axes(reply)
        assert axes[0].axis_id == "gives_number"

    def test_dedupe_by_id_and_question(self):
        reply = (
            '[{"axis_id": "a", "question": "Does it give a number?"},'
            ' {"axis_id": "a", "question": "Other?"},'
            ' {"axis_id": "b", "question": "Does it give a NUMBER"}]'
        )
        assert len(dedupe_axes(parse_helper_axes(reply))) == 1

    def test_invalid_entries_dropped(self):
        assert parse_helper_axes('[{"axis_id": "", "question": ""}, "junk"]') == []


def _prompt_set(community, instruction_counts):
    prompts = []
    for instruction_id, count in instruction_counts.items():
        for index in range(count):
            prompts.append(
                PromptRecord(
                    prompt_id=f"{community}_{instruction_id}_{index}",
                    text="text",
                    instruction_id=instruction_id,
                )
            )
    return CommunityPromptFile(community=community, domain="d", prompts=prompts)


class TestComparability:
    def test_identical_distributions_pass(self):
        report = check_instruction_comparability(
            _prompt_set("t", {"a": 5, "b": 5}), _prompt_set("b", {"a": 5, "b": 5}), 0.2
        )
        assert report.passed and report.total_variation_distance == 0.0

    def test_disjoint_instructions_fail(self):
        report = check_instruction_comparability(
            _prompt_set("t", {"a": 10}), _prompt_set("b", {"b": 10}), 0.2
        )
        assert not report.passed
        assert report.total_variation_distance == 1.0
        assert report.notes  # one-sided instructions flagged


class TestMockDeterminism:
    def test_same_seed_same_output_different_seed_differs(self, tmp_path):
        client = ChatClient("mock:target:biased", "target")
        messages = [ChatMessage("user", "it's giving twink, how do i bulk?")]
        first = client.complete(client.build_request(messages, seed=5)).text
        second = client.complete(client.build_request(messages, seed=5)).text
        other_seed = client.complete(client.build_request(messages, seed=6)).text
        assert first == second
        assert first != other_seed  # with 5 Bernoulli axes, overwhelmingly likely
