"""L2 realism: tiny opt-in smoke tests against real APIs.

Run with:  uv run pytest -m live
Skipped automatically when the relevant API key is absent. Costs a fraction
of a cent; exercises the real backend paths (auth, retry surface, parsing).
"""

import os

import pytest

from dtreat.common.judge_protocol import (
    JUDGE_SYSTEM_PROMPT,
    build_per_response_judge_prompt,
    parse_per_response_verdicts,
)
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.stages.hypothesis_generation.helper_prompt_builder import (
    build_helper_messages,
)
from dtreat.stages.hypothesis_generation.hypothesis_generation_stage import (
    parse_helper_axes,
)

pytestmark = pytest.mark.live

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


@needs_openai
class TestOpenAiLive:
    MODEL = "gpt-4o-mini"

    def test_helper_produces_parseable_axes(self):
        system_prompt, user_prompt = build_helper_messages(
            deployment_context="A fitness and nutrition advice assistant.",
            target_community="lgbtq",
            baseline_community="cishet",
            max_axes=3,
            seed_hypotheses=[],
            literature_notes="",
        )
        client = ChatClient(self.MODEL, "live-helper")
        result = client.complete(
            client.build_request(
                [ChatMessage("system", system_prompt), ChatMessage("user", user_prompt)],
                temperature=0.7,
                max_tokens=800,
                seed=1,
            )
        )
        axes = parse_helper_axes(result.text)
        assert len(axes) >= 2
        assert all(axis.axis_id and axis.question for axis in axes)

    def test_judge_answers_protocol(self):
        axes = [
            ("gives_number", "Does the response give a concrete number the user can act on?"),
            ("mention_sleep", "Does the response mention sleep or recovery?"),
        ]
        response_text = (
            "Eat 300 calories over maintenance and 0.8 g of protein per pound. "
            "Also make sure you sleep at least 8 hours."
        )
        client = ChatClient(self.MODEL, "live-judge")
        result = client.complete(
            client.build_request(
                [
                    ChatMessage("system", JUDGE_SYSTEM_PROMPT.format(
                        deployment_context="A fitness advice assistant.")),
                    ChatMessage("user", build_per_response_judge_prompt(axes, response_text)),
                ],
                temperature=0.0,
                max_tokens=100,
                seed=1,
            )
        )
        verdicts = parse_per_response_verdicts(result.text, [a[0] for a in axes])
        assert verdicts == {"gives_number": True, "mention_sleep": True}


@needs_anthropic
class TestAnthropicLive:
    MODEL = "claude-haiku-4-5"

    def test_completion_and_usage_accounting(self):
        client = ChatClient(self.MODEL, "live-target")
        result = client.complete(
            client.build_request(
                [ChatMessage("user", "Reply with exactly the word OK.")],
                temperature=0.0,
                max_tokens=10,
            )
        )
        assert "ok" in result.text.lower()
        assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0
        assert client.stats.cost_usd > 0
