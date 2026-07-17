"""Anthropic chat backend."""

from __future__ import annotations

import os

import anthropic

from .chat_backend_base import ChatBackend
from .chat_types import ChatRequest, ChatResult, ChatUsage


class AnthropicChatBackend(ChatBackend):
    """Chat completions via the Anthropic Messages API."""

    backend_name = "anthropic"

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set (env or .env). "
                "Required for anthropic-backed models."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, request: ChatRequest) -> ChatResult:
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            # Anthropic temperature range is [0, 1]; clamp silently-documented
            "temperature": min(max(request.temperature, 0.0), 1.0),
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.non_system_messages()
            ],
        }
        system_text = request.system_text()
        if system_text:
            kwargs["system"] = system_text

        response = self._client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        stop_reason = response.stop_reason or "stop"
        refused = stop_reason == "refusal"
        finish_reason = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "refusal": "refusal",
        }.get(stop_reason, stop_reason)

        return ChatResult(
            text=text,
            model=response.model,
            backend=self.backend_name,
            finish_reason=finish_reason,
            refused=refused,
            usage=ChatUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
