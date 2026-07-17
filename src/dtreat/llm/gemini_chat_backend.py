"""Google Gemini chat backend."""

from __future__ import annotations

import os

from google import genai
from google.genai import types as genai_types

from .chat_backend_base import ChatBackend
from .chat_types import ChatRequest, ChatResult, ChatUsage


class GeminiChatBackend(ChatBackend):
    """Chat completions via the Google Gen AI SDK."""

    backend_name = "gemini"

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set (env or .env). "
                "Required for gemini-backed models."
            )
        self._client = genai.Client(api_key=api_key)

    def complete(self, request: ChatRequest) -> ChatResult:
        config = genai_types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=request.system_text(),
            # thinking shares the output-token budget on flash models; a
            # thinking burst can starve the visible reply to a few tokens
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )
        contents = "\n\n".join(
            message.content for message in request.non_system_messages()
        )
        response = self._client.models.generate_content(
            model=request.model, contents=contents, config=config
        )

        finish_reason = "stop"
        refused = False
        text = response.text or ""
        candidates = response.candidates or []
        if candidates:
            raw_reason = str(getattr(candidates[0], "finish_reason", "") or "").lower()
            if "safety" in raw_reason or "prohibited" in raw_reason:
                finish_reason, refused = "refusal", True
            elif "max_tokens" in raw_reason:
                finish_reason = "length"
        elif not text:
            # No candidates at all: the prompt itself was blocked
            finish_reason, refused = "refusal", True

        usage = response.usage_metadata
        return ChatResult(
            text=text,
            model=request.model,
            backend=self.backend_name,
            finish_reason=finish_reason,
            refused=refused,
            usage=ChatUsage(
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            ),
        )
