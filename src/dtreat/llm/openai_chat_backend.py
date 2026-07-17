"""OpenAI chat backend."""

from __future__ import annotations

import os

import openai

from .chat_backend_base import ChatBackend
from .chat_types import ChatRequest, ChatResult, ChatUsage

# Reasoning-model families reject sampling params and use max_completion_tokens
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODEL_PREFIXES)


class OpenAIChatBackend(ChatBackend):
    """Chat completions via the OpenAI Chat Completions API."""

    backend_name = "openai"

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set (env or .env). "
                "Required for openai-backed models."
            )
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, request: ChatRequest) -> ChatResult:
        kwargs: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if _is_reasoning_model(request.model):
            # Reasoning models reject temperature and use a different token cap;
            # sampling diversity then comes only from the server's own stochasticity.
            kwargs["max_completion_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = request.max_tokens
            kwargs["temperature"] = request.temperature
        if request.seed is not None:
            kwargs["seed"] = request.seed

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as error:
            # Content-policy blocks surface as BadRequest: record as refusal,
            # never as a silent empty response (that would bias every statistic
            # downstream if one community trips the filter more often).
            if "content_policy" in str(error).lower() or "invalid_prompt" in str(error).lower():
                return ChatResult(
                    text="",
                    model=request.model,
                    backend=self.backend_name,
                    finish_reason="refusal",
                    refused=True,
                )
            raise

        choice = response.choices[0]
        message_refusal = getattr(choice.message, "refusal", None)
        refused = choice.finish_reason == "content_filter" or bool(message_refusal)
        usage = response.usage

        return ChatResult(
            text=choice.message.content or (message_refusal or ""),
            model=response.model,
            backend=self.backend_name,
            finish_reason="refusal" if refused else (choice.finish_reason or "stop"),
            refused=refused,
            usage=ChatUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
