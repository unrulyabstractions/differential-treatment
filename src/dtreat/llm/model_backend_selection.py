"""Resolve a model spec string to a chat backend + provider model name.

Specs:
    mock:<role>[:variant]      -> MockChatBackend (spec passed through whole)
    anthropic:<model>          -> AnthropicChatBackend with <model>
    openai:<model>             -> OpenAIChatBackend with <model>
    claude*                    -> AnthropicChatBackend
    gpt* / o1* / o3* / o4*     -> OpenAIChatBackend
"""

from __future__ import annotations

from dtreat.common.dotenv_loading import load_dotenv_file

from .anthropic_chat_backend import AnthropicChatBackend
from .chat_backend_base import ChatBackend
from .mock_chat_backend import MockChatBackend
from .openai_chat_backend import OpenAIChatBackend

_BACKEND_CACHE: dict[str, ChatBackend] = {}


def _cached(backend_key: str, factory) -> ChatBackend:
    if backend_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[backend_key] = factory()
    return _BACKEND_CACHE[backend_key]


def resolve_backend(model_spec: str) -> tuple[ChatBackend, str]:
    """Return (backend, model name to send to the provider).

    Loads .env lazily so API keys in the repo root are found without
    polluting mock-only runs.
    """
    if model_spec.startswith("mock:"):
        return _cached("mock", MockChatBackend), model_spec

    load_dotenv_file()

    if model_spec.startswith("anthropic:"):
        return _cached("anthropic", AnthropicChatBackend), model_spec.split(":", 1)[1]
    if model_spec.startswith("openai:"):
        return _cached("openai", OpenAIChatBackend), model_spec.split(":", 1)[1]
    if model_spec.startswith("claude"):
        return _cached("anthropic", AnthropicChatBackend), model_spec
    if model_spec.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return _cached("openai", OpenAIChatBackend), model_spec

    raise ValueError(
        f"Cannot infer a backend for model spec '{model_spec}'. "
        "Use an explicit 'anthropic:<model>', 'openai:<model>', or 'mock:<role>' spec."
    )

