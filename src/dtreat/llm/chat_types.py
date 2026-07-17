"""Typed request/response records for chat-completion calls.

Every LLM interaction in the pipeline flows through these schemas, so calls
are cacheable (deterministic request IDs), traceable, and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class ChatMessage(BaseSchema):
    """One message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatRequest(BaseSchema):
    """A single chat-completion request.

    `seed` identifies the sample: the same (model, messages, seed) is treated
    as the same draw and hits the response cache. Backends that support a
    server-side seed pass it through; the mock backend derives its RNG from it.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None

    def system_text(self) -> str | None:
        """Concatenated system-message content, or None."""
        parts = [m.content for m in self.messages if m.role == "system"]
        return "\n\n".join(parts) if parts else None

    def non_system_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.role != "system"]


@dataclass
class ChatUsage(BaseSchema):
    """Token accounting for one call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResult(BaseSchema):
    """Outcome of a chat-completion call.

    Refusals are first-class: a content-policy block or explicit refusal is
    recorded, never silently converted to an empty response — in a bias audit,
    who gets refused is itself signal, and dropping it would corrupt Stage 5.
    """

    text: str
    model: str
    backend: str
    finish_reason: str = "stop"  # "stop" | "length" | "refusal" | "error"
    refused: bool = False
    usage: ChatUsage = field(default_factory=ChatUsage)
    cached: bool = False


@dataclass
class ChatFailure(BaseSchema):
    """A request that exhausted retries; quarantined, not fatal."""

    job_id: str
    model: str
    error_type: str
    error_message: str
