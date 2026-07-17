"""Abstract chat backend interface.

Deliberately minimal compared to the base repo's token-trajectory Backend ABC:
this pipeline only needs text-in/text-out completions with sampling controls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chat_types import ChatRequest, ChatResult


class ChatBackend(ABC):
    """A provider that can complete chat requests."""

    #: short name recorded in every ChatResult / trace record
    backend_name: str = "abstract"

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResult:
        """Execute one chat completion. Raise on transport errors (the
        client layer retries); return refused=True results for content-policy
        blocks rather than raising."""
        ...
