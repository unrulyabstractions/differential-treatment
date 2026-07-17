"""Shared retry-with-backoff for API backends.

Consolidates the three near-duplicate retry loops found in the base repo's
anthropic/openai/gemini backends into one utility.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from dtreat.common.console_logging import log

RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "429",
    "503",
    "529",
    "500",
    "502",
)


def is_retryable_error(error: Exception) -> bool:
    """Heuristic: retry on rate limits, overloads, and transient transport errors."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def call_with_retry(
    fn: Callable[[], Any],
    max_retries: int = 5,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
    label: str = "api call",
) -> Any:
    """Call fn(), retrying retryable errors with exponential backoff + jitter.

    Non-retryable errors (auth, bad request, content shape) raise immediately.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as error:
            attempt += 1
            if attempt > max_retries or not is_retryable_error(error):
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x]
            log(
                f"  [Retry {attempt}/{max_retries}] {label}: "
                f"{type(error).__name__}: {str(error)[:120]} — sleeping {delay:.1f}s"
            )
            time.sleep(delay)
