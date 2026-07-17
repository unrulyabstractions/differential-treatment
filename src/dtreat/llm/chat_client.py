"""ChatClient: caching, retry, tracing, and cost accounting around a backend.

One client per pipeline role (helper / target / judge). The on-disk response
cache is keyed by the deterministic ChatRequest ID, which makes every LLM
stage resumable for free: re-running a stage replays cached calls and only
pays for what is missing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from dtreat.common.base_schema import BaseSchema
from dtreat.common.file_io import append_jsonl, ensure_dir, load_json, save_json

from .api_retry import call_with_retry
from .chat_types import ChatMessage, ChatRequest, ChatResult
from .llm_pricing import cost_usd
from .model_backend_selection import resolve_backend


@dataclass
class ChatClientStats(BaseSchema):
    """Aggregate accounting for one client (thread-safe via client lock)."""

    role_label: str = ""
    model: str = ""
    calls: int = 0
    cache_hits: int = 0
    refusals: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class _TraceRecord(BaseSchema):
    """One line in the run's llm_trace.jsonl."""

    timestamp: float = 0.0
    role_label: str = ""
    model: str = ""
    request_id: str = ""
    cached: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    refused: bool = False
    error: str | None = None
    preview: str = ""  # first chars of the reply, for quick trace scanning


class ChatClient:
    """Thread-safe wrapper: cache -> retry -> backend, with trace + stats."""

    def __init__(
        self,
        model_spec: str,
        role_label: str,
        cache_dir: Path | None = None,
        trace_path: Path | None = None,
        max_retries: int = 5,
    ):
        self.backend, self.resolved_model = resolve_backend(model_spec)
        self.model_spec = model_spec
        self.role_label = role_label
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.trace_path = Path(trace_path) if trace_path else None
        self.max_retries = max_retries
        self.stats = ChatClientStats(role_label=role_label, model=model_spec)
        self._lock = threading.Lock()
        if self.cache_dir:
            ensure_dir(self.cache_dir)

    def build_request(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> ChatRequest:
        return ChatRequest(
            model=self.resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def complete(self, request: ChatRequest) -> ChatResult:
        request_id = request.get_id()
        started = time.time()

        cached_result = self._cache_load(request_id)
        if cached_result is not None:
            cached_result.cached = True
            self._record(request_id, cached_result, started, cached=True)
            return cached_result

        try:
            result = call_with_retry(
                lambda: self.backend.complete(request),
                max_retries=self.max_retries,
                label=f"{self.role_label} ({self.model_spec})",
            )
        except Exception as error:
            self._record_error(request_id, error, started)
            raise

        self._cache_store(request_id, result)
        self._record(request_id, result, started, cached=False)
        return result

    # ── cache ────────────────────────────────────────────────────────────

    def _cache_path(self, request_id: str) -> Path | None:
        return self.cache_dir / f"{request_id}.json" if self.cache_dir else None

    def _cache_load(self, request_id: str) -> ChatResult | None:
        path = self._cache_path(request_id)
        if path is None or not path.exists():
            return None
        return ChatResult.from_dict(load_json(path))

    def _cache_store(self, request_id: str, result: ChatResult) -> None:
        path = self._cache_path(request_id)
        if path is None:
            return
        temp_path = path.with_suffix(".tmp")
        save_json(result.to_dict(), temp_path, readable_text=False)
        temp_path.replace(path)  # atomic within the same filesystem

    # ── accounting + trace ───────────────────────────────────────────────

    def _record(self, request_id: str, result: ChatResult, started: float, cached: bool) -> None:
        call_cost = 0.0 if cached else cost_usd(self.model_spec, result.usage)
        with self._lock:
            self.stats.calls += 1
            self.stats.cache_hits += int(cached)
            self.stats.refusals += int(result.refused)
            if not cached:
                self.stats.input_tokens += result.usage.input_tokens
                self.stats.output_tokens += result.usage.output_tokens
                self.stats.cost_usd += call_cost
            self._trace(
                _TraceRecord(
                    timestamp=started,
                    role_label=self.role_label,
                    model=self.model_spec,
                    request_id=request_id,
                    cached=cached,
                    latency_ms=int((time.time() - started) * 1000),
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cost_usd=call_cost,
                    finish_reason=result.finish_reason,
                    refused=result.refused,
                    preview=result.text[:80],
                )
            )

    def _record_error(self, request_id: str, error: Exception, started: float) -> None:
        with self._lock:
            self.stats.calls += 1
            self.stats.errors += 1
            self._trace(
                _TraceRecord(
                    timestamp=started,
                    role_label=self.role_label,
                    model=self.model_spec,
                    request_id=request_id,
                    latency_ms=int((time.time() - started) * 1000),
                    finish_reason="error",
                    error=f"{type(error).__name__}: {str(error)[:200]}",
                )
            )

    def _trace(self, record: _TraceRecord) -> None:
        if self.trace_path:
            append_jsonl(record.to_dict(), self.trace_path)
