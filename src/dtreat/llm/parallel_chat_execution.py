"""Threaded fan-out of chat jobs with progress and failure quarantine.

Failures never abort a stage: they are collected as ChatFailure records and
written to the stage's quarantine file so a long run survives flaky calls,
and `dtreat validate` surfaces what is missing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from dtreat.common.base_schema import BaseSchema

from .chat_client import ChatClient
from .chat_types import ChatFailure, ChatRequest, ChatResult


@dataclass
class ChatJob(BaseSchema):
    """One unit of LLM work, identified by a stage-meaningful job id."""

    job_id: str
    request: ChatRequest


def execute_chat_jobs(
    client: ChatClient,
    jobs: list[ChatJob],
    max_workers: int = 8,
    description: str = "llm calls",
    show_progress: bool = True,
) -> tuple[dict[str, ChatResult], list[ChatFailure]]:
    """Run jobs concurrently through the client.

    Returns (results by job_id, failures). Results only contain jobs that
    succeeded; callers decide whether failures are acceptable.
    """
    results: dict[str, ChatResult] = {}
    failures: list[ChatFailure] = []
    if not jobs:
        return results, failures

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(client.complete, job.request): job for job in jobs}
        iterator = as_completed(future_to_job)
        if show_progress:
            iterator = tqdm(iterator, total=len(jobs), desc=description, unit="call")
        for future in iterator:
            job = future_to_job[future]
            try:
                results[job.job_id] = future.result()
            except Exception as error:
                failures.append(
                    ChatFailure(
                        job_id=job.job_id,
                        model=job.request.model,
                        error_type=type(error).__name__,
                        error_message=str(error)[:500],
                    )
                )
    return results, failures
