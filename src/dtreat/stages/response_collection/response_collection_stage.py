"""Stage 3 — response collection (paper §4.3).

Samples K responses per prompt from the target LLM for both communities.
Resumable: the LLM cache replays already-collected samples, and existing
response records are kept on re-runs.
"""

from __future__ import annotations

from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import load_jsonl, save_json, save_jsonl
from dtreat.common.random_seed import derive_seed
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.llm.parallel_chat_execution import ChatJob, execute_chat_jobs
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths
from dtreat.stages.prompt_collection.prompt_set_schemas import (
    PromptRecord,
    PromptStageArtifact,
)

from .response_record_schemas import CollectionManifest, ResponseRecord


def run_response_collection(
    config: ExperimentConfig,
    paths: RunDirectoryPaths,
    limit_prompts: int | None = None,
) -> list[ResponseRecord]:
    """Execute stage 3 and write `stage3_responses/responses.jsonl`.

    Args:
        limit_prompts: optional per-community cap for cheap trial runs
            (`dtreat responses --limit N`).
    """
    log("Stage 3: collecting responses from target LLM")
    artifact = PromptStageArtifact.from_json(paths.prompt_sets_path)
    prompts = _flatten_prompts(artifact, limit_prompts)

    client = ChatClient(
        config.target_model,
        role_label="target",
        cache_dir=paths.llm_cache_dir,
        trace_path=paths.llm_trace_path,
    )

    existing = {
        record["response_id"]: ResponseRecord.from_dict(record)
        for record in load_jsonl(paths.responses_path, default=[])
    }

    jobs: list[ChatJob] = []
    job_meta: dict[str, tuple[PromptRecord, str, int, int]] = {}
    for prompt, community in prompts:
        for sample_index in range(config.samples_per_prompt):
            response_id = f"{prompt.prompt_id}~s{sample_index}"
            if response_id in existing:
                continue
            sample_seed = derive_seed(config.seed, prompt.prompt_id, sample_index)
            job_meta[response_id] = (prompt, community, sample_index, sample_seed)
            jobs.append(
                ChatJob(
                    job_id=response_id,
                    request=client.build_request(
                        [ChatMessage("user", prompt.text)],
                        temperature=config.temperature,
                        max_tokens=config.max_response_tokens,
                        seed=sample_seed,
                    ),
                )
            )

    log(f"  {len(existing)} responses already collected, {len(jobs)} to sample")
    results, failures = execute_chat_jobs(
        client, jobs, max_workers=config.max_workers, description="sampling target"
    )

    records = dict(existing)
    for response_id, result in results.items():
        prompt, community, sample_index, sample_seed = job_meta[response_id]
        records[response_id] = ResponseRecord(
            response_id=response_id,
            prompt_id=prompt.prompt_id,
            community=community,
            instruction_id=prompt.instruction_id,
            sample_index=sample_index,
            seed=sample_seed,
            model=config.target_model,
            text=result.text,
            finish_reason=result.finish_reason,
            refused=result.refused,
            usage=result.usage,
        )

    ordered = sorted(records.values(), key=lambda r: r.response_id)
    save_jsonl([record.to_dict() for record in ordered], paths.responses_path)
    if failures:
        save_jsonl(
            [failure.to_dict() for failure in failures],
            paths.quarantine_path("stage3_responses"),
        )

    manifest = CollectionManifest(
        target_model=config.target_model,
        samples_per_prompt=config.samples_per_prompt,
        temperature=config.temperature,
        expected_responses=len(prompts) * config.samples_per_prompt,
        collected_responses=len(ordered),
        failed_requests=len(failures),
        refusals=sum(record.refused for record in ordered),
        input_tokens=client.stats.input_tokens,
        output_tokens=client.stats.output_tokens,
        estimated_cost_usd=client.stats.cost_usd,
        responses_by_community=_count_by_community(ordered),
    )
    save_json(manifest.to_dict(), paths.collection_manifest_path)
    log_kv(
        {
            "collected": f"{manifest.collected_responses}/{manifest.expected_responses}",
            "failures": manifest.failed_requests,
            "refusals": manifest.refusals,
            "est. cost": f"${manifest.estimated_cost_usd:.4f}",
        }
    )
    log(f"  wrote {paths.responses_path}")
    return ordered


def _flatten_prompts(
    artifact: PromptStageArtifact, limit_prompts: int | None
) -> list[tuple[PromptRecord, str]]:
    target = artifact.target_set.prompts
    baseline = artifact.baseline_set.prompts
    if limit_prompts is not None:
        target = target[:limit_prompts]
        baseline = baseline[:limit_prompts]
    return [(prompt, artifact.target_set.community) for prompt in target] + [
        (prompt, artifact.baseline_set.community) for prompt in baseline
    ]


def _count_by_community(records: list[ResponseRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.community] = counts.get(record.community, 0) + 1
    return counts
