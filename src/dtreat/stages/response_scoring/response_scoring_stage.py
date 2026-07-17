"""Stage 4 — response scoring with an LLM judge (paper §4.4).

Every response is scored along every hypothesized axis. The judge sees the
deployment context and the response, never the community (paper §2.3).
Two modes: one judge call per response answering all axes as JSON (default,
n_axes times cheaper), or one call per (response, axis).
"""

from __future__ import annotations

from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import load_jsonl, save_json, save_jsonl
from dtreat.common.judge_protocol import (
    JUDGE_SYSTEM_PROMPT,
    build_per_axis_judge_prompt,
    build_per_response_judge_prompt,
    parse_per_axis_verdict,
    parse_per_response_verdicts,
)
from dtreat.common.random_seed import derive_seed
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.llm.parallel_chat_execution import ChatJob, execute_chat_jobs
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord

from .scored_response_schemas import ScoredResponse, ScoringManifest


def run_response_scoring(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> list[ScoredResponse]:
    """Execute stage 4 and write `stage4_scores/scored_responses.jsonl`."""
    log("Stage 4: scoring responses with LLM judge")
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    responses = [
        ResponseRecord.from_dict(record)
        for record in load_jsonl(paths.responses_path)
    ]
    axis_pairs = hypothesis_set.axis_pairs()
    axis_ids = hypothesis_set.axis_ids()

    client = ChatClient(
        config.judge_model,
        role_label="judge",
        cache_dir=paths.llm_cache_dir,
        trace_path=paths.llm_trace_path,
    )
    system_prompt = JUDGE_SYSTEM_PROMPT.format(deployment_context=config.deployment_context)

    existing = {
        record["response_id"]: ScoredResponse.from_dict(record)
        for record in load_jsonl(paths.scored_responses_path, default=[])
    }

    # Refused/empty responses are not judged: they carry no behavior to score.
    # They stay visible via the refusal counts in stage 5.
    judgeable = [r for r in responses if not r.refused and r.text.strip()]
    to_score = [r for r in judgeable if r.response_id not in existing]
    log(f"  {len(existing)} already scored, {len(to_score)} to judge on {len(axis_ids)} axes")

    if config.judge_mode == "per_response":
        scored_new, failures, judge_calls = _judge_per_response(
            client, system_prompt, to_score, axis_pairs, axis_ids, config
        )
    else:
        scored_new, failures, judge_calls = _judge_per_axis(
            client, system_prompt, to_score, axis_pairs, config
        )

    scored = {**existing, **{s.response_id: s for s in scored_new}}
    ordered = sorted(scored.values(), key=lambda s: s.response_id)
    save_jsonl([record.to_dict() for record in ordered], paths.scored_responses_path)
    if failures:
        save_jsonl(
            [failure.to_dict() for failure in failures],
            paths.quarantine_path("stage4_scores"),
        )

    manifest = ScoringManifest(
        judge_model=config.judge_model,
        judge_mode=config.judge_mode,
        axis_ids=axis_ids,
        scored_responses=len(ordered),
        skipped_refusals=len(responses) - len(judgeable),
        judge_calls=judge_calls,
        unparsed_verdicts=sum(len(record.unparsed_axes) for record in ordered),
        failed_requests=len(failures),
        input_tokens=client.stats.input_tokens,
        output_tokens=client.stats.output_tokens,
        estimated_cost_usd=client.stats.cost_usd,
    )
    save_json(manifest.to_dict(), paths.scoring_manifest_path)
    log_kv(
        {
            "scored": manifest.scored_responses,
            "skipped refusals/empty": manifest.skipped_refusals,
            "unparsed verdicts": manifest.unparsed_verdicts,
            "failures": manifest.failed_requests,
            "est. cost": f"${manifest.estimated_cost_usd:.4f}",
        }
    )
    log(f"  wrote {paths.scored_responses_path}")
    return ordered


def _judge_per_response(client, system_prompt, to_score, axis_pairs, axis_ids, config):
    """One judge call per response; reply is a JSON verdict object."""
    jobs = []
    by_id = {record.response_id: record for record in to_score}
    for record in to_score:
        prompt_text = build_per_response_judge_prompt(axis_pairs, record.text)
        jobs.append(
            ChatJob(
                job_id=record.response_id,
                request=client.build_request(
                    [ChatMessage("system", system_prompt), ChatMessage("user", prompt_text)],
                    temperature=config.judge_temperature,
                    max_tokens=config.judge_max_tokens,
                    seed=derive_seed(config.seed, "judge", record.response_id),
                ),
            )
        )
    results, failures = execute_chat_jobs(
        client, jobs, max_workers=config.max_workers, description="judging"
    )

    scored = []
    for response_id, result in results.items():
        record = by_id[response_id]
        verdict_map = parse_per_response_verdicts(result.text, axis_ids)
        scored.append(_make_scored(record, verdict_map, result.text))
    return scored, failures, len(jobs)


def _judge_per_axis(client, system_prompt, to_score, axis_pairs, config):
    """One judge call per (response, axis); replies are bare YES/NO."""
    jobs = []
    for record in to_score:
        for axis_id, question in axis_pairs:
            prompt_text = build_per_axis_judge_prompt(axis_id, question, record.text)
            jobs.append(
                ChatJob(
                    job_id=f"{record.response_id}::{axis_id}",
                    request=client.build_request(
                        [
                            ChatMessage("system", system_prompt),
                            ChatMessage("user", prompt_text),
                        ],
                        temperature=config.judge_temperature,
                        max_tokens=16,
                        seed=derive_seed(config.seed, "judge", record.response_id, axis_id),
                    ),
                )
            )
    results, failures = execute_chat_jobs(
        client, jobs, max_workers=config.max_workers, description="judging"
    )

    scored = []
    for record in to_score:
        verdict_map: dict[str, bool | None] = {}
        raw_parts = []
        for axis_id, _question in axis_pairs:
            result = results.get(f"{record.response_id}::{axis_id}")
            verdict_map[axis_id] = parse_per_axis_verdict(result.text) if result else None
            if result:
                raw_parts.append(f"{axis_id}: {result.text.strip()}")
        scored.append(_make_scored(record, verdict_map, "; ".join(raw_parts)))
    return scored, failures, len(jobs)


def _make_scored(
    record: ResponseRecord, verdict_map: dict[str, bool | None], raw_reply: str
) -> ScoredResponse:
    return ScoredResponse(
        response_id=record.response_id,
        prompt_id=record.prompt_id,
        community=record.community,
        instruction_id=record.instruction_id,
        refused=record.refused,
        verdicts={axis: verdict for axis, verdict in verdict_map.items() if verdict is not None},
        unparsed_axes=[axis for axis, verdict in verdict_map.items() if verdict is None],
        raw_judge_reply=raw_reply,
    )
