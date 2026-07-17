"""Stage 4 — response scoring with an LLM judge panel (paper §4.4).

Every response is scored along every hypothesized axis by every configured
judge model. Judges see the deployment context, the axis questions (with
rubrics when present), and the response — never the community (paper §2.3).
Per-judge verdicts are aggregated into the response's behavior vector.

Modes: one call per response answering all axes as JSON (default, n_axes
times cheaper), or one call per (response, axis).
"""

from __future__ import annotations

from dtreat.common.console_logging import log, log_kv
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.file_io import load_jsonl, save_json, save_jsonl
from dtreat.common.judge_protocol import (
    JUDGE_SYSTEM_PROMPT,
    build_per_axis_judge_prompt,
    build_per_response_judge_prompt,
    parse_per_axis_verdict,
    parse_per_response_verdicts,
)
from dtreat.common.random_seed import derive_seed
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatFailure, ChatMessage
from dtreat.llm.parallel_chat_execution import ChatJob, execute_chat_jobs
from dtreat.stages.hypothesis_generation.hypothesis_schemas import HypothesisSet
from dtreat.stages.response_collection.response_record_schemas import ResponseRecord

from .scored_response_schemas import ScoredResponse, ScoringManifest


def run_response_scoring(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> list[ScoredResponse]:
    """Execute stage 4 and write `stage4_scores/scored_responses.jsonl`."""
    log("Stage 4: scoring responses with LLM judge panel")
    hypothesis_set = HypothesisSet.from_json(paths.hypothesis_set_path)
    responses = [
        ResponseRecord.from_dict(record) for record in load_jsonl(paths.responses_path)
    ]
    axis_ids = hypothesis_set.axis_ids()
    judge_models = config.judge_panel()

    # Records with no verdicts at all mean every judge call failed or was
    # unparseable — treat them as not-yet-scored so a transient failure never
    # permanently removes a response from the audit.
    existing = {
        record["response_id"]: ScoredResponse.from_dict(record)
        for record in load_jsonl(paths.scored_responses_path, default=[])
        if record.get("verdicts") or any(record.get("verdicts_by_judge", {}).values())
    }

    # Refused/empty responses are not judged: they carry no behavior to score.
    # They stay visible via the refusal counts in stage 5.
    judgeable = [r for r in responses if not r.refused and r.text.strip()]
    to_score = [r for r in judgeable if r.response_id not in existing]
    log(
        f"  {len(existing)} already scored, {len(to_score)} to judge on "
        f"{len(axis_ids)} axes with {len(judge_models)} judge(s)"
    )

    all_failures: list[ChatFailure] = []
    per_judge_verdicts: dict[str, dict[str, dict[str, bool | None]]] = {}
    raw_replies: dict[str, dict[str, str]] = {}
    judge_calls = 0
    stats_totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    for judge_model in judge_models:
        client = ChatClient(
            judge_model,
            role_label=f"judge:{judge_model}" if len(judge_models) > 1 else "judge",
            cache_dir=paths.llm_cache_dir,
            trace_path=paths.llm_trace_path,
        )
        verdict_maps, replies, failures, calls = judge_all_responses(
            client, config, hypothesis_set, to_score
        )
        per_judge_verdicts[judge_model] = verdict_maps
        for response_id, reply in replies.items():
            raw_replies.setdefault(response_id, {})[judge_model] = reply
        all_failures.extend(failures)
        judge_calls += calls
        stats_totals["input_tokens"] += client.stats.input_tokens
        stats_totals["output_tokens"] += client.stats.output_tokens
        stats_totals["cost_usd"] += client.stats.cost_usd

    # Responses where every judge call failed outright produce no record at
    # all (their failures are quarantined); they will be re-judged on re-run.
    scored_new = [
        _aggregate_scored(record, axis_ids, judge_models, per_judge_verdicts,
                          raw_replies, config.judge_aggregation)
        for record in to_score
        if any(
            record.response_id in per_judge_verdicts[judge_model]
            for judge_model in judge_models
        )
    ]

    scored = {**existing, **{s.response_id: s for s in scored_new}}
    ordered = sorted(scored.values(), key=lambda s: s.response_id)
    save_jsonl([record.to_dict() for record in ordered], paths.scored_responses_path)
    quarantine_file = paths.quarantine_path("stage4_scores")
    if all_failures:
        save_jsonl([failure.to_dict() for failure in all_failures], quarantine_file)
    elif quarantine_file.exists():
        quarantine_file.unlink()  # clean re-run: drop stale failure reports

    manifest = ScoringManifest(
        judge_models=judge_models,
        judge_mode=config.judge_mode,
        judge_aggregation=config.judge_aggregation,
        axis_ids=axis_ids,
        scored_responses=len(ordered),
        skipped_refusals=len(responses) - len(judgeable),
        judge_calls=judge_calls,
        unparsed_verdicts=sum(len(record.unparsed_axes) for record in ordered),
        failed_requests=len(all_failures),
        input_tokens=stats_totals["input_tokens"],
        output_tokens=stats_totals["output_tokens"],
        estimated_cost_usd=stats_totals["cost_usd"],
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


def judge_all_responses(
    client: ChatClient,
    config: ExperimentConfig,
    hypothesis_set: HypothesisSet,
    to_score: list[ResponseRecord],
) -> tuple[dict[str, dict[str, bool | None]], dict[str, str], list[ChatFailure], int]:
    """Run one judge over all responses.

    Returns (verdict map per response_id, raw reply per response_id,
    failures, number of calls issued).
    """
    axis_pairs = hypothesis_set.axis_pairs()
    axis_ids = hypothesis_set.axis_ids()
    rubrics = hypothesis_set.axis_rubrics()
    system_prompt = JUDGE_SYSTEM_PROMPT.format(
        deployment_context=config.deployment_context
    )
    if config.judge_mode == "per_response":
        return _judge_per_response(
            client, system_prompt, to_score, axis_pairs, axis_ids, rubrics, config
        )
    return _judge_per_axis(client, system_prompt, to_score, axis_pairs, rubrics, config)


def _judge_per_response(client, system_prompt, to_score, axis_pairs, axis_ids, rubrics, config):
    """One judge call per response; reply is a JSON verdict object."""
    jobs = []
    for record in to_score:
        prompt_text = build_per_response_judge_prompt(axis_pairs, record.text, rubrics)
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
        client, jobs, max_workers=config.max_workers,
        description=f"judging ({client.model_spec})",
    )
    verdict_maps = {
        response_id: parse_per_response_verdicts(result.text, axis_ids)
        for response_id, result in results.items()
    }
    replies = {response_id: result.text for response_id, result in results.items()}
    return verdict_maps, replies, failures, len(jobs)


def _judge_per_axis(client, system_prompt, to_score, axis_pairs, rubrics, config):
    """One judge call per (response, axis); replies are bare YES/NO."""
    jobs = []
    for record in to_score:
        for axis_id, question in axis_pairs:
            prompt_text = build_per_axis_judge_prompt(
                axis_id, question, record.text, rubrics.get(axis_id, "")
            )
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
        client, jobs, max_workers=config.max_workers,
        description=f"judging ({client.model_spec})",
    )

    verdict_maps: dict[str, dict[str, bool | None]] = {}
    replies: dict[str, str] = {}
    for record in to_score:
        verdict_map: dict[str, bool | None] = {}
        raw_parts = []
        for axis_id, _question in axis_pairs:
            result = results.get(f"{record.response_id}::{axis_id}")
            verdict_map[axis_id] = parse_per_axis_verdict(result.text) if result else None
            if result:
                raw_parts.append(f"{axis_id}: {result.text.strip()}")
        verdict_maps[record.response_id] = verdict_map
        replies[record.response_id] = "; ".join(raw_parts)
    return verdict_maps, replies, failures, len(jobs)


def aggregate_panel_verdict(
    judge_verdicts: list[bool | None], aggregation: str
) -> bool | None:
    """Combine one axis's verdicts across the panel; None = no aggregate."""
    parsed = [v for v in judge_verdicts if v is not None]
    if not parsed:
        return None
    yes = sum(parsed)
    no = len(parsed) - yes
    if aggregation == "majority":
        if yes == no:
            return None  # tie: refuse to guess
        return yes > no
    if aggregation == "unanimous":
        if yes == len(parsed):
            return True
        if no == len(parsed):
            return False
        return None
    if aggregation == "any":
        return yes > 0
    raise ValueError(f"Unknown judge_aggregation '{aggregation}'")


def _aggregate_scored(
    record: ResponseRecord,
    axis_ids: list[str],
    judge_models: list[str],
    per_judge_verdicts: dict[str, dict[str, dict[str, bool | None]]],
    raw_replies: dict[str, dict[str, str]],
    aggregation: str,
) -> ScoredResponse:
    verdicts_by_judge: dict[str, dict] = {}
    for judge_model in judge_models:
        judge_map = per_judge_verdicts[judge_model].get(record.response_id, {})
        verdicts_by_judge[judge_model] = {
            axis: verdict for axis, verdict in judge_map.items() if verdict is not None
        }
    aggregated: dict[str, bool] = {}
    unparsed: list[str] = []
    for axis_id in axis_ids:
        panel = [
            per_judge_verdicts[judge_model].get(record.response_id, {}).get(axis_id)
            for judge_model in judge_models
        ]
        verdict = aggregate_panel_verdict(panel, aggregation)
        if verdict is None:
            unparsed.append(axis_id)
        else:
            aggregated[axis_id] = verdict
    return ScoredResponse(
        response_id=record.response_id,
        prompt_id=record.prompt_id,
        community=record.community,
        instruction_id=record.instruction_id,
        refused=record.refused,
        verdicts=aggregated,
        unparsed_axes=unparsed,
        verdicts_by_judge=verdicts_by_judge,
        raw_judge_replies=raw_replies.get(record.response_id, {}),
    )
