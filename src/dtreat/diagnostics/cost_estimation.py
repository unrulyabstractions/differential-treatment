"""`dtreat estimate-cost` — pre-run call/token/cost estimate per stage.

Rough by design: token counts are estimated at ~1.3 tokens per word from the
actual prompt files, so the order of magnitude is trustworthy before spending.
"""

from __future__ import annotations

from dtreat.common.console_logging import log, log_header, log_kv
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.llm.chat_types import ChatUsage
from dtreat.llm.llm_pricing import cost_usd, is_priced_model
from dtreat.stages.prompt_collection.prompt_collection_stage import (
    load_community_prompt_file,
)

TOKENS_PER_WORD = 1.3


def _words(text: str) -> int:
    return len(text.split())


def print_cost_estimate(config: ExperimentConfig) -> int:
    """Estimate stage-by-stage LLM usage for this config."""
    require_instructions = config.annotate_instructions == "provided"
    target_set = load_community_prompt_file(
        config.target_community.prompt_file, require_instructions
    )
    baseline_set = load_community_prompt_file(
        config.baseline_community.prompt_file, require_instructions
    )
    prompts = target_set.prompts + baseline_set.prompts
    n_prompts = len(prompts)
    n_responses = n_prompts * config.samples_per_prompt
    mean_prompt_tokens = (
        sum(_words(p.text) for p in prompts) / max(1, n_prompts) * TOKENS_PER_WORD
    )

    # Stage 2: one helper call
    helper_usage = ChatUsage(input_tokens=800, output_tokens=120 * config.max_axes)
    helper_cost = cost_usd(config.helper_model, helper_usage)

    # Stage 3: n_responses target calls
    target_usage = ChatUsage(
        input_tokens=int(mean_prompt_tokens * n_responses),
        output_tokens=int(config.max_response_tokens * 0.7) * n_responses,
    )
    target_cost = cost_usd(config.target_model, target_usage)

    # Stage 4: judge calls (per_response: 1/response; per_axis: n_axes/response)
    calls_per_response = 1 if config.judge_mode == "per_response" else config.max_axes
    judge_calls = n_responses * calls_per_response
    judge_input_per_call = int(
        (config.max_response_tokens * 0.7) + 60 * (config.max_axes if calls_per_response == 1 else 1) + 120
    )
    judge_usage = ChatUsage(
        input_tokens=judge_input_per_call * judge_calls,
        output_tokens=(config.judge_max_tokens if calls_per_response == 1 else 4) * judge_calls,
    )
    judge_cost = cost_usd(config.judge_model, judge_usage)

    log_header(f"Cost estimate: {config.run_name}")
    log_kv(
        {
            "prompts (target + baseline)": f"{len(target_set.prompts)} + {len(baseline_set.prompts)}",
            "responses to sample": n_responses,
            "judge calls": judge_calls,
            "helper": _line(config.helper_model, 1, helper_usage, helper_cost),
            "target": _line(config.target_model, n_responses, target_usage, target_cost),
            "judge": _line(config.judge_model, judge_calls, judge_usage, judge_cost),
            "TOTAL estimated cost": f"${helper_cost + target_cost + judge_cost:.2f}",
        }
    )
    for model in (config.helper_model, config.target_model, config.judge_model):
        if not is_priced_model(model):
            log(f"  [warn] no pricing for {model}; its cost shows as $0")
    return 0


def _line(model: str, calls: int, usage: ChatUsage, cost: float) -> str:
    return (
        f"{model}: {calls} calls, ~{usage.input_tokens:,}→{usage.output_tokens:,} "
        f"tokens, ~${cost:.2f}"
    )
