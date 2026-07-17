"""Stage 2 — hypothesis generation (paper §4.2).

Asks the helper LLM for candidate axes of differential treatment, parses and
dedups them, folds in configured seed hypotheses, and writes the stage
artifact.
"""

from __future__ import annotations

import re

from dtreat.common.console_logging import log
from dtreat.common.file_io import save_json
from dtreat.common.json_text_extraction import extract_first_json_array
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths

from .helper_prompt_builder import build_helper_messages
from .hypothesis_schemas import HypothesisAxis, HypothesisSet


def parse_helper_axes(reply: str) -> list[HypothesisAxis]:
    """Parse helper JSON into validated axes (invalid entries are dropped)."""
    entries = extract_first_json_array(reply) or []
    axes = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        axis_id = _normalize_axis_id(str(entry.get("axis_id", "")))
        question = str(entry.get("question", "")).strip()
        if not axis_id or not question:
            continue
        axes.append(
            HypothesisAxis(
                axis_id=axis_id,
                question=question,
                rationale=str(entry.get("rationale", "")).strip(),
                rubric=str(entry.get("rubric", "")).strip(),
                source="helper",
            )
        )
    return axes


def _normalize_axis_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _seed_hypothesis_axis(text: str, index: int) -> HypothesisAxis:
    """Turn a configured seed hypothesis (free text) into an axis."""
    axis_id = _normalize_axis_id("_".join(text.split()[:4])) or f"seed_axis_{index}"
    return HypothesisAxis(axis_id=axis_id, question=text.strip(), source="seed")


def dedupe_axes(axes: list[HypothesisAxis]) -> list[HypothesisAxis]:
    """Drop repeated axis ids and near-identical questions, keeping first."""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    unique = []
    for axis in axes:
        question_key = re.sub(r"\W+", " ", axis.question.lower()).strip()
        if axis.axis_id in seen_ids or question_key in seen_questions:
            continue
        seen_ids.add(axis.axis_id)
        seen_questions.add(question_key)
        unique.append(axis)
    return unique


def run_hypothesis_generation(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> HypothesisSet:
    """Execute stage 2 and write `stage2_hypotheses/hypothesis_set.json`."""
    log("Stage 2: generating hypotheses with helper LLM")
    system_prompt, user_prompt = build_helper_messages(
        deployment_context=config.deployment_context,
        target_community=config.target_community.name,
        baseline_community=config.baseline_community.name,
        max_axes=config.max_axes,
        seed_hypotheses=config.seed_hypotheses,
        literature_notes=config.literature_notes,
    )
    client = ChatClient(
        config.helper_model,
        role_label="helper",
        cache_dir=paths.llm_cache_dir,
        trace_path=paths.llm_trace_path,
    )
    request = client.build_request(
        [ChatMessage("system", system_prompt), ChatMessage("user", user_prompt)],
        temperature=0.7,
        max_tokens=2048,
        seed=config.seed,
    )
    result = client.complete(request)
    if result.refused:
        raise RuntimeError("Helper LLM refused the hypothesis-generation request")

    helper_axes = parse_helper_axes(result.text)
    if not helper_axes:
        raise RuntimeError(
            "Could not parse any axes from the helper reply; inspect "
            f"{paths.llm_trace_path} (request preview in trace) and the raw reply "
            "stored in the artifact."
        )
    seed_axes = [
        _seed_hypothesis_axis(text, index) for index, text in enumerate(config.seed_hypotheses)
    ]
    axes = dedupe_axes(helper_axes + seed_axes)[: config.max_axes]

    hypothesis_set = HypothesisSet(
        deployment_context=config.deployment_context,
        helper_model=config.helper_model,
        axes=axes,
        raw_helper_reply=result.text,
    )
    save_json(hypothesis_set.to_dict(), paths.hypothesis_set_path)
    log(f"  {len(axes)} axes: {', '.join(hypothesis_set.axis_ids())}")
    log(f"  wrote {paths.hypothesis_set_path}")
    return hypothesis_set
