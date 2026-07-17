"""Stage 1 — prompt collection (paper §4.1).

Loads both community prompt files, validates them, annotates instructions
(provided or LLM-extracted), optionally matches instruction frequencies by
subsampling, runs the comparability check, and writes the stage artifact.
"""

from __future__ import annotations

from pathlib import Path

from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import save_json
from dtreat.llm.chat_client import ChatClient
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths

from .instruction_comparability import check_instruction_comparability
from .instruction_extraction import annotate_prompt_sets
from .instruction_frequency_matching import match_instruction_frequencies
from .prompt_set_schemas import (
    CommunityPromptFile,
    FrequencyMatchingReport,
    PromptStageArtifact,
)


def load_community_prompt_file(
    path: str | Path, require_instructions: bool = True
) -> CommunityPromptFile:
    prompt_file = CommunityPromptFile.from_json(Path(path))
    _validate_prompt_file(prompt_file, str(path), require_instructions)
    return prompt_file


def _validate_prompt_file(
    prompt_file: CommunityPromptFile, source: str, require_instructions: bool
) -> None:
    problems = []
    if not prompt_file.prompts:
        problems.append("contains no prompts")
    seen_ids = set()
    for prompt in prompt_file.prompts:
        if not prompt.text.strip():
            problems.append(f"prompt {prompt.prompt_id} has empty text")
        if require_instructions and not prompt.instruction_id.strip():
            problems.append(
                f"prompt {prompt.prompt_id} has no instruction_id "
                "(use annotate_instructions: 'extract' to infer them)"
            )
        if prompt.prompt_id in seen_ids:
            problems.append(f"duplicate prompt_id {prompt.prompt_id}")
        seen_ids.add(prompt.prompt_id)
    if problems:
        raise ValueError(f"Invalid prompt file {source}: " + "; ".join(problems))


def run_prompt_collection(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> PromptStageArtifact:
    """Execute stage 1 and write `stage1_prompts/prompt_sets.json`."""
    log("Stage 1: loading community prompt sets")
    require_instructions = config.annotate_instructions == "provided"
    target_set = load_community_prompt_file(
        config.target_community.prompt_file, require_instructions
    )
    baseline_set = load_community_prompt_file(
        config.baseline_community.prompt_file, require_instructions
    )

    if target_set.community != config.target_community.name:
        raise ValueError(
            f"Target prompt file says community '{target_set.community}' but config "
            f"expects '{config.target_community.name}'"
        )
    if baseline_set.community != config.baseline_community.name:
        raise ValueError(
            f"Baseline prompt file says community '{baseline_set.community}' but config "
            f"expects '{config.baseline_community.name}'"
        )

    if config.annotate_instructions == "extract":
        log(f"  extracting instructions with {config.annotator_model}")
        annotator = ChatClient(
            config.annotator_model,
            role_label="annotator",
            cache_dir=paths.llm_cache_dir,
            trace_path=paths.llm_trace_path,
        )
        annotate_prompt_sets(config, target_set, baseline_set, annotator)

    matching = FrequencyMatchingReport()
    if config.match_instruction_frequencies:
        matching = match_instruction_frequencies(target_set, baseline_set, config.seed)
        log(
            f"  frequency matching kept {len(target_set.prompts)}/side, "
            f"dropped {matching.total_dropped()} prompts"
        )

    comparability = check_instruction_comparability(
        target_set, baseline_set, config.comparability_max_tv_distance
    )
    artifact = PromptStageArtifact(
        target_set=target_set,
        baseline_set=baseline_set,
        comparability=comparability,
        annotator_model=(
            config.annotator_model if config.annotate_instructions == "extract" else ""
        ),
        matching=matching,
    )
    save_json(artifact.to_dict(), paths.prompt_sets_path)

    log_kv(
        {
            "target prompts": len(target_set.prompts),
            "baseline prompts": len(baseline_set.prompts),
            "TV distance": f"{comparability.total_variation_distance:.3f} "
            f"(max {comparability.max_allowed_tv_distance})",
            "chi2 p-value": f"{comparability.chi2_p_value:.3f}",
            "comparability": "PASSED" if comparability.passed else "FAILED",
        }
    )
    for note in comparability.notes:
        log(f"  [Note] {note}")
    if not comparability.passed:
        raise ValueError(
            "Instruction comparability FAILED: the two prompt sets ask different "
            "things at different rates, so behavior differences would be "
            "confounded. Fix the prompt sets or raise comparability_max_tv_distance "
            "if you accept the confound."
        )
    log(f"  wrote {paths.prompt_sets_path}")
    return artifact
