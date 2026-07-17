"""Stage 1 — prompt collection (paper §4.1).

Loads both community prompt files, validates them, runs the
instruction-comparability check, and writes the stage artifact.
"""

from __future__ import annotations

from pathlib import Path

from dtreat.common.console_logging import log, log_kv
from dtreat.common.file_io import save_json
from dtreat.pipeline.experiment_config import ExperimentConfig
from dtreat.pipeline.run_directory_paths import RunDirectoryPaths

from .instruction_comparability import check_instruction_comparability
from .prompt_set_schemas import CommunityPromptFile, PromptStageArtifact


def load_community_prompt_file(path: str | Path) -> CommunityPromptFile:
    prompt_file = CommunityPromptFile.from_json(Path(path))
    _validate_prompt_file(prompt_file, str(path))
    return prompt_file


def _validate_prompt_file(prompt_file: CommunityPromptFile, source: str) -> None:
    problems = []
    if not prompt_file.prompts:
        problems.append("contains no prompts")
    seen_ids = set()
    for prompt in prompt_file.prompts:
        if not prompt.text.strip():
            problems.append(f"prompt {prompt.prompt_id} has empty text")
        if not prompt.instruction_id.strip():
            problems.append(f"prompt {prompt.prompt_id} has no instruction_id")
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
    target_set = load_community_prompt_file(config.target_community.prompt_file)
    baseline_set = load_community_prompt_file(config.baseline_community.prompt_file)

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

    comparability = check_instruction_comparability(
        target_set, baseline_set, config.comparability_max_tv_distance
    )
    artifact = PromptStageArtifact(
        target_set=target_set,
        baseline_set=baseline_set,
        comparability=comparability,
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
