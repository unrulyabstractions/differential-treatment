"""Frequency matching: enforce the instruction-comparability assumption by
subsampling (paper Eq 3).

For every instruction id, keep min(target count, baseline count) prompts on
each side (seeded, deterministic), and drop instructions present in only one
set entirely. The retained sets then have exactly matching instruction
frequency distributions (TV distance 0), at the cost of the dropped prompts —
every drop is recorded in the artifact so nothing disappears silently.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .prompt_set_schemas import (
    CommunityPromptFile,
    FrequencyMatchingReport,
    InstructionMatchOutcome,
    PromptRecord,
)


def match_instruction_frequencies(
    target_set: CommunityPromptFile,
    baseline_set: CommunityPromptFile,
    seed: int,
) -> FrequencyMatchingReport:
    """Subsample both sets in place to identical instruction distributions."""
    target_by_instruction = _group(target_set.prompts)
    baseline_by_instruction = _group(baseline_set.prompts)
    all_ids = sorted(set(target_by_instruction) | set(baseline_by_instruction))

    rng = np.random.default_rng(seed)
    outcomes = []
    kept_target: list[PromptRecord] = []
    kept_baseline: list[PromptRecord] = []
    dropped: list[str] = []

    for instruction_id in all_ids:
        target_prompts = target_by_instruction.get(instruction_id, [])
        baseline_prompts = baseline_by_instruction.get(instruction_id, [])
        keep = min(len(target_prompts), len(baseline_prompts))
        kept_target_here = _sample(target_prompts, keep, rng)
        kept_baseline_here = _sample(baseline_prompts, keep, rng)
        kept_target.extend(kept_target_here)
        kept_baseline.extend(kept_baseline_here)
        kept_ids_here = {p.prompt_id for p in kept_target_here + kept_baseline_here}
        dropped.extend(
            prompt.prompt_id
            for prompt in target_prompts + baseline_prompts
            if prompt.prompt_id not in kept_ids_here
        )
        outcomes.append(
            InstructionMatchOutcome(
                instruction_id=instruction_id,
                kept_per_side=keep,
                dropped_target=len(target_prompts) - keep,
                dropped_baseline=len(baseline_prompts) - keep,
            )
        )

    target_set.prompts = sorted(kept_target, key=lambda p: p.prompt_id)
    baseline_set.prompts = sorted(kept_baseline, key=lambda p: p.prompt_id)
    return FrequencyMatchingReport(
        enabled=True, outcomes=outcomes, dropped_prompt_ids=sorted(dropped)
    )


def _group(prompts: list[PromptRecord]) -> dict[str, list[PromptRecord]]:
    grouped: dict[str, list[PromptRecord]] = defaultdict(list)
    for prompt in prompts:
        grouped[prompt.instruction_id].append(prompt)
    return dict(grouped)


def _sample(prompts: list[PromptRecord], keep: int, rng: np.random.Generator) -> list[PromptRecord]:
    if keep >= len(prompts):
        return list(prompts)
    indices = rng.choice(len(prompts), size=keep, replace=False)
    return [prompts[i] for i in sorted(indices)]
