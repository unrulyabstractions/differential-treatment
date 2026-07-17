"""Instruction-comparability check (paper §3.1, Eq 2–3).

Both prompt sets must make the same underlying requests at the same
frequency, so observed behavior differences cannot be attributed to *what*
was asked. We quantify the gap between the two instruction-frequency
distributions with total-variation distance and a chi-square independence
test, and fail the stage when it exceeds the configured tolerance.
"""

from __future__ import annotations

from collections import Counter

from scipy.stats import chi2_contingency

from .prompt_set_schemas import (
    CommunityPromptFile,
    ComparabilityReport,
    InstructionFrequency,
)


def check_instruction_comparability(
    target_set: CommunityPromptFile,
    baseline_set: CommunityPromptFile,
    max_tv_distance: float,
) -> ComparabilityReport:
    """Compare instruction-frequency distributions of the two sets."""
    target_counts = Counter(p.instruction_id for p in target_set.prompts)
    baseline_counts = Counter(p.instruction_id for p in baseline_set.prompts)
    all_instructions = sorted(set(target_counts) | set(baseline_counts))
    n_target = max(1, len(target_set.prompts))
    n_baseline = max(1, len(baseline_set.prompts))

    frequencies = [
        InstructionFrequency(
            instruction_id=instruction_id,
            target_count=target_counts.get(instruction_id, 0),
            baseline_count=baseline_counts.get(instruction_id, 0),
            target_fraction=target_counts.get(instruction_id, 0) / n_target,
            baseline_fraction=baseline_counts.get(instruction_id, 0) / n_baseline,
        )
        for instruction_id in all_instructions
    ]

    tv_distance = 0.5 * sum(
        abs(f.target_fraction - f.baseline_fraction) for f in frequencies
    )
    chi2_statistic, chi2_p_value = _chi2_independence(frequencies)

    notes = []
    one_sided = [f.instruction_id for f in frequencies if f.target_count == 0 or f.baseline_count == 0]
    if one_sided:
        notes.append(
            "Instructions present in only one set: "
            + ", ".join(one_sided)
            + " — behavior differences on these cannot be separated from what was asked."
        )

    return ComparabilityReport(
        total_variation_distance=tv_distance,
        chi2_statistic=chi2_statistic,
        chi2_p_value=chi2_p_value,
        max_allowed_tv_distance=max_tv_distance,
        passed=tv_distance <= max_tv_distance,
        frequencies=frequencies,
        notes=notes,
    )


def _chi2_independence(frequencies: list[InstructionFrequency]) -> tuple[float, float]:
    """Chi-square test of instruction × community independence.

    Returns (0.0, 1.0) when the table is degenerate (single instruction or
    empty cells everywhere), where the test is uninformative.
    """
    table = [
        [f.target_count for f in frequencies],
        [f.baseline_count for f in frequencies],
    ]
    if len(frequencies) < 2 or sum(table[0]) == 0 or sum(table[1]) == 0:
        return 0.0, 1.0
    try:
        result = chi2_contingency(table)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        # Zero-sum column (instruction absent from both) or similar degeneracy
        return 0.0, 1.0
