"""Inter-judge agreement statistics for calibration (paper §5.3: judges need
their own validation).

Cohen's kappa for judge pairs, Fleiss' kappa for panels, both on binary
verdicts. Hand-implemented so they are unit-testable against textbook values.
"""

from __future__ import annotations


def cohen_kappa(verdicts_a: list[bool], verdicts_b: list[bool]) -> float | None:
    """Chance-corrected agreement between two judges on paired verdicts.

    None when fewer than 2 pairs. When expected agreement is 1 (both judges
    constant), kappa is 1.0 on perfect agreement else 0.0.
    """
    if len(verdicts_a) != len(verdicts_b):
        raise ValueError("Verdict lists must be paired")
    n = len(verdicts_a)
    if n < 2:
        return None
    observed = sum(a == b for a, b in zip(verdicts_a, verdicts_b, strict=True)) / n
    p_yes_a = sum(verdicts_a) / n
    p_yes_b = sum(verdicts_b) / n
    expected = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(yes_counts: list[int], raters_per_item: int) -> float | None:
    """Fleiss' kappa for a panel of `raters_per_item` judges on binary items.

    Args:
        yes_counts: per item, how many judges said YES
        raters_per_item: fixed panel size n (items with missing verdicts must
            be excluded by the caller)

    None when undefined (fewer than 2 items or fewer than 2 raters).
    """
    n_items = len(yes_counts)
    n = raters_per_item
    if n_items < 2 or n < 2:
        return None
    if any(count < 0 or count > n for count in yes_counts):
        raise ValueError("yes count outside [0, raters_per_item]")

    p_yes = sum(yes_counts) / (n_items * n)
    p_no = 1.0 - p_yes
    expected = p_yes * p_yes + p_no * p_no

    per_item_agreement = [
        (count * (count - 1) + (n - count) * (n - count - 1)) / (n * (n - 1))
        for count in yes_counts
    ]
    observed = sum(per_item_agreement) / n_items
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
