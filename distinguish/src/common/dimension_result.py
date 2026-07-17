"""The uniform verdict schema every dimension reduces to for the run summary.

Each dimension produces a rich, dimension-specific result schema (saved as
{dimension}.json) plus one or more DimensionVerdict rows: a flat, comparable
statement of "how distinguishable were the two sets under this test".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema


@dataclass
class DimensionVerdict(BaseSchema):
    """Flat summary of one statistical test run by a dimension."""

    dimension: str  # e.g. "semantic"
    test_name: str  # e.g. "mmd_fuse"
    variant: str  # e.g. "text:all-MiniLM-L6-v2"; "" when there is one variant
    statistic_name: str  # e.g. "c2st_accuracy"
    statistic_value: float
    p_value: float | None  # None when the test yields no single p-value
    significant: bool | None  # None when significance is not defined
    detail: str  # one human-readable sentence
