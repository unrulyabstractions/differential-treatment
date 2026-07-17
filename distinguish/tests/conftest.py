"""Shared fixtures for the validation suite. Run: `uv run pytest tests/ -x -q`.

This suite exists to PROVE the statistical/methodological invariants of the
pipeline (null calibration, known-answer recovery, no author leak, ground-truth
token positions, fault recovery) rather than assert them by inspection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.dataset_tables import PromptDataset

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def synthetic() -> PromptDataset:
    return PromptDataset.load(REPO / "data" / "synthetic")
