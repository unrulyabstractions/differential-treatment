"""Ordered registry of pipeline stages.

Single source of truth mapping CLI names to stage runners, so the CLI,
`run-all`, `status`, and the debug server all agree on what the pipeline is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.run_directory_paths import RunDirectoryPaths
from dtreat.stages.hypothesis_generation.hypothesis_generation_stage import (
    run_hypothesis_generation,
)
from dtreat.stages.prompt_collection.prompt_collection_stage import run_prompt_collection
from dtreat.stages.response_collection.response_collection_stage import (
    run_response_collection,
)
from dtreat.stages.response_scoring.response_scoring_stage import run_response_scoring
from dtreat.stages.treatment_analysis.treatment_analysis_stage import (
    run_treatment_analysis,
)


@dataclass
class StageDefinition:
    """One pipeline stage as the CLI and orchestrator see it."""

    name: str
    title: str
    paper_section: str
    runner: Callable[[ExperimentConfig, RunDirectoryPaths], Any]


# Fig. 1 is a DAG, not a chain: hypothesis generation (2) and response
# collection (3) both depend only on stage 1 and join at scoring (4).
# run-all executes 3 before 2 — a valid topological order — so
# behavior-grounded hypothesis methods can observe real responses.
PIPELINE_STAGES: list[StageDefinition] = [
    StageDefinition(
        name="prompts",
        title="Prompt collection + instruction comparability",
        paper_section="§4.1",
        runner=run_prompt_collection,
    ),
    StageDefinition(
        name="responses",
        title="Response collection (target LLM)",
        paper_section="§4.3",
        runner=run_response_collection,
    ),
    StageDefinition(
        name="hypotheses",
        title="Hypothesis generation (helper LLM, all methods)",
        paper_section="§4.2",
        runner=run_hypothesis_generation,
    ),
    StageDefinition(
        name="score",
        title="Response scoring (LLM judge)",
        paper_section="§4.4",
        runner=run_response_scoring,
    ),
    StageDefinition(
        name="analyze",
        title="Comparing distributions",
        paper_section="§4.5",
        runner=run_treatment_analysis,
    ),
]

STAGES_BY_NAME = {stage.name: stage for stage in PIPELINE_STAGES}
