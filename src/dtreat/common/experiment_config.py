"""Experiment configuration: one JSON file defines a full pipeline run.

Model specs accept `mock:...` for level-0 realism, so the same config format
drives everything from unit tests to full experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dtreat.common.base_schema import BaseSchema


@dataclass
class CommunitySpec(BaseSchema):
    """One community and where its prompt set lives."""

    name: str
    prompt_file: str  # path to a community prompt JSON, relative to cwd or absolute


@dataclass
class ExperimentConfig(BaseSchema):
    """Everything a run needs; stages read only their relevant slice.

    Statistical knobs default to the paper's choices: epsilon = 0.01 profile
    smoothing (§4.5.3), Benjamini–Hochberg FDR over per-axis permutation
    p-values (§4.5.1), C2ST held-out accuracy (§4.5.3).
    """

    run_name: str
    deployment_context: str
    target_community: CommunitySpec
    baseline_community: CommunitySpec

    # LLM roles (paper Fig. 1): helper proposes axes, target is audited,
    # judge scores responses along axes.
    helper_model: str = "mock:helper"
    target_model: str = "mock:target:biased"
    judge_model: str = "mock:judge"

    # Stage 1 — instruction annotation + comparability
    # "provided": prompt files carry instruction_id; "extract": an annotator
    # LLM infers and canonicalizes instructions (paper §3.1's iota mapping)
    annotate_instructions: str = "provided"
    annotator_model: str = "mock:annotator"
    # subsample both sets to exactly matching instruction distributions
    match_instruction_frequencies: bool = False

    # Stage 2 — hypothesis generation
    max_axes: int = 8
    seed_hypotheses: list[str] = field(default_factory=list)
    literature_notes: str = ""

    # Stage 3 — response collection
    samples_per_prompt: int = 3
    temperature: float = 1.0
    max_response_tokens: int = 512

    # Stage 4 — judge
    judge_mode: str = "per_response"  # "per_response" (one call, all axes) | "per_axis"
    judge_max_tokens: int = 300
    judge_temperature: float = 0.0
    # panel: extra judge models scored simultaneously alongside judge_model
    judge_models: list[str] = field(default_factory=list)
    judge_aggregation: str = "majority"  # "majority" | "unanimous" | "any"

    # Stage 5 — analysis
    epsilon: float = 0.01
    n_permutations: int = 1000
    fdr_alpha: float = 0.05
    permutation_unit: str = "prompt"  # "prompt" (cluster-respecting) | "response"
    c2st_test_fraction: float = 0.3

    # Execution
    seed: int = 0
    max_workers: int = 8
    comparability_max_tv_distance: float = 0.2  # Eq 3 tolerance before stage 1 fails

    def judge_panel(self) -> list[str]:
        """All judge models for stage 4, primary first, de-duplicated."""
        panel = [self.judge_model] + [m for m in self.judge_models if m != self.judge_model]
        return list(dict.fromkeys(panel))

    @classmethod
    def from_config_file(cls, path: str | Path) -> ExperimentConfig:
        config = cls.from_json(Path(path))
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on configs that would produce meaningless statistics."""
        problems = []
        if self.samples_per_prompt < 1:
            problems.append("samples_per_prompt must be >= 1")
        if self.annotate_instructions not in ("provided", "extract"):
            problems.append(f"unknown annotate_instructions '{self.annotate_instructions}'")
        if self.judge_mode not in ("per_response", "per_axis"):
            problems.append(f"unknown judge_mode '{self.judge_mode}'")
        if self.judge_aggregation not in ("majority", "unanimous", "any"):
            problems.append(f"unknown judge_aggregation '{self.judge_aggregation}'")
        if self.permutation_unit not in ("prompt", "response"):
            problems.append(f"unknown permutation_unit '{self.permutation_unit}'")
        if not 0.0 < self.c2st_test_fraction < 1.0:
            problems.append("c2st_test_fraction must be in (0, 1)")
        if self.target_community.name == self.baseline_community.name:
            problems.append("target and baseline communities must have distinct names")
        if problems:
            raise ValueError("Invalid experiment config: " + "; ".join(problems))
