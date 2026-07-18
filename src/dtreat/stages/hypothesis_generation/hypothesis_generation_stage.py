"""Stage 2 — hypothesis generation (paper §4.2).

ALL applicable generation methods run by default (zero_context, literature,
grounded, seeded, two_stage, response_grounded); the union of their axes —
with per-method provenance on every axis — becomes the run's hypothesis set.
Per-condition proposals live in helper_study.json; stage 5 appends the
per-method downstream comparison.
"""

from __future__ import annotations

from dtreat.common.console_logging import log
from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.run_directory_paths import RunDirectoryPaths

from .helper_condition_study import applicable_conditions, run_helper_conditions
from .hypothesis_schemas import HypothesisSet


def run_hypothesis_generation(
    config: ExperimentConfig, paths: RunDirectoryPaths
) -> HypothesisSet:
    """Execute stage 2 and write the multi-method union as
    `stage2_hypotheses/hypothesis_set.json`."""
    log("Stage 2: generating hypotheses (all methods, union with provenance)")
    conditions = applicable_conditions(config)
    _report, union = run_helper_conditions(config, paths, conditions)
    if not union.axes:
        raise RuntimeError(
            "No hypothesis-generation method produced any axes; inspect "
            f"{paths.llm_trace_path} and the per-condition artifacts under "
            f"{paths.helper_condition_path('x').parent}"
        )
    return union
