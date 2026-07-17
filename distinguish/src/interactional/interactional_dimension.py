"""Interactional dimension: how prompts engage the chatbot (paper 3.3.6).

For EVERY configured annotation backend: annotate each prompt with one option
per interactional facet (speech act, self-disclosure depth,
anthropomorphization), compare per-set option-share distributions with
Jensen-Shannon divergence, and test each facet's divergence with an
author-level permutation test that respects within-author dependence.
Backends whose API key is missing are skipped and recorded in
skipped_variants, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet, qualified_author_ids
from src.common.run_config import InteractionalConfig
from src.common.stats_utils import (
    interleave_texts,
    jensen_shannon_divergence,
    permutation_p_value,
    permute_labels_by_author,
)
from src.inference.embedding_store import embedder_unavailable_reason
from src.interactional.facet_annotation import annotate_facets, annotation_model_name
from src.interactional.interaction_facets import INTERACTION_FACETS

if TYPE_CHECKING:  # annotation-only: a runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

# Per-(backend, facet) salts keep the interactional null draws independent of
# each other and of the other dimensions (which use 4, 50+, 331-333).
_PERMUTATION_SALT_BASE = 60


@dataclass
class FacetShare(BaseSchema):
    """Share of one facet option within each prompt set."""

    facet: str
    option: str
    share_target: float
    share_baseline: float
    count_target: int
    count_baseline: int


@dataclass
class FacetTest(BaseSchema):
    """The divergence test of one facet's option distributions."""

    facet: str
    jsd_bits: float
    p_value: float
    significant: bool


@dataclass
class InteractionalBackendResult(BaseSchema):
    """One annotation backend's facet tests and option-share tables."""

    backend: str  # spec: "embedding" | "openai:<model>"
    annotation_model: str  # concrete model the spec resolved to
    facet_tests: list[FacetTest] = field(default_factory=list)
    share_rows: list[FacetShare] = field(default_factory=list)

    def rows_for_facet(self, facet: str) -> list[FacetShare]:
        return [row for row in self.share_rows if row.facet == facet]


@dataclass
class InteractionalResult(BaseSchema):
    """Full output of the interactional dimension for one pair of prompt sets."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    n_permutations: int
    significance_alpha: float
    backend_results: list[InteractionalBackendResult] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)  # "spec (reason)"

    def to_verdicts(self) -> list[DimensionVerdict]:
        verdicts = []
        for backend in self.backend_results:
            for test in backend.facet_tests:
                top = max(
                    backend.rows_for_facet(test.facet),
                    key=lambda row: abs(row.share_target - row.share_baseline),
                )
                detail = (
                    f"{test.facet} distributions of {self.target_name} and "
                    f"{self.baseline_name} diverge by {test.jsd_bits:.3f} bits "
                    f"under the '{backend.backend}' annotation; largest gap at "
                    f"'{top.option}' "
                    f"({top.share_target:.0%} vs {top.share_baseline:.0%})."
                )
                verdicts.append(
                    DimensionVerdict(
                        dimension="interactional",
                        test_name="facet_jsd",
                        variant=f"{backend.backend}·{test.facet}",
                        statistic_name="jensen_shannon_divergence_bits",
                        statistic_value=test.jsd_bits,
                        p_value=test.p_value,
                        significant=test.significant,
                        detail=detail,
                    )
                )
        return verdicts


def _shares(option_ids: NDArray[np.integer], n_options: int) -> NDArray[np.floating]:
    counts = np.bincount(option_ids, minlength=n_options)
    return counts / counts.sum()


def compute_interactional(
    target: PromptSet,
    baseline: PromptSet,
    config: InteractionalConfig,
    context: PipelineContext,
) -> InteractionalResult:
    """Annotate and test each facet's JSD, per configured backend."""
    result = InteractionalResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        n_permutations=config.n_permutations,
        significance_alpha=config.significance_alpha,
    )
    for backend_index, backend in enumerate(config.annotation_backends):
        reason = embedder_unavailable_reason(backend)
        if reason:
            log(f"Interactional: skipping backend '{backend}' ({reason})")
            result.skipped_variants.append(f"{backend} ({reason})")
            continue
        log(
            f"Interactional: annotating {len(target.prompts)}+"
            f"{len(baseline.prompts)} prompts via '{backend}' backend"
        )
        result.backend_results.append(
            _run_backend(backend_index, backend, target, baseline, config, context)
        )
    return result


def _run_backend(
    backend_index: int,
    backend: str,
    target: PromptSet,
    baseline: PromptSet,
    config: InteractionalConfig,
    context: PipelineContext,
) -> InteractionalBackendResult:
    """One backend's facet annotation and per-facet permutation tests."""
    # Pooled, alternating annotation: batch-level LLM noise must not confound
    # set identity (see topical_dimension for the same guard).
    pooled_texts, unpool = interleave_texts(target.texts, baseline.texts)
    pooled = annotate_facets(pooled_texts, backend, config, context)
    annotations_target, annotations_baseline = pooled.split(unpool)

    # Null: reassign authors (not prompts) to sets, so within-author habits
    # (e.g. one chatty author) cannot masquerade as a set-level difference.
    authors = qualified_author_ids(target, baseline)
    labels = np.concatenate(
        [
            np.zeros(len(target.prompts), dtype=np.int64),
            np.ones(len(baseline.prompts), dtype=np.int64),
        ]
    )

    backend_result = InteractionalBackendResult(
        backend=backend,
        annotation_model=annotation_model_name(backend, config),
    )
    for facet_index, (facet, options) in enumerate(INTERACTION_FACETS.items()):
        ids_target = np.asarray(annotations_target.for_facet(facet))
        ids_baseline = np.asarray(annotations_baseline.for_facet(facet))
        n_options = len(options)
        counts_target = np.bincount(ids_target, minlength=n_options)
        counts_baseline = np.bincount(ids_baseline, minlength=n_options)
        shares_target = counts_target / counts_target.sum()
        shares_baseline = counts_baseline / counts_baseline.sum()
        jsd_bits = jensen_shannon_divergence(shares_target, shares_baseline)

        pooled_ids = np.concatenate([ids_target, ids_baseline])
        salt = (
            _PERMUTATION_SALT_BASE
            + backend_index * len(INTERACTION_FACETS)
            + facet_index
        )
        rng = context.make_rng(salt)
        null_jsds = np.empty(config.n_permutations)
        for i in range(config.n_permutations):
            permuted = permute_labels_by_author(authors, labels, rng)
            null_jsds[i] = jensen_shannon_divergence(
                _shares(pooled_ids[permuted == 0], n_options),
                _shares(pooled_ids[permuted == 1], n_options),
            )
        p_value = permutation_p_value(jsd_bits, null_jsds)

        backend_result.facet_tests.append(
            FacetTest(
                facet=facet,
                jsd_bits=jsd_bits,
                p_value=p_value,
                significant=p_value <= config.significance_alpha,
            )
        )
        backend_result.share_rows.extend(
            FacetShare(
                facet=facet,
                option=option.option,
                share_target=float(shares_target[i]),
                share_baseline=float(shares_baseline[i]),
                count_target=int(counts_target[i]),
                count_baseline=int(counts_baseline[i]),
            )
            for i, option in enumerate(options)
        )
    return backend_result
