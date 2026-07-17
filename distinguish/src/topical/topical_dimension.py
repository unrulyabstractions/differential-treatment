"""Topical dimension: divergence between the topic distributions of two sets.

Implements the paper's Section 3.3.5 for EVERY configured assignment backend:
assign each prompt to a topic, compare per-set topic distributions with
Jensen-Shannon divergence, and test significance with an author-level
permutation test that respects within-author dependence. Survey backends use
the fixed 15-topic catalog (with a domain-level JSD too); the "topicgpt:<model>"
backend first GENERATES an intent/style catalog from the pooled corpus and
reports only a topic-level JSD. Backends whose API key is missing are skipped
and recorded in skipped_variants, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.common.base_schema import BaseSchema
from src.common.dataset_annotations import DOMAINS
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet, qualified_author_ids
from src.common.run_config import TopicalConfig
from src.common.stats_utils import (
    interleave_texts,
    jensen_shannon_divergence,
    permutation_p_value,
    permute_labels_by_author,
)
from src.topical.survey_topic_catalog import SURVEY_TOPICS
from src.topical.topic_assignment import (
    assign_topics,
    assignment_model_name,
    assignment_unavailable_reason,
    is_topicgpt_backend,
)
from src.topical.topicgpt_taxonomy import GeneratedTaxonomy, build_taxonomy

if TYPE_CHECKING:  # annotation-only: a runtime import would cycle through
    from src.pipeline.pipeline_context import PipelineContext  # src.pipeline

# Base salt; + backend index keeps each backend's null draws independent of
# the other backends and of every other dimension's randomness.
_PERMUTATION_SALT = 50
# Separate salt space for taxonomy-subsampling so it never collides with the
# permutation draws above (backends << 100 apart).
_TAXONOMY_SALT = 150

# topic_id -> index into DOMAINS, as an array so pooled ids map vectorized
_DOMAIN_INDEX_BY_TOPIC = np.zeros(len(SURVEY_TOPICS) + 1, dtype=np.int64)
for _topic in SURVEY_TOPICS:
    _DOMAIN_INDEX_BY_TOPIC[_topic.topic_id] = DOMAINS.index(_topic.domain)


@dataclass
class TopicShare(BaseSchema):
    """Share of one survey topic within each prompt set."""

    topic_id: int
    domain: str
    short_name: str
    proportion_target: float
    proportion_baseline: float
    count_target: int
    count_baseline: int


@dataclass
class DomainShare(BaseSchema):
    """Share of one survey domain (MH/GSH/REL) within each prompt set."""

    domain: str
    proportion_target: float
    proportion_baseline: float
    count_target: int
    count_baseline: int


@dataclass
class TopicalBackendResult(BaseSchema):
    """One assignment backend's divergence tests and share tables."""

    backend: str  # spec: "embedding" | "openai:<model>" | "topicgpt:<model>"
    assignment_model: str  # concrete model the spec resolved to
    topic_jsd: float  # bits, over the topic catalog
    domain_jsd: float | None  # bits over 3 survey domains; None for generated catalogs
    p_value: float
    significant: bool
    topic_rows: list[TopicShare] = field(default_factory=list)
    domain_rows: list[DomainShare] = field(default_factory=list)
    # Generated taxonomy for "topicgpt:<model>" backends (None for survey
    # backends); persisted with the section JSON so every run is auditable.
    taxonomy: GeneratedTaxonomy | None = None

    def most_divergent_topics(self, top_n: int = 2) -> list[TopicShare]:
        """Topics with the largest absolute proportion gap between sets."""
        by_gap = sorted(
            self.topic_rows,
            key=lambda row: abs(row.proportion_target - row.proportion_baseline),
            reverse=True,
        )
        return by_gap[:top_n]


@dataclass
class TopicalResult(BaseSchema):
    """Full output of the topical dimension for one pair of prompt sets."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    n_permutations: int
    significance_alpha: float
    backend_results: list[TopicalBackendResult] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)  # "spec (reason)"

    def to_verdicts(self) -> list[DimensionVerdict]:
        verdicts = []
        for backend in self.backend_results:
            gaps = " and ".join(
                f"'{row.short_name}' "
                f"({row.proportion_target:.0%} vs {row.proportion_baseline:.0%})"
                for row in backend.most_divergent_topics()
            )
            # Generated (topicgpt) catalogs have no survey-domain grouping.
            generated = backend.domain_jsd is None
            catalog_kind = "generated-topic" if generated else "survey-topic"
            domain_clause = (
                "" if generated else (f" (domain-level {backend.domain_jsd:.3f})")
            )
            detail = (
                f"{catalog_kind.capitalize()} distributions of {self.target_name} "
                f"and {self.baseline_name} diverge by {backend.topic_jsd:.3f} bits"
                f"{domain_clause} under the '{backend.backend}' assignment; "
                f"largest gaps at {gaps}."
            )
            verdicts.append(
                DimensionVerdict(
                    dimension="topical",
                    test_name="topic_jsd",
                    variant=backend.backend,
                    statistic_name="jensen_shannon_divergence_bits",
                    statistic_value=backend.topic_jsd,
                    p_value=backend.p_value,
                    significant=backend.significant,
                    detail=detail,
                )
            )
        return verdicts


def _topic_counts(
    topic_ids: NDArray[np.integer], n_topics: int = len(SURVEY_TOPICS)
) -> NDArray[np.int64]:
    """Counts aligned with topic ids 1..n_topics (survey catalog by default)."""
    return np.bincount(topic_ids, minlength=n_topics + 1)[1 : n_topics + 1]


def _domain_counts(topic_ids: NDArray[np.integer]) -> NDArray[np.int64]:
    """Counts aligned with DOMAINS order (MH, GSH, REL)."""
    return np.bincount(_DOMAIN_INDEX_BY_TOPIC[topic_ids], minlength=len(DOMAINS))


def _proportions(counts: NDArray[np.integer]) -> NDArray[np.floating]:
    return counts / counts.sum()


def compute_topical(
    target: PromptSet,
    baseline: PromptSet,
    config: TopicalConfig,
    context: PipelineContext,
) -> TopicalResult:
    """Assign, compare, and permutation-test per configured backend."""
    result = TopicalResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        n_permutations=config.n_permutations,
        significance_alpha=config.significance_alpha,
    )
    for backend_index, backend in enumerate(config.assignment_backends):
        reason = assignment_unavailable_reason(backend)
        if reason:
            log(f"Topical: skipping backend '{backend}' ({reason})")
            result.skipped_variants.append(f"{backend} ({reason})")
            continue
        log(
            f"Topical: assigning {len(target.prompts)}+{len(baseline.prompts)} "
            f"prompts via '{backend}' backend"
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
    config: TopicalConfig,
    context: PipelineContext,
) -> TopicalBackendResult:
    """One backend's assignment, JSDs, and author-level permutation test.

    Survey backends map onto the fixed 15-topic catalog (with a domain-level
    JSD too); the "topicgpt:<model>" backend first GENERATES a catalog from the
    pooled corpus (orthogonal intent/style topics) and reports only a topic-JSD.
    """
    # Annotate BOTH sets in one pooled, alternating call: LLM backends sample
    # nondeterministically, so per-set calls would confound batch-level
    # annotation noise with set identity (observed as null-pair false positives).
    pooled_texts, unpool = interleave_texts(target.texts, baseline.texts)
    taxonomy: GeneratedTaxonomy | None = None
    if is_topicgpt_backend(backend):
        # Build the taxonomy ONCE on the interleaved pool so its construction
        # cannot leak which side a text came from.
        taxonomy = build_taxonomy(
            pooled_texts,
            assignment_model_name(backend, config),
            config,
            context.make_rng(_TAXONOMY_SALT + backend_index),
            context,
        )
    n_topics = len(taxonomy.topics) if taxonomy else len(SURVEY_TOPICS)

    pooled_ids = assign_topics(pooled_texts, backend, config, context, taxonomy)
    ids_target_list, ids_baseline_list = unpool(pooled_ids)
    ids_target = np.asarray(ids_target_list, dtype=np.int64)
    ids_baseline = np.asarray(ids_baseline_list, dtype=np.int64)
    counts_target = _topic_counts(ids_target, n_topics)
    counts_baseline = _topic_counts(ids_baseline, n_topics)
    shares_target = _proportions(counts_target)
    shares_baseline = _proportions(counts_baseline)
    topic_jsd = jensen_shannon_divergence(shares_target, shares_baseline)

    # Null: reassign authors (not prompts) to sets, so within-author topic
    # repetition cannot masquerade as a set-level difference.
    pooled = np.concatenate([ids_target, ids_baseline])
    labels = np.concatenate(
        [
            np.zeros(len(ids_target), dtype=np.int64),
            np.ones(len(ids_baseline), dtype=np.int64),
        ]
    )
    authors = qualified_author_ids(target, baseline)
    rng = context.make_rng(_PERMUTATION_SALT + backend_index)
    null_jsds = np.empty(config.n_permutations)
    for i in range(config.n_permutations):
        permuted = permute_labels_by_author(authors, labels, rng)
        null_jsds[i] = jensen_shannon_divergence(
            _proportions(_topic_counts(pooled[permuted == 0], n_topics)),
            _proportions(_topic_counts(pooled[permuted == 1], n_topics)),
        )
    p_value = permutation_p_value(topic_jsd, null_jsds)

    topic_rows = _topic_rows(
        taxonomy, shares_target, shares_baseline, counts_target, counts_baseline
    )
    domain_jsd, domain_rows = _domain_summary(taxonomy, ids_target, ids_baseline)
    return TopicalBackendResult(
        backend=backend,
        assignment_model=assignment_model_name(backend, config),
        topic_jsd=topic_jsd,
        domain_jsd=domain_jsd,
        p_value=p_value,
        significant=p_value <= config.significance_alpha,
        topic_rows=topic_rows,
        domain_rows=domain_rows,
        taxonomy=taxonomy,
    )


def _topic_rows(
    taxonomy: GeneratedTaxonomy | None,
    shares_target: NDArray[np.floating],
    shares_baseline: NDArray[np.floating],
    counts_target: NDArray[np.integer],
    counts_baseline: NDArray[np.integer],
) -> list[TopicShare]:
    """Per-topic share rows for the survey catalog or the generated one."""
    if taxonomy is None:
        return [
            TopicShare(
                topic_id=topic.topic_id,
                domain=topic.domain,
                short_name=topic.short_name,
                proportion_target=float(shares_target[i]),
                proportion_baseline=float(shares_baseline[i]),
                count_target=int(counts_target[i]),
                count_baseline=int(counts_baseline[i]),
            )
            for i, topic in enumerate(SURVEY_TOPICS)
        ]
    return [
        TopicShare(
            topic_id=i + 1,
            domain="",  # generated topics are not grouped into survey domains
            short_name=topic.label,
            proportion_target=float(shares_target[i]),
            proportion_baseline=float(shares_baseline[i]),
            count_target=int(counts_target[i]),
            count_baseline=int(counts_baseline[i]),
        )
        for i, topic in enumerate(taxonomy.topics)
    ]


def _domain_summary(
    taxonomy: GeneratedTaxonomy | None,
    ids_target: NDArray[np.integer],
    ids_baseline: NDArray[np.integer],
) -> tuple[float | None, list[DomainShare]]:
    """Survey-domain JSD + rows; None and no rows for a generated catalog."""
    if taxonomy is not None:
        return None, []
    domain_counts_target = _domain_counts(ids_target)
    domain_counts_baseline = _domain_counts(ids_baseline)
    domain_shares_target = _proportions(domain_counts_target)
    domain_shares_baseline = _proportions(domain_counts_baseline)
    domain_jsd = jensen_shannon_divergence(domain_shares_target, domain_shares_baseline)
    domain_rows = [
        DomainShare(
            domain=domain,
            proportion_target=float(domain_shares_target[i]),
            proportion_baseline=float(domain_shares_baseline[i]),
            count_target=int(domain_counts_target[i]),
            count_baseline=int(domain_counts_baseline[i]),
        )
        for i, domain in enumerate(DOMAINS)
    ]
    return domain_jsd, domain_rows
