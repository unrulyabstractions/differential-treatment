"""Distributional dimension: classifier two-sample tests (paper section 3.3.4).

One linear C2ST runs per configured embedding space (provider specs); spaces
whose API key is missing are recorded in `skipped_variants`, never crashed
on. The fine-tuned ModernBERT variant joins when config.classifiers includes
"modernbert". Only linear variants carry a permutation p-value — see
modernbert_c2st.py for why the fine-tuned one reports accuracy only.
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
from src.common.run_config import DistributionalConfig
from src.distributional.c2st_linear import run_linear_c2st
from src.distributional.modernbert_c2st import run_modernbert_c2st
from src.inference.embedding_store import embedder_unavailable_reason
from src.inference.residual_stream_extractor import ResidualStreamExtractor

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

_DIMENSION_NAME = "distributional"
# Per-embedder salts from a dimension-owned block keep every variant's
# permutation stream independent of the others and of other dimensions.
_RNG_SALT_BASE = 440


def _majority_chance(labels: NDArray[np.integer]) -> float:
    """Majority-class accuracy — the real no-skill reference under imbalance.

    A balanced split gives 0.5; an imbalanced one gives max(n0,n1)/n > 0.5, which
    is where the permutation null actually centers.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.5
    ones = int(labels.sum())
    return max(ones, labels.size - ones) / labels.size


@dataclass
class C2stVariantResult(BaseSchema):
    """One (classifier x representation) C2ST variant with its evidence."""

    classifier: str  # "linear" | "modernbert"
    representation: str  # embedder provider spec or fine-tuned model name
    accuracy: float
    chance_level: float
    fold_accuracies: list[float]
    p_value: float | None  # None for modernbert (no permutation test)
    n_permutations: int
    significant: bool | None
    # Permutation-null accuracies (empty for modernbert), kept so the
    # histogram plot and any p-value audit never re-run the permutations.
    null_accuracies: list[float] = field(default_factory=list)
    # Pooled held-out P(target) scores and true labels (1 = target),
    # prompt-aligned, kept for the ROC overlay.
    heldout_scores: list[float] = field(default_factory=list)
    heldout_labels: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Verdict-facing variant id, e.g. "linear:openai:text-embedding-3-small"."""
        return f"{self.classifier}:{self.representation}"

    @property
    def short_label(self) -> str:
        """Compact plot label: the representation without its org prefix."""
        return f"{self.classifier}:{self.representation.split('/')[-1]}"


@dataclass
class DistributionalResult(BaseSchema):
    """Full distributional-dimension result (saved as distributional.json)."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    cv_folds: int
    n_permutations: int
    significance_alpha: float
    variants: list[C2stVariantResult] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)  # "spec (reason)"

    def to_verdicts(self) -> list[DimensionVerdict]:
        verdicts = []
        for variant in self.variants:
            if variant.p_value is None:
                detail = (
                    f"accuracy {variant.accuracy:.2f} vs chance "
                    f"{variant.chance_level:.1f}; no permutation test "
                    "(per-permutation fine-tuning cost)"
                )
            else:
                detail = (
                    f"accuracy {variant.accuracy:.2f} vs chance "
                    f"{variant.chance_level:.1f}, p={variant.p_value:.3f} over "
                    f"{variant.n_permutations} author-level permutations"
                )
            verdicts.append(
                DimensionVerdict(
                    dimension=_DIMENSION_NAME,
                    test_name="c2st",
                    variant=variant.label,
                    statistic_name="held_out_accuracy",
                    statistic_value=variant.accuracy,
                    p_value=variant.p_value,
                    significant=variant.significant,
                    detail=detail,
                )
            )
        return verdicts


def compute_distributional(
    target: PromptSet,
    baseline: PromptSet,
    config: DistributionalConfig,
    context: PipelineContext,
) -> DistributionalResult:
    """Classifier two-sample tests between the target and baseline sets."""
    texts = target.texts + baseline.texts
    # Target is the positive class, so held-out scores read as P(target).
    labels = np.array([1] * len(target.prompts) + [0] * len(baseline.prompts))
    # Side-qualified author ids: identically named authors in different sets
    # are different people and must group/permute independently.
    author_ids = qualified_author_ids(target, baseline)

    result = DistributionalResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        cv_folds=config.cv_folds,
        n_permutations=config.n_permutations,
        significance_alpha=config.significance_alpha,
    )
    if "linear" in config.classifiers:
        for index, spec in enumerate(config.embedders):
            reason = embedder_unavailable_reason(spec)
            if reason:
                log(f"distributional: skipping linear C2ST on {spec} ({reason})")
                result.skipped_variants.append(f"{spec} ({reason})")
                continue
            result.variants.append(
                _linear_variant(texts, labels, author_ids, spec, index, config, context)
            )
        # Residual-stream C2ST (paper §3.3.4): same linear C2ST on the residual
        # embeddings semantic uses, one causal LM resident at a time.
        for offset, model_name in enumerate(config.residual_models):
            log(f"distributional: residual streams via {model_name} for C2ST")
            extractor = ResidualStreamExtractor(
                model_name, config.residual_layer_fraction
            )
            embeddings = extractor.extract(texts)
            extractor.cleanup()
            representation = f"residual:{model_name}@{config.residual_layer_fraction:g}"
            result.variants.append(
                _c2st_variant(
                    embeddings,
                    representation,
                    labels,
                    author_ids,
                    len(config.embedders) + offset,
                    config,
                    context,
                )
            )
    if "modernbert" in config.classifiers:
        result.variants.append(_modernbert_variant(texts, labels, author_ids, config))
    return result


def _linear_variant(
    texts: list[str],
    labels: NDArray[np.integer],
    author_ids: list[str],
    spec: str,
    index: int,
    config: DistributionalConfig,
    context: PipelineContext,
) -> C2stVariantResult:
    log(f"distributional: linear C2ST on {spec}")
    embeddings = context.embedding_store.get_text_embeddings(texts, spec)
    return _c2st_variant(embeddings, spec, labels, author_ids, index, config, context)


def _c2st_variant(
    embeddings: NDArray[np.floating],
    representation: str,
    labels: NDArray[np.integer],
    author_ids: list[str],
    salt_offset: int,
    config: DistributionalConfig,
    context: PipelineContext,
) -> C2stVariantResult:
    """Linear C2ST result for one already-computed embedding space."""
    outcome = run_linear_c2st(
        embeddings,
        labels,
        author_ids,
        cv_folds=config.cv_folds,
        n_permutations=config.n_permutations,
        rng=context.make_rng(_RNG_SALT_BASE + salt_offset),
    )
    return C2stVariantResult(
        classifier="linear",
        representation=representation,
        accuracy=outcome.accuracy,
        chance_level=_majority_chance(labels),
        fold_accuracies=outcome.fold_accuracies,
        p_value=outcome.p_value,
        n_permutations=outcome.n_permutations,
        significant=bool(outcome.p_value <= config.significance_alpha),
        null_accuracies=outcome.null_accuracies,
        heldout_scores=outcome.scores,
        heldout_labels=outcome.true_labels,
    )


def _modernbert_variant(
    texts: list[str],
    labels: NDArray[np.integer],
    author_ids: list[str],
    config: DistributionalConfig,
) -> C2stVariantResult:
    outcome = run_modernbert_c2st(texts, labels, author_ids, config)
    return C2stVariantResult(
        classifier="modernbert",
        representation=outcome.model_name,
        accuracy=outcome.accuracy,
        chance_level=_majority_chance(labels),
        fold_accuracies=outcome.fold_accuracies,
        p_value=None,
        n_permutations=0,
        significant=None,
        heldout_scores=outcome.scores,
        heldout_labels=outcome.true_labels,
    )
