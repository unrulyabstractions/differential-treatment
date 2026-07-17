"""Semantic dimension: MMD-Fuse two-sample tests over shared embedding spaces.

Implements paper section 3.3.3: each prompt is embedded into every configured
shared space — one variant per text embedder (sentence-transformers, OpenAI,
Cohere provider specs) and one per residual-stream model at change-of-turn
positions — and per space MMD-Fuse tests whether the two embedding clouds
share one distribution. Text embedders whose API key is missing are skipped
and recorded in `skipped_variants`, never dropped silently. Each variant also
stores a joint 2D PCA projection and downsampled pairwise cosine-similarity
summaries so plotting never needs the raw embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import PromptSet
from src.common.run_config import SemanticConfig
from src.inference.embedding_store import embedder_unavailable_reason
from src.inference.residual_stream_extractor import ResidualStreamExtractor
from src.semantic.mmd_fuse_runner import run_mmd_fuse

if TYPE_CHECKING:  # annotation-only: runtime import would cycle via src.pipeline
    from src.pipeline.pipeline_context import PipelineContext

# Salts derive from each variant's position in the CONFIGURED list (text
# embedders first, then residual models), so a skipped provider never shifts
# another variant's random stream, and streams stay independent across
# variants and of every other dimension's randomness.
_VARIANT_SALT_BASE = 330
_VARIANT_SALT_STRIDE = 2  # slot 0: MMD-Fuse seed, slot 1: similarity subsample
# Pair counts grow quadratically with set size; a density histogram (the only
# consumer) is visually stable at 500 samples, so storing more would only
# bloat semantic.json.
_MAX_SIMILARITY_PAIRS = 500


@dataclass
class PcaProjection2D(BaseSchema):
    """2D PCA of both sets' embeddings (fit on the stacked cloud, split back)."""

    target_x: list[float]
    target_y: list[float]
    baseline_x: list[float]
    baseline_y: list[float]
    variance_ratio_1: float
    variance_ratio_2: float


@dataclass
class SimilaritySummary(BaseSchema):
    """Pairwise cosine similarities within and between the two embedding clouds.

    Each group is subsampled (without replacement) to at most
    _MAX_SIMILARITY_PAIRS values with the run's seeded RNG.
    """

    within_target: list[float]
    within_baseline: list[float]
    between: list[float]


@dataclass
class SemanticVariantResult(BaseSchema):
    """MMD-Fuse outcome for one embedding space."""

    variant: str  # "text:<spec>" | "residual:<model>@<layer fraction>"
    embedding_model: str  # provider spec or residual model name
    embedding_dim: int
    p_value: float
    rejected: bool
    significant: bool  # MMD-Fuse reject at significance_alpha (p_value <= alpha)
    projection: PcaProjection2D
    similarity: SimilaritySummary


@dataclass
class SemanticResult(BaseSchema):
    """Full semantic-dimension result: one MMD-Fuse test per embedding space."""

    target_name: str
    baseline_name: str
    target_label: str
    baseline_label: str
    n_prompts_target: int
    n_prompts_baseline: int
    significance_alpha: float
    residual_layer_fraction: float
    variants: list[SemanticVariantResult] = field(default_factory=list)
    skipped_variants: list[str] = field(default_factory=list)  # "spec (reason)"

    def to_verdicts(self) -> list[DimensionVerdict]:
        return [
            DimensionVerdict(
                dimension="semantic",
                test_name="mmd_fuse",
                variant=v.variant,
                statistic_name="mmd_fuse_rejected",
                statistic_value=1.0 if v.rejected else 0.0,
                p_value=v.p_value,
                significant=v.significant,
                detail=(
                    f"MMD-Fuse on {v.variant} embeddings ({v.embedding_dim}d) "
                    f"{'rejects' if v.rejected else 'does not reject'} "
                    f"H0 of a shared distribution (p = {v.p_value:.4g})."
                ),
            )
            for v in self.variants
        ]


def compute_semantic(
    target: PromptSet,
    baseline: PromptSet,
    config: SemanticConfig,
    context: PipelineContext,
) -> SemanticResult:
    """Run one MMD-Fuse two-sample test per configured embedding space."""
    result = SemanticResult(
        target_name=target.name,
        baseline_name=baseline.name,
        target_label=target.label,
        baseline_label=baseline.label,
        n_prompts_target=len(target.prompts),
        n_prompts_baseline=len(baseline.prompts),
        significance_alpha=config.significance_alpha,
        residual_layer_fraction=config.residual_layer_fraction,
    )

    for index, spec in enumerate(config.text_embedders):
        reason = embedder_unavailable_reason(spec)
        if reason:
            log(f"semantic: skipping {spec} ({reason})")
            result.skipped_variants.append(f"{spec} ({reason})")
            continue
        log(f"semantic: text embeddings via {spec}")
        embeddings_target = context.embedding_store.get_text_embeddings(
            target.texts, spec
        )
        embeddings_baseline = context.embedding_store.get_text_embeddings(
            baseline.texts, spec
        )
        result.variants.append(
            _test_embedding_space(
                f"text:{spec}",
                spec,
                embeddings_target,
                embeddings_baseline,
                config,
                context,
                variant_index=index,
            )
        )

    for offset, model_name in enumerate(config.residual_models):
        log(f"semantic: residual streams via {model_name}")
        # One extractor per model serves both sets, released before the next
        # model loads so only one causal LM is ever resident.
        extractor = ResidualStreamExtractor(model_name, config.residual_layer_fraction)
        embeddings_target = extractor.extract(target.texts)
        embeddings_baseline = extractor.extract(baseline.texts)
        extractor.cleanup()
        result.variants.append(
            _test_embedding_space(
                f"residual:{model_name}@{config.residual_layer_fraction:g}",
                model_name,
                embeddings_target,
                embeddings_baseline,
                config,
                context,
                variant_index=len(config.text_embedders) + offset,
            )
        )

    return result


def _test_embedding_space(
    variant_label: str,
    model_name: str,
    embeddings_target: NDArray[np.floating],
    embeddings_baseline: NDArray[np.floating],
    config: SemanticConfig,
    context: PipelineContext,
    variant_index: int,
) -> SemanticVariantResult:
    # Seed flows through make_rng so results follow the run's global seed.
    salt = _VARIANT_SALT_BASE + _VARIANT_SALT_STRIDE * variant_index
    seed = int(context.make_rng(salt).integers(2**31))
    # One alpha governs both the MMD-Fuse reject verdict and `significant` below.
    outcome = run_mmd_fuse(
        embeddings_target, embeddings_baseline, seed, config.significance_alpha
    )
    log(f"  {variant_label}: p = {outcome.p_value:.4g}, rejected = {outcome.rejected}")
    similarity_rng = context.make_rng(salt + 1)
    return SemanticVariantResult(
        variant=variant_label,
        embedding_model=model_name,
        embedding_dim=int(embeddings_target.shape[1]),
        p_value=outcome.p_value,
        rejected=outcome.rejected,
        significant=outcome.rejected,  # p_value <= alpha, same threshold as MMD-Fuse
        projection=_project_to_2d(embeddings_target, embeddings_baseline),
        similarity=_summarize_similarities(
            embeddings_target, embeddings_baseline, similarity_rng
        ),
    )


def _project_to_2d(
    embeddings_target: NDArray[np.floating],
    embeddings_baseline: NDArray[np.floating],
) -> PcaProjection2D:
    """Joint PCA keeps both sets in one comparable 2D coordinate frame."""
    stacked = np.vstack([embeddings_target, embeddings_baseline]).astype(np.float64)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(stacked)
    n_target = embeddings_target.shape[0]
    return PcaProjection2D(
        target_x=coords[:n_target, 0].tolist(),
        target_y=coords[:n_target, 1].tolist(),
        baseline_x=coords[n_target:, 0].tolist(),
        baseline_y=coords[n_target:, 1].tolist(),
        variance_ratio_1=float(pca.explained_variance_ratio_[0]),
        variance_ratio_2=float(pca.explained_variance_ratio_[1]),
    )


def _summarize_similarities(
    embeddings_target: NDArray[np.floating],
    embeddings_baseline: NDArray[np.floating],
    rng: np.random.Generator,
) -> SimilaritySummary:
    """Flat pairwise cosine samples for the similarity-histogram plot."""
    unit_target = _unit_rows(embeddings_target)
    unit_baseline = _unit_rows(embeddings_baseline)
    upper_target = np.triu_indices(unit_target.shape[0], k=1)
    upper_baseline = np.triu_indices(unit_baseline.shape[0], k=1)
    return SimilaritySummary(
        within_target=_subsample((unit_target @ unit_target.T)[upper_target], rng),
        within_baseline=_subsample(
            (unit_baseline @ unit_baseline.T)[upper_baseline], rng
        ),
        between=_subsample((unit_target @ unit_baseline.T).ravel(), rng),
    )


def _unit_rows(embeddings: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _subsample(values: NDArray[np.floating], rng: np.random.Generator) -> list[float]:
    if values.size > _MAX_SIMILARITY_PAIRS:
        values = rng.choice(values, size=_MAX_SIMILARITY_PAIRS, replace=False)
    return [float(v) for v in values]
