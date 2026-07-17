"""Pipeline configuration: complete-analysis defaults, one JSON-able schema.

The defaults run EVERYTHING the paper specifies — every embedder (local,
OpenAI, Cohere), every residual-stream model, both classifiers, both
assignment backends, plus the codedness sweep and identity-slice
explorations. Variants whose API key is missing are skipped and recorded, not
silently dropped. configs/config.json mirrors these defaults.

Provider specs: "openai:<model>", "cohere:<model>", anything else =
sentence-transformers model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

ALL_DIMENSIONS = [
    "lexical",
    "syntactic",
    "semantic",
    "distributional",
    "topical",
    "interactional",
]
DEFAULT_TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_EMBEDDERS = [
    DEFAULT_TEXT_EMBEDDING_MODEL,
    "openai:text-embedding-3-small",
    "cohere:embed-v4.0",
]
DEFAULT_RESIDUAL_MODELS = [
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
]
DEFAULT_LLM_BACKENDS = ["embedding", "openai:gpt-5-mini"]


@dataclass
class LexicalConfig(BaseSchema):
    """Calibrated marked words (log-odds, hybrid Dirichlet prior) + BH FDR."""

    min_word_count: int = 2
    # Prior calibration mode: "mickel" = More of the Same Algorithm 3 (per-side
    # regularizers over a calibration word set; suppresses common/register
    # words); "fixed" = legacy fixed-strength hybrid prior below.
    prior_calibration: str = "mickel"
    # Reference corpus for the hybrid prior: "wordfreq:<lang>" or a path to a
    # JSON file mapping word -> relative frequency (or raw counts).
    reference_corpus: str = "wordfreq:en"
    # Operating point tuned so register words (pronouns, imperatives) do not
    # flag while content signatures do — MotS's own "binary-search C on your
    # data" methodology (their published value is C=0.25875, reference_prior_
    # weight=0.85; the calibration-sweep plot justifies this choice per corpus).
    reference_prior_weight: float = 0.6  # count-space weight on the reference
    # "auto" binary-searches C per corpus so no register/function word flags
    # (the faithful MotS find_optimal_alpha); or set a fixed float.
    calibration_constant: float | str = "auto"
    # "fixed" mode knobs (kept for comparison runs):
    english_prior_weight: float = 0.5
    prior_strength: float = 500.0
    # Significance: "bh_fdr" (our default) or "raw_z" (MotS replication, z>=1.96)
    significance: str = "bh_fdr"
    fdr_alpha: float = 0.05
    top_words_reported: int = 25
    calibration_plots: bool = True  # emit lexical/calibration/ justification plots


@dataclass
class SyntacticConfig(BaseSchema):
    """NeuroBiber binary style features compared via smoothed log-odds ratios."""

    model_name: str = "Blablablab/neurobiber"
    batch_size: int = 16
    smoothing_count: float = 0.5
    fdr_alpha: float = 0.05
    top_features_reported: int = 20


@dataclass
class SemanticConfig(BaseSchema):
    """MMD-Fuse two-sample tests over every configured embedding space."""

    text_embedders: list[str] = field(
        default_factory=lambda: list(DEFAULT_TEXT_EMBEDDERS)
    )
    residual_models: list[str] = field(
        default_factory=lambda: list(DEFAULT_RESIDUAL_MODELS)
    )
    residual_layer_fraction: float = 0.75
    significance_alpha: float = 0.05


@dataclass
class DistributionalConfig(BaseSchema):
    """C2ST: linear per embedding space + fine-tuned ModernBERT."""

    embedders: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_EMBEDDERS))
    # Residual-stream spaces the linear C2ST also runs on (paper §3.3.4: C2ST on
    # "the semantic embeddings of §3.3.3" = text AND residual). Empty by default
    # because residual extraction is heavy; set to enable per-model residual C2ST.
    residual_models: list[str] = field(default_factory=list)
    residual_layer_fraction: float = 0.75
    classifiers: list[str] = field(default_factory=lambda: ["linear", "modernbert"])
    cv_folds: int = 5
    n_permutations: int = 200
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    modernbert_epochs: int = 3
    modernbert_learning_rate: float = 2e-5
    significance_alpha: float = 0.05


@dataclass
class TopicalConfig(BaseSchema):
    """Survey-topic assignment (every backend) + Jensen-Shannon divergence."""

    assignment_backends: list[str] = field(
        default_factory=lambda: [*DEFAULT_LLM_BACKENDS, "topicgpt:gpt-5-mini"]
    )
    embedding_model: str = DEFAULT_TEXT_EMBEDDING_MODEL
    n_permutations: int = 1000
    significance_alpha: float = 0.05
    # TopicGPT backend (taxonomy generated from the pooled corpus, seeds
    # orthogonal to the survey topics — see docs/ITERATION4_PLAN.md):
    topicgpt_merge_similarity: float = 0.5
    topicgpt_prune_fraction: float = 0.01
    topicgpt_early_stop: int = 25
    topicgpt_max_generation_docs: int = 400


@dataclass
class InteractionalConfig(BaseSchema):
    """Speech acts, disclosure depth, anthropomorphization (paper 3.3.6)."""

    annotation_backends: list[str] = field(
        default_factory=lambda: list(DEFAULT_LLM_BACKENDS)
    )
    embedding_model: str = DEFAULT_TEXT_EMBEDDING_MODEL
    n_permutations: int = 1000
    significance_alpha: float = 0.05


@dataclass
class UsageConfig(BaseSchema):
    """Usage frequency / aversion / satisfaction from interaction context c."""

    fdr_alpha: float = 0.05


@dataclass
class AuthorSliceSpec(BaseSchema):
    """One identity/demographic slice: filter authors, rerun the sections."""

    facet: str  # grouping directory: gender | race | age | transgender | ...
    name: str  # value directory within the facet, e.g. "women"
    z_field: str  # AuthorIdentity/AuthorDemographics field name
    op: str  # "eq" | "contains" | "not_contains" | "in" | "not_in"
    value: str  # for in/not_in: comma-separated accepted values
    apply_to: str = "both"  # "both" | "target" (e.g. transgender only exists there)


def _default_slices() -> list[AuthorSliceSpec]:
    return [
        AuthorSliceSpec("transgender", "trans", "transgender", "eq", "1", "target"),
        AuthorSliceSpec("gender", "women", "gender", "contains", "woman", "both"),
        AuthorSliceSpec("gender", "men", "gender", "contains", "man", "both"),
        AuthorSliceSpec(
            "gender", "nonbinary", "gender", "contains", "nonbinary", "target"
        ),
        AuthorSliceSpec(
            "race", "poc", "race", "not_contains", "White/Caucasian", "both"
        ),
        AuthorSliceSpec("race", "white", "race", "contains", "white", "both"),
        AuthorSliceSpec("age", "under35", "age", "in", "13-17,18-24,25-34", "both"),
        AuthorSliceSpec(
            "age",
            "35plus",
            "age",
            "in",
            "35-44,45-54,55-64,65-74,75-84,85-94,95+",
            "both",
        ),
    ]


@dataclass
class ExplorationsConfig(BaseSchema):
    """How each statistic moves under implicitness filters and identity slices."""

    # implicit/: breakdown over the y annotations — codedness thresholds plus
    # markedness splits (implicit-only / marked-only), per H1/H2.
    run_implicit_breakdown: bool = True
    codedness_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    include_markedness_splits: bool = True
    run_slices: bool = True
    slices: list[AuthorSliceSpec] = field(default_factory=_default_slices)
    # conditional/: distinguishability WITHIN strata of a content variable, to
    # separate topic-choice from coded style (marginal vs conditional). Each
    # named prompt-level field partitions the prompts into strata.
    run_conditional: bool = True
    conditioning_variables: list[str] = field(
        default_factory=lambda: ["domain", "provenance"]
    )
    full_outputs: bool = True  # write each rerun's full section JSON + plots
    n_permutations: int = 500  # lighter permutation budget for reruns
    min_prompts_per_side: int = 8  # below this a filtered rerun is skipped


@dataclass
class AttributionalConfig(BaseSchema):
    """Token attribution of the residual probe's decision (paper §3.3.5)."""

    probe_model: str = "google/gemma-2-2b-it"  # residual probe representation
    layer_fraction: float = 0.75
    n_examples: int = 12  # top LGBTQ+-activating prompts (and as many contrasts)
    n_highlight_tokens: int = 6  # strongest-contributing tokens marked per prompt
    concept_model: str = "claude-opus-4-8"  # reads both sets, names the concept


@dataclass
class PipelineConfig(BaseSchema):
    """Full configuration for one dataset run."""

    run_name: str = ""  # "" -> dataset name
    dimensions: list[str] = field(default_factory=lambda: list(ALL_DIMENSIONS))
    include_usage_attitudes: bool = True
    include_attributional: bool = False  # residual probe; needs a causal LM
    random_seed: int = 0
    lexical: LexicalConfig = field(default_factory=LexicalConfig)
    syntactic: SyntacticConfig = field(default_factory=SyntacticConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)
    topical: TopicalConfig = field(default_factory=TopicalConfig)
    interactional: InteractionalConfig = field(default_factory=InteractionalConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    attributional: AttributionalConfig = field(default_factory=AttributionalConfig)
    explorations: ExplorationsConfig = field(default_factory=ExplorationsConfig)

    def validate(self) -> None:
        unknown = [d for d in self.dimensions if d not in ALL_DIMENSIONS]
        if unknown:
            raise ValueError(f"Unknown dimensions {unknown}; valid: {ALL_DIMENSIONS}")
