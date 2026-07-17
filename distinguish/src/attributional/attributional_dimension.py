"""Attributional analysis (paper §3.3.5): name the concept that most
distinguishes the target prompts by attributing the residual probe's decision to
individual tokens. The paper's motivating target is LGBTQ+ prompts, but the group
names come from the dataset, so this runs on any target/baseline contrast.

The distributional probe applies a linear head w to the mean-pooled residual
stream, so its score decomposes EXACTLY into per-token contributions
a_t = (1/T) w·h_t (h_t the residual for token t, T the prompt length). A positive
a_t pushes toward the target class, negative toward the baseline, and they sum to
the probe score — exact, not estimated. We rank prompts by that score, take the
most target-activating prompts plus a contrasting non-activating set, highlight
each prompt's strongest-contributing tokens inline, and have Claude Opus 4.8 read
both sets and name the discriminating concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.attributional.concept_namer import name_concept
from src.common.base_schema import BaseSchema
from src.common.dimension_result import DimensionVerdict
from src.common.logging_utils import log
from src.common.prompt_set_schema import qualified_author_ids
from src.distributional.c2st_linear import run_linear_c2st
from src.inference.residual_stream_extractor import ResidualStreamExtractor

if TYPE_CHECKING:  # runtime import would cycle via src.pipeline
    from src.common.prompt_set_schema import PromptSet
    from src.common.run_config import AttributionalConfig
    from src.pipeline.pipeline_context import PipelineContext

_DIMENSION_NAME = "attributional"


@dataclass
class TokenContribution(BaseSchema):
    """One token's exact contribution to the probe score."""

    token: str
    contribution: float  # a_t = (1/T) w·h_t; + toward target, - toward baseline


@dataclass
class AttributedPrompt(BaseSchema):
    """One prompt, its probe score, and its per-token attribution."""

    prompt_id: str
    cohort: str
    text: str
    probe_score: float  # sum of a_t over the prompt (toward target if > 0)
    highlighted: str  # user text with the strongest tokens marked «like_this»
    top_tokens: list[TokenContribution] = field(default_factory=list)


@dataclass
class AttributionalResult(BaseSchema):
    """The named discriminating concept plus the evidence it was read from."""

    probe_model: str
    layer_fraction: float
    probe_accuracy: float  # in-sample separability of the linear probe (~1.0)
    concept: str = ""  # Claude Opus 4.8's name for the discriminating concept
    concept_explanation: str = ""
    probe_heldout_accuracy: float = 0.5  # held-out author-grouped separability (real)
    target_label: str = ""  # dataset's target group name (framing, not hard-coded)
    baseline_label: str = ""
    activating: list[AttributedPrompt] = field(default_factory=list)  # target end
    contrasting: list[AttributedPrompt] = field(default_factory=list)  # baseline end
    skipped_variants: list[str] = field(default_factory=list)

    def to_verdicts(self) -> list[DimensionVerdict]:
        return [
            DimensionVerdict(
                dimension=_DIMENSION_NAME,
                test_name="token_attribution",
                variant=self.probe_model,
                statistic_name="probe_heldout_accuracy",
                statistic_value=self.probe_heldout_accuracy,
                p_value=None,
                significant=None,
                detail=(
                    f"concept: {self.concept} "
                    f"(held-out {self.probe_heldout_accuracy:.2f}, in-sample "
                    f"{self.probe_accuracy:.2f})"
                ),
            )
        ]


def _highlight(
    tokens: list[str], contributions: NDArray, span: tuple[int, int], n_top: int
) -> tuple[str, list[TokenContribution]]:
    """Rebuild the user text, marking the top-|a_t| content tokens «token»."""
    start, end = span
    content = list(range(start, min(end, len(tokens))))
    if not content:
        content = list(range(len(tokens)))
    ranked = sorted(content, key=lambda i: -abs(contributions[i]))[:n_top]
    mark = set(ranked)
    pieces = [f"«{tokens[i]}»" if i in mark else tokens[i] for i in content]
    top = [
        TokenContribution(token=tokens[i].strip(), contribution=float(contributions[i]))
        for i in ranked
    ]
    return "".join(pieces).strip(), top


def compute_attributional(
    target: PromptSet,
    baseline: PromptSet,
    config: AttributionalConfig,
    context: PipelineContext,
) -> AttributionalResult:
    """Fit the residual probe, decompose per token, name the concept."""
    log(f"attributional: residual probe via {config.probe_model}")
    extractor = ResidualStreamExtractor(config.probe_model, config.layer_fraction)

    records = [(p.prompt_id, target.name, p.text) for p in target.prompts]
    records += [(p.prompt_id, baseline.name, p.text) for p in baseline.prompts]
    labels = np.array([1] * len(target.prompts) + [0] * len(baseline.prompts))

    per_token, spans, token_strs, pooled = [], [], [], []
    for _, _, text in records:
        toks, hidden, span = extractor.extract_per_token(text)
        per_token.append(hidden)
        spans.append(span)
        token_strs.append(toks)
        pooled.append(hidden.mean(axis=0))
    extractor.cleanup()
    pooled = np.stack(pooled)

    # The §3.3.4 linear probe: StandardScaler -> LogisticRegression on the pooled
    # residual. Fold the scaler into an effective raw-space head so per-token
    # contributions decompose exactly.
    scaler = StandardScaler().fit(pooled)
    probe = LogisticRegression(max_iter=2000).fit(scaler.transform(pooled), labels)
    accuracy = float(probe.score(scaler.transform(pooled), labels))  # in-sample
    w_eff = probe.coef_[0] / scaler.scale_  # decision w on the raw residual

    # Held-out author-grouped separability of the SAME probe (matches the §3.3.4
    # C2ST). This is the honest measure of how real the concept is: the in-sample
    # accuracy is ~1.0 for any high-dim probe, so the plot leads with this instead.
    author_ids = qualified_author_ids(target, baseline)
    heldout = float(
        run_linear_c2st(
            pooled, labels, author_ids, 5, 1, np.random.default_rng(0)
        ).accuracy
    )

    scored = []
    for i, (pid, cohort, text) in enumerate(records):
        hidden = per_token[i]
        contributions = (hidden @ w_eff) / hidden.shape[0]  # a_t, sums to w_eff·pooled
        highlighted, top = _highlight(
            token_strs[i], contributions, spans[i], config.n_highlight_tokens
        )
        scored.append(
            AttributedPrompt(
                prompt_id=pid,
                cohort=cohort,
                text=text,
                probe_score=float(contributions.sum()),
                highlighted=highlighted,
                top_tokens=top,
            )
        )

    scored.sort(key=lambda a: -a.probe_score)
    activating = scored[: config.n_examples]
    contrasting = scored[-config.n_examples :][::-1]

    concept, explanation, skipped = name_concept(
        activating, contrasting, config.concept_model, target.label, baseline.label
    )
    return AttributionalResult(
        probe_model=config.probe_model,
        layer_fraction=config.layer_fraction,
        probe_accuracy=accuracy,
        probe_heldout_accuracy=heldout,
        concept=concept,
        concept_explanation=explanation,
        target_label=target.label,
        baseline_label=baseline.label,
        activating=activating,
        contrasting=contrasting,
        skipped_variants=skipped,
    )
