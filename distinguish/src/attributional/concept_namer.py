"""Claude Opus 4.8 reads the two attributed prompt sets and names the concept.

The activating set (most target-leaning by the residual probe) and the contrasting
set (most baseline-leaning) are shown with each prompt's strongest-contributing
tokens marked «like_this». The model names the single concept the highlighted
tokens have in common on the target side. Group names are taken from the dataset,
so this works for any target/baseline contrast, not just the paper's LGBTQ+ one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anthropic import Anthropic

from src.common.logging_utils import log

if TYPE_CHECKING:
    from src.attributional.attributional_dimension import AttributedPrompt

_SYSTEM = """\
You are analysing a linear probe trained on language-model residual streams to
distinguish two groups of prompts: a TARGET group ({target}) from a BASELINE
group ({baseline}). You are shown the prompts the probe scores MOST toward the
target class and the prompts it scores MOST toward the baseline. In each prompt
the tokens that contributed most to the probe's decision are marked with
«guillemets».

Name the SINGLE discriminating concept that the highlighted tokens on the TARGET
side share and that the baseline side lacks — the concept the probe has learned.
Be specific and grounded in the highlighted tokens, not a generic label.

Reply with ONLY a JSON object, no code fences:
{{"concept": "<a short noun phrase>", "explanation": "<one or two sentences citing the highlighted tokens>"}}"""


def _render(prompts: list[AttributedPrompt], limit: int = 900) -> str:
    lines = []
    for p in prompts:
        toks = ", ".join(t.token for t in p.top_tokens[:5])
        lines.append(f"- {p.highlighted[:limit]}   [top tokens: {toks}]")
    return "\n".join(lines)


def name_concept(
    activating: list[AttributedPrompt],
    contrasting: list[AttributedPrompt],
    model: str,
    target_label: str = "the target group",
    baseline_label: str = "the baseline group",
) -> tuple[str, str, list[str]]:
    """Return (concept, explanation, skipped). Skipped is non-empty on failure."""
    if not activating or not contrasting:
        return "", "", ["attributional (no prompts to attribute)"]
    user = (
        f"PROMPTS THE PROBE SCORES TOWARD THE TARGET ({target_label}) — highlighted "
        f"tokens drove the decision:\n{_render(activating)}\n\n"
        f"PROMPTS THE PROBE SCORES TOWARD THE BASELINE ({baseline_label}):\n"
        f"{_render(contrasting)}\n\n"
        "Name the discriminating concept."
    )
    try:
        client = Anthropic()
        log(f"attributional: naming the concept via {model}")
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=_SYSTEM.format(target=target_label, baseline=baseline_label),
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```")
        parsed = json.loads(cleaned.strip())
        return str(parsed["concept"]), str(parsed.get("explanation", "")), []
    except Exception as error:  # report, never crash the pipeline
        log(f"attributional: concept naming failed ({error})")
        return "", "", [f"{model} ({type(error).__name__})"]
