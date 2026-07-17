"""Approximate per-token pricing for cost tracking and pre-run estimation.

Prices are USD per million tokens (input, output), matched by model-name
prefix, first match wins. Edit freely — this table is advisory, used for
`dtreat estimate-cost` and run summaries, never for correctness.
"""

from __future__ import annotations

from .chat_types import ChatUsage

PRICE_TABLE: list[tuple[str, float, float]] = [
    ("mock:", 0.0, 0.0),
    ("claude-opus", 15.0, 75.0),
    ("claude-sonnet", 3.0, 15.0),
    ("claude-haiku", 1.0, 5.0),
    ("claude-3-5-haiku", 0.8, 4.0),
    ("gpt-5-nano", 0.05, 0.4),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-4o-mini", 0.15, 0.6),
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4.1-mini", 0.4, 1.6),
    ("gpt-4.1", 2.0, 8.0),
    ("o3", 2.0, 8.0),
    ("o4-mini", 1.1, 4.4),
    ("gemini-3.5-flash", 0.3, 2.5),
    ("gemini-3.1-flash-lite", 0.1, 0.4),
    ("gemini-3-flash", 0.3, 2.5),
    ("gemini-2.5-flash-lite", 0.1, 0.4),
    ("gemini-2.5-flash", 0.3, 2.5),
    ("gemini-2.5-pro", 1.25, 10.0),
    ("gemini-2.0-flash", 0.1, 0.4),
]


def price_for_model(model: str) -> tuple[float, float] | None:
    """(input, output) USD/MTok for the first matching prefix, else None."""
    # provider-prefixed specs ("anthropic:claude-x") price by the bare model
    model = model.split(":", 1)[1] if model.split(":", 1)[0] in ("anthropic", "openai", "google") else model
    for prefix, price_in, price_out in PRICE_TABLE:
        if model.startswith(prefix):
            return (price_in, price_out)
    return None


def is_priced_model(model: str) -> bool:
    return price_for_model(model) is not None


def cost_usd(model: str, usage: ChatUsage) -> float:
    """Estimated cost of one call; 0.0 for unpriced models."""
    prices = price_for_model(model)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
