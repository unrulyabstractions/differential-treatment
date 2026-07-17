"""The fixed interactional facets (paper 3.3.6): how a prompt engages the chatbot.

Every prompt receives exactly one option per facet, so any two prompt sets are
described in the same closed option space and their per-facet distributions are
directly comparable. Descriptions are one-sentence behavioral definitions,
written to be usable verbatim for zero-shot matching by either backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema


@dataclass
class FacetOption(BaseSchema):
    """One option of one interactional facet."""

    facet: str
    option: str
    description: str


INTERACTION_FACETS: dict[str, list[FacetOption]] = {
    "speech_act": [
        FacetOption(
            "speech_act", "information_seeking", "asks for facts or explanations"
        ),
        FacetOption("speech_act", "advice_seeking", "asks what the author should do"),
        FacetOption(
            "speech_act",
            "emotional_disclosure",
            "shares feelings or personal struggles seeking support",
        ),
        FacetOption(
            "speech_act",
            "venting",
            "expresses frustration without asking for anything",
        ),
    ],
    "disclosure_depth": [
        FacetOption("disclosure_depth", "none", "no personal information"),
        FacetOption(
            "disclosure_depth", "factual", "shares life facts or circumstances"
        ),
        FacetOption(
            "disclosure_depth",
            "emotional",
            "shares inner feelings, fears, or vulnerabilities",
        ),
    ],
    "anthropomorphization": [
        FacetOption(
            "anthropomorphization", "impersonal", "treats the assistant as a tool"
        ),
        FacetOption(
            "anthropomorphization",
            "personified",
            "addresses the assistant like a person — greetings, direct 'you', "
            "asking its opinions or care",
        ),
    ],
}

FACET_NAMES: list[str] = list(INTERACTION_FACETS)
