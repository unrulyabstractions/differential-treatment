"""Per-prompt and per-author annotation schemas, mirroring the paper's dataset.

D = {(x_i, y_i, z_i, d_i, c_i)}: prompt text x, coarse labels y, detailed
self-reported identity z, demographics d, and interaction context c (Tables 1
and 2 of the paper). The string "*" marks a prefer-not-to-answer response; 0
marks an unrecorded ordinal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

PREFER_NOT_TO_ANSWER = "*"

DOMAINS = ["MH", "GSH", "REL"]  # mental health, gender/sexual health, relationships
PROVENANCE_VALUES = ["real", "hyp"]  # recalled real prompt vs hypothetical

# Ordinal scale labels (Table 2b). Index 0 is unused; scales are 1-based.
ATTITUDE_SCALE = [
    "",
    "very negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "very positive",
]  # A: 1-5
FREQUENCY_SCALE = [
    "",
    "very rarely",
    "~once/yr",
    "~once/mo",
    "~once/wk",
    "~once/day",
    "1-5/day",
    "5-15/day",
    ">15/day",
]  # F: 1-8


@dataclass
class PromptLabels(BaseSchema):
    """y — coarse group membership and per-prompt annotations (Table 1a)."""

    lgbtq: int  # y^lgbtq: 1 = target (LGBTQ+), 0 = baseline
    markedness: int  # y^mark: explicit identity signal in the text
    codedness: float  # y^code in [0,1]: implicit identity signal strength


@dataclass
class AuthorIdentity(BaseSchema):
    """z — detailed self-reported LGBTQ+ identity (Table 1b)."""

    transgender: str = PREFER_NOT_TO_ANSWER  # "0" | "1" | "*"
    gender: list[str] = field(
        default_factory=list
    )  # subset of woman/man/nonbinary/other
    orientation: list[str] = field(default_factory=list)
    pronouns: list[str] = field(default_factory=list)


@dataclass
class AuthorDemographics(BaseSchema):
    """d — self-reported demographics (Table 1c)."""

    race: list[str] = field(default_factory=list)  # multi-select
    age: str = PREFER_NOT_TO_ANSWER  # bracket, e.g. "25-34"
    disability: str = PREFER_NOT_TO_ANSWER  # "0" | "1" | "*"
    education: str = PREFER_NOT_TO_ANSWER  # ordinal bracket name
    income: str = PREFER_NOT_TO_ANSWER  # ordinal bracket name


@dataclass
class InteractionContext(BaseSchema):
    """c — interaction context recorded per prompt (Table 2a).

    Ordinals use 0 for "not recorded"; topic_id indexes the fixed survey
    catalog (src/topical/survey_topic_catalog.py) and domain is its group.
    """

    topic_id: int = 0  # c^top: 1-15
    domain: str = ""  # c^dom: MH | GSH | REL
    provenance: str = ""  # c^prov: real | hyp
    adoption: int = 0  # c^hist: 1-5, how long ago the author adopted chatbots
    general_freq: int = 0  # c^gen: F 1-8
    llm_freq: int = 0  # c^freq: F 1-8, chatbot use for this domain
    professional_freq: int = 0  # c^prof: F 1-8, professional help for this domain
    aversion: int = 0  # c^avr: A 1-5
    satisfaction: int = 0  # c^sat: A 1-5
