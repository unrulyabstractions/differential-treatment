"""The paper's fixed 15 survey topics (Table 2c), as data.

The topical dimension always assigns prompts into this closed catalog, so any
two prompt sets are described in the same topic space and their distributions
are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema
from src.common.dataset_annotations import DOMAINS

__all__ = ["DOMAINS", "SURVEY_TOPICS", "SurveyTopic"]


@dataclass
class SurveyTopic(BaseSchema):
    """One fixed survey topic; descriptions are verbatim from the paper."""

    topic_id: int
    domain: str  # one of DOMAINS
    short_name: str  # concise axis/report label
    description: str


SURVEY_TOPICS: list[SurveyTopic] = [
    SurveyTopic(
        1,
        "MH",
        "social anxiety",
        "isolation, anxiety, depression, or panic in social settings",
    ),
    SurveyTopic(
        2,
        "MH",
        "life transitions",
        "coping with a major life transition or shift in self-view",
    ),
    SurveyTopic(3, "MH", "starting therapy", "whether to start seeing a therapist"),
    SurveyTopic(
        4,
        "MH",
        "values vs. family",
        "values conflicting with family or community expectations",
    ),
    SurveyTopic(5, "MH", "body image", "improving body image"),
    SurveyTopic(
        6,
        "GSH",
        "gender & orientation",
        "understanding one's gender or sexual orientation",
    ),
    SurveyTopic(7, "GSH", "STI concerns", "STI symptoms, causes, or treatments"),
    SurveyTopic(
        8, "GSH", "gender-affirming care", "understanding gender-affirming care"
    ),
    SurveyTopic(
        9,
        "GSH",
        "medication side effects",
        "side effects of gender/sexual-health medications (PrEP, estrogen)",
    ),
    SurveyTopic(
        10, "GSH", "finding providers", "finding knowledgeable healthcare providers"
    ),
    SurveyTopic(
        11,
        "REL",
        "new relationship",
        "communication, intimacy, or boundaries in a new relationship",
    ),
    SurveyTopic(12, "REL", "new dynamic", "exploring a new relationship dynamic"),
    SurveyTopic(13, "REL", "communicating needs", "communicating needs to a partner"),
    SurveyTopic(
        14, "REL", "relatable media", "media featuring relationships like one's own"
    ),
    SurveyTopic(
        15, "REL", "new sexual experience", "preparing for a new sexual experience"
    ),
]
