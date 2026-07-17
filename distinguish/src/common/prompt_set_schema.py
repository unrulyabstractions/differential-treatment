"""In-memory prompt sets in the paper's format: D = {(x, y, z, d, c)}.

A PromptSet is one side of a comparison (target or baseline), built from a
dataset's Parquet tables (src/common/dataset_tables.py). Author-level
attributes (identity z, demographics d) live in the authors list; per-prompt
records carry the text x, labels y, and interaction context c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema
from src.common.dataset_annotations import (
    AuthorDemographics,
    AuthorIdentity,
    InteractionContext,
    PromptLabels,
)

GROUP_TARGET = "target"  # LGBTQ+ community
GROUP_BASELINE = "baseline"  # cis-heterosexual comparison group


@dataclass
class AuthorProfile(BaseSchema):
    """One survey respondent: identity z and demographics d."""

    author_id: str
    identity: AuthorIdentity = field(default_factory=AuthorIdentity)
    demographics: AuthorDemographics = field(default_factory=AuthorDemographics)


@dataclass
class PromptRecord(BaseSchema):
    """One prompt x with its labels y and interaction context c."""

    prompt_id: str
    author_id: str
    text: str
    labels: PromptLabels = field(default_factory=lambda: PromptLabels(0, 0, 0.0))
    context: InteractionContext = field(default_factory=InteractionContext)


@dataclass
class PromptSet(BaseSchema):
    """A named collection of prompts from one population."""

    name: str  # short slug, e.g. "target"
    group: str  # GROUP_TARGET or GROUP_BASELINE
    display_name: str = ""  # e.g. "Target (LGBTQ+)"; defaults to name
    description: str = ""
    authors: list[AuthorProfile] = field(default_factory=list)
    prompts: list[PromptRecord] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.prompts]

    @property
    def author_ids(self) -> list[str]:
        return [p.author_id for p in self.prompts]

    def validate(self) -> None:
        """Raise ValueError on structural problems."""
        if not self.prompts:
            raise ValueError(f"Prompt set '{self.name}' has no prompts")
        if self.group not in (GROUP_TARGET, GROUP_BASELINE):
            raise ValueError(f"Unknown group '{self.group}' in set '{self.name}'")
        profile_ids = {a.author_id for a in self.authors}
        if len(profile_ids) != len(self.authors):
            raise ValueError(f"Duplicate author profiles in set '{self.name}'")
        seen_prompt_ids = set()
        for record in self.prompts:
            if not record.text.strip():
                raise ValueError(f"Empty text in prompt '{record.prompt_id}'")
            if record.prompt_id in seen_prompt_ids:
                raise ValueError(f"Duplicate prompt_id '{record.prompt_id}'")
            seen_prompt_ids.add(record.prompt_id)
            if record.author_id not in profile_ids:
                raise ValueError(
                    f"Prompt '{record.prompt_id}' references unknown author "
                    f"'{record.author_id}'"
                )
            if not 0.0 <= record.labels.codedness <= 1.0:
                raise ValueError(f"codedness out of [0,1] in '{record.prompt_id}'")


def qualified_author_ids(set_a: PromptSet, set_b: PromptSet) -> list[str]:
    """Author ids made globally unique across the two sets, prompt-aligned.

    Identically named authors in different sets are different people, and the
    author-level permutation machinery requires one label per author — so ids
    are qualified with a fixed side prefix (set names alone could collide too).
    """
    return [f"a:{set_a.name}:{a}" for a in set_a.author_ids] + [
        f"b:{set_b.name}:{a}" for a in set_b.author_ids
    ]
