"""One dataset = two Parquet tables + a manifest, scalable to many samples.

    data/{dataset}/
    ├── dataset.json       # manifest: cohorts + comparisons (this schema)
    ├── prompts.parquet    # one row per prompt: x, y, c + cohort + author_id
    └── authors.parquet    # one row per author: z, d + cohort

Cohorts are labeled subsets (e.g. target / baseline / target_twin); comparisons
name a target and baseline cohort. In-memory analysis still uses PromptSet —
this module builds them from table slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.common.base_schema import BaseSchema
from src.common.dataset_annotations import (
    AuthorDemographics,
    AuthorIdentity,
    InteractionContext,
    PromptLabels,
)
from src.common.file_io import load_json
from src.common.prompt_set_schema import AuthorProfile, PromptRecord, PromptSet

_PROMPT_COLUMNS = [
    "prompt_id",
    "author_id",
    "cohort",
    "text",
    "lgbtq",
    "markedness",
    "codedness",
    "topic_id",
    "domain",
    "provenance",
    "adoption",
    "general_freq",
    "llm_freq",
    "professional_freq",
    "aversion",
    "satisfaction",
]
_AUTHOR_COLUMNS = [
    "author_id",
    "cohort",
    "transgender",
    "gender",
    "orientation",
    "pronouns",
    "race",
    "age",
    "disability",
    "education",
    "income",
]


@dataclass
class CohortSpec(BaseSchema):
    """One labeled subset of the dataset."""

    name: str
    group: str  # target | baseline
    display_name: str = ""
    description: str = ""


@dataclass
class ComparisonSpec(BaseSchema):
    """One comparison the dataset supports."""

    name: str  # e.g. "target_vs_baseline"; becomes the run subdirectory
    target_cohort: str
    baseline_cohort: str
    expectation: str = ""  # "distinguishable" | "null" | ""
    explorations: bool = True  # run implicit breakdown + slices for this pair
    expected_accuracy: float = 0.0  # published separability target (0 = none)
    notes: str = ""  # validation role + caveats (e.g. label circularity)


@dataclass
class DatasetManifest(BaseSchema):
    """dataset.json: describes the tables and the intended comparisons."""

    name: str
    description: str = ""
    cohorts: list[CohortSpec] = field(default_factory=list)
    comparisons: list[ComparisonSpec] = field(default_factory=list)


class PromptDataset:
    """A loaded dataset: manifest + tables, and PromptSet builders."""

    def __init__(
        self, manifest: DatasetManifest, prompts: pd.DataFrame, authors: pd.DataFrame
    ):
        self.manifest = manifest
        self.prompts = prompts
        self.authors = authors

    @classmethod
    def load(cls, dataset_dir: Path) -> PromptDataset:
        dataset_dir = Path(dataset_dir)
        manifest = DatasetManifest.from_dict(load_json(dataset_dir / "dataset.json"))
        prompts = pd.read_parquet(dataset_dir / "prompts.parquet")
        authors = pd.read_parquet(dataset_dir / "authors.parquet")
        dataset = cls(manifest, prompts, authors)
        dataset.validate()
        return dataset

    def validate(self) -> None:
        for column in _PROMPT_COLUMNS:
            if column not in self.prompts.columns:
                raise ValueError(f"prompts.parquet missing column '{column}'")
        for column in _AUTHOR_COLUMNS:
            if column not in self.authors.columns:
                raise ValueError(f"authors.parquet missing column '{column}'")
        cohort_names = {c.name for c in self.manifest.cohorts}
        table_cohorts = set(self.prompts["cohort"]) | set(self.authors["cohort"])
        if not table_cohorts <= cohort_names:
            raise ValueError(
                f"Unknown cohorts in tables: {table_cohorts - cohort_names}"
            )
        if self.prompts["prompt_id"].duplicated().any():
            raise ValueError("Duplicate prompt_id in prompts.parquet")
        known_authors = set(self.authors["author_id"])
        missing = set(self.prompts["author_id"]) - known_authors
        if missing:
            raise ValueError(
                f"Prompts reference unknown authors: {sorted(missing)[:5]}"
            )
        for comparison in self.manifest.comparisons:
            for cohort in (comparison.target_cohort, comparison.baseline_cohort):
                if cohort not in cohort_names:
                    raise ValueError(
                        f"Comparison '{comparison.name}' references unknown cohort '{cohort}'"
                    )

    def cohort_spec(self, name: str) -> CohortSpec:
        return next(c for c in self.manifest.cohorts if c.name == name)

    def prompt_set(
        self,
        cohort: str,
        author_mask: pd.Series | None = None,
        prompt_mask: pd.Series | None = None,
    ) -> PromptSet:
        """Build the in-memory PromptSet for a cohort, optionally filtered.

        Masks are boolean Series aligned to the full tables; rows outside the
        cohort are ignored. Authors left without prompts are dropped.
        """
        spec = self.cohort_spec(cohort)
        authors = self.authors[self.authors["cohort"] == cohort]
        prompts = self.prompts[self.prompts["cohort"] == cohort]
        if author_mask is not None:
            authors = authors[author_mask.reindex(authors.index, fill_value=False)]
            prompts = prompts[prompts["author_id"].isin(set(authors["author_id"]))]
        if prompt_mask is not None:
            prompts = prompts[prompt_mask.reindex(prompts.index, fill_value=False)]
        authors = authors[authors["author_id"].isin(set(prompts["author_id"]))]
        return PromptSet(
            name=spec.name,
            group=spec.group,
            display_name=spec.display_name,
            description=spec.description,
            authors=[_author_profile(row) for row in authors.to_dict("records")],
            prompts=[_prompt_record(row) for row in prompts.to_dict("records")],
        )


def _as_list(value) -> list[str]:
    return [str(v) for v in value] if value is not None else []


def _author_profile(row: dict) -> AuthorProfile:
    return AuthorProfile(
        author_id=row["author_id"],
        identity=AuthorIdentity(
            transgender=str(row["transgender"]),
            gender=_as_list(row["gender"]),
            orientation=_as_list(row["orientation"]),
            pronouns=_as_list(row["pronouns"]),
        ),
        demographics=AuthorDemographics(
            race=_as_list(row["race"]),
            age=str(row["age"]),
            disability=str(row["disability"]),
            education=str(row["education"]),
            income=str(row["income"]),
        ),
    )


def _prompt_record(row: dict) -> PromptRecord:
    return PromptRecord(
        prompt_id=row["prompt_id"],
        author_id=row["author_id"],
        text=row["text"],
        labels=PromptLabels(
            lgbtq=int(row["lgbtq"]),
            markedness=int(row["markedness"]),
            codedness=float(row["codedness"]),
        ),
        context=InteractionContext(
            topic_id=int(row["topic_id"]),
            domain=str(row["domain"]),
            provenance=str(row["provenance"]),
            adoption=int(row["adoption"]),
            general_freq=int(row["general_freq"]),
            llm_freq=int(row["llm_freq"]),
            professional_freq=int(row["professional_freq"]),
            aversion=int(row["aversion"]),
            satisfaction=int(row["satisfaction"]),
        ),
    )
