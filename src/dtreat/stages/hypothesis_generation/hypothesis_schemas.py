"""Schemas for hypothesized axes of differential treatment."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtreat.common.base_schema import BaseSchema


@dataclass
class HypothesisAxis(BaseSchema):
    """One interpretable axis: a yes/no question askable of any response
    (the Lambda_j dimensions of the behavior characterization, §3.2).

    The rubric sharpens judging: what counts as YES, what does not, edge
    cases. Judges see it; it never mentions communities."""

    axis_id: str
    question: str
    rationale: str = ""
    rubric: str = ""
    source: str = "helper"  # primary generation method (first proposer)
    sources: list[str] = field(default_factory=list)  # ALL methods that proposed it


@dataclass
class HypothesisSet(BaseSchema):
    """Stage-2 output: the axes every response will be scored along."""

    deployment_context: str
    helper_model: str
    axes: list[HypothesisAxis] = field(default_factory=list)
    raw_helper_reply: str = ""

    def axis_pairs(self) -> list[tuple[str, str]]:
        """(axis_id, question) pairs in stable order, for the judge protocol."""
        return [(axis.axis_id, axis.question) for axis in self.axes]

    def axis_ids(self) -> list[str]:
        return [axis.axis_id for axis in self.axes]

    def axis_rubrics(self) -> dict[str, str]:
        """axis_id -> rubric, only for axes that have one."""
        return {axis.axis_id: axis.rubric for axis in self.axes if axis.rubric.strip()}
