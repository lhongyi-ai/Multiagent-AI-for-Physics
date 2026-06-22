from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


HypothesisStatus = Literal["active", "repaired", "branched", "combined", "rejected", "finalist"]
GenerationStrategy = Literal["mechanistic", "analogy", "contrarian", "minimal-explanation", "repair", "branch", "combine"]


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    core_claim: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    novelty_statement: str = Field(min_length=1)
    testable_predictions: list[str] = Field(min_length=1)
    falsification_criteria: list[str] = Field(min_length=1)
    proposed_experiments: list[str] = Field(min_length=1)
    uncertainty: float = Field(ge=0.0, le=1.0)
    generation_strategy: GenerationStrategy
    parent_ids: list[str] = Field(default_factory=list)
    version: int = Field(ge=1)
    status: HypothesisStatus = "active"
    change_summary: str | None = None


class HypothesisBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypotheses: list[Hypothesis] = Field(min_length=1)
