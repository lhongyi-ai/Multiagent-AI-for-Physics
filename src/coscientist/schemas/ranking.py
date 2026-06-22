from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HypothesisRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: str = Field(min_length=1)
    correctness: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    testability: float = Field(ge=0, le=10)
    explanatory_power: float = Field(ge=0, le=10)
    feasibility: float = Field(ge=0, le=10)
    discriminative_power: float = Field(ge=0, le=10)
    evidence_quality: float = Field(ge=0, le=10)
    impact: float = Field(ge=0, le=10)
    parsimony: float = Field(ge=0, le=10)
    weighted_total: float = Field(ge=0, le=10)
    pairwise_wins: int = Field(default=0, ge=0)
    pairwise_losses: int = Field(default=0, ge=0)
    judge_notes: list[str] = Field(default_factory=list)


class RankingBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rankings: list[HypothesisRanking] = Field(min_length=1)


PairwiseWinner = Literal["a", "b", "tie"]


class PairwiseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_a_id: str = Field(min_length=1)
    hypothesis_b_id: str = Field(min_length=1)
    winner: PairwiseWinner
    judge_notes: str = Field(min_length=1)


class PairwiseBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    comparisons: list[PairwiseComparison] = Field(default_factory=list)
