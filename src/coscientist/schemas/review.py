from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Recommendation = Literal["keep", "repair", "reject"]


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: str = Field(min_length=1)
    fatal_flaws: list[str] = Field(default_factory=list)
    nonfatal_weaknesses: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    novelty_concerns: list[str] = Field(default_factory=list)
    suggested_repairs: list[str] = Field(default_factory=list)
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviews: list[Review] = Field(min_length=1)
