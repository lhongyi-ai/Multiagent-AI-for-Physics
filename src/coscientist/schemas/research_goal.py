from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResearchGoal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    background: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    desired_attributes: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    prohibited_methods: list[str] = Field(default_factory=list)
    max_rounds: int = Field(default=2, ge=0)
