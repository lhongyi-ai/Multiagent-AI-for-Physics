from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coscientist.schemas.scholarly import ProjectLiteratureConfig
from coscientist.schemas.v15b import V15BConfig
from coscientist.schemas.v15c import V15CConfig


TaskType = Literal[
    "explanation",
    "prediction",
    "discovery",
    "proof",
    "design",
    "optimization",
    "identification",
    "reproduction",
    "anomaly_investigation",
]


class ResearchProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    task_type: TaskType
    background: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    known_observations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    excluded_directions: list[str] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)
    desired_output: str = Field(min_length=1)
    evaluation_criteria: list[str] = Field(min_length=1)
    validation_constraints: list[str] = Field(default_factory=list)
    maximum_literature_budget: int = Field(ge=0)
    maximum_model_call_budget: int = Field(gt=0)
    maximum_evolution_rounds: int = Field(ge=0, le=5)
    literature_fixture_path: str | None = None
    rubric_path: str | None = None
    literature: ProjectLiteratureConfig = Field(default_factory=ProjectLiteratureConfig)
    v15b: V15BConfig = Field(default_factory=V15BConfig)
    v15c: V15CConfig = Field(default_factory=V15CConfig)
    created_at: datetime
    schema_version: str = "v1"

    @field_validator("research_question")
    @classmethod
    def question_must_be_substantive(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("research_question must be substantive")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> ResearchProjectSpec:
        overlap = set(self.constraints).intersection(self.excluded_directions)
        if overlap:
            raise ValueError(f"constraints and excluded_directions overlap: {sorted(overlap)}")
        if self.maximum_evolution_rounds > self.maximum_model_call_budget:
            raise ValueError("maximum_evolution_rounds cannot exceed maximum_model_call_budget")
        return self
