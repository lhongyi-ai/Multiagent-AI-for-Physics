from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvaluationSubject = Literal["initial", "reviewed", "evolution_round_1", "evolution_round_2", "final"]


class RubricScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dimension: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=10.0)
    rationale: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    hypothesis_id: str = Field(min_length=1)
    round_label: EvaluationSubject
    evaluator: str = "deterministic-pilot-evaluator"
    evaluator_type: str = "mock_rule_based"
    rubric_version: str = "pilot-v1"
    scores: list[RubricScore] = Field(min_length=1)
    aggregate_score: float = Field(ge=0.0, le=10.0)
    rationale: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)
    model_metadata: dict[str, str] = Field(default_factory=dict)
    score_kind: Literal["absolute", "pairwise"] = "absolute"
    evaluated_at: datetime


class RoundEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    round_label: EvaluationSubject
    records: list[EvaluationRecord] = Field(default_factory=list)
    mean_scores: dict[str, float] = Field(default_factory=dict)


class RoundComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_id: str = Field(min_length=1)
    baseline_round: EvaluationSubject = "initial"
    final_round: EvaluationSubject = "final"
    score_changes_by_dimension: dict[str, float] = Field(default_factory=dict)
    citation_coverage: dict[str, float] = Field(default_factory=dict)
    unsupported_claim_count: dict[str, int] = Field(default_factory=dict)
    hypothesis_diversity: dict[str, float] = Field(default_factory=dict)
    duplicate_hypotheses: list[list[str]] = Field(default_factory=list)
    prediction_specificity: dict[str, float] = Field(default_factory=dict)
    falsification_plan_quality: dict[str, float] = Field(default_factory=dict)
    final_lineage: dict[str, list[str]] = Field(default_factory=dict)
    repaired_or_rejected: dict[str, list[str]] = Field(default_factory=dict)
    evaluator_self_preference_note: str
    generated_at: datetime


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    schema_version: str = "v1"
    offline_mode: bool = True
    live_network_enabled: bool = False
    live_model_enabled: bool = False
    model_mode: str = "mock"
    model_provider: str = "mock"
    sanitized_model_base_url: str | None = None
    requested_model: str | None = None
    returned_models: list[str] = Field(default_factory=list)
    literature_mode: str = "fixture"
    model_call_budget: int | None = None
    model_usage: dict[str, Any] = Field(default_factory=dict)
    provider_failures: int = 0
    structured_output_failures: int = 0
    repair_attempts: int = 0
    run_status: str = "complete"
    project_version: str = "v1"
    rubric_version: str = "pilot-v1"
    artifact_schema_version: str = "v1"
    created_at: datetime
    completed_at: datetime | None = None
    artifacts: list[str] = Field(default_factory=list)
