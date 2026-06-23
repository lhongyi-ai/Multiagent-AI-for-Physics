from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RankingMode = Literal["bounded", "elo", "bradley_terry"]
ModelRole = Literal["generation", "review", "comparison", "deep_reasoning", "novelty_audit", "meta_review"]
ReproductionOutcome = Literal["reproduced", "partially_reproduced", "not_reproduced", "inconclusive"]


class CandidateRating(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: str
    rating: float = 1000.0
    uncertainty: float = Field(default=350.0, ge=0.0)
    comparisons: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)


class EloTournamentState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v20"
    ranking_mode: RankingMode
    ratings: list[CandidateRating]
    completed_pairings: list[str] = Field(default_factory=list)
    deep_comparison_pairings: list[str] = Field(default_factory=list)
    maximum_comparisons: int = Field(ge=0)
    validation_status: str = "validated"

    @model_validator(mode="after")
    def validate_pairing_limits(self) -> EloTournamentState:
        if len(set(self.completed_pairings)) != len(self.completed_pairings):
            raise ValueError("completed pairings must be unique")
        if len(self.completed_pairings) > self.maximum_comparisons:
            raise ValueError("completed pairings exceed maximum comparison budget")
        return self


class StrategyAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    strategy: str
    historical_yield: float = Field(ge=0.0, le=1.0)
    duplicate_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    falsification_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    token_budget: int = Field(ge=0)
    model_call_budget: int = Field(ge=0)
    verifier_call_budget: int = Field(ge=0)
    preserve_branch: bool = False
    rationale: str


class AdaptiveBudgetAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v20"
    project_id: str
    run_id: str
    total_token_budget: int = Field(ge=0)
    total_model_call_budget: int = Field(ge=0)
    total_verifier_call_budget: int = Field(ge=0)
    allocations: list[StrategyAllocation]
    verifier_stage_yield: dict[str, float] = Field(default_factory=dict)
    role_yield: dict[str, float] = Field(default_factory=dict)
    lineage_yield: dict[str, float] = Field(default_factory=dict)
    validation_status: str = "validated"

    @model_validator(mode="after")
    def validate_budget_bounds(self) -> AdaptiveBudgetAllocation:
        if sum(item.token_budget for item in self.allocations) > self.total_token_budget:
            raise ValueError("allocated token budget exceeds total")
        if sum(item.model_call_budget for item in self.allocations) > self.total_model_call_budget:
            raise ValueError("allocated model-call budget exceeds total")
        if sum(item.verifier_call_budget for item in self.allocations) > self.total_verifier_call_budget:
            raise ValueError("allocated verifier budget exceeds total")
        return self


class RoleModelRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    role: ModelRole
    provider: Literal["mock", "openai_compatible"] = "mock"
    model: str = "deterministic-mock"
    model_mode: Literal["mock", "live"] = "mock"
    max_context_characters: int = Field(default=12000, ge=0)
    max_output_tokens: int = Field(default=900, ge=0)
    live_permission_required: bool = False
    fallback_provider: Literal["mock"] = "mock"


class ProviderRoutingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v20"
    project_id: str
    run_id: str
    live_model_enabled: bool = False
    routes: list[RoleModelRoute]
    validation_status: str = "validated"

    @model_validator(mode="after")
    def validate_live_permissions(self) -> ProviderRoutingPlan:
        if not self.live_model_enabled:
            live_routes = [route.role for route in self.routes if route.model_mode == "live" or route.provider != "mock"]
            if live_routes:
                raise ValueError(f"live routes require explicit live_model_enabled: {live_routes}")
        return self


class ReproductionRun(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    path_id: str
    method: str
    assumptions: list[str] = Field(default_factory=list)
    output_value: float | None = None
    output_summary: str
    package_versions: dict[str, str] = Field(default_factory=dict)


class ImplementationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: str
    compared_path_ids: list[str]
    tolerance: float = Field(ge=0.0)
    absolute_difference: float | None = None
    discrepancy_summary: str


class ReproductionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v20"
    reproduction_result_id: str
    candidate_id: str
    requested_at: datetime
    runs: list[ReproductionRun] = Field(min_length=1)
    comparison: ImplementationComparison
    outcome: ReproductionOutcome
    validation_status: str = "validated"
