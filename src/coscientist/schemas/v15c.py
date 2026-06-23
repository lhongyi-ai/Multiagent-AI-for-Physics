from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RecommendationActionType = Literal[
    "increase_generator_strategy",
    "decrease_generator_strategy",
    "explore_underrepresented_cluster",
    "suppress_duplicate_cluster",
    "add_targeted_search_query",
    "repair_hypothesis",
    "branch_hypothesis",
    "combine_hypotheses",
    "hold_hypothesis",
    "request_more_evidence",
    "request_human_review",
    "preserve_current_strategy",
]
RecommendationPriority = Literal["low", "medium", "high", "critical"]
RecommendationTargetType = Literal["strategy", "cluster", "hypothesis", "evidence", "verification", "metric", "project"]
RecommendationStatus = Literal["proposed", "validated", "rejected", "planned", "executed", "advisory_only"]
RecommendationDecisionValue = Literal[
    "accepted",
    "accepted_with_modification",
    "rejected_invalid_reference",
    "rejected_constraint_violation",
    "rejected_budget",
    "rejected_conflict",
    "rejected_duplicate",
    "advisory_only",
]
FeedbackMode = Literal["advisory", "controlled_feedback"]
ValidationStatus = Literal["unvalidated", "validated", "invalid", "partially_validated"]
ExecutionStatus = Literal["executed", "recorded_noop", "skipped", "failed"]
FeedbackOutcomeLabel = Literal["improved", "mixed", "no_material_change", "regressed", "insufficient_evidence"]


class ControlledFeedbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = False
    require_explicit_enable: bool = True
    max_generator_reallocation: int = Field(default=4, ge=0, le=20)
    max_targeted_search_queries: int = Field(default=5, ge=0, le=50)
    max_repairs: int = Field(default=3, ge=0, le=50)
    max_branches: int = Field(default=3, ge=0, le=50)
    max_combinations: int = Field(default=2, ge=0, le=25)
    max_holds: int = Field(default=5, ge=0, le=50)
    max_recommendations: int = Field(default=10, ge=1, le=100)
    max_cluster_resource_fraction: float = Field(default=0.6, ge=0.1, le=1.0)
    preserve_strategy_diversity: bool = True
    require_valid_artifact_references: bool = True
    allow_single_cluster_dominance: bool = False


class FeedbackABConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    random_seed: int = 11
    control_label: str = "control_advisory"
    treatment_label: str = "treatment_controlled_feedback"
    max_evolution_rounds: int = Field(default=1, ge=1, le=3)
    force_mock_model: bool = True
    force_offline_literature: bool = True


class V15CConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    controlled_feedback: ControlledFeedbackConfig = Field(default_factory=ControlledFeedbackConfig)
    ab_experiment: FeedbackABConfig = Field(default_factory=FeedbackABConfig)


class MetaReviewRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15c"
    recommendation_id: str
    round_number: int = Field(ge=0)
    action_type: RecommendationActionType
    priority: RecommendationPriority = "medium"
    target_type: RecommendationTargetType
    target_ids: list[str] = Field(default_factory=list)
    requested_change: str
    rationale: str
    source_hypothesis_ids: list[str] = Field(default_factory=list)
    source_cluster_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_verification_ids: list[str] = Field(default_factory=list)
    source_metric_names: list[str] = Field(default_factory=list)
    expected_effect: str
    confidence: float = Field(ge=0.0, le=1.0)
    constraints: list[str] = Field(default_factory=list)
    status: RecommendationStatus = "proposed"


class RecommendationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15c"
    recommendation_id: str
    decision: RecommendationDecisionValue
    reason: str
    validator_checks: list[str] = Field(default_factory=list)
    normalized_action: dict[str, str | int | float | bool | list[str] | list[list[str]]] = Field(default_factory=dict)
    conflicts: list[str] = Field(default_factory=list)
    budget_effect: dict[str, int] = Field(default_factory=dict)
    decided_at: datetime


class NextRoundPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15c"
    project_id: str
    run_id: str
    branch: str
    source_round: int = Field(ge=0)
    target_round: int = Field(ge=1)
    mode: FeedbackMode
    accepted_recommendation_ids: list[str] = Field(default_factory=list)
    generator_allocation: dict[str, int] = Field(default_factory=dict)
    target_clusters: list[str] = Field(default_factory=list)
    suppressed_clusters: list[str] = Field(default_factory=list)
    targeted_search_queries: list[str] = Field(default_factory=list)
    repair_hypothesis_ids: list[str] = Field(default_factory=list)
    branch_hypothesis_ids: list[str] = Field(default_factory=list)
    combine_pairs: list[list[str]] = Field(default_factory=list)
    hold_hypothesis_ids: list[str] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)
    human_review_requests: list[str] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)
    safeguards: list[str] = Field(default_factory=list)
    plan_hash: str
    validation_status: ValidationStatus = "unvalidated"

    @model_validator(mode="after")
    def validate_pairs(self) -> NextRoundPlan:
        for pair in self.combine_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("combine_pairs must contain two distinct hypothesis ids")
        return self


class FeedbackExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15c"
    target_round: int = Field(ge=1)
    recommendation_id: str
    planned_action: str
    actual_action: str
    affected_agent: str
    affected_hypothesis_ids: list[str] = Field(default_factory=list)
    before_state: dict[str, str | int | float | list[str]] = Field(default_factory=dict)
    after_state: dict[str, str | int | float | list[str]] = Field(default_factory=dict)
    execution_status: ExecutionStatus
    notes: str = ""


class FeedbackBranchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v15c"
    project_id: str
    run_id: str
    branch: str
    designation: Literal["control", "treatment"]
    source_round: int
    target_round: int
    model_mode: str
    literature_mode: str
    random_seed: int
    recommendation_count: int = Field(ge=0)
    accepted_recommendation_count: int = Field(ge=0)
    executed_action_count: int = Field(ge=0)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    validation_status: ValidationStatus = "unvalidated"
    created_at: datetime


class FeedbackABManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v15c"
    experiment_id: str
    project_id: str
    project_title: str
    created_at: datetime
    model_mode: str
    literature_mode: str
    random_seed: int
    control_run_id: str
    treatment_run_id: str
    control_branch: str
    treatment_branch: str
    shared_research_question: str
    shared_initial_hypothesis_ids: list[str]
    shared_budget: dict[str, int]
    permission_guarantees: list[str]
    configuration_hash: str
    validation_status: ValidationStatus = "unvalidated"


class FeedbackABComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15c"
    experiment_id: str
    project_id: str
    control_run_id: str
    treatment_run_id: str
    metrics: dict[str, dict[str, float | int | str]] = Field(default_factory=dict)
    deltas: dict[str, float | int] = Field(default_factory=dict)
    outcome_label: FeedbackOutcomeLabel
    outcome_rationale: str
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime
    validation_status: ValidationStatus = "unvalidated"
