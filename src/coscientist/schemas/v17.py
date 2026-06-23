from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProblemType = Literal[
    "mechanism_discovery",
    "counterexample_search",
    "candidate_construction",
    "model_selection",
    "experiment_design",
    "inverse_problem",
]
CandidateType = Literal[
    "hypothesis",
    "counterexample",
    "construction",
    "mechanistic_model",
    "experiment_plan",
    "analytic_derivation",
    "numerical_candidate",
]
CandidateStatus = Literal[
    "proposed",
    "cheap_filter_failed",
    "awaiting_grounding",
    "awaiting_verification",
    "partially_verified",
    "falsified",
    "promising",
    "strong_verification_passed",
    "expert_review_required",
    "expert_validated",
    "archived",
]
SearchTaskType = Literal[
    "formalize_problem",
    "generate_candidates",
    "cheap_filter",
    "retrieve_evidence",
    "novelty_check",
    "verify_candidate",
    "search_counterexample",
    "compare_candidates",
    "evolve_candidate",
    "repair_candidate",
    "combine_candidates",
    "cross_domain_transfer",
    "summarize_search_state",
    "request_expert_review",
]
TaskStatus = Literal["pending", "ready", "running", "completed", "failed", "cancelled", "blocked"]
SearchStrategyName = Literal[
    "mainstream_extension",
    "counterexample_search",
    "assumption_relaxation",
    "extreme_case_search",
    "repair_failed_candidate",
    "combine_partial_solutions",
    "cross_domain_transfer",
]
VerifierStage = Literal["cheap", "standard", "strong"]
VerifierVerdict = Literal["pass", "partial", "fail", "inconclusive", "error"]


class ProblemConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    constraint_id: str
    description: str
    hard: bool = True
    provenance: list[str] = Field(default_factory=list)


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    criterion_id: str
    description: str
    measurable: bool = True


class FailureCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    criterion_id: str
    description: str
    severity: Literal["low", "medium", "high"] = "medium"


class ObservableTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observable_id: str
    description: str
    expected_direction: str | None = None
    units: str | None = None


class ScientificProblem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    problem_id: str
    title: str
    precise_statement: str
    problem_type: ProblemType
    scientific_domain: str
    candidate_types: list[CandidateType] = Field(min_length=1)
    known_constraints: list[ProblemConstraint] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    failure_criteria: list[FailureCriterion] = Field(default_factory=list)
    observable_targets: list[ObservableTarget] = Field(default_factory=list)
    accepted_evidence_types: list[str] = Field(default_factory=list)
    excluded_claims: list[str] = Field(default_factory=list)
    known_baselines: list[str] = Field(default_factory=list)
    corpus_scope: str
    human_notes: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class ProblemFormalization(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    problem_id: str
    created_at: datetime
    formal_statement: str
    normalized_constraints: list[str] = Field(default_factory=list)
    search_space_summary: str
    evaluator_only_fields_present: bool = False
    validation_status: str = "validated"


class CandidateSolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    candidate_id: str
    problem_id: str
    candidate_type: CandidateType
    title: str
    summary: str
    formal_representation: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    construction_or_model: str | None = None
    derivation_steps: list[str] = Field(default_factory=list)
    predicted_observables: list[str] = Field(default_factory=list)
    falsification_conditions: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    root_candidate_id: str | None = None
    lineage_depth: int = Field(default=0, ge=0)
    generation_strategy: SearchStrategyName
    linked_evidence_ids: list[str] = Field(default_factory=list)
    linked_cluster_ids: list[str] = Field(default_factory=list)
    verification_result_ids: list[str] = Field(default_factory=list)
    failure_reason_ids: list[str] = Field(default_factory=list)
    novelty_status: Literal["unknown", "known", "possibly_novel", "duplicate"] = "unknown"
    scientific_status: CandidateStatus = "proposed"
    component_scores: dict[str, float] = Field(default_factory=dict)
    aggregate_search_score: float = Field(default=0.0, ge=0.0, le=1.0)
    structured_model: dict[str, Any] = Field(default_factory=dict)
    created_step: int = Field(ge=0)
    updated_step: int = Field(ge=0)
    provenance: list[str] = Field(default_factory=list)


class CandidateStatusEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    candidate_id: str
    from_status: CandidateStatus | None = None
    to_status: CandidateStatus
    step: int
    reason: str


class CandidateArchiveSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    problem_id: str
    candidates: list[CandidateSolution] = Field(default_factory=list)
    status_history: list[CandidateStatusEvent] = Field(default_factory=list)
    duplicate_groups: list[list[str]] = Field(default_factory=list)
    validation_status: str = "validated"


class TaskBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    token_budget: int = Field(default=1000, ge=0)
    model_call_budget: int = Field(default=0, ge=0)
    verifier_call_budget: int = Field(default=1, ge=0)


class SearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    task_id: str
    problem_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    task_type: SearchTaskType
    priority: int = Field(default=100, ge=0)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    retry_count: int = Field(default=0, ge=0)
    maximum_retries: int = Field(default=1, ge=0)
    budget: TaskBudget = Field(default_factory=TaskBudget)
    created_step: int = Field(ge=0)
    started_step: int | None = None
    completed_step: int | None = None
    result_artifact_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    task_id: str
    status: TaskStatus
    candidate_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class SearchStrategyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v17"
    strategy: SearchStrategyName
    candidates_generated: int = 0
    cheap_filter_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_yield: float = Field(default=0.0, ge=0.0, le=1.0)
    score_improvement: float = 0.0
    token_cost: int = 0
    model_calls: int = 0
    surviving_lineages: int = 0


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    mode: Literal["beam"] = "beam"
    max_steps: int = Field(default=12, ge=1, le=200)
    beam_width: int = Field(default=4, ge=1, le=50)
    max_candidates_total: int = Field(default=40, ge=1, le=500)
    max_children_per_candidate: int = Field(default=3, ge=0, le=20)
    max_lineage_depth: int = Field(default=5, ge=0, le=50)
    preserve_diverse_clusters: bool = True
    preserve_counterexample_branch: bool = True
    tournament_max_candidates: int = Field(default=6, ge=0, le=50)
    tournament_max_comparisons: int = Field(default=8, ge=0, le=200)
    tournament_debate_turns: int = Field(default=1, ge=1, le=5)
    tournament_ranking_mode: Literal["bounded", "elo", "bradley_terry"] = "bounded"
    tournament_initial_rating: float = Field(default=1000.0, ge=0.0)
    tournament_k_factor: float = Field(default=24.0, ge=0.0, le=100.0)
    tournament_initial_uncertainty: float = Field(default=350.0, ge=0.0)
    tournament_close_match_gap: float = Field(default=35.0, ge=0.0)
    tournament_max_deep_comparisons: int = Field(default=2, ge=0, le=50)
    adaptive_compute_enabled: bool = True
    preserve_minimum_contrarian_branches: int = Field(default=1, ge=0, le=10)
    role_model_routing: dict[str, str] = Field(default_factory=dict)
    plateau_window: int = Field(default=3, ge=1, le=20)
    plateau_minimum_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    token_budget: int = Field(default=12000, ge=0)
    model_call_budget: int = Field(default=0, ge=0)
    verifier_call_budget: int = Field(default=80, ge=0)


class VerifierRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    verifier_id: str
    candidate_id: str
    stage: VerifierStage
    capability: str


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    verifier_id: str
    verifier_version: str
    verifier_result_id: str
    candidate_id: str
    stage: VerifierStage
    verdict: VerifierVerdict
    score: float = Field(ge=0.0, le=1.0)
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    counterexample_found: bool = False
    reproducible_output_artifact_ids: list[str] = Field(default_factory=list)
    runtime_ms: float = Field(default=0.0, ge=0.0)
    tool_calls: int = Field(default=0, ge=0)
    provenance: list[str] = Field(default_factory=list)


class BeamSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    step: int
    selected_candidate_ids: list[str]
    component_formula: str
    component_inputs: dict[str, dict[str, float]]
    validation_status: str = "validated"


class TournamentComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    comparison_id: str
    candidate_a_id: str
    candidate_b_id: str
    winner_id: str | None = None
    rationale: str
    single_turn: bool = True


class SearchCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v17"
    project_id: str
    problem_id: str
    run_id: str
    random_seed: int
    current_step: int
    queue: list[SearchTask]
    completed_task_ids: list[str]
    archive: CandidateArchiveSnapshot
    strategy_metrics: list[SearchStrategyMetrics]
    verifier_results: list[VerifierResult]
    grounding_references: list[str] = Field(default_factory=list)
    budgets_spent: dict[str, int] = Field(default_factory=dict)
    plateau_history: list[dict[str, float | int | str]] = Field(default_factory=list)
    project_hash: str
    corpus_hash: str
    created_at: datetime
    resume_count: int = 0
    validation_status: str = "validated"


class DiscoveryProject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v17"
    project_id: str
    title: str
    model_mode: Literal["mock", "live"] = "mock"
    literature_mode: Literal["fixture", "existing"] = "fixture"
    grounding_mode: Literal["strict"] = "strict"
    random_seed: int = 11
    problem: ScientificProblem
    initial_candidates: list[CandidateSolution] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evaluator_only_ground_truth: dict[str, Any] = Field(default_factory=dict)
    search: SearchConfig = Field(default_factory=SearchConfig)
    enabled_verifiers: list[str] = Field(default_factory=lambda: [
        "schema_constraint",
        "logical_consistency",
        "evidence_consistency",
        "counterexample_hook",
        "experimental_consistency",
        "materials_formula",
    ])
    created_at: datetime

    @model_validator(mode="after")
    def validate_offline_defaults(self) -> DiscoveryProject:
        if self.model_mode != "mock":
            raise ValueError("V1.7 discovery projects default to mock model mode unless a future live runner explicitly opts in")
        return self
