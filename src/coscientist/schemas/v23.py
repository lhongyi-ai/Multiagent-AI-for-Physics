from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AtomicClaimType = Literal[
    "definitional",
    "mathematical",
    "mechanistic",
    "empirical",
    "numerical",
    "predictive",
    "novelty",
    "experimental_feasibility",
    "parameter_plausibility",
    "data_interpretation",
]
ClaimCheckVerdict = Literal["pass", "fail", "uncertain", "contradicted", "not_applicable"]
ValidationOutcome = Literal["internally_validated", "needs_experiment", "refuted", "insufficient_evidence", "verifier_insufficient"]
AgentRoleV23 = Literal["principal_investigator", "theorist", "critic", "experimentalist", "curator", "repair_agent"]


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    agent_id: str
    title: str
    expertise: str
    goal: str
    role: AgentRoleV23
    provider: Literal["mock", "openrouter", "openai_compatible"] = "mock"
    model: str = "deterministic-mock"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=600, ge=1, le=4000)


class MeetingAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    agent_id: str
    role: AgentRoleV23
    concise_message: str
    candidate_model_ids: list[str] = Field(default_factory=list)
    proposed_hamiltonian_terms: list[str] = Field(default_factory=list)
    proposed_verifier_tasks: list[str] = Field(default_factory=list)
    proposed_held_out_predictions: list[str] = Field(default_factory=list)
    unresolved_derivation_gaps: list[str] = Field(default_factory=list)
    cited_artifact_ids: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    proposed_repairs: list[str] = Field(default_factory=list)
    changed_by_critic: bool = False


class MeetingSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    meeting_id: str
    research_question: str
    current_round: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=2, ge=1, le=8)
    status: Literal["fixture", "live", "live_connection_blocked", "failed", "completed", "stopped_no_progress"] = "fixture"
    stopping_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class MeetingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    message_id: str
    meeting_id: str
    round_number: int = Field(ge=0)
    agent_id: str
    role: AgentRoleV23
    provider: str
    model: str
    content: str
    cited_artifact_ids: list[str] = Field(default_factory=list)
    critic_influenced: bool = False
    created_at: datetime


class ProviderCallRecordV23(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    call_id: str
    meeting_id: str
    agent_id: str
    provider: str
    model: str
    remote_response_id: str | None = None
    timestamp: datetime
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_hash: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    parsing_result: Literal["parsed", "blocked", "failed"] = "parsed"
    permission_mode: Literal["fixture", "live_model", "blocked"] = "fixture"


class ClaimRepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    repair_request_id: str
    candidate_id: str
    claim_id: str
    reason: str
    requested_by: str = "frontend"
    created_at: datetime


class ClaimRepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    repair_result_id: str
    parent_candidate_id: str
    child_candidate_id: str
    claim_id: str
    changed_fields: list[str]
    invalidated_claim_ids: list[str]
    outcome: Literal["accepted", "rejected_no_op", "blocked"]
    created_at: datetime


class CandidateModelSketch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    model_id: str
    meeting_id: str
    source_message_ids: list[str]
    model_family: str
    hamiltonian_terms: list[str]
    energy_decomposition_targets: list[str]
    assumptions: list[str]
    derivation_gaps: list[str]
    status: Literal["sketch_only", "ready_for_verifier", "invalid"] = "sketch_only"


class VerifierTaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    task_id: str
    meeting_id: str
    model_id: str
    verifier_type: Literal[
        "microscopic_hamiltonian_audit",
        "mean_field_derivation",
        "free_energy_closure",
        "current_operator_consistency",
        "optical_sum_rule",
        "held_out_prediction_gate",
        "competitor_model_check",
        "data_coverage_audit",
        "peierls_gauge_coupling",
        "continuity_equation",
        "optical_sum_rule_gauge_check",
        "representation_counterexample",
        "exact_diagonalization_counterexample",
        "hellmann_feynman_diagnostic",
        "observable_classification",
        "invariant_theorem_obligation",
    ]
    required_inputs: list[str]
    required_outputs: list[str]
    blocking: bool = True
    status: Literal["queued", "blocked_missing_input", "complete"] = "queued"


class HeldOutPredictionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    prediction_id: str
    meeting_id: str
    model_id: str
    observable: str
    material_family: str
    doping_or_parameter: str
    prediction_statement: str
    preregistered_before_data: bool = True
    status: Literal["placeholder_requires_model", "preregistered", "revealed"] = "placeholder_requires_model"


class ScienceProgressDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    meeting_id: str
    outcome: Literal[
        "no_new_scientific_knowledge",
        "candidate_requires_verification",
        "ready_for_frozen_campaign",
        "insufficient_agent_output",
    ]
    repetition_score: float = Field(ge=0.0, le=1.0)
    valid_model_sketch_count: int = Field(ge=0)
    verifier_task_count: int = Field(ge=0)
    held_out_prediction_count: int = Field(ge=0)
    failure_modes: list[str]
    required_next_artifacts: list[str]
    concise_assessment: str
    created_at: datetime


class FormalScientificClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    claim_id: str
    candidate_id: str
    scoped_main_claim: str
    falsification_conditions: list[str]
    assumptions: list[str]
    provenance: list[str] = Field(default_factory=list)


class AtomicScientificClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    claim_id: str
    parent_claim_id: str
    candidate_id: str
    claim_type: AtomicClaimType
    statement: str
    load_bearing: bool
    assumptions: list[str] = Field(default_factory=list)
    falsification_conditions: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    verifier_requirements: list[str] = Field(default_factory=list)
    uncertainty: str = "not quantified"
    repairable: bool = True
    provenance: list[str] = Field(default_factory=list)


class ClaimDependency(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    parent_claim_id: str
    child_claim_id: str
    dependency_type: Literal["requires", "supports", "contradicts", "context"]
    load_bearing_path: bool = False


class ClaimDAG(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    candidate_id: str
    main_claim_id: str
    claim_ids: list[str]
    dependency_edges: list[ClaimDependency]
    cycles: list[list[str]] = Field(default_factory=list)
    missing_dependencies: list[str] = Field(default_factory=list)
    orphan_claim_ids: list[str] = Field(default_factory=list)
    load_bearing_paths: list[list[str]] = Field(default_factory=list)
    weakest_link_claim_id: str | None = None
    maturity: Literal["draft", "checkable", "blocked", "terminal"] = "draft"


class ClaimCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    request_id: str
    claim_id: str
    verifier_id: str
    required_independent: bool = True
    input_artifact_ids: list[str] = Field(default_factory=list)


class ClaimCheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    check_id: str
    claim_id: str
    verdict: ClaimCheckVerdict
    severity: Literal["load_bearing", "supporting", "context"]
    confidence: float = Field(ge=0.0, le=1.0)
    basis_artifact_ids: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)
    repairable: bool = True
    missing_information: list[str] = Field(default_factory=list)
    concise_justification: str
    verifier_name: str
    verifier_version: str = "v23"
    cached_or_recomputed: Literal["cached", "recomputed", "fixture"] = "fixture"

    @model_validator(mode="after")
    def missing_basis_degrades_to_uncertain(self) -> ClaimCheckRecord:
        if self.verdict == "pass" and not self.basis_artifact_ids:
            self.verdict = "uncertain"
            self.missing_information = [*self.missing_information, "missing basis artifact"]
        return self


class ClaimContradiction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    contradiction_id: str
    claim_id: str
    source_artifact_id: str
    severity: Literal["fatal", "major", "minor"]
    description: str
    resolved: bool = False


class IndependentCheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    independence_id: str
    claim_id: str
    check_ids: list[str] = Field(min_length=2)
    implementation_paths: list[str] = Field(min_length=2)
    shared_dependencies: list[str] = Field(default_factory=list)
    independence_limitations: list[str] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)
    reconciliation_status: Literal["reconciled", "unresolved", "not_independent"] = "reconciled"


class ValidationBlocker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    blocker_id: str
    claim_id: str | None = None
    blocker_type: Literal["contradiction", "missing_evidence", "missing_independent_check", "verifier_gap", "experiment_required", "unphysical_parameter"]
    description: str
    terminal_impact: ValidationOutcome


class ValidationVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    candidate_id: str
    terminal_status: ValidationOutcome
    selected_rule: str
    blocking_claim_ids: list[str] = Field(default_factory=list)
    blocker_ids: list[str] = Field(default_factory=list)
    rule_trace: list[str] = Field(default_factory=list)
    arbiter_summary: str = ""


class DeepValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v23"
    request_id: str
    run_id: str
    candidate_id: str
    top_k_context_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    live_model_enabled: bool = False
    live_network_enabled: bool = False
