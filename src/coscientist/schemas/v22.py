from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AgentRole = Literal[
    "generator",
    "theory_reviewer",
    "experimental_reviewer",
    "prior_art_reviewer",
    "adversarial_reviewer",
    "evolution",
    "ranker",
    "supervisor",
    "meta_review",
]
ProviderKind = Literal["mock", "openrouter", "openai_compatible"]
ConnectionStatus = Literal["connected", "authentication_required", "rate_limited", "unavailable", "fixture_only", "failed", "blocked"]
TheoryFamily = Literal[
    "phonon_dominated_bcs",
    "correlated_hopping_hirsch_style",
    "mixed_phonon_correlated_hopping",
    "phenomenological_mixed_kernel",
    "non_hirsch_kinetic_competitor",
    "overparameterized_competitor",
    "null_or_inadequate_model",
]
ObservableName = Literal[
    "tc",
    "gap",
    "symmetry",
    "isotope_coefficient",
    "doping_dependence",
    "electron_hole_asymmetry",
    "kinetic_energy_change",
    "interaction_energy_change",
    "correlated_hopping_energy_change",
    "condensation_energy",
    "optical_partial_sum",
    "superfluid_weight",
    "penetration_depth",
    "specific_heat",
    "resistivity",
    "pressure",
]


class PerRoleModelRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    role: AgentRole
    provider: ProviderKind = "mock"
    model: str = "deterministic-mock"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=900, ge=1, le=8000)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)


class LiveRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    live_model_enabled: bool = False
    live_network_enabled: bool = False
    max_total_model_calls: int = Field(default=20, ge=0, le=1000)
    max_total_tokens: int = Field(default=45000, ge=0)
    max_estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    routes: list[PerRoleModelRoute] = Field(default_factory=list)

    @model_validator(mode="after")
    def forbid_live_routes_without_permission(self) -> LiveRuntimeConfig:
        if not self.live_model_enabled:
            live_routes = [route.role for route in self.routes if route.provider != "mock"]
            if live_routes:
                raise ValueError(f"live model routes require live_model_enabled: {live_routes}")
        return self


class ProviderConnectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    provider: str
    provider_type: Literal["scholarly", "materials", "model"]
    fixture_status: ConnectionStatus = "fixture_only"
    live_status: ConnectionStatus = "blocked"
    authentication_status: Literal["not_required", "configured", "missing", "unknown"] = "unknown"
    record_count: int = Field(default=0, ge=0)
    request_parameters: dict[str, Any] = Field(default_factory=dict)
    snapshot_path: str | None = None
    snapshot_sha256: str | None = None
    retrieved_at: str | None = None
    error: str | None = None


class LiveModelSmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    role: AgentRole
    status: Literal["ok", "blocked", "failed"]
    summary: str
    safe_to_continue: bool


class AgentDialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    turn_id: str
    candidate_id: str | None = None
    role: AgentRole
    provider: ProviderKind = "mock"
    model: str = "deterministic-mock"
    status: Literal["completed", "skipped", "blocked", "schema_failed"] = "completed"
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    summary: str
    objections: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, ge=0.0)


class MicroscopicHamiltonian(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    model_id: str
    family: TheoryFamily
    real_space_hamiltonian: str
    momentum_representation: str
    dispersion_renormalization: str
    anomalous_vertex: str
    current_operator: str
    kinetic_operator: str
    correlated_hopping_expectation: str | None = None
    electron_hole_transform_behavior: str
    assumptions: list[str] = Field(default_factory=list)
    dropped_terms: list[str] = Field(default_factory=list)
    validity_scope: list[str] = Field(default_factory=list)


class MicroscopicDerivation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    derivation_id: str
    model_id: str
    derivation_type: Literal["microscopic", "phenomenological", "null"]
    mean_field_decoupling: str
    gap_equation: str
    number_equation: str
    free_energy_functional: str
    sign_conventions: list[str]
    verifier_status: Literal["pass", "partial", "fail", "not_applicable"]
    unresolved_issues: list[str] = Field(default_factory=list)


class MaterialFamilyCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    tc_records: int = Field(default=0, ge=0)
    optical_records: int = Field(default=0, ge=0)
    isotope_records: int = Field(default=0, ge=0)
    thermodynamic_records: int = Field(default=0, ge=0)
    penetration_depth_records: int = Field(default=0, ge=0)
    doping_points: int = Field(default=0, ge=0)
    source_quality: Literal["high", "medium", "low", "fixture_only"] = "fixture_only"
    missing_observables: list[ObservableName] = Field(default_factory=list)


class MaterialFamilyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    family_id: str
    name: str
    rationale: str
    coverage: MaterialFamilyCoverage
    selected: bool = False


class MaterialFamilySelectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    selected_family_id: str
    compared_family_ids: list[str]
    decision_basis: list[str]
    limitations: list[str] = Field(default_factory=list)


class ExpertDossierRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    record_id: str
    family_id: str
    source_id: str
    source_type: Literal["direct_measurement", "derived", "author_interpretation", "review", "computed", "manual", "metadata"]
    observable: ObservableName
    doping_label: str | None = None
    original_value: str
    normalized_value: float | None = None
    normalized_unit: str | None = None
    uncertainty: str = "not quantified"
    split: Literal["fit", "validation", "held_out", "expert_only"] = "fit"
    curation_note: str
    provenance: list[str] = Field(default_factory=list)


class ParameterPrior(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    parameter_id: str
    model_id: str
    parameter_name: str
    source_supported_min: float | None = None
    source_supported_max: float | None = None
    unit: str
    provenance: list[str] = Field(default_factory=list)


class ParameterPlausibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    model_id: str
    parameter_id: str
    fitted_value: float
    unit: str
    classification: Literal["source_supported", "plausible", "weakly_constrained", "boundary_seeking", "implausible", "unphysical"]
    rationale: str


class ObservableFingerprintComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    observable: ObservableName
    direction_or_value: str
    required_conditions: list[str] = Field(default_factory=list)
    uncertainty: str = "not quantified"
    source_ids: list[str] = Field(default_factory=list)


class MechanismFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    fingerprint_id: str
    model_id: str
    family: TheoryFamily
    components: list[ObservableFingerprintComponent]
    missing_observables: list[ObservableName] = Field(default_factory=list)


class FingerprintPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    prediction_id: str
    model_id: str
    observable: ObservableName
    conditions: str
    predicted_value: str
    preregistered_before_holdout: bool = True
    held_out_record_ids: list[str] = Field(default_factory=list)


class FingerprintComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    model_id: str
    joint_consistency_score: float = Field(ge=0.0, le=1.0)
    tensions: list[str] = Field(default_factory=list)
    missing_observables: list[ObservableName] = Field(default_factory=list)
    source_conflicts: list[str] = Field(default_factory=list)


class TheoryDiscriminationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    selected_family_id: str
    status: Literal["falsified", "survives_within_scope", "observationally_equivalent", "requires_unphysical_parameters", "data_insufficient", "verifier_insufficient"]
    candidate_rankings: list[dict[str, Any]]
    equivalence_classes: list[list[str]] = Field(default_factory=list)
    nontrivial_outputs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AdversarialTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    test_id: str
    model_id: str
    attack: str
    outcome: Literal["falsified", "survives_within_scope", "observationally_equivalent", "requires_unphysical_parameters", "data_insufficient", "verifier_insufficient"]
    impact: str
    unresolved_issue: str | None = None


class IndependentReproductionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    conclusion_id: str
    paths: list[str] = Field(min_length=2)
    discrepancy: str
    outcome: Literal["reproduced", "discrepant", "not_reproduced", "insufficient"]


class ExperimentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v22"
    proposal_id: str
    selected_material_family: str
    composition_or_doping_points: list[str] = Field(min_length=1)
    sample_requirements: list[str] = Field(min_length=1)
    observable: ObservableName
    conditions: dict[str, str]
    theory_predictions: dict[str, str]
    expected_separation: str
    required_precision: str
    confounders: list[str]
    falsification_logic: list[str]
    feasibility: Literal["high", "medium", "low", "unknown"]
    source_support: list[str]
