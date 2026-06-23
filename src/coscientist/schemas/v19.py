from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coscientist.schemas.v18 import SupportedUnit


SourceType = Literal["public_reference", "curated_table", "manual_transcription", "method_reference", "human_curation_note"]
ObservationKind = Literal[
    "energy_level",
    "hyperfine_interval",
    "transition_frequency",
    "wavelength",
    "Zeeman_slope",
    "relative_strength",
    "parity",
    "Landé_g_factor",
    "selection_rule_label",
]
EvaluationStatus = Literal["measured", "critically_evaluated", "fitted", "calculated", "derived_ritz", "model_dependent", "manually_transcribed"]
AgentVisibility = Literal["agent_visible", "evaluator_only"]
DataSplit = Literal["train", "validation", "test", "reference"]
CampaignStatus = Literal["drafted", "curated", "validated", "ready", "running", "paused", "completed", "failed_validation", "awaiting_expert_review", "closed"]


def _safe_text(value: str) -> str:
    lowered = value.lower()
    forbidden = ["openai_api_key", "sk-", "bearer ", "http://secret", "lambda", "exec(", "eval(", "subprocess"]
    if any(token in lowered for token in forbidden):
        raise ValueError("unsafe text in campaign schema")
    return value


class ScientificDataSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    source_id: str
    source_type: SourceType
    title: str
    authors_or_owner: list[str] = Field(default_factory=list)
    version_or_revision: str
    access_date: date
    original_locator: str
    local_snapshot_path: str
    checksum: str | None = None
    license_or_usage_note: str
    citation_text: str
    fields_used: list[str]
    transformations_applied: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @field_validator("source_id", "title", "version_or_revision", "original_locator", "local_snapshot_path", "license_or_usage_note", "citation_text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _safe_text(value)


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    source_id: str
    local_snapshot_path: str
    checksum: str
    byte_count: int = Field(ge=0)
    created_by: str
    provenance: list[str] = Field(default_factory=list)


class AtomicObservationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    observation_id: str
    source_id: str
    species: str
    isotope: str
    charge_state: str = "neutral"
    manifold: str
    lower_state: str | None = None
    upper_state: str | None = None
    observable_type: ObservationKind
    original_value: float
    original_unit: SupportedUnit
    value: float
    unit: SupportedUnit
    uncertainty: float = Field(ge=0.0)
    uncertainty_type: str
    field_value: float | None = None
    field_unit: SupportedUnit | None = None
    polarization: Literal["pi", "sigma+", "sigma-", "unknown"] = "unknown"
    temperature: float | None = None
    measurement_context: str
    evaluation_status: EvaluationStatus
    agent_visibility: AgentVisibility
    split: DataSplit
    provenance: list[str] = Field(default_factory=list)

    @field_validator("observation_id", "source_id", "species", "isotope", "charge_state", "manifold", "measurement_context", "uncertainty_type")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _safe_text(value)


class AtomicLevelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    level_id: str
    observation_id: str
    quantum_numbers: dict[str, float | int | str]
    parity: str = "unknown"


class AtomicTransitionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    transition_id: str
    observation_id: str
    lower_state: str
    upper_state: str
    selection_rule_label: str = "unknown"


class FieldSweepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    sweep_id: str
    observation_ids: list[str]
    field_unit: SupportedUnit
    split: DataSplit


class DataSplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    split_manifest_id: str
    train_observation_ids: list[str]
    validation_observation_ids: list[str]
    test_observation_ids: list[str]
    split_rationale: str
    split_hashes: dict[str, str]
    leakage_checks_passed: bool
    hidden_test_answers: bool = True

    @model_validator(mode="after")
    def validate_disjoint(self) -> DataSplitManifest:
        sets = [set(self.train_observation_ids), set(self.validation_observation_ids), set(self.test_observation_ids)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("data splits must be disjoint")
        return self


class CurationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    decision_id: str
    observation_id: str | None = None
    decision_type: Literal["include", "exclude", "normalize_unit", "mark_conflict", "assign_split", "preserve_uncertainty"]
    rationale: str
    transformation: str | None = None
    provenance: list[str] = Field(default_factory=list)


class CurationConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    conflict_id: str
    observation_ids: list[str]
    conflict_type: Literal["duplicate", "value_disagreement", "split_leakage", "unit_equivalent_duplicate"]
    description: str
    resolved: bool = False
    resolution: str | None = None


class AtomicDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    dataset_manifest_id: str
    campaign_id: str
    species: str
    isotope: str
    source_ids: list[str]
    observation_count: int = Field(ge=0)
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    checksum: str
    limitations: list[str] = Field(default_factory=list)


class ScientificCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    campaign_id: str
    problem_id: str
    domain: str
    pilot_name: str
    research_question: str
    dataset_manifest_id: str
    train_split_id: str
    validation_split_id: str
    test_split_id: str
    candidate_families: list[str]
    allowed_hamiltonian_terms: list[str]
    allowed_verifiers: list[str]
    search_configuration: dict[str, Any] = Field(default_factory=dict)
    budget_configuration: dict[str, int] = Field(default_factory=dict)
    success_criteria: list[str]
    failure_criteria: list[str]
    novelty_policy: str
    expert_review_policy: str
    stopping_policy: str
    status: CampaignStatus = "drafted"
    provenance: list[str] = Field(default_factory=list)


class OpenProblemCampaignTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v19"
    template_id: str
    precise_unresolved_claim: str
    current_best_known_baselines: list[str]
    candidate_representation: str
    automated_verifier_coverage: list[str]
    unresolved_uncertainty: list[str]
    novelty_definition: str
    falsification_criteria: list[str]
    expert_contact_or_review_plan: str
    allowed_external_tools: list[str]
    stopping_conditions: list[str]
    publication_and_disclosure_policy: str
