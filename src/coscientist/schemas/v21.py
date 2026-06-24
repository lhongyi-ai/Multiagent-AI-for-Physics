from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SuperconductivityModelFamily = Literal[
    "phonon_only_bcs",
    "correlated_hopping_only",
    "mixed_phonon_correlated_hopping",
    "interaction_energy_dominated",
    "kinetic_energy_dominated",
    "mixed_underdetermined",
    "overparameterized_mixed_model",
    "null_normal_state",
]
LatticeKind = Literal["constant_dos", "one_dimensional_tight_binding", "two_dimensional_square"]
EnergyConvention = Literal["fixed_particle_number", "fixed_chemical_potential"]
DataSourceKind = Literal["supercon", "materials_project", "nomad", "optimade", "scholarly", "manual_note"]


class LatticeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lattice_id: str
    kind: LatticeKind = "constant_dos"
    dimension: int = Field(default=0, ge=0, le=3)
    k_grid_size: int = Field(default=64, ge=4, le=512)
    density_of_states: float = Field(default=1.0, gt=0.0)
    bandwidth_ev: float = Field(default=1.0, gt=0.0)


class DispersionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dispersion_id: str
    nearest_neighbor_t_ev: float = Field(default=0.25, gt=0.0)
    next_nearest_neighbor_t_ev: float = 0.0
    electron_hole_asymmetry: float = Field(default=0.0, ge=-2.0, le=2.0)
    sign_convention: str = "negative hopping lowers band energy"


class FillingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    filling: float = Field(default=1.0, ge=0.0, le=2.0)
    temperature_k: float = Field(default=0.0, ge=0.0)
    fixed_quantity: EnergyConvention = "fixed_particle_number"


class PhononAttractionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    coupling_ev: float = Field(default=0.0, ge=0.0, le=5.0)
    cutoff_ev: float = Field(default=0.08, gt=0.0, le=1.0)
    isotope_sensitive: bool = True
    provenance: list[str] = Field(default_factory=list)


class CorrelatedHoppingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    coupling_ev: float = Field(default=0.0, ge=0.0, le=5.0)
    asymmetry_parameter: float = Field(default=0.0, ge=-2.0, le=2.0)
    operator_structure: str = "density-dependent nearest-neighbor hopping projected to a pairing channel"
    provenance: list[str] = Field(default_factory=list)


class PairingChannelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    channel_id: str
    symmetry: Literal["s_wave", "extended_s", "d_wave_proxy"] = "s_wave"
    contributes_to: Literal["interaction", "kinetic", "mixed"] = "mixed"
    sign_convention: str = "positive coupling means attractive contribution in the effective pairing kernel"


class PairingKernelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kernel_id: str
    phonon: PhononAttractionSpec = Field(default_factory=PhononAttractionSpec)
    correlated_hopping: CorrelatedHoppingSpec = Field(default_factory=CorrelatedHoppingSpec)
    channels: list[PairingChannelSpec] = Field(default_factory=lambda: [PairingChannelSpec(channel_id="s")])
    separable: bool = True


class MeanFieldDecouplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approximation_id: str = "bounded_bcs_mean_field"
    approximations: list[str] = Field(default_factory=lambda: ["separable kernel", "constant density-of-states or bounded lattice grid", "real scalar gap"])
    phenomenological_terms: list[str] = Field(default_factory=list)


class SelfConsistencySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_iterations: int = Field(default=200, ge=1, le=5000)
    tolerance: float = Field(default=1e-8, gt=0.0, le=1e-2)
    initial_gaps_ev: list[float] = Field(default_factory=lambda: [1e-4, 1e-3, 1e-2])


class EnergyDecompositionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    convention: EnergyConvention = "fixed_particle_number"
    normal_reference: str = "same filling normal state at same temperature"
    include_correlated_hopping_term: bool = True
    prevent_double_counting: bool = True


class OpticalSumRuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    direction: Literal["x", "y", "isotropic"] = "isotropic"
    cutoffs_ev: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2])
    include_interband: bool = False


class MaterialMappingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    material_id: str
    formula: str
    family: str
    tc_k: float | None = Field(default=None, ge=0.0)
    doping_label: str | None = None
    mapping_assumptions: list[str] = Field(default_factory=list)
    mapping_uncertainty: str = "high"
    unsupported_fields: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class SuperconductivityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation_id: str
    material_id: str | None = None
    observable_type: str
    value: float | str
    unit: str
    uncertainty: float | None = Field(default=None, ge=0.0)
    doping_label: str | None = None
    source_id: str
    split: Literal["train", "validation", "test"] = "train"
    computed_or_experimental: Literal["computed", "experimental", "manual_note"] = "manual_note"


class SuperconductivityCorpusItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    citation: str
    doi_or_identifier: str
    local_snapshot: str
    section_or_page: str
    claim_type: Literal["direct_result", "derivation", "model_assumption", "author_interpretation", "experimental_observation", "review_statement", "contradiction", "failed_approach", "metadata_only"]
    exact_excerpt_or_manual_note: str
    candidate_family_relevance: list[SuperconductivityModelFamily]
    support_or_contradiction: Literal["support", "contradiction", "mixed", "context", "unusable_metadata"]
    material_scope: str
    doping_scope: str
    temperature_scope: str
    frequency_cutoff: str | None = None
    uncertainty: str = "not quantified"
    curation_status: Literal["strict_support", "context_only", "metadata_only", "needs_review"] = "context_only"
    provenance: list[str] = Field(default_factory=list)


class SuperconductivityModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v21"
    model_id: str
    candidate_id: str
    family: SuperconductivityModelFamily
    lattice: LatticeSpec = Field(default_factory=LatticeSpec)
    dispersion: DispersionSpec = Field(default_factory=lambda: DispersionSpec(dispersion_id="dispersion"))
    filling: FillingSpec = Field(default_factory=FillingSpec)
    pairing_kernel: PairingKernelSpec = Field(default_factory=lambda: PairingKernelSpec(kernel_id="kernel"))
    mean_field: MeanFieldDecouplingSpec = Field(default_factory=MeanFieldDecouplingSpec)
    self_consistency: SelfConsistencySpec = Field(default_factory=SelfConsistencySpec)
    energy_decomposition: EnergyDecompositionSpec = Field(default_factory=EnergyDecompositionSpec)
    optical_sum_rule: OpticalSumRuleSpec = Field(default_factory=OpticalSumRuleSpec)
    material_mappings: list[MaterialMappingSpec] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_family_channels(self) -> SuperconductivityModelSpec:
        phonon = self.pairing_kernel.phonon.coupling_ev
        ch = self.pairing_kernel.correlated_hopping.coupling_ev
        if self.family == "phonon_only_bcs" and ch != 0:
            raise ValueError("phonon_only_bcs cannot include correlated-hopping coupling")
        if self.family == "correlated_hopping_only" and phonon != 0:
            raise ValueError("correlated_hopping_only cannot include phonon coupling")
        if self.family == "null_normal_state" and (phonon != 0 or ch != 0):
            raise ValueError("null_normal_state cannot include pairing couplings")
        return self


class PairingKernelResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_id: str
    effective_coupling_ev: float
    phonon_contribution_ev: float
    correlated_hopping_contribution_ev: float
    equation: str
    approximations: list[str]


class SelfConsistencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_id: str
    gap_ev: float
    tc_k: float
    chemical_potential_ev: float
    converged: bool
    iterations: int
    diagnostics: list[str] = Field(default_factory=list)


class EnergyDecompositionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_id: str
    normal_reference: str
    convention: EnergyConvention
    delta_kinetic_ev: float
    delta_interaction_ev: float
    delta_correlated_hopping_ev: float
    delta_pairing_mean_field_ev: float
    total_internal_energy_change_ev: float
    free_energy_change_ev: float
    condensation_energy_closure_error_ev: float
    sign_convention: str


class OpticalSumRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_id: str
    full_sum: float
    partial_sum_by_cutoff: dict[str, float]
    delta_sum: float
    direction: str
    normal_reference: str
    superconducting_reference: str
    interpretation_warnings: list[str]


class DopingSweepPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    filling: float
    gap_ev: float
    tc_k: float
    condensation_energy_ev: float
    delta_kinetic_ev: float
    delta_interaction_ev: float
    delta_correlated_hopping_ev: float
    optical_proxy: float
    convergence_status: str


class DopingSweepResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_id: str
    points: list[DopingSweepPoint]
    sign_changes: list[str] = Field(default_factory=list)
    non_monotonic_metrics: list[str] = Field(default_factory=list)
    non_identifiable_regions: list[str] = Field(default_factory=list)


class IdentifiabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    group_id: str
    model_ids: list[str]
    status: Literal["identifiable", "nearly_equivalent", "underdetermined"]
    observables_compared: list[str]
    required_precision: float | None = None
    discriminating_observable: str | None = None


class SuperconductivityScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    candidate_id: str
    model_id: str
    hamiltonian_validity: float = Field(ge=0.0, le=1.0)
    mean_field_derivation: float = Field(ge=0.0, le=1.0)
    self_consistency: float = Field(ge=0.0, le=1.0)
    free_energy_stability: float = Field(ge=0.0, le=1.0)
    limiting_case_score: float = Field(ge=0.0, le=1.0)
    energy_closure_score: float = Field(ge=0.0, le=1.0)
    optical_consistency: float = Field(ge=0.0, le=1.0)
    doping_trend_consistency: float = Field(ge=0.0, le=1.0)
    material_mapping_quality: float = Field(ge=0.0, le=1.0)
    identifiability: float = Field(ge=0.0, le=1.0)
    counterexample_survival: float = Field(ge=0.0, le=1.0)
    literature_support: float = Field(ge=0.0, le=1.0)
    literature_contradiction: float = Field(ge=0.0, le=1.0)
    complexity_penalty: float = Field(ge=0.0, le=1.0)
    reproduction_status: Literal["reproduced", "partially_reproduced", "not_reproduced", "inconclusive"]
    aggregate_score: float = Field(ge=0.0, le=1.0)


class ScientificIndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v21-sqlite-1"
    database_path: str
    artifact_root: str
    artifact_hash: str
    table_counts: dict[str, int]
    stale: bool = False


class SuperconductivityVerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    verifier_id: str
    model_id: str
    stage: Literal["cheap", "standard", "strong"]
    verdict: Literal["pass", "partial", "fail", "inconclusive"]
    score: float = Field(ge=0.0, le=1.0)
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
