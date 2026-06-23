from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AtomicModelFamily = Literal[
    "finite_level_hamiltonian",
    "zeeman_splitting",
    "hyperfine_structure",
    "driven_two_level",
    "driven_multilevel",
    "lindblad_open_system",
    "effective_spectroscopy_model",
]
HamiltonianTermType = Literal[
    "diagonal_energy",
    "coherent_coupling",
    "zeeman_linear",
    "zeeman_quadratic",
    "hyperfine_scalar",
    "stark_linear",
    "stark_quadratic",
    "detuning",
    "rabi_drive",
    "custom_matrix_literal",
]
ObservableType = Literal[
    "eigenenergies",
    "transition_frequencies",
    "transition_strengths",
    "field_sweep_spectrum",
    "time_population",
    "expectation_value",
    "steady_state_population",
    "linewidth_proxy",
    "spectrum_residual",
    "parameter_identifiability",
]
SupportedUnit = Literal[
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "rad/s",
    "seconds",
    "microseconds",
    "nanoseconds",
    "tesla",
    "gauss",
    "V/m",
    "dimensionless",
]


def _reject_code_like(value: str) -> str:
    lowered = value.lower()
    forbidden = ["import ", "lambda", "__", "open(", "exec(", "eval(", "subprocess", "http://", "https://", "file://", ";"]
    if any(token in lowered for token in forbidden):
        raise ValueError("code-like or external-access string is not allowed in atomic model specs")
    return value


class QuantumNumbers(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    J: float | None = None
    L: int | None = Field(default=None, ge=0)
    S: float | None = None
    F: float | None = None
    mJ: float | None = None
    mF: float | None = None
    parity: Literal["even", "odd", "unknown"] = "unknown"
    manifold: str | None = None

    @field_validator("manifold")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        return _reject_code_like(value) if value else value

    @model_validator(mode="after")
    def validate_m_quantum_numbers(self) -> QuantumNumbers:
        if self.J is not None and self.mJ is not None and abs(self.mJ) > self.J:
            raise ValueError("|mJ| cannot exceed J")
        if self.F is not None and self.mF is not None and abs(self.mF) > self.F:
            raise ValueError("|mF| cannot exceed F")
        return self


class AtomicBasisState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state_id: str
    label: str
    energy_reference: float = 0.0
    quantum_numbers: QuantumNumbers = Field(default_factory=QuantumNumbers)

    @field_validator("state_id", "label")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _reject_code_like(value)


class ModelParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    parameter_id: str
    value: float
    unit: SupportedUnit = "dimensionless"
    bounds: list[float] | None = Field(default=None, min_length=2, max_length=2)
    fitted: bool = False

    @field_validator("parameter_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _reject_code_like(value)

    @model_validator(mode="after")
    def validate_bounds(self) -> ModelParameter:
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError("parameter lower bound cannot exceed upper bound")
        return self


class HamiltonianTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    term_id: str
    term_type: HamiltonianTermType
    coefficient: float = 0.0
    unit: SupportedUnit = "Hz"
    state_ids: list[str] = Field(default_factory=list)
    parameter_refs: list[str] = Field(default_factory=list)
    matrix_elements: list[list[float]] = Field(default_factory=list)
    hermitian_conjugate: bool = True
    provenance: list[str] = Field(default_factory=list)

    @field_validator("term_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _reject_code_like(value)

    @field_validator("state_ids", "parameter_refs", "provenance")
    @classmethod
    def validate_text_list(cls, value: list[str]) -> list[str]:
        return [_reject_code_like(item) for item in value]

    @model_validator(mode="after")
    def validate_term_shape(self) -> HamiltonianTerm:
        if self.term_type == "custom_matrix_literal":
            if not self.matrix_elements:
                raise ValueError("custom_matrix_literal requires matrix_elements")
            if len(self.matrix_elements) > 8 or any(len(row) != len(self.matrix_elements) for row in self.matrix_elements):
                raise ValueError("custom_matrix_literal must be a square matrix of dimension <= 8")
        elif self.term_type == "coherent_coupling" and len(self.state_ids) != 2:
            raise ValueError("coherent_coupling requires exactly two state IDs")
        elif self.term_type in {"diagonal_energy", "detuning"} and len(self.state_ids) != 1:
            raise ValueError(f"{self.term_type} requires exactly one state ID")
        return self


class ObservableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observable_id: str
    observable_type: ObservableType
    state_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)
    unit: SupportedUnit = "Hz"

    @field_validator("observable_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _reject_code_like(value)


class SpectrumObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation_id: str
    frequencies: list[float] = Field(min_length=1)
    unit: SupportedUnit = "Hz"
    uncertainty: float = Field(default=0.0, ge=0.0)
    relative_intensities: list[float] = Field(default_factory=list)
    held_out: bool = False

    @model_validator(mode="after")
    def validate_lengths(self) -> SpectrumObservation:
        if self.relative_intensities and len(self.relative_intensities) != len(self.frequencies):
            raise ValueError("relative_intensities length must match frequencies")
        return self


class TransitionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation_id: str
    from_state_id: str | None = None
    to_state_id: str | None = None
    frequency: float
    unit: SupportedUnit = "Hz"
    uncertainty: float = Field(default=0.0, ge=0.0)
    polarization: Literal["pi", "sigma+", "sigma-", "unknown"] = "unknown"


class DynamicsObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation_id: str
    times: list[float] = Field(default_factory=list)
    populations: list[float] = Field(default_factory=list)
    time_unit: SupportedUnit = "seconds"

    @model_validator(mode="after")
    def validate_lengths(self) -> DynamicsObservation:
        if len(self.times) != len(self.populations):
            raise ValueError("times and populations must have the same length")
        if len(self.times) > 512:
            raise ValueError("dynamics observations are bounded to 512 samples")
        return self


class ParameterBound(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    parameter_id: str
    lower: float
    upper: float
    unit: SupportedUnit = "dimensionless"

    @model_validator(mode="after")
    def validate_bound(self) -> ParameterBound:
        if self.lower > self.upper:
            raise ValueError("lower cannot exceed upper")
        return self


class AtomicModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v18"
    model_id: str
    candidate_id: str
    model_family: AtomicModelFamily
    basis_states: list[AtomicBasisState] = Field(min_length=1, max_length=8)
    hamiltonian_terms: list[HamiltonianTerm] = Field(default_factory=list, max_length=32)
    parameters: list[ModelParameter] = Field(default_factory=list, max_length=32)
    requested_observables: list[ObservableRequest] = Field(default_factory=list, max_length=16)
    field_configuration: dict[str, float] = Field(default_factory=dict)
    units: dict[str, SupportedUnit] = Field(default_factory=lambda: {"hamiltonian": "Hz"})
    solver_preferences: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    limits_to_check: dict[str, float] = Field(default_factory=dict)

    @field_validator("model_id", "candidate_id", "assumptions")
    @classmethod
    def validate_text_fields(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            return [_reject_code_like(item) for item in value]
        return _reject_code_like(value)

    @model_validator(mode="after")
    def validate_references(self) -> AtomicModelSpec:
        state_ids = [state.state_id for state in self.basis_states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("basis state IDs must be unique")
        parameter_ids = {parameter.parameter_id for parameter in self.parameters}
        for term in self.hamiltonian_terms:
            missing_states = set(term.state_ids) - set(state_ids)
            if missing_states:
                raise ValueError(f"term {term.term_id} references unknown states: {sorted(missing_states)}")
            missing_parameters = set(term.parameter_refs) - parameter_ids
            if missing_parameters:
                raise ValueError(f"term {term.term_id} references unknown parameters: {sorted(missing_parameters)}")
            if term.term_type == "custom_matrix_literal" and len(term.matrix_elements) != len(state_ids):
                raise ValueError("custom_matrix_literal dimension must match basis size")
        for observable in self.requested_observables:
            missing = set(observable.state_ids) - set(state_ids)
            if missing:
                raise ValueError(f"observable {observable.observable_id} references unknown states: {sorted(missing)}")
        return self


class AtomicVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v18"
    candidate_id: str
    model_spec: AtomicModelSpec
    spectrum_observations: list[SpectrumObservation] = Field(default_factory=list)
    transition_observations: list[TransitionObservation] = Field(default_factory=list)
    parameter_bounds: list[ParameterBound] = Field(default_factory=list)
    max_dimension: int = Field(default=8, ge=1, le=16)
    max_time_steps: int = Field(default=256, ge=1, le=2048)
    max_fit_evaluations: int = Field(default=200, ge=1, le=5000)


class AtomicVerificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v18"
    candidate_id: str
    model_id: str
    backend: str
    backend_version: str
    verdict: Literal["pass", "partial", "fail", "inconclusive", "error"]
    score: float = Field(ge=0.0, le=1.0)
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
