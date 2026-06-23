from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

from coscientist.schemas.v18 import AtomicModelSpec, SupportedUnit


class UnitConverter:
    """Small explicit converter. Internal Hamiltonian convention: ordinary frequency in Hz."""

    _frequency_to_hz = {
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
        "rad/s": 1.0 / (2.0 * np.pi),
    }
    _time_to_seconds = {
        "seconds": 1.0,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
    }
    _field_to_tesla = {
        "tesla": 1.0,
        "gauss": 1e-4,
    }

    def __init__(self) -> None:
        self.records: list[dict[str, float | str]] = []

    def frequency_to_hz(self, value: float, unit: SupportedUnit) -> float:
        if unit not in self._frequency_to_hz:
            raise ValueError(f"unit {unit} is not a supported Hamiltonian frequency unit")
        converted = value * self._frequency_to_hz[unit]
        self.records.append({"value": value, "from_unit": unit, "to_unit": "Hz", "converted": converted})
        return converted

    def time_to_seconds(self, value: float, unit: SupportedUnit) -> float:
        if unit not in self._time_to_seconds:
            raise ValueError(f"unit {unit} is not a supported time unit")
        converted = value * self._time_to_seconds[unit]
        self.records.append({"value": value, "from_unit": unit, "to_unit": "seconds", "converted": converted})
        return converted

    def field_to_tesla(self, value: float, unit: SupportedUnit) -> float:
        if unit not in self._field_to_tesla:
            raise ValueError(f"unit {unit} is not a supported field unit")
        converted = value * self._field_to_tesla[unit]
        self.records.append({"value": value, "from_unit": unit, "to_unit": "tesla", "converted": converted})
        return converted


@dataclass
class AtomicBuildResult:
    model: AtomicModelSpec
    basis_index: dict[str, int]
    hamiltonian_hz: np.ndarray
    collapse_operators: list[np.ndarray] = field(default_factory=list)
    conversion_records: list[dict[str, float | str]] = field(default_factory=list)
    summary: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "v18",
            "model_id": self.model.model_id,
            "candidate_id": self.model.candidate_id,
            "basis_index": self.basis_index,
            "hamiltonian_hz": np.round(self.hamiltonian_hz, 12).tolist(),
            "shape": list(self.hamiltonian_hz.shape),
            "conversion_records": self.conversion_records,
            "summary": self.summary,
            "package_versions": self.package_versions,
        }


class AtomicModelBuilder:
    def __init__(self, *, max_dimension: int = 8) -> None:
        self.max_dimension = max_dimension

    def build(self, model: AtomicModelSpec) -> AtomicBuildResult:
        if len(model.basis_states) > self.max_dimension:
            raise ValueError(f"basis dimension {len(model.basis_states)} exceeds max_dimension {self.max_dimension}")
        basis_index = {state.state_id: index for index, state in enumerate(model.basis_states)}
        converter = UnitConverter()
        hamiltonian = np.zeros((len(basis_index), len(basis_index)), dtype=float)
        for state in model.basis_states:
            hamiltonian[basis_index[state.state_id], basis_index[state.state_id]] += state.energy_reference
        parameters = {parameter.parameter_id: converter.frequency_to_hz(parameter.value, parameter.unit) if parameter.unit in {"Hz", "kHz", "MHz", "GHz", "rad/s"} else parameter.value for parameter in model.parameters}
        for term in model.hamiltonian_terms:
            coefficient = parameters[term.parameter_refs[0]] if term.parameter_refs else converter.frequency_to_hz(term.coefficient, term.unit)
            if term.term_type in {"diagonal_energy", "detuning", "zeeman_linear", "zeeman_quadratic", "hyperfine_scalar", "stark_linear", "stark_quadratic"}:
                hamiltonian[basis_index[term.state_ids[0]], basis_index[term.state_ids[0]]] += coefficient
            elif term.term_type in {"coherent_coupling", "rabi_drive"}:
                left, right = basis_index[term.state_ids[0]], basis_index[term.state_ids[1]]
                hamiltonian[left, right] += coefficient
                if term.hermitian_conjugate:
                    hamiltonian[right, left] += coefficient
            elif term.term_type == "custom_matrix_literal":
                matrix = np.array(term.matrix_elements, dtype=float)
                if matrix.shape != hamiltonian.shape:
                    raise ValueError("custom_matrix_literal shape does not match basis")
                hamiltonian += matrix
            else:
                raise ValueError(f"unsupported Hamiltonian term type: {term.term_type}")
        return AtomicBuildResult(
            model=model,
            basis_index=basis_index,
            hamiltonian_hz=hamiltonian,
            conversion_records=converter.records,
            summary=f"{model.model_family} with {len(basis_index)} states and {len(model.hamiltonian_terms)} terms",
            package_versions=_package_versions(["numpy", "scipy", "sympy", "qutip"]),
        )


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions
