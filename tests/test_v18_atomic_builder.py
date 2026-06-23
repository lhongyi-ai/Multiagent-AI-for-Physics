from __future__ import annotations

import numpy as np
import pytest

from coscientist.atomic.builder import AtomicModelBuilder, UnitConverter
from coscientist.schemas.v18 import AtomicBasisState, AtomicModelSpec, HamiltonianTerm


def _model(*, conjugate: bool = True) -> AtomicModelSpec:
    return AtomicModelSpec(
        model_id="m",
        candidate_id="c",
        model_family="driven_two_level",
        basis_states=[AtomicBasisState(state_id="g", label="g"), AtomicBasisState(state_id="e", label="e")],
        hamiltonian_terms=[
            HamiltonianTerm(term_id="e", term_type="diagonal_energy", coefficient=10.0, unit="MHz", state_ids=["e"]),
            HamiltonianTerm(term_id="c", term_type="coherent_coupling", coefficient=2.0, unit="MHz", state_ids=["g", "e"], hermitian_conjugate=conjugate),
        ],
    )


def test_builder_constructs_two_level_hamiltonian_in_hz() -> None:
    result = AtomicModelBuilder().build(_model())
    assert result.basis_index == {"g": 0, "e": 1}
    assert np.allclose(result.hamiltonian_hz, [[0.0, 2e6], [2e6, 10e6]])
    assert result.package_versions["numpy"] != "unavailable"


def test_builder_can_reveal_missing_conjugate_nonhermiticity() -> None:
    result = AtomicModelBuilder().build(_model(conjugate=False))
    assert not np.allclose(result.hamiltonian_hz, result.hamiltonian_hz.T)


def test_unit_converter_records_conversions() -> None:
    converter = UnitConverter()
    assert converter.frequency_to_hz(1.0, "MHz") == 1e6
    assert converter.time_to_seconds(2.0, "microseconds") == 2e-6
    assert converter.field_to_tesla(10.0, "gauss") == 1e-3
    assert len(converter.records) == 3


def test_builder_rejects_oversized_basis() -> None:
    model = AtomicModelSpec(
        model_id="big",
        candidate_id="c",
        model_family="finite_level_hamiltonian",
        basis_states=[AtomicBasisState(state_id=f"s{i}", label=f"s{i}") for i in range(3)],
    )
    with pytest.raises(ValueError):
        AtomicModelBuilder(max_dimension=2).build(model)
