from __future__ import annotations

import pytest
from pydantic import ValidationError

from coscientist.schemas.v18 import AtomicBasisState, AtomicModelSpec, HamiltonianTerm, QuantumNumbers


def _model() -> AtomicModelSpec:
    return AtomicModelSpec(
        model_id="m",
        candidate_id="c",
        model_family="driven_two_level",
        basis_states=[
            AtomicBasisState(state_id="g", label="ground", quantum_numbers=QuantumNumbers(J=0.0, mJ=0.0, parity="even")),
            AtomicBasisState(state_id="e", label="excited", quantum_numbers=QuantumNumbers(J=1.0, mJ=0.0, parity="odd")),
        ],
        hamiltonian_terms=[
            HamiltonianTerm(term_id="e", term_type="diagonal_energy", coefficient=10.0, unit="MHz", state_ids=["e"]),
            HamiltonianTerm(term_id="c", term_type="coherent_coupling", coefficient=2.0, unit="MHz", state_ids=["g", "e"]),
        ],
    )


def test_atomic_model_spec_accepts_safe_two_level_model() -> None:
    model = _model()
    assert model.model_family == "driven_two_level"
    assert len(model.basis_states) == 2


def test_duplicate_state_ids_are_rejected() -> None:
    payload = _model().model_dump()
    payload["basis_states"][1]["state_id"] = "g"
    with pytest.raises(ValidationError):
        AtomicModelSpec.model_validate(payload)


def test_forbidden_code_like_fields_are_rejected() -> None:
    payload = _model().model_dump()
    payload["basis_states"][0]["label"] = "lambda x: x"
    with pytest.raises(ValidationError):
        AtomicModelSpec.model_validate(payload)


def test_oversized_matrix_literal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HamiltonianTerm(term_id="m", term_type="custom_matrix_literal", unit="Hz", matrix_elements=[[0.0] * 9 for _ in range(9)])


def test_invalid_quantum_number_combination_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QuantumNumbers(J=1.0, mJ=2.0)
