from __future__ import annotations

import importlib.util

import pytest

from coscientist.discovery import load_discovery_project
from coscientist.verifiers.atomic import (
    CounterexampleParameterSearchVerifier,
    NumericalDiagonalizationVerifier,
    ParameterFitVerifier,
    QuTiPDynamicsVerifier,
    QuTiPEigenVerifier,
    SelectionRuleVerifier,
    SpectrumConsistencyVerifier,
    SymbolicHamiltonianVerifier,
)


FIXTURE = "examples/atomic_spectroscopy_fixture/project.yaml"


def _candidate(candidate_id: str):
    project = load_discovery_project(FIXTURE)
    return next(candidate for candidate in project.initial_candidates if candidate.candidate_id == candidate_id), project.problem


def test_symbolic_and_numerical_verifiers_pass_coupled_case() -> None:
    candidate, problem = _candidate("case-a-coupled-true")
    symbolic = SymbolicHamiltonianVerifier().verify(candidate, problem, evidence_ids=set())
    numerical = NumericalDiagonalizationVerifier().verify(candidate, problem, evidence_ids=set())
    assert symbolic.verdict == "pass"
    assert "two_level_analytic_eigenvalues" in symbolic.checks_passed
    assert numerical.verdict == "pass"
    assert any(item.startswith("transitions_hz") for item in numerical.assumptions)


def test_spectrum_verifier_penalizes_wrong_uncoupled_case() -> None:
    candidate, problem = _candidate("case-a-uncoupled")
    result = SpectrumConsistencyVerifier().verify(candidate, problem, evidence_ids=set())
    assert result.verdict == "partial"
    assert "spectrum_residual_exceeds_tolerance" in result.checks_failed
    assert result.score < 0.1


def test_selection_rules_detect_forbidden_same_parity_coupling() -> None:
    candidate, problem = _candidate("case-b-ambiguous-zero-field")
    payload = candidate.model_copy(deep=True)
    model = payload.structured_model["atomic_model"]
    model["hamiltonian_terms"].append({"term_id": "bad-coupling", "term_type": "coherent_coupling", "coefficient": 1.0, "unit": "MHz", "state_ids": ["b", "c"], "hermitian_conjugate": True})
    result = SelectionRuleVerifier().verify(payload, problem, evidence_ids=set())
    assert result.verdict == "fail"
    assert any("electric_dipole_no_parity_change" in item for item in result.checks_failed)


def test_parameter_fit_and_counterexample_are_bounded() -> None:
    candidate, problem = _candidate("case-a-coupled-true")
    payload = candidate.model_copy(deep=True)
    payload.structured_model["atomic_model"]["parameters"] = [{"parameter_id": "omega", "value": 1.0, "unit": "MHz", "bounds": [1.5, 2.5], "fitted": True}]
    payload.structured_model["atomic_model"]["hamiltonian_terms"][1]["parameter_refs"] = ["omega"]
    fit = ParameterFitVerifier().verify(payload, problem, evidence_ids=set())
    counterexample = CounterexampleParameterSearchVerifier().verify(payload, problem, evidence_ids=set())
    assert fit.verdict in {"pass", "partial"}
    assert any(item.startswith("nfev:") for item in fit.assumptions)
    assert counterexample.verdict == "pass"


def test_qutip_verifiers_skip_cleanly_when_unavailable() -> None:
    candidate, problem = _candidate("case-a-coupled-true")
    if importlib.util.find_spec("qutip") is not None:
        pytest.skip("environment has qutip; unavailable-path test is not applicable")
    eigen = QuTiPEigenVerifier().verify(candidate, problem, evidence_ids=set())
    dynamics = QuTiPDynamicsVerifier().verify(candidate, problem, evidence_ids=set())
    assert eigen.verdict == "inconclusive"
    assert dynamics.verdict == "inconclusive"
    assert "qutip_unavailable" in eigen.checks_failed
