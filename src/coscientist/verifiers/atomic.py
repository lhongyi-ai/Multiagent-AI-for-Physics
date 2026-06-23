from __future__ import annotations

import hashlib
import json
from importlib import metadata
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from coscientist.atomic.builder import AtomicModelBuilder
from coscientist.schemas.v17 import CandidateSolution, ScientificProblem
from coscientist.schemas.v18 import AtomicModelSpec, SpectrumObservation
from coscientist.verifiers.base import ScientificVerifier


def atomic_model_from_candidate(candidate: CandidateSolution) -> AtomicModelSpec | None:
    payload = candidate.structured_model.get("atomic_model") if candidate.structured_model else None
    if payload is None:
        return None
    return AtomicModelSpec.model_validate(payload)


def spectrum_observations_from_candidate(candidate: CandidateSolution) -> list[SpectrumObservation]:
    payload = candidate.structured_model.get("spectrum_observations", []) if candidate.structured_model else []
    return [SpectrumObservation.model_validate(item) for item in payload]


class AtomicSchemaVerifier(ScientificVerifier):
    verifier_id = "atomic_schema"
    stage = "cheap"
    capability = "atomic_schema"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        return "pass", 1.0, [f"atomic_model_valid:{model.model_family}", f"basis_dimension:{len(model.basis_states)}"], [], [], False


class SymbolicHamiltonianVerifier(ScientificVerifier):
    verifier_id = "symbolic_hamiltonian"
    stage = "standard"
    capability = "sympy"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        build = AtomicModelBuilder().build(model)
        matrix = sp.Matrix(build.hamiltonian_hz)
        passed, failed = [], []
        if matrix == matrix.T.conjugate():
            passed.append("symbolic_hermitian")
        else:
            failed.append("symbolic_non_hermitian")
        if matrix.shape[0] == len(model.basis_states):
            passed.append("symbolic_dimension_matches_basis")
        if matrix.shape == (2, 2):
            eigenvalues = sorted([sp.simplify(value) for value in matrix.eigenvals().keys()], key=str)
            trace = sp.simplify(sum(eigenvalues) - matrix.trace())
            determinant = sp.simplify(sp.prod(eigenvalues) - matrix.det())
            if abs(float(sp.N(trace))) < 1e-6:
                passed.append("trace_identity")
            else:
                failed.append("trace_identity_failed")
            if abs(float(sp.N(determinant))) < 1e-3:
                passed.append("determinant_identity")
            else:
                failed.append("determinant_identity_failed")
            passed.append("two_level_analytic_eigenvalues")
        if any(term.term_type in {"coherent_coupling", "rabi_drive"} and term.coefficient == 0 for term in model.hamiltonian_terms):
            passed.append("zero_coupling_limit_encoded")
        verdict = "pass" if not failed else "fail"
        score = len(passed) / max(1, len(passed) + len(failed))
        return verdict, round(score, 3), passed, failed, [f"sympy:{_version('sympy')}"], False


class NumericalDiagonalizationVerifier(ScientificVerifier):
    verifier_id = "numerical_diagonalization"
    stage = "standard"
    capability = "numpy_scipy"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        build = AtomicModelBuilder().build(model)
        hamiltonian = build.hamiltonian_hz
        passed, failed = [], []
        if np.all(np.isfinite(hamiltonian)):
            passed.append("finite_matrix_values")
        else:
            failed.append("non_finite_matrix_values")
        if np.allclose(hamiltonian, hamiltonian.T.conjugate(), atol=1e-9):
            passed.append("numeric_hermitian")
        else:
            failed.append("numeric_non_hermitian")
        eigenvalues, _ = np.linalg.eigh(hamiltonian)
        if np.all(np.diff(eigenvalues) >= -1e-9):
            passed.append("eigenvalues_ordered")
        transition_frequencies = _transition_frequencies(eigenvalues)
        if transition_frequencies:
            passed.append("transition_frequencies_computed")
        condition = np.linalg.cond(hamiltonian) if hamiltonian.size else 0.0
        if np.isfinite(condition):
            passed.append("condition_number_finite")
        verdict = "pass" if not failed else "fail"
        score = len(passed) / max(1, len(passed) + len(failed))
        assumptions = [f"eigenvalues_hz:{_round_list(eigenvalues)}", f"transitions_hz:{_round_list(transition_frequencies)}", f"numpy:{_version('numpy')}", f"scipy:{_version('scipy')}"]
        return verdict, round(score, 3), passed, failed, assumptions, False


class SpectrumConsistencyVerifier(ScientificVerifier):
    verifier_id = "spectrum_consistency"
    stage = "standard"
    capability = "spectrum"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        observations = spectrum_observations_from_candidate(candidate)
        if model is None or not observations:
            return "inconclusive", 0.5, [], ["missing_model_or_spectrum_observations"], [], False
        predicted = _predicted_transitions(model)
        observed = observations[0]
        obs_hz = [_frequency_to_hz(value, observed.unit) for value in observed.frequencies]
        assignment, unmatched_predicted, unmatched_observed = _match_lines(predicted, obs_hz)
        residuals = [abs(predicted_value - observed_value) for predicted_value, observed_value in assignment]
        rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
        tolerance = max(_frequency_to_hz(observed.uncertainty, observed.unit), 1e-9)
        passed, failed = [], []
        if rms <= tolerance:
            passed.append("spectrum_residual_within_tolerance")
        else:
            failed.append("spectrum_residual_exceeds_tolerance")
        if unmatched_observed:
            failed.append("unmatched_observed_lines")
        if unmatched_predicted:
            failed.append("unmatched_predicted_lines")
        ambiguity = _ambiguity_count(predicted, obs_hz, tolerance)
        if ambiguity:
            passed.append(f"ambiguous_assignments:{ambiguity}")
        score = 1.0 / (1.0 + rms / max(tolerance, 1e-9))
        if failed:
            score *= 0.5
        assumptions = [f"weighted_rms_hz:{round(rms, 6)}", f"assignments:{len(assignment)}", f"unmatched_observed:{len(unmatched_observed)}", f"unmatched_predicted:{len(unmatched_predicted)}"]
        return ("pass" if not failed else "partial" if assignment else "fail"), round(float(score), 3), passed, failed, assumptions, False


class SelectionRuleVerifier(ScientificVerifier):
    verifier_id = "selection_rules"
    stage = "cheap"
    capability = "selection_rules"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        by_id = {state.state_id: state for state in model.basis_states}
        passed, failed, inconclusive = [], [], []
        for term in model.hamiltonian_terms:
            if term.term_type not in {"coherent_coupling", "rabi_drive"} or len(term.state_ids) != 2:
                continue
            left, right = by_id[term.state_ids[0]], by_id[term.state_ids[1]]
            ql, qr = left.quantum_numbers, right.quantum_numbers
            if ql.parity != "unknown" and qr.parity != "unknown":
                if ql.parity != qr.parity:
                    passed.append(f"electric_dipole_parity_change:{term.term_id}")
                else:
                    failed.append(f"electric_dipole_no_parity_change:{term.term_id}")
            if ql.J is not None and qr.J is not None:
                if abs(ql.J - qr.J) <= 1:
                    passed.append(f"delta_J_supported:{term.term_id}")
                else:
                    failed.append(f"delta_J_forbidden:{term.term_id}")
            else:
                inconclusive.append(f"missing_J:{term.term_id}")
        if failed:
            return "fail", 0.2, passed, failed, inconclusive, False
        if passed:
            return "pass", 1.0, passed, failed, inconclusive, False
        return "inconclusive", 0.5, [], ["insufficient_quantum_numbers"], inconclusive, False


class LimitingCaseVerifier(ScientificVerifier):
    verifier_id = "limiting_cases"
    stage = "standard"
    capability = "limits"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        passed, failed = [], []
        for limit_name, expected in model.limits_to_check.items():
            if limit_name == "zero_coupling_minimum_transition_hz":
                zeroed = _copy_with_zero_couplings(model)
                transitions = _predicted_transitions(zeroed)
                minimum = min(transitions) if transitions else 0.0
                if abs(minimum - expected) <= max(1e-6, abs(expected) * 1e-6):
                    passed.append("zero_coupling_limit")
                else:
                    failed.append("zero_coupling_limit_failed")
        if not passed and not failed:
            return "inconclusive", 0.5, [], ["no_supported_limits_configured"], [], False
        return ("pass" if not failed else "fail"), len(passed) / max(1, len(passed) + len(failed)), passed, failed, [], False


class ParameterFitVerifier(ScientificVerifier):
    verifier_id = "parameter_fit"
    stage = "strong"
    capability = "scipy_fit"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        observations = spectrum_observations_from_candidate(candidate)
        if model is None or not observations:
            return "inconclusive", 0.5, [], ["missing_model_or_observations"], [], False
        fitted = [parameter for parameter in model.parameters if parameter.fitted and parameter.bounds is not None]
        if not fitted:
            return "inconclusive", 0.5, [], ["no_bounded_fitted_parameters"], [], False
        observed = [_frequency_to_hz(value, observations[0].unit) for value in observations[0].frequencies]
        x0 = np.array([(parameter.bounds[0] + parameter.bounds[1]) / 2 for parameter in fitted], dtype=float)
        bounds = ([parameter.bounds[0] for parameter in fitted], [parameter.bounds[1] for parameter in fitted])

        def residual(values: np.ndarray) -> np.ndarray:
            replacements = dict(zip([parameter.parameter_id for parameter in fitted], values))
            trial = _copy_with_parameter_values(model, replacements)
            predicted = _predicted_transitions(trial)[: len(observed)]
            if len(predicted) < len(observed):
                predicted = predicted + [0.0] * (len(observed) - len(predicted))
            return np.array(predicted) - np.array(observed)

        result = least_squares(residual, x0, bounds=bounds, max_nfev=200)
        rms = float(np.sqrt(np.mean(np.square(result.fun)))) if result.fun.size else 0.0
        tolerance = max(_frequency_to_hz(observations[0].uncertainty, observations[0].unit), 1e-9)
        passed = ["bounded_fit_completed"] if result.success else []
        failed = [] if rms <= tolerance else ["fit_residual_exceeds_tolerance"]
        if len(fitted) >= len(observed):
            failed.append("parameter_identifiability_warning")
        score = 1.0 / (1.0 + rms / tolerance)
        return ("pass" if not failed else "partial"), round(float(score), 3), passed, failed, [f"rms_hz:{round(rms, 6)}", f"nfev:{result.nfev}"], False


class CounterexampleParameterSearchVerifier(ScientificVerifier):
    verifier_id = "counterexample_parameter_search"
    stage = "strong"
    capability = "counterexample_grid"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        fitted = [parameter for parameter in model.parameters if parameter.bounds is not None]
        if not fitted:
            return "inconclusive", 0.5, [], ["no_bounded_parameters"], [], False
        for parameter in fitted[:2]:
            lower, upper = parameter.bounds
            for value in np.linspace(lower, upper, 5):
                trial = _copy_with_parameter_values(model, {parameter.parameter_id: float(value)})
                hamiltonian = AtomicModelBuilder().build(trial).hamiltonian_hz
                if not np.allclose(hamiltonian, hamiltonian.T.conjugate(), atol=1e-9):
                    return "fail", 0.0, [], [f"counterexample_non_hermitian:{parameter.parameter_id}={value}"], [], True
        return "pass", 1.0, ["no_counterexample_in_bounded_grid"], [], [], False


class QuTiPEigenVerifier(ScientificVerifier):
    verifier_id = "qutip_eigen"
    stage = "strong"
    capability = "qutip"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        try:
            import qutip as qt  # type: ignore
        except Exception:
            return "inconclusive", 0.5, [], ["qutip_unavailable"], ["optional dependency"], False
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        build = AtomicModelBuilder().build(model)
        qobj = qt.Qobj(build.hamiltonian_hz)
        qutip_values = np.sort(np.array(qobj.eigenenergies(), dtype=float))
        numpy_values = np.sort(np.linalg.eigvalsh(build.hamiltonian_hz))
        if np.allclose(qutip_values, numpy_values, atol=1e-8):
            return "pass", 1.0, ["qutip_numpy_eigen_agreement"], [], [f"qutip:{_version('qutip')}"], False
        return "fail", 0.0, [], ["qutip_numpy_eigen_disagreement"], [f"qutip:{_version('qutip')}"], False


class QuTiPDynamicsVerifier(ScientificVerifier):
    verifier_id = "qutip_dynamics"
    stage = "strong"
    capability = "qutip"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        try:
            import qutip as qt  # type: ignore
        except Exception:
            return "inconclusive", 0.5, [], ["qutip_unavailable"], ["optional dependency"], False
        model = atomic_model_from_candidate(candidate)
        if model is None:
            return "inconclusive", 0.5, [], ["no_atomic_model_spec"], [], False
        build = AtomicModelBuilder().build(model)
        if build.hamiltonian_hz.shape[0] > 4:
            return "inconclusive", 0.5, [], ["dimension_exceeds_dynamics_limit"], [], False
        hamiltonian = qt.Qobj(build.hamiltonian_hz)
        initial = qt.basis(build.hamiltonian_hz.shape[0], 0)
        times = np.linspace(0, 1e-6, 16)
        result = qt.sesolve(hamiltonian, initial, times)
        norms = [state.norm() for state in result.states]
        if all(abs(norm - 1.0) < 1e-8 for norm in norms):
            return "pass", 1.0, ["trace_or_norm_preserved"], [], [f"time_steps:{len(times)}"], False
        return "partial", 0.5, [], ["norm_drift_detected"], [f"time_steps:{len(times)}"], False


def _predicted_transitions(model: AtomicModelSpec) -> list[float]:
    hamiltonian = AtomicModelBuilder().build(model).hamiltonian_hz
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    return _transition_frequencies(eigenvalues)


def _transition_frequencies(eigenvalues: Any) -> list[float]:
    values = sorted(float(value) for value in eigenvalues)
    transitions = []
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            transitions.append(abs(right - left))
    return sorted(transitions)


def _match_lines(predicted: list[float], observed: list[float]) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    remaining = predicted[:]
    assignment: list[tuple[float, float]] = []
    for obs in sorted(observed):
        if not remaining:
            break
        best = min(remaining, key=lambda pred: (abs(pred - obs), pred))
        assignment.append((best, obs))
        remaining.remove(best)
    matched_observed = [obs for _, obs in assignment]
    unmatched_observed = [obs for obs in observed if obs not in matched_observed]
    return assignment, remaining, unmatched_observed


def _ambiguity_count(predicted: list[float], observed: list[float], tolerance: float) -> int:
    return sum(1 for obs in observed if sum(1 for pred in predicted if abs(pred - obs) <= tolerance) > 1)


def _copy_with_zero_couplings(model: AtomicModelSpec) -> AtomicModelSpec:
    data = model.model_dump()
    for term in data["hamiltonian_terms"]:
        if term["term_type"] in {"coherent_coupling", "rabi_drive"}:
            term["coefficient"] = 0.0
            term["parameter_refs"] = []
    return AtomicModelSpec.model_validate(data)


def _copy_with_parameter_values(model: AtomicModelSpec, replacements: dict[str, float]) -> AtomicModelSpec:
    data = model.model_dump()
    for parameter in data["parameters"]:
        if parameter["parameter_id"] in replacements:
            parameter["value"] = replacements[parameter["parameter_id"]]
    return AtomicModelSpec.model_validate(data)


def _frequency_to_hz(value: float, unit: str) -> float:
    factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "rad/s": 1.0 / (2.0 * np.pi)}
    return value * factors.get(unit, 1.0)


def _round_list(values: Any) -> str:
    return json.dumps([round(float(value), 6) for value in values])


def _version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def atomic_equivalence_key(candidate: CandidateSolution) -> str:
    model = atomic_model_from_candidate(candidate)
    if model is None:
        return ""
    transitions = _predicted_transitions(model)
    payload = [round(value, 6) for value in transitions]
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]
