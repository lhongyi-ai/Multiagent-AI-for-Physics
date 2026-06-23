from __future__ import annotations

from coscientist.schemas.v17 import CandidateSolution, ScientificProblem
from coscientist.verifiers.registry import default_verifier_registry


def _problem() -> ScientificProblem:
    return ScientificProblem(
        problem_id="p",
        title="Materials fixture",
        precise_statement="Bounded verifier problem.",
        problem_type="mechanism_discovery",
        scientific_domain="materials",
        candidate_types=["hypothesis", "mechanistic_model"],
        corpus_scope="fixture",
    )


def _candidate() -> CandidateSolution:
    return CandidateSolution(
        candidate_id="cand-bad",
        problem_id="p",
        candidate_type="mechanistic_model",
        title="Contradictory candidate",
        summary="A novel Ca0Fe4Al8 model with known failure.",
        formal_representation="Ca0Fe4Al8",
        assumptions=["Ca is present", "not Ca is present"],
        predicted_observables=["observed: no Ca contradicts target CaFe4Al8."],
        falsification_conditions=["observed: no Ca contradicts target CaFe4Al8."],
        generation_strategy="extreme_case_search",
        linked_evidence_ids=["exp-edx-1"],
        novelty_status="duplicate",
        created_step=0,
        updated_step=0,
    )


def test_default_verifiers_find_fixture_failures_without_crashing() -> None:
    registry = default_verifier_registry()
    results = [verifier.verify(_candidate(), _problem(), evidence_ids={"exp-edx-1"}) for verifier in registry.enabled(["logical_consistency", "counterexample_hook", "materials_formula"])]
    failed_checks = {check for result in results for check in result.checks_failed}
    assert "prediction_matches_falsification_condition" in failed_checks
    assert "bounded_counterexample_marker_found" in failed_checks
    assert "composition_inconsistent_with_ca_containing_target" in failed_checks
    assert all(result.tool_calls == 0 for result in results)
