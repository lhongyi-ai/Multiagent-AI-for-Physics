from __future__ import annotations

from coscientist.schemas.v17 import CandidateSolution, ScientificProblem
from coscientist.verifiers.base import ScientificVerifier


class SchemaConstraintVerifier(ScientificVerifier):
    verifier_id = "schema_constraint"
    stage = "cheap"
    capability = "schema"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        passed, failed = [], []
        if candidate.candidate_type in problem.candidate_types:
            passed.append("candidate_type_allowed")
        else:
            failed.append("unsupported_candidate_type")
        if candidate.formal_representation:
            passed.append("formal_representation_present")
        else:
            failed.append("missing_formal_representation")
        if candidate.predicted_observables:
            passed.append("predicted_observables_present")
        else:
            failed.append("missing_predicted_observables")
        if candidate.falsification_conditions:
            passed.append("falsification_conditions_present")
        else:
            failed.append("missing_falsification_conditions")
        if candidate.lineage_depth <= 5:
            passed.append("lineage_depth_within_default_bound")
        else:
            failed.append("lineage_depth_overflow")
        verdict = "pass" if not failed else "fail"
        score = len(passed) / max(1, len(passed) + len(failed))
        return verdict, round(score, 3), passed, failed, [], False


class LogicalConsistencyVerifier(ScientificVerifier):
    verifier_id = "logical_consistency"
    stage = "cheap"
    capability = "consistency"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        passed, failed = ["candidate_loaded"], []
        assumptions = [item.lower() for item in candidate.assumptions]
        for assumption in assumptions:
            if assumption.startswith("not ") and assumption[4:] in assumptions:
                failed.append(f"mutually_incompatible_assumption:{assumption}")
        joined_predictions = " ".join(candidate.predicted_observables).lower()
        for condition in candidate.falsification_conditions:
            if condition.lower() in joined_predictions:
                failed.append("prediction_matches_falsification_condition")
        if candidate.novelty_status == "duplicate" and "novel" in candidate.summary.lower():
            failed.append("duplicate_presented_as_novel")
        verdict = "pass" if not failed else "fail"
        return verdict, 1.0 if not failed else 0.2, passed, failed, [], False


class EvidenceConsistencyVerifier(ScientificVerifier):
    verifier_id = "evidence_consistency"
    stage = "standard"
    capability = "evidence"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        passed, failed = [], []
        missing = sorted(set(candidate.linked_evidence_ids) - evidence_ids)
        if missing:
            failed.append(f"missing_evidence_ids:{','.join(missing)}")
        else:
            passed.append("evidence_ids_exist")
        if any(item.startswith("metadata-only") for item in candidate.linked_evidence_ids):
            failed.append("metadata_only_evidence_used_as_verified_support")
        else:
            passed.append("no_metadata_only_verified_support")
        if len(candidate.linked_evidence_ids) != len(set(candidate.linked_evidence_ids)):
            failed.append("duplicate_evidence_references")
        else:
            passed.append("duplicate_evidence_discounted")
        verdict = "pass" if not failed else "partial" if passed else "fail"
        score = len(passed) / max(1, len(passed) + len(failed))
        return verdict, round(score, 3), passed, failed, [], False


class CounterexampleHookVerifier(ScientificVerifier):
    verifier_id = "counterexample_hook"
    stage = "standard"
    capability = "counterexample"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        text = " ".join([candidate.summary, candidate.formal_representation or "", *candidate.assumptions]).lower()
        if "counterexample:" in text or "known failure" in text:
            return "fail", 0.1, [], ["bounded_counterexample_marker_found"], ["fixture marker only"], True
        return "inconclusive", 0.5, ["no_fixture_counterexample_marker"], [], ["not a full counterexample search"], False


class ExperimentalConsistencyVerifier(ScientificVerifier):
    verifier_id = "experimental_consistency"
    stage = "standard"
    capability = "experiment"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        passed, failed = [], []
        predictions = " ".join(candidate.predicted_observables).lower()
        if "observed:" in predictions and "contradicts" in predictions:
            failed.append("predicted_outcome_contradicts_structured_observation")
        elif candidate.predicted_observables:
            passed.append("predicted_observables_structured")
        else:
            failed.append("missing_structured_predictions")
        verdict = "pass" if not failed else "fail"
        return verdict, 0.8 if not failed else 0.2, passed, failed, [], False
