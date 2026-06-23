from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from coscientist.schemas.v17 import CandidateSolution, DiscoveryProject, ScientificProblem


def _problem() -> ScientificProblem:
    return ScientificProblem(
        problem_id="p",
        title="Fixture problem",
        precise_statement="Search a bounded fixture space.",
        problem_type="mechanism_discovery",
        scientific_domain="materials",
        candidate_types=["hypothesis", "counterexample"],
        corpus_scope="fixture",
    )


def _candidate(candidate_id: str = "cand-a") -> CandidateSolution:
    return CandidateSolution(
        candidate_id=candidate_id,
        problem_id="p",
        candidate_type="hypothesis",
        title="Candidate",
        summary="Bounded candidate",
        formal_representation="CaFe4Al8",
        assumptions=["local fixture"],
        predicted_observables=["observable"],
        falsification_conditions=["falsifier"],
        generation_strategy="mainstream_extension",
        created_step=0,
        updated_step=0,
    )


def test_v17_project_is_strict_and_offline_by_default() -> None:
    project = DiscoveryProject(project_id="demo", title="Demo", problem=_problem(), initial_candidates=[_candidate()], created_at=datetime.now(UTC))
    assert project.model_mode == "mock"
    assert project.grounding_mode == "strict"


def test_live_discovery_project_requires_future_explicit_runner() -> None:
    with pytest.raises(ValidationError):
        DiscoveryProject(project_id="demo", title="Demo", model_mode="live", problem=_problem(), created_at=datetime.now(UTC))


def test_invalid_candidate_type_is_rejected_by_schema() -> None:
    payload = _candidate().model_dump()
    payload["candidate_type"] = "unbounded_answer"
    with pytest.raises(ValidationError):
        CandidateSolution.model_validate(payload)
