from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from coscientist.pilot.project_io import load_project_spec
from coscientist.schemas.project import ResearchProjectSpec


def _spec(**updates):
    data = {
        "project_id": "p",
        "title": "Project",
        "research_question": "What mechanism explains this safe pilot observation?",
        "task_type": "explanation",
        "background": "Background",
        "scope": "Scope",
        "known_observations": [],
        "constraints": ["offline"],
        "excluded_directions": ["unsafe"],
        "available_evidence": [],
        "desired_output": "Hypotheses",
        "evaluation_criteria": ["relevance"],
        "validation_constraints": [],
        "maximum_literature_budget": 1,
        "maximum_model_call_budget": 10,
        "maximum_evolution_rounds": 2,
        "created_at": datetime.now(UTC),
    }
    data.update(updates)
    return data


def test_project_spec_loads_pilot_fixture() -> None:
    project = load_project_spec("research-projects/interdisciplinary_fixture/project.yaml")
    assert project.project_id == "urban-heat-island-mitigation"
    assert project.maximum_evolution_rounds == 2


def test_project_spec_rejects_missing_question() -> None:
    data = _spec(research_question="")
    with pytest.raises(ValidationError):
        ResearchProjectSpec.model_validate(data)


def test_project_spec_rejects_negative_budget() -> None:
    data = _spec(maximum_literature_budget=-1)
    with pytest.raises(ValidationError):
        ResearchProjectSpec.model_validate(data)


def test_project_spec_rejects_unsupported_task_type() -> None:
    data = _spec(task_type="astrology")
    with pytest.raises(ValidationError):
        ResearchProjectSpec.model_validate(data)


def test_project_spec_rejects_inconsistent_constraints() -> None:
    data = _spec(constraints=["offline"], excluded_directions=["offline"])
    with pytest.raises(ValidationError):
        ResearchProjectSpec.model_validate(data)


def test_project_spec_rejects_invalid_round_limits() -> None:
    data = _spec(maximum_model_call_budget=1, maximum_evolution_rounds=2)
    with pytest.raises(ValidationError):
        ResearchProjectSpec.model_validate(data)
