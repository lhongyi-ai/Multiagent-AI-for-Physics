from __future__ import annotations

import pytest
from pydantic import ValidationError

from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.research_goal import ResearchGoal


def test_research_goal_valid() -> None:
    goal = ResearchGoal(
        id="g1",
        title="Goal",
        question="What mechanism is testable?",
        background="Background",
        constraints=[],
        desired_attributes=[],
        evaluation_criteria=[],
        prohibited_methods=[],
        max_rounds=2,
    )
    assert goal.id == "g1"


def test_research_goal_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchGoal.model_validate({
            "id": "g1",
            "title": "Goal",
            "question": "Question",
            "background": "Background",
            "max_rounds": 2,
            "unexpected": "nope",
        })


def test_hypothesis_is_immutable() -> None:
    hypothesis = Hypothesis(
        id="h1",
        title="Title",
        core_claim="Claim",
        mechanism="Mechanism",
        assumptions=["a"],
        supporting_evidence=[],
        contradicting_evidence=[],
        novelty_statement="Novel",
        testable_predictions=["p"],
        falsification_criteria=["f"],
        proposed_experiments=["e"],
        uncertainty=0.5,
        generation_strategy="mechanistic",
        parent_ids=[],
        version=1,
        status="active",
    )
    with pytest.raises(ValidationError):
        hypothesis.title = "Changed"  # type: ignore[misc]
