from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from coscientist.schemas.v16 import AnswerOption, ClosedQuestion, FinalAnswer, GroundTruth


def test_closed_question_schema_validates_question_types_and_answer_counts() -> None:
    question = ClosedQuestion(
        question_id="q",
        prompt="Pick one.",
        question_type="single_choice",
        answer_options=[AnswerOption(answer_id="a", statement="A")],
        allowed_answer_count=1,
        evaluation_method="exact_match",
    )
    assert question.question_type == "single_choice"
    with pytest.raises(ValidationError):
        ClosedQuestion(
            question_id="bad",
            prompt="Pick one.",
            question_type="single_choice",
            answer_options=[AnswerOption(answer_id="a", statement="A")],
            allowed_answer_count=2,
            evaluation_method="exact_match",
        )


def test_numeric_question_requires_tolerance() -> None:
    with pytest.raises(ValidationError):
        ClosedQuestion(
            question_id="num",
            prompt="Estimate.",
            question_type="numeric",
            answer_options=[],
            allowed_answer_count=1,
            evaluation_method="numeric_tolerance",
        )


def test_ground_truth_must_remain_hidden() -> None:
    with pytest.raises(ValidationError):
        GroundTruth(question_id="q", correct_answer_ids=["a"], rationale="test", hidden_from_agents=False)


def test_final_answer_bounds_confidence() -> None:
    with pytest.raises(ValidationError):
        FinalAnswer(
            project_id="p",
            run_id="r",
            question_id="q",
            confidence=1.5,
            rationale_summary="bad",
            recommended_next_action="review",
            created_at=datetime.now(UTC),
        )
