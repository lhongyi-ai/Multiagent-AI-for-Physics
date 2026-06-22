from __future__ import annotations

import pytest
import asyncio

from coscientist.agents.supervisor import BudgetExhausted
from coscientist.config import WorkflowConfig
from coscientist.orchestration.workflow import run_workflow
from coscientist.providers.mock import MockProvider
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.storage.local_store import LocalStore


def _goal() -> ResearchGoal:
    return ResearchGoal(
        id="budget",
        title="Budget",
        question="Can the budget stop cleanly?",
        background="Synthetic",
        constraints=[],
        desired_attributes=[],
        evaluation_criteria=[],
        prohibited_methods=[],
        max_rounds=2,
    )


def test_budget_limit_enforced(tmp_path) -> None:
    config = WorkflowConfig(max_llm_calls=3)
    with pytest.raises(BudgetExhausted):
        asyncio.run(run_workflow(_goal(), MockProvider(), config, LocalStore(tmp_path), run_id="too-small"))
