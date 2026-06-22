from __future__ import annotations

import asyncio

from coscientist.config import WorkflowConfig
from coscientist.orchestration.workflow import run_workflow
from coscientist.providers.mock import MockProvider
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.storage.local_store import LocalStore


def test_mock_literature_workflow_persists_artifacts(tmp_path) -> None:
    goal = ResearchGoal(
        id="lit",
        title="Literature",
        question="What mock sources exist?",
        background="Synthetic",
        max_rounds=2,
    )
    config = WorkflowConfig()
    config = config.model_copy(update={"literature": config.literature.model_copy(update={"enabled": True})})
    asyncio.run(run_workflow(goal, MockProvider(), config, LocalStore(tmp_path), run_id="lit-run"))
    run_dir = tmp_path / "lit-run"
    assert (run_dir / "papers_normalized_round_0.json").exists()
    assert (run_dir / "metadata_resolutions_round_0.json").exists()
    assert (run_dir / "full_text_locations_round_0.json").exists()
    report = (run_dir / "final_report.md").read_text()
    assert "Source-Backed Observations" in report
    assert "Search result rank, metadata, and PDF availability are not treated as claim support." in report
