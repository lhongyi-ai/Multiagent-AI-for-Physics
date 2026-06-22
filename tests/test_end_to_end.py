from __future__ import annotations

import json
import asyncio
from pathlib import Path

from coscientist.config import WorkflowConfig
from coscientist.orchestration.workflow import run_workflow
from coscientist.providers.mock import MockProvider
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.storage.local_store import LocalStore


def _goal() -> ResearchGoal:
    return ResearchGoal(
        id="demo",
        title="Demo",
        question="What hypotheses are testable?",
        background="Synthetic background",
        constraints=["local only"],
        desired_attributes=["testable"],
        evaluation_criteria=["feasible"],
        prohibited_methods=["unsafe"],
        max_rounds=2,
    )


def test_complete_mock_workflow_persists_artifacts(tmp_path: Path) -> None:
    config = WorkflowConfig()
    result = asyncio.run(run_workflow(_goal(), MockProvider(), config, LocalStore(tmp_path), run_id="demo-run"))
    run_dir = tmp_path / "demo-run"
    assert result.run_id == "demo-run"
    assert len(result.finalists) == config.final_top_k
    assert len(json.loads((run_dir / "hypotheses_initial.json").read_text())) == 12
    assert (run_dir / "reviews_round_0.json").exists()
    assert (run_dir / "ranking_round_0.json").exists()
    assert (run_dir / "hypotheses_round_1.json").exists()
    assert (run_dir / "hypotheses_round_2.json").exists()
    assert (run_dir / "run_log.jsonl").exists()
    report = (run_dir / "final_report.md").read_text()
    assert report.count("### ") == config.final_top_k
    assert "Mock-mode outputs are synthetic" in report


def test_final_report_contains_exactly_configured_top_hypotheses(tmp_path: Path) -> None:
    config = WorkflowConfig(final_top_k=2)
    asyncio.run(run_workflow(_goal(), MockProvider(), config, LocalStore(tmp_path), run_id="two-finalists"))
    report = (tmp_path / "two-finalists" / "final_report.md").read_text()
    assert report.count("### ") == 2
