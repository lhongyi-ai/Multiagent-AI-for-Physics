from __future__ import annotations

from pathlib import Path

from coscientist.discovery import run_discovery_project, resume_discovery_checkpoint, validate_discovery_artifacts
from coscientist.pilot.artifacts import read_json, read_jsonl


FIXTURE = "examples/discovery_search_fixture/project.yaml"


def test_discovery_fixture_runs_and_validates(tmp_path: Path) -> None:
    run_dir = run_discovery_project(FIXTURE, runs_dir=tmp_path, run_id="discovery")
    assert validate_discovery_artifacts(run_dir) == []
    usage = read_json(run_dir / "model_usage.json")
    assert usage["call_count"] == 0
    verifier_results = read_jsonl(run_dir / "verifier_results.jsonl")
    assert verifier_results
    candidates = read_jsonl(run_dir / "candidate_archive.jsonl")
    assert any(candidate["verification_result_ids"] for candidate in candidates)
    report = (run_dir / "discovery_report.md").read_text(encoding="utf-8")
    assert "Model calls: 0" in report


def test_discovery_checkpoint_resume_preserves_project_hash(tmp_path: Path) -> None:
    run_dir = run_discovery_project(FIXTURE, runs_dir=tmp_path, run_id="interrupted", stop_after_tasks=2)
    partial_checkpoint = read_json(run_dir / "search_checkpoint.json")
    assert partial_checkpoint["completed_task_ids"] == ["task-cheap-filter", "task-formalize"]
    resumed = resume_discovery_checkpoint(run_dir / "search_checkpoint.json")
    assert resumed == run_dir
    assert validate_discovery_artifacts(run_dir) == []
    final_checkpoint = read_json(run_dir / "search_checkpoint.json")
    assert final_checkpoint["resume_count"] == 1
    assert len(final_checkpoint["completed_task_ids"]) >= len(partial_checkpoint["completed_task_ids"])
