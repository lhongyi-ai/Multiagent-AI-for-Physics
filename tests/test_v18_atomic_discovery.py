from __future__ import annotations

from pathlib import Path

from coscientist.atomic.discovery import compare_atomic_verifiers, refresh_atomic_artifacts_if_present, run_atomic_discovery_project, validate_atomic_discovery_artifacts
from coscientist.discovery import resume_discovery_checkpoint
from coscientist.pilot.artifacts import read_json, read_jsonl


FIXTURE = "examples/atomic_spectroscopy_fixture/project.yaml"


def test_atomic_discovery_benchmark_recovers_cases_and_validates(tmp_path: Path) -> None:
    run_dir = run_atomic_discovery_project(FIXTURE, runs_dir=tmp_path, run_id="atomic")
    assert validate_atomic_discovery_artifacts(run_dir) == []
    metrics = read_json(run_dir / "atomic_benchmark_metrics.json")
    assert metrics["hidden_model_family_recovery"] == 1.0
    assert metrics["cases"]["case_a"]["recovered"] is True
    assert metrics["cases"]["case_b"]["recovered"] is True
    assert metrics["cases"]["case_c"]["recovered"] is True
    assert metrics["model_calls"] == 0
    assert read_jsonl(run_dir / "atomic_model_specs.jsonl")
    assert (run_dir / "atomic_discovery_report.md").exists()


def test_atomic_checkpoint_resume_then_artifacts_can_be_completed(tmp_path: Path) -> None:
    run_dir = run_atomic_discovery_project(FIXTURE, runs_dir=tmp_path, run_id="atomic-interrupted", stop_after_tasks=2)
    resumed = resume_discovery_checkpoint(run_dir / "search_checkpoint.json")
    assert resumed == run_dir
    assert refresh_atomic_artifacts_if_present(run_dir) is True
    assert validate_atomic_discovery_artifacts(run_dir) == []
    assert read_json(run_dir / "atomic_benchmark_metrics.json")["hidden_model_family_recovery"] == 1.0


def test_atomic_verifier_comparison_records_improvement(tmp_path: Path) -> None:
    experiment = compare_atomic_verifiers(FIXTURE, runs_dir=tmp_path, experiment_id="compare")
    comparison = read_json(experiment / "atomic_benchmark_comparison.json")
    assert comparison["bounded_outcome"] == "improved"
    assert comparison["v18_atomic_verifier_pack"]["recovery"] == 1.0
