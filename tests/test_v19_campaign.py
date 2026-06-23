from __future__ import annotations

from pathlib import Path

from coscientist.atomic.campaign import compare_atomic_campaign, resume_atomic_campaign_checkpoint, run_atomic_campaign_project, validate_atomic_campaign_artifacts
from coscientist.pilot.artifacts import read_json, read_jsonl


PROJECT = "examples/rb87_real_spectroscopy/project.yaml"


def test_rb87_campaign_runs_validates_and_selects_bounded_family(tmp_path: Path) -> None:
    run_dir = run_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="rb87")
    assert validate_atomic_campaign_artifacts(run_dir) == []
    comparison = read_json(run_dir / "model_comparison.json")
    assert comparison["selected_family"] == "hyperfine_linear_field"
    assert comparison["outcome"] == "correct_unique_recovery"
    metrics = read_json(run_dir / "campaign_metrics.json")
    assert metrics["model_calls"] == 0
    assert metrics["expert_review_ready"] is True


def test_identifiability_equivalence_and_discriminating_proposals_exist(tmp_path: Path) -> None:
    run_dir = run_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="rb87")
    identifiability = read_jsonl(run_dir / "identifiability_results.jsonl")
    assert any(item["warning"] == "correlated_or_unstable" for item in identifiability)
    equivalence = read_json(run_dir / "equivalence_classes.json")
    assert equivalence["equivalence_classes"]
    proposals = read_jsonl(run_dir / "discriminating_observable_proposals.jsonl")
    assert proposals
    assert proposals[0]["observable_type"] == "Zeeman_slope"
    assert proposals[0]["predicted_separation"] > 0


def test_stress_and_counterexample_artifacts_are_written(tmp_path: Path) -> None:
    run_dir = run_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="rb87")
    stress = read_jsonl(run_dir / "stress_test_results.jsonl")
    counterexamples = read_jsonl(run_dir / "counterexample_search_results.jsonl")
    loo = read_jsonl(run_dir / "leave_one_observation_out.jsonl")
    assert stress
    assert counterexamples
    assert loo
    assert any(item["counterexample_found"] for item in counterexamples)


def test_campaign_checkpoint_resume_completes_without_losing_metrics(tmp_path: Path) -> None:
    run_dir = run_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="rb87", stop_after_stage="curation")
    checkpoint = read_json(run_dir / "campaign_checkpoint.json")
    assert checkpoint["stage"] == "curation"
    resumed = resume_atomic_campaign_checkpoint(run_dir / "campaign_checkpoint.json")
    assert resumed == run_dir
    assert validate_atomic_campaign_artifacts(run_dir) == []
    checkpoint = read_json(run_dir / "campaign_checkpoint.json")
    assert checkpoint["resume_count"] == 1
    assert read_json(run_dir / "campaign_metrics.json")["correct_unique_recovery"] is True


def test_campaign_baseline_comparison_records_improvement(tmp_path: Path) -> None:
    experiment = compare_atomic_campaign(PROJECT, runs_dir=tmp_path, experiment_id="compare")
    comparison = read_json(experiment / "campaign_baseline_comparison.json")
    assert comparison["bounded_outcome"] == "improved"
    assert comparison["v19_campaign_aware"]["outcome"] == "correct_unique_recovery"
