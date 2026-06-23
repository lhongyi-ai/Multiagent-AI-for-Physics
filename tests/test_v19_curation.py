from __future__ import annotations

from pathlib import Path

from coscientist.atomic.campaign import curate_atomic_campaign_project, validate_atomic_campaign_artifacts
from coscientist.pilot.artifacts import read_json, read_jsonl


PROJECT = "examples/rb87_real_spectroscopy/project.yaml"


def test_rb87_curation_preserves_sources_hashes_uncertainties_and_conflicts(tmp_path: Path) -> None:
    run_dir = curate_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="curation")
    manifest = read_json(run_dir / "source_manifest.json")
    assert len(manifest["sources"]) == 2
    for source in manifest["sources"]:
        assert source["checksum"]
        assert (run_dir / source["local_snapshot_path"]).exists()
    observations = read_jsonl(run_dir / "atomic_transitions_normalized.jsonl")
    assert any(item["original_unit"] == "MHz" and item["unit"] == "MHz" for item in observations)
    assert all("uncertainty" in item for item in observations)
    conflicts = read_jsonl(run_dir / "curation_conflicts.jsonl")
    assert conflicts
    assert conflicts[0]["conflict_type"] == "duplicate"
    assert "no averaging" in conflicts[0]["description"]


def test_agent_visible_observations_hide_test_records(tmp_path: Path) -> None:
    run_dir = curate_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="curation")
    visible = read_jsonl(run_dir / "agent_visible_observations.jsonl")
    assert visible
    assert all(item["agent_visibility"] == "agent_visible" for item in visible)
    assert all(item["split"] != "test" for item in visible)
    split = read_json(run_dir / "data_split_manifest.json")
    assert split["leakage_checks_passed"] is True
    assert set(split["train_observation_ids"]).isdisjoint(split["test_observation_ids"])


def test_curation_only_run_is_not_complete_campaign(tmp_path: Path) -> None:
    run_dir = curate_atomic_campaign_project(PROJECT, runs_dir=tmp_path, run_id="curation")
    errors = validate_atomic_campaign_artifacts(run_dir)
    assert any("missing campaign artifact" in error for error in errors)
