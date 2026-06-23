from __future__ import annotations

import json
from pathlib import Path

import pytest

from coscientist.cli import main
from coscientist.pilot.artifacts import validate_v1_artifacts
from coscientist.pilot.runner import CompletedRunError, run_pilot_project_sync


PROJECT = "research-projects/interdisciplinary_fixture/project.yaml"


def test_deterministic_pilot_execution_and_artifacts(tmp_path: Path) -> None:
    run_dir = run_pilot_project_sync(PROJECT, runs_dir=tmp_path, run_id="pilot")
    assert validate_v1_artifacts(run_dir) == []
    final = json.loads((run_dir / "hypotheses_final.json").read_text())
    assert len(final) == 3
    assert final[0]["evidence_links"]
    comparison = json.loads((run_dir / "round_comparison.json").read_text())
    assert comparison["evaluator_self_preference_note"]
    assert (run_dir / "human_review.md").read_text().count("Reviewer Decision") == 4


def test_completed_run_is_immutable_without_force(tmp_path: Path) -> None:
    run_pilot_project_sync(PROJECT, runs_dir=tmp_path, run_id="pilot")
    with pytest.raises(CompletedRunError):
        run_pilot_project_sync(PROJECT, runs_dir=tmp_path, run_id="pilot")


def test_validate_incomplete_or_corrupted_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{bad", encoding="utf-8")
    errors = validate_v1_artifacts(run_dir)
    assert any("missing artifact" in error for error in errors)
    assert any("invalid manifest JSON" in error or "corrupt JSON" in error for error in errors)


def test_v1_cli_commands(tmp_path: Path, capsys) -> None:
    run_id = "cli-pilot"
    assert main(["project-show", PROJECT]) == 0
    assert main(["run-project", PROJECT, "--runs-dir", str(tmp_path), "--run-id", run_id]) == 0
    run_dir = tmp_path / run_id
    assert main(["validate-artifacts", str(run_dir)]) == 0
    assert main(["verify-evidence", str(run_dir)]) == 0
    assert main(["evaluate-run", str(run_dir)]) == 0
    assert main(["compare-rounds", str(run_dir)]) == 0
    assert main(["build-review-package", str(run_dir)]) == 0
    assert "Human review package" in capsys.readouterr().out


def test_live_literature_mode_requires_network_opt_in_for_run_project(tmp_path: Path) -> None:
    assert main([
        "run-project",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--literature-mode",
        "live",
        "--search-providers",
        "openalex",
    ]) == 2
