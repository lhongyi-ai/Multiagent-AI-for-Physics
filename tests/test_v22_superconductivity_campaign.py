from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.schemas.v22 import LiveRuntimeConfig, PerRoleModelRoute
from coscientist.superconductivity import query_scientific_index, run_v22_campaign, test_data_connections as run_data_connection_check, validate_v22_campaign


FIXTURE = "examples/v22_superconductivity_real_data/project.yaml"


def test_v22_runtime_schema_requires_explicit_live_permission_for_live_routes() -> None:
    with pytest.raises(ValidationError):
        LiveRuntimeConfig(routes=[PerRoleModelRoute(role="generator", provider="openrouter", model="openrouter/test")])


def test_v22_data_connections_are_fixture_only_without_live_network(tmp_path: Path) -> None:
    results = run_data_connection_check(live_network=False, snapshots_dir=tmp_path)
    assert results
    assert all(item.fixture_status == "fixture_only" for item in results)
    assert all(item.live_status == "blocked" for item in results)
    assert all(item.snapshot_sha256 for item in results)


def test_v22_campaign_runs_validates_and_preserves_honest_live_status(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    assert validate_v22_campaign(run_dir) == []
    registration = read_json(run_dir / "v22_campaign_registration.json")
    assert registration["live_model_enabled"] is False
    providers = read_json(run_dir / "provider_connection_results.json")["providers"]
    assert any(item["provider"] == "openalex" for item in providers)
    assert all(item["live_status"] != "connected" for item in providers)
    discrimination = read_json(run_dir / "theory_discrimination_results.json")
    assert discrimination["status"] == "data_insufficient"
    assert discrimination["nontrivial_outputs"]
    proposal = read_jsonl(run_dir / "experiment_proposals.jsonl")[0]
    assert proposal["composition_or_doping_points"]
    assert "falsification_logic" in proposal
    calls = read_jsonl(run_dir / "model_call_records.jsonl")
    assert all(item["live_call_executed"] is False for item in calls)


def test_v22_scientific_index_contains_new_tables(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22-index")
    provider_rows = query_scientific_index(run_dir, "provider_connection_status", limit=5)
    dialogue_rows = query_scientific_index(run_dir, "dialogue_turns", limit=5)
    assert provider_rows
    assert dialogue_rows


def test_v22_frontend_facade_uses_backend_artifacts(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_v22_fixture(FIXTURE, run_id="v22-ui"))
    assert app.validate_v22(run_dir) == []
    assert app.v22_live_agent_rows(run_dir)
    assert app.v22_database_status_rows(run_dir)
    assert app.v22_material_family_rows(run_dir)
    assert app.v22_fingerprint_rows(run_dir)
    assert app.v22_adversarial_rows(run_dir)
    assert app.v22_experiment_proposal_rows(run_dir)
