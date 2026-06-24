from __future__ import annotations

from pathlib import Path

from coscientist.claim_dag import create_claim_dag_artifacts_from_run, rebuild_claim_dag_database
from coscientist.frontend import create_app
from coscientist.live_agents import (
    claim_dag_mermaid,
    run_live_agent_meeting,
    stream_live_agent_meeting,
    targeted_repair_from_claim_dag,
    validate_meeting_artifacts,
)
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.superconductivity import run_v22_campaign


FIXTURE = "examples/v22_superconductivity_real_data/project.yaml"


def test_v23_fixture_agent_meeting_streams_and_validates(tmp_path: Path) -> None:
    run_dir = tmp_path / "meeting"
    run_live_agent_meeting(run_dir, "Can mixed BCS and correlated hopping be distinguished?", max_rounds=2)
    assert validate_meeting_artifacts(run_dir) == []
    session = read_json(run_dir / "meeting_session.json")
    messages = read_jsonl(run_dir / "meeting_messages.jsonl")
    calls = read_jsonl(run_dir / "provider_calls.jsonl")
    assert session["status"] == "completed"
    assert len(messages) >= 10
    assert any(item["role"] == "critic" for item in messages)
    assert any(item["critic_influenced"] for item in messages)
    assert all(item["permission_mode"] == "fixture" for item in calls)


def test_v23_agent_meeting_stream_yields_incremental_messages(tmp_path: Path) -> None:
    run_dir = tmp_path / "stream-meeting"
    events = stream_live_agent_meeting(run_dir, "Stream one round of agent debate.", max_rounds=1)
    first_transcript, first_rows, first_calls, first_status = next(events)
    assert first_transcript.startswith("# Live Agent Meeting Room")
    assert len(first_rows) == 1
    assert len(first_calls) == 1
    assert first_status["current_round"] == 1
    remaining = list(events)
    assert remaining[-1][3]["status"] == "completed"
    assert validate_meeting_artifacts(run_dir) == []


def test_v23_live_meeting_without_credentials_is_blocked_not_crashed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "blocked-meeting"
    run_live_agent_meeting(run_dir, "Live meeting should require credentials.", live_model=True, max_rounds=1)
    assert validate_meeting_artifacts(run_dir) == []
    session = read_json(run_dir / "meeting_session.json")
    calls = read_jsonl(run_dir / "provider_calls.jsonl")
    assert session["status"] == "live_connection_blocked"
    assert all(item["permission_mode"] == "blocked" for item in calls)
    assert all(item["parsing_result"] == "blocked" for item in calls)


def test_v23_targeted_repair_uses_claim_dag_blocker(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    create_claim_dag_artifacts_from_run(run_dir)
    rebuild_claim_dag_database(run_dir)
    result = targeted_repair_from_claim_dag(run_dir)
    assert result["outcome"] == "accepted"
    assert Path(run_dir / "repair_requests.jsonl").exists()
    assert Path(run_dir / "repair_results.jsonl").exists()
    repairs = read_jsonl(run_dir / "repair_results.jsonl")
    assert repairs[0]["claim_id"] == result["claim_id"]
    assert "evidence_requirements" in repairs[0]["changed_fields"]


def test_v23_claim_dag_mermaid_graph_is_copyable(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    create_claim_dag_artifacts_from_run(run_dir)
    graph = claim_dag_mermaid(run_dir)
    assert graph.startswith("graph TD")
    assert "Total gate:" in graph
    assert "claim-main-" in graph


def test_v23_frontend_facade_exposes_meeting_repair_and_graph(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_v22_fixture(FIXTURE, run_id="v22-ui"))
    app.build_claim_dag_database(run_dir, force=True)
    app.run_live_agent_meeting(run_dir, "Discuss the strongest blocker.", max_rounds=1)
    assert app.validate_live_agent_meeting(run_dir) == []
    assert app.live_agent_message_rows(run_dir)
    assert app.live_agent_provider_rows(run_dir)
    assert "Live Agent Meeting Room" in app.live_agent_transcript(run_dir)
    repair = app.run_targeted_repair(run_dir)
    assert repair["outcome"] == "accepted"
    assert app.claim_dag_mermaid(run_dir).startswith("graph TD")
