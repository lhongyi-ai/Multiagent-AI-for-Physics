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
    assert messages[0]["content"].startswith("Tool Call: phase1_minimal_mixed_bcs_solver")
    tool_context = read_json(run_dir / "meeting_tool_context.json")
    assert tool_context["enabled"] is True
    assert tool_context["base_solution"]["stable_superconducting_solution"] is True
    tool_verifiers = read_jsonl(run_dir / "phase1_minimal_bcs_tool" / "minimal_bcs_verifier_results.jsonl")
    assert any(item["verifier_id"] == "hamiltonian_term_verifier" and item["verdict"] == "pass" for item in tool_verifiers)
    assert len(messages) >= 10
    assert any(item["role"] == "critic" for item in messages)
    assert any(item["critic_influenced"] for item in messages)
    assert all(item["permission_mode"] == "fixture" for item in calls)
    diagnostic = read_json(run_dir / "meeting_science_diagnostic.json")
    sketches = read_jsonl(run_dir / "meeting_candidate_model_sketches.jsonl")
    verifier_tasks = read_jsonl(run_dir / "meeting_verifier_tasks.jsonl")
    predictions = read_jsonl(run_dir / "meeting_held_out_predictions.jsonl")
    assert diagnostic["outcome"] == "candidate_requires_verification"
    assert diagnostic["valid_model_sketch_count"] == 1
    assert diagnostic["verifier_task_count"] >= 6
    assert sketches[0]["status"] == "sketch_only"
    assert any(task["verifier_type"] == "free_energy_closure" for task in verifier_tasks)
    assert predictions


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


def test_v23_agent_meeting_imports_phase2_doping_series(tmp_path: Path) -> None:
    csv_path = tmp_path / "phase2.csv"
    csv_path.write_text(
        "\n".join(
            [
                "observation_id,material_family,material_id,doping,observable,value,unit,uncertainty,split,source_id,provenance,usable_for_fit",
                "o1,cuprate,LSCO,x=0.10,tc_k,28,K,,train,local,fixture,true",
                "o2,cuprate,LSCO,x=0.10,gap_ev,0.009,eV,,train,local,fixture,true",
                "o3,cuprate,LSCO,x=0.10,penetration_depth_nm,310,nm,,train,local,fixture,true",
                "o4,cuprate,LSCO,x=0.10,isotope_alpha,0.16,dimensionless,,train,local,fixture,true",
                "o5,cuprate,LSCO,x=0.10,optical_spectral_weight_proxy,0.010,relative,,train,local,fixture,true",
                "o6,cuprate,LSCO,x=0.15,tc_k,38,K,,train,local,fixture,true",
                "o7,cuprate,LSCO,x=0.15,gap_ev,0.012,eV,,train,local,fixture,true",
                "o8,cuprate,LSCO,x=0.15,penetration_depth_nm,250,nm,,train,local,fixture,true",
                "o9,cuprate,LSCO,x=0.15,isotope_alpha,0.08,dimensionless,,train,local,fixture,true",
                "o10,cuprate,LSCO,x=0.15,optical_spectral_weight_proxy,0.020,relative,,train,local,fixture,true",
                "o11,cuprate,LSCO,x=0.20,tc_k,30,K,,test,local,fixture,true",
                "o12,cuprate,LSCO,x=0.20,gap_ev,0.010,eV,,test,local,fixture,true",
                "o13,cuprate,LSCO,x=0.20,penetration_depth_nm,280,nm,,test,local,fixture,true",
                "o14,cuprate,LSCO,x=0.20,isotope_alpha,0.12,dimensionless,,test,local,fixture,true",
                "o15,cuprate,LSCO,x=0.20,optical_spectral_weight_proxy,0.015,relative,,test,local,fixture,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "meeting-phase2"
    run_live_agent_meeting(
        run_dir,
        "Can mixed BCS and correlated hopping be distinguished across doping?",
        max_rounds=1,
        phase2_data_path=csv_path,
    )
    assert validate_meeting_artifacts(run_dir) == []
    context = read_json(run_dir / "meeting_tool_context.json")
    data_tool = context["phase2_data_tool"]
    assert data_tool["status"] == "sufficient_for_held_out_comparison"
    assert data_tool["coverage"]["complete_doping_point_count"] == 3
    observations = read_jsonl(run_dir / "phase2_data_coverage_tool" / "phase2_imported_observations.jsonl")
    missing = read_jsonl(run_dir / "phase2_data_coverage_tool" / "phase2_missing_observables.jsonl")
    assert len(observations) == 15
    assert missing == []


def test_v23_live_meeting_without_credentials_is_blocked_not_crashed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "blocked-meeting"
    run_live_agent_meeting(run_dir, "Live meeting should require credentials.", live_model=True, max_rounds=1)
    assert validate_meeting_artifacts(run_dir) == []
    session = read_json(run_dir / "meeting_session.json")
    messages = read_jsonl(run_dir / "meeting_messages.jsonl")
    calls = read_jsonl(run_dir / "provider_calls.jsonl")
    diagnostic = read_json(run_dir / "meeting_science_diagnostic.json")
    sketches = read_jsonl(run_dir / "meeting_candidate_model_sketches.jsonl")
    assert session["status"] == "live_connection_blocked"
    assert messages == []
    assert all(item["permission_mode"] == "blocked" for item in calls)
    assert all(item["parsing_result"] == "blocked" for item in calls)
    assert diagnostic["outcome"] == "no_new_scientific_knowledge"
    assert sketches == []


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
    app.run_live_agent_meeting(run_dir, "Can a generalized BCS state use phonon attraction and electron-hole-asymmetric correlated hopping?", max_rounds=1)
    assert app.validate_live_agent_meeting(run_dir) == []
    assert app.live_agent_message_rows(run_dir)
    assert app.live_agent_provider_rows(run_dir)
    assert "Live Agent Meeting Room" in app.live_agent_transcript(run_dir)
    assert app.science_progress_diagnostic(run_dir)["outcome"] == "candidate_requires_verification"
    assert app.candidate_model_sketch_rows(run_dir)
    assert app.verifier_task_rows(run_dir)
    assert app.held_out_prediction_rows(run_dir)
    assert app.live_agent_tool_context(run_dir)["enabled"] is True
    assert app.live_agent_tool_call_rows(run_dir)
    assert any(item["verifier_id"] == "free_energy_stability_verifier" for item in app.live_agent_bcs_verifier_rows(run_dir))
    assert app.live_agent_phase2_missing_rows(run_dir)
    assert app.live_agent_phase2_candidate_source_rows(run_dir)
    repair = app.run_targeted_repair(run_dir)
    assert repair["outcome"] == "accepted"
    assert app.claim_dag_mermaid(run_dir).startswith("graph TD")
