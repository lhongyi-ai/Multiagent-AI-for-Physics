from __future__ import annotations

from pathlib import Path

from coscientist.core.action_execution import (
    ScientificAction,
    ScientificActionExecutor,
    ScientificToolRegistry,
    action_execution_rows,
    execute_next_scientific_action,
    validate_action_execution_bundle,
)
from coscientist.frontend import create_app
from coscientist.live_agents import run_live_agent_meeting, validate_meeting_artifacts
from coscientist.pilot.artifacts import read_json, read_jsonl, write_jsonl


DECOMPOSITION_QUESTION = (
    "Determine whether the decomposition of superconducting condensation energy into bare kinetic-energy, "
    "phonon-interaction-energy, and correlated-hopping-energy contributions is uniquely defined, gauge invariant, "
    "and physically observable."
)


def _queue(root: Path, rows: list[dict[str, object]]) -> None:
    queue_dir = root / "optimizer_v2_tool"
    queue_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(queue_dir / "information_gain_queue.jsonl", rows)


def _action(action_id: str = "eig-demo", *, tool_id: str = "phase1_minimal_mixed_bcs_solver", priority: float = 0.9) -> dict[str, object]:
    return ScientificAction(
        action_id=action_id,
        hypothesis_id="hyp-demo",
        domain_id="superconductivity_lsco",
        action_type="run_phase1_mixed_bcs_solver",
        tool_id=tool_id,
        expected_information_gain=0.8,
        success_probability=0.9,
        feasibility=0.9,
        estimated_cost=1.0,
        priority=priority,
        rationale="fixture action",
        expected_artifact_types=["verifier_results"],
        target_claims=["claim-hyp-demo"],
        provenance=["test"],
        idempotency_key=f"idempotent-{action_id}",
    ).model_dump(mode="json")


def test_v28_selects_one_action_generates_bundle_and_updates_dag_optimizer(tmp_path: Path) -> None:
    _queue(tmp_path, [_action("eig-low", priority=0.1), _action("eig-high", priority=0.95)])
    state = execute_next_scientific_action(tmp_path, round_number=1)

    assert state.selected_action_id == "eig-high"
    assert state.status == "succeeded"
    assert state.bundle_dir
    bundle = tmp_path / state.bundle_dir
    assert validate_action_execution_bundle(bundle) == []
    assert (bundle / "action_request.json").exists()
    assert (bundle / "claim_dag_diff.json").exists()
    assert (bundle / "optimizer_diff.json").exists()
    assert read_json(bundle / "claim_dag_diff.json")["added_check_ids"]
    assert read_json(bundle / "optimizer_diff.json")["completed_action_ids"] == ["eig-high"]
    assert (tmp_path / "claim_dag.sqlite").exists()


def test_v28_missing_tool_and_permission_blocked_are_structured(tmp_path: Path) -> None:
    _queue(tmp_path, [_action("eig-missing", tool_id="missing_tool")])
    missing = ScientificActionExecutor().execute_next(tmp_path, round_number=1)
    assert missing.status == "no_eligible_action"
    assert "registered executor" in missing.reason or "missing tool" in missing.reason

    permitted = tmp_path / "permission"
    row = _action("eig-live", tool_id="phase2_data_coverage_tool")
    row["required_permissions"] = ["live_network"]
    _queue(permitted, [row])
    blocked = execute_next_scientific_action(permitted, round_number=1, live_network=False)
    assert blocked.status == "no_eligible_action"
    assert "permission" in blocked.reason


def test_v28_duplicate_deterministic_tool_is_not_repeated(tmp_path: Path) -> None:
    _queue(tmp_path, [_action("eig-a"), _action("eig-b", priority=0.8)])
    first = execute_next_scientific_action(tmp_path, round_number=1)
    second = execute_next_scientific_action(tmp_path, round_number=2)
    assert first.status == "succeeded"
    assert second.status == "no_eligible_action"
    assert "already" in second.reason or "duplicate" in second.reason


def test_v28_energy_audit_guardrail_adds_blocker_and_repair_action(tmp_path: Path) -> None:
    _queue(tmp_path, [_action("eig-decomp", tool_id="energy_decomposition_audit_tool")])
    state = execute_next_scientific_action(tmp_path, round_number=1, phase2_data_path="data/phase2_lsco.csv")
    assert state.status == "succeeded"
    assert state.claim_dag_diff
    assert state.claim_dag_diff["added_blocker_ids"]
    assert "unique_phonon_vs_kinetic_percentage_claim" in state.claim_dag_diff["invalidated_claim_ids"]
    assert state.optimizer_diff
    assert state.optimizer_diff["new_action_ids"]
    blockers = read_jsonl(tmp_path / "validation_blockers.jsonl")
    assert any("representation/model dependent" in row["description"] for row in blockers)


def test_v28_live_agent_demo_executes_actions_and_serializes_frontend_rows(tmp_path: Path) -> None:
    run_dir = run_live_agent_meeting(
        tmp_path / "meeting",
        DECOMPOSITION_QUESTION,
        live_model=False,
        max_rounds=3,
        force=True,
        phase2_data_path="data/phase2_lsco.csv",
    )
    assert validate_meeting_artifacts(run_dir) == []
    rows = action_execution_rows(run_dir)
    assert rows
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["policy_decision"]["execution_mode"] == "deterministic"
    assert "OPENROUTER_API_KEY" not in (Path(run_dir) / "closed_loop_action_executions.jsonl").read_text(encoding="utf-8")
    session = read_json(Path(run_dir) / "meeting_session.json")
    assert session["status"] == "stopped_no_progress"

    service = create_app(runs_dir=tmp_path)
    action_rows = service.live_agent_action_rows(run_dir)
    assert action_rows
    assert action_rows[0]["tool_id"] == "energy_decomposition_audit_tool"
    transcript = service.live_agent_transcript(run_dir)
    assert "Closed-loop Scientific Action Execution" in transcript
    assert "representation dependent" in transcript.lower()


def test_v28_phase2_action_preserves_tier_c_block(tmp_path: Path) -> None:
    _queue(tmp_path, [_action("eig-phase2", tool_id="phase2_data_coverage_tool")])
    state = execute_next_scientific_action(tmp_path, round_number=1, phase2_data_path="data/phase2_lsco.csv")
    assert state.status == "succeeded"
    assert state.claim_dag_diff
    assert "material_level_tier_c_quantitative_separation_claim" in state.claim_dag_diff["invalidated_claim_ids"]
    result = state.execution_result or {}
    assert "Tier C remains blocked" in result["summary"]
