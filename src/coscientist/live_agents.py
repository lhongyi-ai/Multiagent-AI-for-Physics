from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.providers.base import ProviderError
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.core.action_execution import (
    action_execution_rows,
    execute_next_scientific_action,
    latest_action_state,
    validate_action_execution_bundle,
)
from coscientist.core import run_optimizer_v2
from coscientist.superconductivity.energy_decomposition import run_energy_decomposition_audit, validate_energy_decomposition_audit
from coscientist.superconductivity.minimal_model import run_minimal_mixed_bcs_project, validate_minimal_mixed_bcs_run
from coscientist.superconductivity.phase2_data import run_phase2_data_coverage_tool, validate_phase2_data_coverage_tool
from coscientist.schemas.v23 import (
    AgentDefinition,
    CandidateModelSketch,
    ClaimRepairRequest,
    ClaimRepairResult,
    HeldOutPredictionSpec,
    MeetingAgentResponse,
    MeetingMessage,
    MeetingSession,
    ProviderCallRecordV23,
    ScienceProgressDiagnostic,
    VerifierTaskSpec,
)


MEETING_ARTIFACTS = [
    "agent_definitions.jsonl",
    "meeting_session.json",
    "meeting_messages.jsonl",
    "meeting_agent_responses.jsonl",
    "provider_calls.jsonl",
    "meeting_checkpoints.jsonl",
    "meeting_science_diagnostic.json",
    "meeting_candidate_model_sketches.jsonl",
    "meeting_verifier_tasks.jsonl",
    "meeting_held_out_predictions.jsonl",
    "meeting_tool_context.json",
    "meeting_tool_calls.jsonl",
    "closed_loop_action_state.json",
    "closed_loop_action_executions.jsonl",
]


def default_agent_definitions(*, live_model: bool = False) -> list[AgentDefinition]:
    provider = "openrouter" if live_model else "mock"
    model = _live_model_name() if live_model else "deterministic-mock"
    return [
        AgentDefinition(
            agent_id="pi",
            title="Principal investigator",
            expertise="Frames the research question, preserves scope, and turns debate into next actions.",
            goal="Keep the meeting focused on falsifiable scientific progress.",
            role="principal_investigator",
            provider=provider,
            model=model,
        ),
        AgentDefinition(
            agent_id="theorist",
            title="Theory builder",
            expertise="Builds compact mechanistic and mathematical hypotheses.",
            goal="Propose a testable hypothesis and note what would falsify it.",
            role="theorist",
            provider=provider,
            model=model,
        ),
        AgentDefinition(
            agent_id="critic",
            title="Adversarial critic",
            expertise="Finds unsupported assumptions, duplicate support, and missing controls.",
            goal="Identify the strongest blocker before the group over-commits.",
            role="critic",
            provider=provider,
            model=model,
            temperature=0.1,
        ),
        AgentDefinition(
            agent_id="experimentalist",
            title="Experiment designer",
            expertise="Turns disagreements into targeted measurements or computations.",
            goal="Propose the most discriminating next test.",
            role="experimentalist",
            provider=provider,
            model=model,
        ),
        AgentDefinition(
            agent_id="curator",
            title="Evidence curator",
            expertise="Separates supplied artifacts from inference and flags evidence gaps.",
            goal="Ground the discussion in visible artifacts only.",
            role="curator",
            provider=provider,
            model=model,
            temperature=0.0,
        ),
        AgentDefinition(
            agent_id="closed_loop_action_executor",
            title="Closed-loop action executor",
            expertise="Runs one selected optimizer action before agent discussion and records artifacts.",
            goal="Convert optimizer actions into validated scientific state.",
            role="curator",
            provider="mock",
            model="scientific_action_executor_v28",
            temperature=0.0,
        ),
    ]


def run_live_agent_meeting(
    run_dir: str | Path,
    research_question: str,
    *,
    live_model: bool = False,
    max_rounds: int = 2,
    force: bool = False,
    phase2_data_path: str | Path | None = None,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_meeting_runtime_artifacts(root)
    research_question = research_question.strip()
    if not research_question:
        raise ValueError("research question is required for an agent meeting")
    if (root / "meeting_session.json").exists() and not force:
        return root
    max_rounds = max(1, min(int(max_rounds), 8))
    now = datetime.now(UTC)
    session = MeetingSession(
        meeting_id=f"meeting-{_digest(research_question)}",
        research_question=research_question,
        max_rounds=max_rounds,
        status="live" if live_model else "fixture",
        created_at=now,
        updated_at=now,
    )
    agents = default_agent_definitions(live_model=live_model)
    discussion_agents = [agent for agent in agents if agent.agent_id != "closed_loop_action_executor"]
    messages: list[MeetingMessage] = []
    responses: list[MeetingAgentResponse] = []
    calls: list[ProviderCallRecordV23] = []
    checkpoints: list[dict[str, object]] = []
    tool_context = _prepare_meeting_tool_context(root, session, phase2_data_path=phase2_data_path)
    if tool_context.get("enabled"):
        messages.append(_tool_context_message(session, tool_context))
    live_available = live_model and _live_credentials_available()
    if live_model and not live_available:
        session.status = "live_connection_blocked"
        session.stopping_reason = "live_model_requested_but_openrouter_or_openai_credentials_missing"
        _write_empty_closed_loop_state(root, "not_started_live_model_blocked")
        _write_meeting_artifacts(root, agents, session, messages, responses, _blocked_call_records(session, agents), checkpoints)
        _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=False)
        return root

    for round_number in range(1, max_rounds + 1):
        session.current_round = round_number
        action_state = execute_next_scientific_action(
            root,
            round_number=round_number,
            live_model=live_available,
            live_network=False,
            phase2_data_path=phase2_data_path,
        )
        tool_context = _with_closed_loop_state(tool_context, action_state.model_dump(mode="json"))
        if action_state.selected_action_id:
            action_message = _closed_loop_action_message(session, round_number, action_state.model_dump(mode="json"))
            messages.append(action_message)
        checkpoints.append(
            {
                "schema_version": "v28",
                "meeting_id": session.meeting_id,
                "round_number": round_number,
                "event": "closed_loop_action_execution",
                "selected_action_id": action_state.selected_action_id,
                "status": action_state.status,
                "bundle_dir": action_state.bundle_dir,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        critic_objections = [message.content for message in messages if message.role == "critic"][-2:]
        for agent in discussion_agents:
            prompt = _meeting_prompt(
                session,
                agent,
                round_number=round_number,
                previous_messages=messages[-8:],
                critic_objections=critic_objections,
                tool_context=tool_context,
            )
            try:
                response, call = _call_agent_or_fixture(
                    session,
                    agent,
                    prompt,
                    live_model=live_available,
                    round_number=round_number,
                    critic_objections=critic_objections,
                )
            except ProviderError as exc:
                session.status = "failed"
                session.stopping_reason = f"live_model_call_failed:{_safe_status(str(exc))}"
                calls.append(_call_record(session.meeting_id, agent, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), parsing_result="failed", permission_mode="live_model"))
                _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
                _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=False)
                return root
            content = _format_agent_response(response)
            message = MeetingMessage(
                message_id=f"msg-{round_number:02d}-{agent.agent_id}",
                meeting_id=session.meeting_id,
                round_number=round_number,
                agent_id=agent.agent_id,
                role=agent.role,
                provider=agent.provider,
                model=agent.model,
                content=content,
                cited_artifact_ids=response.cited_artifact_ids,
                critic_influenced=response.changed_by_critic,
                created_at=datetime.now(UTC),
            )
            messages.append(message)
            responses.append(response)
            calls.append(call)
            checkpoints.append(
                {
                    "schema_version": "v23",
                    "meeting_id": session.meeting_id,
                    "round_number": round_number,
                    "agent_id": agent.agent_id,
                    "message_count": len(messages),
                    "critic_influenced": message.critic_influenced,
                    "created_at": message.created_at.isoformat(),
                }
            )
        if _should_stop_no_progress(session, round_number, root):
            session.status = "stopped_no_progress"
            session.stopping_reason = "STOPPED_NO_PROGRESS: two consecutive meeting rounds produced no new executable artifact, counterexample, derivation artifact, or Claim DAG status change after the round-zero tool audit."
            break
    if session.status not in {"stopped_no_progress", "failed", "live_connection_blocked"}:
        session.status = "live_connection_blocked" if live_model and not live_available else "completed"
    session.stopping_reason = session.stopping_reason or "max_rounds_complete"
    session.updated_at = datetime.now(UTC)
    _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
    _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=not live_model)
    return root


def stream_live_agent_meeting(
    run_dir: str | Path,
    research_question: str,
    *,
    live_model: bool = False,
    max_rounds: int = 2,
    force: bool = True,
    phase2_data_path: str | Path | None = None,
) -> Iterator[tuple[str, list[dict[str, object]], list[dict[str, object]], dict[str, object]]]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_meeting_runtime_artifacts(root)
    research_question = research_question.strip()
    if not research_question:
        raise ValueError("research question is required for an agent meeting")
    if (root / "meeting_session.json").exists() and not force:
        messages = read_jsonl(root / "meeting_messages.jsonl")
        yield _transcript_markdown(messages), _meeting_rows(messages), _provider_rows(root), read_json(root / "meeting_session.json")
        return
    max_rounds = max(1, min(int(max_rounds), 8))
    now = datetime.now(UTC)
    session = MeetingSession(
        meeting_id=f"meeting-{_digest(research_question)}",
        research_question=research_question,
        max_rounds=max_rounds,
        status="live" if live_model else "fixture",
        created_at=now,
        updated_at=now,
    )
    agents = default_agent_definitions(live_model=live_model)
    discussion_agents = [agent for agent in agents if agent.agent_id != "closed_loop_action_executor"]
    messages: list[MeetingMessage] = []
    responses: list[MeetingAgentResponse] = []
    calls: list[ProviderCallRecordV23] = []
    checkpoints: list[dict[str, object]] = []
    tool_context = _prepare_meeting_tool_context(root, session, phase2_data_path=phase2_data_path)
    if tool_context.get("enabled"):
        messages.append(_tool_context_message(session, tool_context))
    live_available = live_model and _live_credentials_available()
    if live_model and not live_available:
        session.status = "live_connection_blocked"
        session.stopping_reason = "live_model_requested_but_openrouter_or_openai_credentials_missing"
        calls = _blocked_call_records(session, agents)
        _write_empty_closed_loop_state(root, "not_started_live_model_blocked")
        _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
        _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=False)
        yield _status_transcript(session), [], _provider_rows(root), read_json(root / "meeting_session.json")
        return
    _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
    if messages:
        message_rows = [item.model_dump(mode="json") for item in messages]
        yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
    for round_number in range(1, max_rounds + 1):
        session.current_round = round_number
        action_state = execute_next_scientific_action(
            root,
            round_number=round_number,
            live_model=live_available,
            live_network=False,
            phase2_data_path=phase2_data_path,
        )
        tool_context = _with_closed_loop_state(tool_context, action_state.model_dump(mode="json"))
        if action_state.selected_action_id:
            action_message = _closed_loop_action_message(session, round_number, action_state.model_dump(mode="json"))
            messages.append(action_message)
        checkpoints.append(
            {
                "schema_version": "v28",
                "meeting_id": session.meeting_id,
                "round_number": round_number,
                "event": "closed_loop_action_execution",
                "selected_action_id": action_state.selected_action_id,
                "status": action_state.status,
                "bundle_dir": action_state.bundle_dir,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        session.updated_at = datetime.now(UTC)
        if action_state.selected_action_id:
            _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
            message_rows = [item.model_dump(mode="json") for item in messages]
            yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
        critic_objections = [message.content for message in messages if message.role == "critic"][-2:]
        for agent in discussion_agents:
            prompt = _meeting_prompt(
                session,
                agent,
                round_number=round_number,
                previous_messages=messages[-8:],
                critic_objections=critic_objections,
                tool_context=tool_context,
            )
            try:
                response, call = _call_agent_or_fixture(
                    session,
                    agent,
                    prompt,
                    live_model=live_available,
                    round_number=round_number,
                    critic_objections=critic_objections,
                )
            except ProviderError as exc:
                session.status = "failed"
                session.stopping_reason = f"live_model_call_failed:{_safe_status(str(exc))}"
                calls.append(_call_record(session.meeting_id, agent, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), parsing_result="failed", permission_mode="live_model"))
                _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
                _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=False)
                message_rows = [item.model_dump(mode="json") for item in messages]
                yield _transcript_markdown(message_rows) if message_rows else _status_transcript(session), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
                return
            message = MeetingMessage(
                message_id=f"msg-{round_number:02d}-{agent.agent_id}",
                meeting_id=session.meeting_id,
                round_number=round_number,
                agent_id=agent.agent_id,
                role=agent.role,
                provider=agent.provider,
                model=agent.model,
                content=_format_agent_response(response),
                cited_artifact_ids=response.cited_artifact_ids,
                critic_influenced=response.changed_by_critic,
                created_at=datetime.now(UTC),
            )
            messages.append(message)
            responses.append(response)
            calls.append(call)
            checkpoints.append(
                {
                    "schema_version": "v23",
                    "meeting_id": session.meeting_id,
                    "round_number": round_number,
                    "agent_id": agent.agent_id,
                    "message_count": len(messages),
                    "critic_influenced": message.critic_influenced,
                    "created_at": message.created_at.isoformat(),
                }
            )
            session.updated_at = datetime.now(UTC)
            _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
            message_rows = [item.model_dump(mode="json") for item in messages]
            yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
        if _should_stop_no_progress(session, round_number, root):
            session.status = "stopped_no_progress"
            session.stopping_reason = "STOPPED_NO_PROGRESS: two consecutive meeting rounds produced no new executable artifact, counterexample, derivation artifact, or Claim DAG status change after the round-zero tool audit."
            session.updated_at = datetime.now(UTC)
            _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
            _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=not live_model)
            message_rows = [item.model_dump(mode="json") for item in messages]
            yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
            break
    if session.status not in {"stopped_no_progress", "failed", "live_connection_blocked"}:
        session.status = "live_connection_blocked" if live_model and not live_available else "completed"
    session.stopping_reason = session.stopping_reason or "max_rounds_complete"
    session.updated_at = datetime.now(UTC)
    _write_meeting_artifacts(root, agents, session, messages, responses, calls, checkpoints)
    _write_science_progress_artifacts(root, session, messages, responses, allow_deterministic_scaffold=not live_model)
    message_rows = [item.model_dump(mode="json") for item in messages]
    yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")


def validate_meeting_artifacts(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    errors = [f"missing meeting artifact: {name}" for name in MEETING_ARTIFACTS if not (root / name).exists()]
    if errors:
        return errors
    try:
        session = MeetingSession.model_validate_json((root / "meeting_session.json").read_text(encoding="utf-8"))
        agents = _validate_jsonl(root / "agent_definitions.jsonl", AgentDefinition)
        messages = _validate_jsonl(root / "meeting_messages.jsonl", MeetingMessage)
        _validate_jsonl(root / "meeting_agent_responses.jsonl", MeetingAgentResponse)
        calls = _validate_jsonl(root / "provider_calls.jsonl", ProviderCallRecordV23)
        diagnostic = ScienceProgressDiagnostic.model_validate_json((root / "meeting_science_diagnostic.json").read_text(encoding="utf-8"))
        _validate_jsonl(root / "meeting_candidate_model_sketches.jsonl", CandidateModelSketch)
        _validate_jsonl(root / "meeting_verifier_tasks.jsonl", VerifierTaskSpec)
        _validate_jsonl(root / "meeting_held_out_predictions.jsonl", HeldOutPredictionSpec)
        state = latest_action_state(root)
        if not state:
            errors.append("missing closed-loop action state")
    except Exception as exc:
        return [f"invalid meeting artifact: {exc}"]
    agent_ids = {agent.agent_id for agent in agents}
    if not messages and session.status not in {"live_connection_blocked", "failed", "stopped_no_progress"}:
        errors.append("meeting has no messages")
    for message in messages:
        if message.meeting_id != session.meeting_id:
            errors.append(f"message references wrong meeting: {message.message_id}")
        if message.agent_id not in agent_ids:
            errors.append(f"message references missing agent: {message.agent_id}")
    if session.status == "live_connection_blocked" and any(call.permission_mode == "live_model" for call in calls):
        errors.append("blocked live meeting recorded live model calls")
    if session.status == "completed" and session.current_round > session.max_rounds:
        errors.append("meeting exceeded max rounds")
    if diagnostic.outcome == "ready_for_frozen_campaign" and diagnostic.valid_model_sketch_count == 0:
        errors.append("ready campaign diagnostic lacks a model sketch")
    if _contains_secret_like_text(root, ["meeting_session.json", "meeting_messages.jsonl", "provider_calls.jsonl"]):
        errors.append("secret-like content appears in meeting artifacts")
    for bundle in sorted((root / "action_executions").glob("*")) if (root / "action_executions").exists() else []:
        if bundle.is_dir():
            errors.extend(f"{bundle.name}: {error}" for error in validate_action_execution_bundle(bundle))
    return errors


def targeted_repair_from_claim_dag(
    run_dir: str | Path,
    *,
    claim_id: str | None = None,
    reason: str = "",
    requested_by: str = "frontend",
) -> dict[str, object]:
    root = Path(run_dir)
    blockers = read_jsonl(root / "validation_blockers.jsonl") if (root / "validation_blockers.jsonl").exists() else []
    claims = read_jsonl(root / "atomic_claims.jsonl") if (root / "atomic_claims.jsonl").exists() else []
    selected = claim_id or next((str(item.get("claim_id")) for item in blockers if item.get("claim_id")), "")
    if not selected:
        return {"outcome": "blocked", "reason": "no claim blocker is available for targeted repair"}
    candidate_id = next((str(item.get("candidate_id")) for item in claims if item.get("claim_id") == selected), "candidate")
    blocker_text = "; ".join(str(item.get("description")) for item in blockers if item.get("claim_id") == selected)
    repair_reason = reason.strip() or blocker_text or "verifier requested targeted repair"
    now = datetime.now(UTC)
    digest = _digest(f"{selected}:{repair_reason}:{now.isoformat()}")[:10]
    request = ClaimRepairRequest(
        repair_request_id=f"repair-request-{digest}",
        candidate_id=candidate_id,
        claim_id=selected,
        reason=repair_reason,
        requested_by=requested_by,
        created_at=now,
    )
    result = ClaimRepairResult(
        repair_result_id=f"repair-result-{digest}",
        parent_candidate_id=candidate_id,
        child_candidate_id=f"{candidate_id}-repair-{digest}",
        claim_id=selected,
        changed_fields=["falsification_conditions", "evidence_requirements", "required_measurements"],
        invalidated_claim_ids=[selected],
        outcome="accepted",
        created_at=now,
    )
    _append_jsonl(root / "repair_requests.jsonl", request)
    _append_jsonl(root / "repair_results.jsonl", result)
    _write_repair_chain(root, result)
    return {
        "outcome": result.outcome,
        "repair_request_id": request.repair_request_id,
        "repair_result_id": result.repair_result_id,
        "parent_candidate_id": result.parent_candidate_id,
        "child_candidate_id": result.child_candidate_id,
        "claim_id": selected,
        "changed_fields": result.changed_fields,
        "reason": repair_reason,
    }


def claim_dag_mermaid(run_dir: str | Path) -> str:
    root = Path(run_dir)
    if not (root / "claim_dag.json").exists():
        return "graph TD\n  missing[Claim DAG not built yet]"
    dag = read_json(root / "claim_dag.json")
    nodes = {read_json(root / "formal_claim.json")["claim_id"]: read_json(root / "formal_claim.json")["scoped_main_claim"]}
    for claim in read_jsonl(root / "atomic_claims.jsonl"):
        nodes[claim["claim_id"]] = claim["statement"]
    lines = ["graph TD"]
    for claim_id, statement in nodes.items():
        label = _mermaid_label(claim_id, statement)
        lines.append(f"  { _mermaid_id(claim_id) }[{label}]")
    for edge in dag.get("dependency_edges", []):
        lines.append(f"  {_mermaid_id(edge['parent_claim_id'])} -->|{edge['dependency_type']}| {_mermaid_id(edge['child_claim_id'])}")
    if (root / "total_gate_result.json").exists():
        gate = read_json(root / "total_gate_result.json")
        lines.append(f"  gate[Total gate: {gate.get('terminal_status')}]")
        lines.append(f"  gate -. blocks .-> {_mermaid_id(dag.get('weakest_link_claim_id') or dag.get('main_claim_id'))}")
    return "\n".join(lines)


def meeting_message_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "meeting_messages.jsonl"
    return _meeting_rows(read_jsonl(path)) if path.exists() else []


def provider_call_rows(run_dir: str | Path) -> list[dict[str, object]]:
    return _provider_rows(Path(run_dir))


def meeting_transcript(run_dir: str | Path) -> str:
    path = Path(run_dir) / "meeting_messages.jsonl"
    return _transcript_markdown(read_jsonl(path)) if path.exists() else ""


def science_progress_diagnostic(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir) / "meeting_science_diagnostic.json"
    return read_json(path) if path.exists() else {}


def candidate_model_sketch_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "meeting_candidate_model_sketches.jsonl"
    return read_jsonl(path) if path.exists() else []


def verifier_task_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "meeting_verifier_tasks.jsonl"
    return read_jsonl(path) if path.exists() else []


def held_out_prediction_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "meeting_held_out_predictions.jsonl"
    return read_jsonl(path) if path.exists() else []


def meeting_tool_context(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir) / "meeting_tool_context.json"
    return read_json(path) if path.exists() else {}


def meeting_tool_call_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "meeting_tool_calls.jsonl"
    return read_jsonl(path) if path.exists() else []


def closed_loop_action_rows(run_dir: str | Path) -> list[dict[str, object]]:
    rows = []
    for item in action_execution_rows(run_dir):
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        policy = item.get("policy_decision") if isinstance(item.get("policy_decision"), dict) else {}
        result = item.get("execution_result") if isinstance(item.get("execution_result"), dict) else {}
        dag = item.get("claim_dag_diff") if isinstance(item.get("claim_dag_diff"), dict) else {}
        optimizer = item.get("optimizer_diff") if isinstance(item.get("optimizer_diff"), dict) else {}
        rows.append(
            {
                "selected_action_id": item.get("selected_action_id"),
                "tool_id": action.get("tool_id"),
                "status": item.get("status"),
                "expected_information_gain": action.get("expected_information_gain"),
                "priority": action.get("priority"),
                "policy_status": policy.get("status"),
                "execution_mode": policy.get("execution_mode"),
                "duration_ms": result.get("duration_ms"),
                "generated_artifacts": result.get("generated_artifacts"),
                "verifier_count": len(result.get("verifier_results") or []),
                "claim_dag_checks": dag.get("added_check_ids"),
                "claim_dag_blockers": dag.get("added_blocker_ids"),
                "optimizer_new_actions": optimizer.get("new_action_ids"),
                "bundle_dir": item.get("bundle_dir"),
            }
        )
    return rows


def closed_loop_latest_state(run_dir: str | Path) -> dict[str, object]:
    return latest_action_state(run_dir)


def meeting_bcs_verifier_result_rows(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "phase1_minimal_bcs_tool" / "minimal_bcs_verifier_results.jsonl"
    return read_jsonl(path) if path.exists() else []


def _prepare_meeting_tool_context(root: Path, session: MeetingSession, *, phase2_data_path: str | Path | None = None) -> dict[str, object]:
    if not _is_bcs_question(session.research_question):
        context = {
            "schema_version": "v24-meeting-tool-context",
            "enabled": False,
            "reason": "no domain-specific executable tool matched the research question",
        }
        write_json(root / "meeting_tool_context.json", context)
        write_jsonl(root / "meeting_tool_calls.jsonl", [])
        return context

    tool_run_id = "phase1_minimal_bcs_tool"
    tool_dir = run_minimal_mixed_bcs_project(runs_dir=root, run_id=tool_run_id, force=True)
    validation_errors = validate_minimal_mixed_bcs_run(tool_dir)
    phase2_tool_id = "phase2_data_coverage_tool"
    phase2_tool_dir = run_phase2_data_coverage_tool(root / phase2_tool_id, source_path=phase2_data_path)
    phase2_tool_errors = validate_phase2_data_coverage_tool(phase2_tool_dir)
    decomposition_tool_id = "energy_decomposition_audit_tool"
    decomposition_enabled = _is_decomposition_uniqueness_question(session.research_question)
    decomposition_tool_dir = run_energy_decomposition_audit(root / decomposition_tool_id) if decomposition_enabled else None
    decomposition_errors = validate_energy_decomposition_audit(decomposition_tool_dir) if decomposition_tool_dir else []
    decomposition_summary = read_json(decomposition_tool_dir / "audit_summary.json") if decomposition_tool_dir else {}
    decomposition_outcome = read_json(decomposition_tool_dir / "final_theory_outcome.json") if decomposition_tool_dir else {}
    decomposition_verifiers = read_jsonl(decomposition_tool_dir / "energy_decomposition_verifier_results.jsonl") if decomposition_tool_dir else []
    optimizer_tool_id = "optimizer_v2_tool"
    optimizer_result = run_optimizer_v2(
        _meeting_optimizer_hypotheses(session.research_question),
        run_dir=root / optimizer_tool_id,
        domain_id="superconductivity_lsco",
        run_id=optimizer_tool_id,
    )
    hamiltonian = read_json(tool_dir / "hamiltonian_spec.json")
    base = read_json(tool_dir / "base_solution.json")
    phase = read_json(tool_dir / "phase_diagram_summary.json")
    verifier_results = read_jsonl(tool_dir / "minimal_bcs_verifier_results.jsonl")
    phase2 = read_json(tool_dir / "phase2_held_out_evaluation.json")
    phase2_data_eval = read_json(phase2_tool_dir / "phase2_data_tool_evaluation.json")
    phase2_coverage = read_json(phase2_tool_dir / "phase2_data_coverage.json")
    phase2_missing = read_jsonl(phase2_tool_dir / "phase2_missing_observables.jsonl")
    passed = [item["verifier_id"] for item in verifier_results if item["verdict"] == "pass"]
    blocked = [item["verifier_id"] for item in verifier_results if item["verdict"] in {"fail", "inconclusive"}]
    context = {
        "schema_version": "v24-meeting-tool-context",
        "enabled": True,
        "tool_name": "phase1_minimal_mixed_bcs_solver",
        "tool_run_dir": tool_run_id,
        "validation": "valid" if not validation_errors else "invalid",
        "validation_errors": validation_errors,
        "artifact_ids": [
            f"{tool_run_id}/hamiltonian_spec.json",
            f"{tool_run_id}/solution_method.json",
            f"{tool_run_id}/base_solution.json",
            f"{tool_run_id}/energy_ledger.json",
            f"{tool_run_id}/four_model_comparison.jsonl",
            f"{tool_run_id}/phase_diagram_summary.json",
            f"{tool_run_id}/minimal_bcs_verifier_results.jsonl",
            f"{tool_run_id}/phase2_held_out_evaluation.json",
            f"{phase2_tool_id}/phase2_data_tool_evaluation.json",
            f"{phase2_tool_id}/phase2_data_coverage.json",
            f"{phase2_tool_id}/phase2_imported_observations.jsonl",
            f"{phase2_tool_id}/phase2_missing_observables.jsonl",
            f"{phase2_tool_id}/phase2_data_template.csv",
            f"{optimizer_tool_id}/optimizer_v2_summary.json",
            f"{optimizer_tool_id}/hypothesis_portfolio_v2.jsonl",
            f"{optimizer_tool_id}/information_gain_queue.jsonl",
            f"{optimizer_tool_id}/failure_memory.jsonl",
            f"{optimizer_tool_id}/hard_gate_results.jsonl",
            *(
                [
                    f"{decomposition_tool_id}/research_objective.json",
                    f"{decomposition_tool_id}/gauge_coupled_hamiltonian.json",
                    f"{decomposition_tool_id}/electromagnetic_response_derivation.json",
                    f"{decomposition_tool_id}/representation_counterexample.json",
                    f"{decomposition_tool_id}/hellmann_feynman_diagnostics.json",
                    f"{decomposition_tool_id}/observable_classification.json",
                    f"{decomposition_tool_id}/energy_decomposition_verifier_results.jsonl",
                    f"{decomposition_tool_id}/claim_dag_seed_tasks.jsonl",
                    f"{decomposition_tool_id}/final_theory_outcome.json",
                    f"{decomposition_tool_id}/audit_summary.json",
                ]
                if decomposition_enabled
                else []
            ),
        ],
        "hamiltonian_terms": hamiltonian.get("terms", {}),
        "base_solution": {
            "gap_ev": base.get("gap_ev"),
            "stable_superconducting_solution": base.get("stable_superconducting_solution"),
            "free_energy_change_ev": base.get("free_energy_change_ev"),
            "delta_kinetic_ev": base.get("delta_kinetic_ev"),
            "delta_phonon_interaction_ev": base.get("delta_phonon_interaction_ev"),
            "delta_correlated_hopping_ev": base.get("delta_correlated_hopping_ev"),
            "isotope_exponent": base.get("isotope_exponent"),
        },
        "phase_diagram_summary": {
            "point_count": phase.get("point_count"),
            "stable_point_count": phase.get("stable_point_count"),
            "mixed_kinetic_phonon_correlated_gain_count": phase.get("mixed_kinetic_phonon_correlated_gain_count"),
        },
        "verifier_summary": {
            "passed": passed,
            "blocked_or_inconclusive": blocked,
        },
        "phase2_status": phase2_data_eval.get("status"),
        "phase2_conclusion": phase2_data_eval.get("conclusion"),
        "phase1_material_level_phase2_status": phase2.get("status"),
        "phase1_material_level_phase2_conclusion": phase2.get("conclusion"),
        "phase2_data_tool": {
            "validation": "valid" if not phase2_tool_errors else "invalid",
            "validation_errors": phase2_tool_errors,
            "status": phase2_data_eval.get("status"),
            "import_status": phase2_data_eval.get("import_status"),
            "coverage": phase2_coverage,
            "missing_observable_count": len(phase2_missing),
            "missing_observables_preview": phase2_missing[:5],
        },
        "optimizer_v2": {
            "tool_run_dir": optimizer_tool_id,
            "hypothesis_count": optimizer_result.hypothesis_count,
            "killed_count": optimizer_result.killed_count,
            "pareto_frontier_ids": optimizer_result.pareto_frontier_ids,
            "portfolio": [item.model_dump(mode="json") for item in optimizer_result.portfolio[:5]],
            "top_information_gain_actions": [item.model_dump(mode="json") for item in optimizer_result.information_gain_actions[:5]],
            "failure_memory": [item.model_dump(mode="json") for item in optimizer_result.failure_memory[:5]],
        },
        "energy_decomposition_audit": {
            "enabled": decomposition_enabled,
            "tool_run_dir": decomposition_tool_id if decomposition_enabled else None,
            "validation": "valid" if decomposition_enabled and not decomposition_errors else "not_run" if not decomposition_enabled else "invalid",
            "validation_errors": decomposition_errors,
            "summary": decomposition_summary,
            "outcome": decomposition_outcome,
            "verifier_results": decomposition_verifiers,
        },
    }
    write_json(root / "meeting_tool_context.json", context)
    write_jsonl(
        root / "meeting_tool_calls.jsonl",
        [
            {
                "schema_version": "v24",
                "tool_call_id": f"tool-{session.meeting_id}-phase1-bcs",
                "meeting_id": session.meeting_id,
                "tool_name": "phase1_minimal_mixed_bcs_solver",
                "status": "complete" if not validation_errors else "invalid",
                "artifact_dir": tool_run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "summary": _tool_prompt_summary(context),
            },
            {
                "schema_version": "v24",
                "tool_call_id": f"tool-{session.meeting_id}-phase2-data",
                "meeting_id": session.meeting_id,
                "tool_name": "phase2_data_coverage_tool",
                "status": "complete" if not phase2_tool_errors else "invalid",
                "artifact_dir": phase2_tool_id,
                "created_at": datetime.now(UTC).isoformat(),
                "summary": _phase2_tool_summary(context),
            },
            {
                "schema_version": "v26",
                "tool_call_id": f"tool-{session.meeting_id}-optimizer-v2",
                "meeting_id": session.meeting_id,
                "tool_name": "hypothesis_optimizer_v2",
                "status": "complete",
                "artifact_dir": optimizer_tool_id,
                "created_at": datetime.now(UTC).isoformat(),
                "summary": _optimizer_tool_summary(context),
            },
            *(
                [
                    {
                        "schema_version": "v27",
                        "tool_call_id": f"tool-{session.meeting_id}-energy-decomposition",
                        "meeting_id": session.meeting_id,
                        "tool_name": "energy_decomposition_audit",
                        "status": "complete" if not decomposition_errors else "invalid",
                        "artifact_dir": decomposition_tool_id,
                        "created_at": datetime.now(UTC).isoformat(),
                        "summary": _energy_decomposition_summary(context),
                    }
                ]
                if decomposition_enabled
                else []
            ),
        ],
    )
    return context


def _tool_context_message(session: MeetingSession, tool_context: dict[str, object]) -> MeetingMessage:
    return MeetingMessage(
        message_id="msg-00-executable-theory-tools",
        meeting_id=session.meeting_id,
        round_number=0,
        agent_id="curator",
        role="curator",
        provider="deterministic-tool",
        model="phase1_minimal_mixed_bcs_solver+energy_decomposition_audit",
        content="Tool Call: phase1_minimal_mixed_bcs_solver\nTool Call: executable_theory_tools\n" + _tool_prompt_summary(tool_context),
        cited_artifact_ids=list(tool_context.get("artifact_ids", [])),
        critic_influenced=False,
        created_at=datetime.now(UTC),
    )


def _with_closed_loop_state(tool_context: dict[str, object], action_state: dict[str, object]) -> dict[str, object]:
    updated = dict(tool_context)
    prior = updated.get("closed_loop_actions")
    actions = list(prior) if isinstance(prior, list) else []
    actions.append(action_state)
    updated["closed_loop_actions"] = actions[-4:]
    updated["latest_closed_loop_action"] = action_state
    artifact_ids = list(updated.get("artifact_ids", []))
    bundle = action_state.get("bundle_dir")
    if isinstance(bundle, str) and bundle:
        artifact_ids.extend(
            [
                f"{bundle}/action_request.json",
                f"{bundle}/execution_result.json",
                f"{bundle}/verifier_results.jsonl",
                f"{bundle}/claim_dag_diff.json",
                f"{bundle}/optimizer_diff.json",
                f"{bundle}/execution_summary.json",
            ]
        )
    updated["artifact_ids"] = sorted(dict.fromkeys(str(item) for item in artifact_ids))
    return updated


def _closed_loop_action_message(session: MeetingSession, round_number: int, action_state: dict[str, object]) -> MeetingMessage:
    return MeetingMessage(
        message_id=f"msg-{round_number:02d}-closed-loop-action",
        meeting_id=session.meeting_id,
        round_number=round_number,
        agent_id="closed_loop_action_executor",
        role="curator",
        provider="deterministic-tool",
        model="scientific_action_executor_v28",
        content=_closed_loop_action_summary(action_state),
        cited_artifact_ids=_closed_loop_artifact_ids(action_state),
        critic_influenced=False,
        created_at=datetime.now(UTC),
    )


def _closed_loop_artifact_ids(action_state: dict[str, object]) -> list[str]:
    bundle = action_state.get("bundle_dir")
    if not isinstance(bundle, str) or not bundle:
        return []
    return [
        f"{bundle}/action_request.json",
        f"{bundle}/execution_result.json",
        f"{bundle}/verifier_results.jsonl",
        f"{bundle}/claim_dag_diff.json",
        f"{bundle}/optimizer_diff.json",
        f"{bundle}/execution_summary.json",
    ]


def _closed_loop_action_summary(action_state: dict[str, object]) -> str:
    action = action_state.get("action") if isinstance(action_state.get("action"), dict) else {}
    result = action_state.get("execution_result") if isinstance(action_state.get("execution_result"), dict) else {}
    policy = action_state.get("policy_decision") if isinstance(action_state.get("policy_decision"), dict) else {}
    dag = action_state.get("claim_dag_diff") if isinstance(action_state.get("claim_dag_diff"), dict) else {}
    optimizer = action_state.get("optimizer_diff") if isinstance(action_state.get("optimizer_diff"), dict) else {}
    next_actions = action_state.get("next_eligible_actions") if isinstance(action_state.get("next_eligible_actions"), list) else []
    lines = [
        "Closed-loop Scientific Action Execution",
        f"- selected_action_id: {action_state.get('selected_action_id')}",
        f"- status: {action_state.get('status')}",
        f"- reason: {action_state.get('reason')}",
    ]
    if action:
        lines.extend(
            [
                f"- action_type: {action.get('action_type')}",
                f"- tool_id: {action.get('tool_id')}",
                f"- expected_information_gain: {action.get('expected_information_gain')}",
                f"- priority: {action.get('priority')}",
                f"- rationale: {action.get('rationale')}",
            ]
        )
    if policy:
        lines.extend(
            [
                f"- policy: {policy.get('status')}",
                f"- required_permissions: {policy.get('required_permissions')}",
                f"- execution_mode: {policy.get('execution_mode')}",
            ]
        )
    if result:
        lines.extend(
            [
                f"- generated_artifacts: {result.get('generated_artifacts')}",
                f"- verifier_count: {len(result.get('verifier_results') or [])}",
                f"- errors: {result.get('errors')}",
            ]
        )
    if dag:
        lines.extend(
            [
                f"- claim_dag_checks_added: {dag.get('added_check_ids')}",
                f"- claim_dag_blockers_added: {dag.get('added_blocker_ids')}",
                f"- invalidated_claims: {dag.get('invalidated_claim_ids')}",
            ]
        )
    if optimizer:
        lines.extend(
            [
                f"- optimizer_completed_actions: {optimizer.get('completed_action_ids')}",
                f"- optimizer_new_actions: {optimizer.get('new_action_ids')}",
                f"- optimizer_downgraded_actions: {optimizer.get('downgraded_action_ids')}",
            ]
        )
    if next_actions:
        lines.append(f"- next_eligible_actions: {[item.get('action_id') for item in next_actions[:3] if isinstance(item, dict)]}")
    lines.append("Agents must treat this as completed state, not as future work.")
    return "\n".join(lines)


def _tool_prompt_summary(tool_context: dict[str, object]) -> str:
    if not tool_context.get("enabled"):
        return str(tool_context.get("reason") or "none")
    terms = tool_context.get("hamiltonian_terms", {})
    base = tool_context.get("base_solution", {})
    phase = tool_context.get("phase_diagram_summary", {})
    verifiers = tool_context.get("verifier_summary", {})
    artifact_ids = tool_context.get("artifact_ids", [])
    lines = [
        "Executable Phase-1 BCS tool has already run. Do not describe Hamiltonian construction as future work.",
        "Hamiltonian terms:",
    ]
    if isinstance(terms, dict):
        for key in ["H_t", "H_U", "H_ph", "H_e_ph", "H_corr"]:
            if key in terms:
                lines.append(f"- {key}: {terms[key]}")
    if isinstance(base, dict):
        lines.extend(
            [
                "Base numerical solution:",
                f"- gap_ev: {base.get('gap_ev')}",
                f"- stable_superconducting_solution: {base.get('stable_superconducting_solution')}",
                f"- free_energy_change_ev: {base.get('free_energy_change_ev')}",
                f"- delta_kinetic_ev: {base.get('delta_kinetic_ev')}",
                f"- delta_phonon_interaction_ev: {base.get('delta_phonon_interaction_ev')}",
                f"- delta_correlated_hopping_ev: {base.get('delta_correlated_hopping_ev')}",
                f"- isotope_exponent: {base.get('isotope_exponent')}",
            ]
        )
    if isinstance(phase, dict):
        lines.extend(
            [
                "Parameter scan:",
                f"- point_count: {phase.get('point_count')}",
                f"- stable_point_count: {phase.get('stable_point_count')}",
                f"- mixed_kinetic_phonon_correlated_gain_count: {phase.get('mixed_kinetic_phonon_correlated_gain_count')}",
            ]
        )
    if isinstance(verifiers, dict):
        lines.extend(
            [
                "Executable verifier summary:",
                f"- passed: {', '.join(verifiers.get('passed', []))}",
                f"- blocked_or_inconclusive: {', '.join(verifiers.get('blocked_or_inconclusive', []))}",
            ]
        )
    lines.extend(
        [
            f"Phase 2 status: {tool_context.get('phase2_status')}",
            f"Phase 2 conclusion: {tool_context.get('phase2_conclusion')}",
            _phase2_tool_summary(tool_context),
            _energy_decomposition_summary(tool_context),
            _optimizer_tool_summary(tool_context),
            _closed_loop_prompt_summary(tool_context),
            "Citable artifact IDs: " + ", ".join(str(item) for item in artifact_ids),
        ]
    )
    return "\n".join(lines)


def _closed_loop_prompt_summary(tool_context: dict[str, object]) -> str:
    state = tool_context.get("latest_closed_loop_action")
    if not isinstance(state, dict):
        return "Closed-loop action execution: no action has executed yet"
    action = state.get("action") if isinstance(state.get("action"), dict) else {}
    result = state.get("execution_result") if isinstance(state.get("execution_result"), dict) else {}
    dag = state.get("claim_dag_diff") if isinstance(state.get("claim_dag_diff"), dict) else {}
    optimizer = state.get("optimizer_diff") if isinstance(state.get("optimizer_diff"), dict) else {}
    lines = [
        "Closed-loop action execution:",
        f"- selected_action_id: {state.get('selected_action_id')}",
        f"- status: {state.get('status')}",
        f"- tool_id: {action.get('tool_id') if action else None}",
        f"- summary: {result.get('summary') if result else state.get('reason')}",
        f"- generated_artifacts: {result.get('generated_artifacts') if result else []}",
        f"- claim_dag_diff: checks={dag.get('added_check_ids') if dag else []}, blockers={dag.get('added_blocker_ids') if dag else []}, invalidated={dag.get('invalidated_claim_ids') if dag else []}",
        f"- optimizer_diff: completed={optimizer.get('completed_action_ids') if optimizer else []}, new={optimizer.get('new_action_ids') if optimizer else []}",
        "Hard rule: do not restate this selected action as something to do next; use its result to choose the next claim, equation, counterexample, or blocker.",
    ]
    return "\n".join(lines)


def _energy_decomposition_summary(tool_context: dict[str, object]) -> str:
    audit = tool_context.get("energy_decomposition_audit")
    if not isinstance(audit, dict) or not audit.get("enabled"):
        return "Energy decomposition audit: not run"
    summary = audit.get("summary", {})
    outcome = audit.get("outcome", {})
    verifiers = audit.get("verifier_results", [])
    passed = [item.get("verifier_id") for item in verifiers if item.get("verdict") == "pass"]
    failed = [item.get("verifier_id") for item in verifiers if item.get("verdict") == "fail"]
    lines = [
        "Energy decomposition audit:",
        f"- status: {summary.get('status') if isinstance(summary, dict) else None}",
        f"- valid_outcome_type: {summary.get('valid_outcome_type') if isinstance(summary, dict) else None}",
        f"- claim: {outcome.get('claim') if isinstance(outcome, dict) else None}",
        f"- passed_verifiers: {passed}",
        f"- failed_or_refuted_claims: {failed}",
        f"- invariant_quantities: {outcome.get('invariant_quantities') if isinstance(outcome, dict) else []}",
        f"- model_dependent_quantities: {outcome.get('model_dependent_quantities') if isinstance(outcome, dict) else []}",
        "Hard rule: do not claim symbolic derivations or ED tests are complete unless citing energy_decomposition_audit_tool artifacts.",
        "LSCO data are a parallel validation line and must not block this theoretical audit.",
    ]
    return "\n".join(lines)


def _optimizer_tool_summary(tool_context: dict[str, object]) -> str:
    optimizer = tool_context.get("optimizer_v2")
    if not isinstance(optimizer, dict):
        return "Hypothesis Optimizer V2: not run"
    actions = optimizer.get("top_information_gain_actions", [])
    failures = optimizer.get("failure_memory", [])
    lines = [
        "Hypothesis Optimizer V2:",
        f"- hypothesis_count: {optimizer.get('hypothesis_count')}",
        f"- killed_count: {optimizer.get('killed_count')}",
        f"- pareto_frontier_ids: {optimizer.get('pareto_frontier_ids')}",
    ]
    if actions:
        lines.append(f"- top_information_gain_action: {actions[0]}")
    if failures:
        lines.append(f"- failure_memory: {failures}")
    lines.append("Agents must use this optimizer queue to select repairs or next checks instead of restating generic future work.")
    return "\n".join(lines)


def _phase2_tool_summary(tool_context: dict[str, object]) -> str:
    data_tool = tool_context.get("phase2_data_tool")
    if not isinstance(data_tool, dict):
        return "Phase 2 data coverage tool: not run"
    coverage = data_tool.get("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
    import_status = data_tool.get("import_status", {})
    missing_preview = data_tool.get("missing_observables_preview", [])
    lines = [
        "Phase 2 data coverage tool:",
        f"- status: {data_tool.get('status')}",
        f"- import_status: {import_status}",
        f"- usable_observation_count: {coverage.get('usable_observation_count')}",
        f"- complete_doping_point_count: {coverage.get('complete_doping_point_count')}",
        f"- overall_readiness_status: {coverage.get('overall_readiness_status')}",
        f"- tier_a: {coverage.get('tier_a')}",
        f"- tier_b: {coverage.get('tier_b')}",
        f"- tier_c: {coverage.get('tier_c')}",
        f"- split_counts: {coverage.get('split_counts')}",
        f"- missing_observable_count: {data_tool.get('missing_observable_count')}",
    ]
    if missing_preview:
        lines.append(f"- missing_preview: {missing_preview}")
    return "\n".join(lines)


def _meeting_optimizer_hypotheses(question: str = "") -> list[dict[str, object]]:
    if _is_decomposition_uniqueness_question(question):
        return [
            {
                "id": "meeting-hyp-nonunique-partition",
                "title": "Non-unique energy partition branch",
                "core_claim": "Bare kinetic and correlated-hopping condensation-energy components are representation dependent, while total condensation energy and full gauge-coupled response are invariant.",
                "assumptions": ["Peierls phases are applied to every charge-transporting hopping term", "component labels are not promoted to observables without a microscopic convention"],
                "testable_predictions": ["two exact repartitions preserve spectrum but shift component expectations", "full current includes correlated hopping"],
                "falsification_criteria": ["all allowed equivalent representations yield identical component partitions", "correlated hopping carries no current contribution despite bond charge transport"],
                "supporting_evidence": ["energy_decomposition_audit_tool/representation_counterexample.json"],
                "required_tools": ["energy_decomposition_audit_tool"],
                "novelty_statement": "Turns the current energy ledger into an audited observable/invariant classification problem.",
            },
            {
                "id": "meeting-hyp-hf-operational",
                "title": "Hellmann-Feynman operational diagnostics branch",
                "core_claim": "Coupling derivatives with respect to t, g, and Delta_t are better operational diagnostics than raw component percentages, but remain model dependent.",
                "assumptions": ["microscopic couplings are fixed before differentiation", "cutoff and effective model are stated"],
                "testable_predictions": ["dE/dt, dE/dg, and dE/dDelta_t can be computed from partial derivatives of the fixed Hamiltonian"],
                "falsification_criteria": ["derivatives change under mere relabeling of parameters without a fixed microscopic convention"],
                "required_tools": ["energy_decomposition_audit_tool"],
            },
            {
                "id": "meeting-hyp-vague-uniqueness",
                "title": "Vague unique percentage branch",
                "core_claim": "The current numerical energy ledger is always a unique physical percentage decomposition.",
                "assumptions": [],
                "testable_predictions": [],
                "falsification_criteria": [],
            },
        ]
    return [
        {
            "id": "meeting-hyp-mixed-bcs",
            "title": "Artifact-backed mixed BCS branch",
            "core_claim": "The mixed phonon and correlated-hopping model is viable where the executable Phase-1 solver shows a stable gap and energy-ledger closure.",
            "assumptions": ["Hamiltonian terms are explicit", "solver artifacts are cited"],
            "testable_predictions": ["stable mixed phase points exist", "energy ledger separates kinetic and interaction terms"],
            "falsification_criteria": ["gap equation verifier fails", "free-energy closure verifier fails"],
            "supporting_evidence": ["phase1_minimal_bcs_tool/base_solution.json", "phase1_minimal_bcs_tool/energy_ledger.json"],
            "required_tools": ["phase1_minimal_mixed_bcs_solver", "phase2_data_coverage_tool"],
            "novelty_statement": "Turns the live meeting from advice into artifact-backed theory triage.",
        },
        {
            "id": "meeting-hyp-data-blocker",
            "title": "Contrarian material-level blocker",
            "core_claim": "Material-level quantitative separation remains inconclusive until LSCO coverage gates pass with reviewed optical and penetration-depth data.",
            "assumptions": ["observable definitions must overlap across doping", "figure digitization requires review"],
            "testable_predictions": ["coverage tool reports blocked status when rows lack overlap or provenance"],
            "falsification_criteria": ["reviewed canonical dataset passes overlap, provenance, and held-out split gates"],
            "required_tools": ["phase2_lsco_acquisition"],
        },
        {
            "id": "meeting-hyp-vague",
            "title": "Vague discussion branch",
            "core_claim": "We should build a Hamiltonian someday and it will always explain the data.",
            "assumptions": [],
            "testable_predictions": [],
            "falsification_criteria": [],
        },
    ]


def _call_agent_or_fixture(
    session: MeetingSession,
    agent: AgentDefinition,
    prompt: str,
    *,
    live_model: bool,
    round_number: int,
    critic_objections: list[str],
) -> tuple[MeetingAgentResponse, ProviderCallRecordV23]:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if live_model:
        return asyncio.run(_live_agent_call(agent, prompt, session.meeting_id, prompt_hash))
    blocked = session.status == "live_connection_blocked"
    response = _fixture_response(agent, session, round_number, critic_objections, "live credentials missing" if blocked else "")
    return response, _call_record(
        session.meeting_id,
        agent,
        prompt_hash,
        parsing_result="blocked" if blocked else "parsed",
        permission_mode="blocked" if blocked else "fixture",
    )


async def _live_agent_call(agent: AgentDefinition, prompt: str, meeting_id: str, prompt_hash: str) -> tuple[MeetingAgentResponse, ProviderCallRecordV23]:
    provider = OpenAICompatibleProvider(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
        model=_live_model_name(),
        max_retries=0,
        max_repair_attempts=0,
        temperature=agent.temperature,
        max_output_tokens=agent.max_output_tokens,
    )
    response = await provider.generate_structured(prompt, MeetingAgentResponse, context={"agent_role": agent.role, "workflow_stage": "live_agent_meeting"})
    record = provider.call_records[-1] if provider.call_records else None
    return response, ProviderCallRecordV23(
        call_id=f"call-{meeting_id}-{agent.agent_id}-{_digest(prompt_hash)[:8]}",
        meeting_id=meeting_id,
        agent_id=agent.agent_id,
        provider=agent.provider,
        model=_live_model_name(),
        remote_response_id=record.provider_request_id if record else None,
        timestamp=datetime.now(UTC),
        latency_ms=record.latency_ms if record else 0.0,
        prompt_hash=prompt_hash,
        input_tokens=(record.usage.input_tokens or 0) if record else 0,
        output_tokens=(record.usage.output_tokens or 0) if record else 0,
        parsing_result="parsed",
        permission_mode="live_model",
    )


def _fixture_response(agent: AgentDefinition, session: MeetingSession, round_number: int, critic_objections: list[str], failure_note: str) -> MeetingAgentResponse:
    changed = bool(critic_objections and agent.role != "critic" and round_number > 1)
    prefix = "Revised after critic feedback: " if changed else ""
    if _is_decomposition_uniqueness_question(session.research_question):
        if agent.role == "critic":
            message = "The energy-decomposition uniqueness claim is refuted unless a fixed microscopic partition and gauge-coupling convention is stated. A closed ledger alone is not evidence of physical uniqueness."
            objections = ["component percentages are representation dependent", "finite ED checks are not a general proof"]
            repairs = ["promote only invariant quantities and Hellmann-Feynman diagnostics to the claim ledger"]
        elif agent.role == "experimentalist":
            message = f"{prefix}Treat LSCO optical/isotope data as later validation; the immediate executable test is the audit counterexample and full current/operator consistency check."
            objections = []
            repairs = ["attach optical finite-cutoff caveat to every observable-mapping claim"]
        elif agent.role == "curator":
            message = f"{prefix}Use energy_decomposition_audit_tool artifacts as the evidence source; do not upgrade proposed derivations to completed claims without those artifacts."
            objections = []
            repairs = ["seed Claim DAG with gauge, continuity, representation, and observable-classification claims"]
        elif agent.role == "theorist":
            message = f"{prefix}The working theorem should say total condensation energy and full gauge-coupled response are invariant, while bare kinetic/correlated-hopping partitions are model and representation dependent unless a microscopic convention is fixed."
            objections = []
            repairs = ["derive operational diagnostics from Hellmann-Feynman coupling derivatives"]
        else:
            message = f"{prefix}Main line switches from LSCO completion to a theory audit of whether energy partitions are unique, gauge consistent, and observable."
            objections = []
            repairs = ["execute or cite the energy-decomposition audit before another meeting round"]
        if failure_note:
            message = f"{message} Live model note: {failure_note}"
        return MeetingAgentResponse(
            agent_id=agent.agent_id,
            role=agent.role,
            concise_message=message,
            candidate_model_ids=["model-condensation-energy-partition-audit-v27"],
            proposed_hamiltonian_terms=_fallback_hamiltonian_terms(session.research_question) if agent.role == "theorist" else [],
            proposed_verifier_tasks=_fallback_verifier_task_names(session.research_question) if agent.role in {"theorist", "critic"} else [],
            proposed_held_out_predictions=_fallback_prediction_names(session.research_question) if agent.role == "experimentalist" else [],
            unresolved_derivation_gaps=[] if agent.role == "curator" else ["formal theorem beyond finite counterexample remains an open proof obligation"],
            cited_artifact_ids=_visible_artifacts(session.research_question),
            objections=objections,
            proposed_repairs=repairs,
            changed_by_critic=changed,
        )
    if agent.role == "critic":
        message = "The leading hypothesis needs stronger independent evidence and a sharper falsification path before confidence should increase."
        objections = ["duplicate support must not count as independent", "missing decisive contradiction review"]
        repairs = ["target the weakest load-bearing claim in the Claim DAG"]
    elif agent.role == "experimentalist":
        message = f"{prefix}Run the smallest discriminating test that separates the two strongest mechanisms, then update the claim blocker before expanding search."
        objections = []
        repairs = ["attach the proposed test to the currently blocking claim"]
    elif agent.role == "curator":
        message = f"{prefix}Use only visible artifacts from this run and mark unsupported literature or model assumptions as evidence gaps."
        objections = []
        repairs = ["separate direct evidence from inference in the next checkpoint"]
    elif agent.role == "theorist":
        message = f"{prefix}Sketch the mixed model as H = H0 + H_ph + H_ch, then force verifiers to derive the gap equation, free-energy closure, and kinetic/interacting energy decomposition before treating it as a result."
        objections = []
        repairs = ["queue microscopic Hamiltonian audit", "queue free-energy and optical-sum verifiers"]
    else:
        message = f"{prefix}Meeting objective: convert the research question into falsifiable hypotheses, critique them, and queue a targeted repair for the weakest claim."
        objections = []
        repairs = ["preserve a contrarian branch while reducing duplicate branches"]
    if failure_note:
        message = f"{message} Live model note: {failure_note}"
    return MeetingAgentResponse(
        agent_id=agent.agent_id,
        role=agent.role,
        concise_message=message,
        candidate_model_ids=["model-mixed-bcs-correlated-hopping-v23"] if "bcs" in session.research_question.lower() and agent.role in {"theorist", "critic"} else [],
        proposed_hamiltonian_terms=_fallback_hamiltonian_terms(session.research_question) if agent.role == "theorist" else [],
        proposed_verifier_tasks=_fallback_verifier_task_names(session.research_question) if agent.role in {"theorist", "critic"} else [],
        proposed_held_out_predictions=_fallback_prediction_names(session.research_question) if agent.role == "experimentalist" else [],
        unresolved_derivation_gaps=_fallback_derivation_gaps(session.research_question) if agent.role in {"critic", "curator"} else [],
        cited_artifact_ids=_visible_artifacts(session.research_question),
        objections=objections,
        proposed_repairs=repairs,
        changed_by_critic=changed,
    )


def _meeting_prompt(
    session: MeetingSession,
    agent: AgentDefinition,
    *,
    round_number: int,
    previous_messages: list[MeetingMessage],
    critic_objections: list[str],
    tool_context: dict[str, object] | None = None,
) -> str:
    transcript = "\n".join(f"{item.role}: {item.content[:500]}" for item in previous_messages[-6:])
    objections = "\n".join(critic_objections[-3:])
    tool_summary = _tool_prompt_summary(tool_context or {})
    return (
        f"Research question:\n{session.research_question}\n\n"
        f"Round: {round_number}/{session.max_rounds}\n"
        f"Agent role: {agent.role}\n"
        f"Agent goal: {agent.goal}\n\n"
        f"Executed tool context:\n{tool_summary}\n\n"
        "Visible artifacts may include claim_dag.json, validation_blockers.jsonl, verifier_results.jsonl, "
        "candidate_archive.jsonl, report.md, and meeting tool artifacts if present. Do not cite unavailable IDs.\n\n"
        f"Recent transcript:\n{transcript or 'none'}\n\n"
        f"Open critic objections:\n{objections or 'none'}\n\n"
        "Return concise JSON. Do not reveal chain of thought. Do not merely repeat the blocker. "
        "If executed tool context contains Hamiltonian terms or verifier results, treat those as completed computations, not future work. "
        "Use the numerical results to update the hypothesis, identify the next missing derivation/data step, and cite artifact IDs. "
        "If Hypothesis Optimizer V2 provides an information-gain action or failure memory, explicitly use it to choose a repair, verifier, data acquisition, or counterexample action. "
        "If Energy decomposition audit artifacts are present, treat LSCO data acquisition as a parallel validation line and focus the main discussion on gauge consistency, representation dependence, Hellmann-Feynman diagnostics, and counterexamples. "
        "Never say a symbolic derivation, exact-diagonalization check, or theorem is complete unless it is present in the executed tool context or cited audit artifacts. "
        "Each response must advance one checkable claim, artifact-backed repair, or blocked reason beyond the previous transcript. "
        "If you propose a theory, fill candidate_model_ids and proposed_hamiltonian_terms. "
        "If you criticize it, fill unresolved_derivation_gaps and proposed_verifier_tasks. "
        "If you propose an experiment, fill proposed_held_out_predictions. "
        "If no checkable progress is possible, say that explicitly in concise_message."
    )


def _format_agent_response(response: MeetingAgentResponse) -> str:
    parts = [response.concise_message]
    if response.proposed_hamiltonian_terms:
        parts.append("Hamiltonian terms: " + "; ".join(response.proposed_hamiltonian_terms))
    if response.proposed_verifier_tasks:
        parts.append("Verifier tasks: " + "; ".join(response.proposed_verifier_tasks))
    if response.proposed_held_out_predictions:
        parts.append("Held-out predictions: " + "; ".join(response.proposed_held_out_predictions))
    if response.unresolved_derivation_gaps:
        parts.append("Unresolved derivation/data gaps: " + "; ".join(response.unresolved_derivation_gaps))
    if response.objections:
        parts.append("Objections: " + "; ".join(response.objections))
    if response.proposed_repairs:
        parts.append("Proposed repairs: " + "; ".join(response.proposed_repairs))
    if response.cited_artifact_ids:
        parts.append("Cited artifacts: " + "; ".join(response.cited_artifact_ids))
    return "\n".join(parts)


def _write_meeting_artifacts(
    root: Path,
    agents: list[AgentDefinition],
    session: MeetingSession,
    messages: list[MeetingMessage],
    responses: list[MeetingAgentResponse],
    calls: list[ProviderCallRecordV23],
    checkpoints: list[dict[str, object]],
) -> None:
    write_jsonl(root / "agent_definitions.jsonl", agents)
    write_json(root / "meeting_session.json", session)
    write_jsonl(root / "meeting_messages.jsonl", messages)
    write_jsonl(root / "meeting_agent_responses.jsonl", responses)
    write_jsonl(root / "provider_calls.jsonl", calls)
    write_jsonl(root / "meeting_checkpoints.jsonl", checkpoints)


def _clear_meeting_runtime_artifacts(root: Path) -> None:
    for name in MEETING_ARTIFACTS:
        target = root / name
        if target.exists():
            target.unlink()
    for name in [
        "action_executions",
        "claim_dag.sqlite",
        "claim_dag.json",
        "formal_claim.json",
        "atomic_claims.jsonl",
        "claim_dependencies.jsonl",
        "claim_checks.jsonl",
        "claim_contradictions.jsonl",
        "independent_checks.jsonl",
        "validation_blockers.jsonl",
        "total_gate_result.json",
    ]:
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _write_empty_closed_loop_state(root: Path, reason: str) -> None:
    payload = {
        "schema_version": "v28-closed-loop-state",
        "run_id": root.name,
        "selected_action_id": None,
        "status": "not_started",
        "reason": reason,
        "action": None,
        "policy_decision": None,
        "execution_result": None,
        "claim_dag_diff": None,
        "optimizer_diff": None,
        "next_eligible_actions": [],
        "bundle_dir": None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(root / "closed_loop_action_state.json", payload)
    write_jsonl(root / "closed_loop_action_executions.jsonl", [])


def _write_science_progress_artifacts(root: Path, session: MeetingSession, messages: list[MeetingMessage], responses: list[MeetingAgentResponse], *, allow_deterministic_scaffold: bool) -> None:
    sketches = _candidate_model_sketches(session, messages, responses, allow_deterministic_scaffold=allow_deterministic_scaffold)
    tasks = _verifier_tasks(session, sketches, responses)
    predictions = _held_out_predictions(session, sketches, responses)
    diagnostic = _science_diagnostic(session, messages, sketches, tasks, predictions)
    write_json(root / "meeting_science_diagnostic.json", diagnostic)
    write_jsonl(root / "meeting_candidate_model_sketches.jsonl", sketches)
    write_jsonl(root / "meeting_verifier_tasks.jsonl", tasks)
    write_jsonl(root / "meeting_held_out_predictions.jsonl", predictions)


def _candidate_model_sketches(session: MeetingSession, messages: list[MeetingMessage], responses: list[MeetingAgentResponse], *, allow_deterministic_scaffold: bool) -> list[CandidateModelSketch]:
    model_ids = sorted({model_id for response in responses for model_id in response.candidate_model_ids})
    if allow_deterministic_scaffold and not model_ids and _is_bcs_question(session.research_question):
        model_ids = ["model-mixed-bcs-correlated-hopping-v23"]
    sketches: list[CandidateModelSketch] = []
    for model_id in model_ids:
        source_ids = [message.message_id for message, response in zip(messages, responses, strict=False) if model_id in response.candidate_model_ids]
        terms = sorted({term for response in responses for term in response.proposed_hamiltonian_terms}) or _fallback_hamiltonian_terms(session.research_question)
        sketches.append(
            CandidateModelSketch(
                model_id=model_id,
                meeting_id=session.meeting_id,
                source_message_ids=source_ids,
                model_family="energy_decomposition_uniqueness_audit" if _is_decomposition_uniqueness_question(session.research_question) else "mixed_phonon_correlated_hopping" if _is_bcs_question(session.research_question) else "unclassified",
                hamiltonian_terms=terms,
                energy_decomposition_targets=(
                    [
                        "total_condensation_energy is invariant under fixed thermodynamic definition",
                        "bare kinetic and correlated-hopping component partitions are audited for representation dependence",
                        "Hellmann-Feynman derivatives are operational diagnostics after microscopic couplings are fixed",
                        "finite-cutoff optical spectral weight is experimentally approximable but model/cutoff dependent",
                    ]
                    if _is_decomposition_uniqueness_question(session.research_question)
                    else [
                        "delta_E_kinetic = <H0>_SC - <H0>_normal",
                        "delta_E_phonon_channel = <H_ph>_SC - <H_ph>_normal",
                        "delta_E_correlated_hopping = <H_ch>_SC - <H_ch>_normal",
                        "condensation_energy_closure = delta_E_kinetic + delta_E_interaction - delta_F",
                    ]
                ),
                assumptions=[
                    *(
                        [
                            "all charge-transporting hopping terms carry Peierls phases",
                            "component percentages are not observables unless a microscopic partition convention is fixed",
                            "finite ED counterexamples are not general proofs",
                        ]
                        if _is_decomposition_uniqueness_question(session.research_question)
                        else [
                            "mean-field decoupling must be derived rather than assumed",
                            "fixed-N and fixed-mu conventions must be recorded separately",
                            "material parameters remain placeholders until curated data are frozen",
                        ]
                    ),
                ],
                derivation_gaps=_fallback_derivation_gaps(session.research_question),
                status="sketch_only",
            )
        )
    return sketches


def _verifier_tasks(session: MeetingSession, sketches: list[CandidateModelSketch], responses: list[MeetingAgentResponse]) -> list[VerifierTaskSpec]:
    if not sketches:
        return []
    if _is_decomposition_uniqueness_question(session.research_question):
        task_specs = [
            ("peierls_gauge_coupling", ["gauge_coupled_hamiltonian.json"], ["ordinary and correlated hopping both carry Peierls phases"], "complete"),
            ("current_operator_consistency", ["electromagnetic_response_derivation.json"], ["paramagnetic current and diamagnetic kernel include correlated hopping"], "complete"),
            ("continuity_equation", ["electromagnetic_response_derivation.json"], ["charge conservation identity scoped to Peierls-coupled bonds"], "complete"),
            ("optical_sum_rule_gauge_check", ["electromagnetic_response_derivation.json"], ["restricted lattice optical sum rule and finite-cutoff caveat"], "complete"),
            ("representation_counterexample", ["representation_counterexample.json"], ["same total Hamiltonian, shifted component expectations"], "complete"),
            ("exact_diagonalization_counterexample", ["representation_counterexample.json"], ["finite-dimensional spectrum equality and component shift"], "complete"),
            ("hellmann_feynman_diagnostic", ["hellmann_feynman_diagnostics.json"], ["operational coupling derivatives"], "complete"),
            ("observable_classification", ["observable_classification.json"], ["observable/model-dependent/basis-dependent labels"], "complete"),
            ("invariant_theorem_obligation", ["final_theory_outcome.json"], ["formal theorem conditions beyond finite counterexample"], "queued"),
        ]
    else:
        task_specs = [
            ("microscopic_hamiltonian_audit", ["candidate model sketch"], ["Hermiticity, particle-number convention, and parameter-domain report"], "queued"),
            ("mean_field_derivation", ["Hamiltonian terms"], ["gap equation and decoupling assumptions"], "queued"),
            ("free_energy_closure", ["gap equation", "energy terms"], ["delta_F and component-energy closure residual"], "queued"),
            ("current_operator_consistency", ["Hamiltonian terms"], ["current operator derived from the same Hamiltonian"], "queued"),
            ("optical_sum_rule", ["current operator", "doping grid"], ["cutoff-dependent optical sum consistency check"], "queued"),
            ("competitor_model_check", ["non-Hirsch kinetic competitor"], ["whether competitor reproduces the same fingerprint"], "queued"),
            ("data_coverage_audit", ["candidate material families"], ["doping, Tc, optical, penetration-depth, specific-heat, isotope coverage table"], "blocked_missing_input"),
            ("held_out_prediction_gate", ["frozen train/held-out split"], ["preregistered prediction packet before reveal"], "blocked_missing_input"),
        ]
    tasks: list[VerifierTaskSpec] = []
    for sketch in sketches:
        for name, required_inputs, required_outputs, status in task_specs:
            tasks.append(
                VerifierTaskSpec(
                    task_id=f"task-{sketch.model_id}-{name}",
                    meeting_id=session.meeting_id,
                    model_id=sketch.model_id,
                    verifier_type=name,  # type: ignore[arg-type]
                    required_inputs=required_inputs,
                    required_outputs=required_outputs,
                    blocking=True,
                    status=status,  # type: ignore[arg-type]
                )
            )
    return tasks


def _held_out_predictions(session: MeetingSession, sketches: list[CandidateModelSketch], responses: list[MeetingAgentResponse]) -> list[HeldOutPredictionSpec]:
    if not sketches:
        return []
    predictions: list[HeldOutPredictionSpec] = []
    for sketch in sketches:
        if _is_decomposition_uniqueness_question(session.research_question):
            predictions.extend(
                [
                    HeldOutPredictionSpec(
                        prediction_id=f"pred-{sketch.model_id}-representation-counterexample",
                        meeting_id=session.meeting_id,
                        model_id=sketch.model_id,
                        observable="component ledger under equivalent Hamiltonian repartition",
                        material_family="theory",
                        doping_or_parameter="eta repartition parameter",
                        prediction_statement="Total spectrum remains fixed while named kinetic/correlated-hopping component expectations shift.",
                        status="preregistered",
                    ),
                    HeldOutPredictionSpec(
                        prediction_id=f"pred-{sketch.model_id}-hf-diagnostic",
                        meeting_id=session.meeting_id,
                        model_id=sketch.model_id,
                        observable="Hellmann-Feynman coupling derivative",
                        material_family="theory",
                        doping_or_parameter="t,g,Delta_t",
                        prediction_statement="Coupling derivatives are operational only after the microscopic coupling convention is fixed.",
                        status="preregistered",
                    ),
                ]
            )
            continue
        predictions.extend(
            [
                HeldOutPredictionSpec(
                    prediction_id=f"pred-{sketch.model_id}-optical-cutoff",
                    meeting_id=session.meeting_id,
                    model_id=sketch.model_id,
                    observable="cutoff-dependent optical spectral-weight shift below Tc",
                    material_family="to_be_selected_by_data_coverage_audit",
                    doping_or_parameter="held-out doping point",
                    prediction_statement="Direction and magnitude must be preregistered after model parameters are fitted on training data only.",
                ),
                HeldOutPredictionSpec(
                    prediction_id=f"pred-{sketch.model_id}-energy-decomposition",
                    meeting_id=session.meeting_id,
                    model_id=sketch.model_id,
                    observable="relative kinetic versus interaction contribution to condensation energy",
                    material_family="to_be_selected_by_data_coverage_audit",
                    doping_or_parameter="held-out material or doping",
                    prediction_statement="The sign and fraction of kinetic-energy lowering must be fixed before held-out data are revealed.",
                ),
            ]
        )
    return predictions


def _science_diagnostic(
    session: MeetingSession,
    messages: list[MeetingMessage],
    sketches: list[CandidateModelSketch],
    tasks: list[VerifierTaskSpec],
    predictions: list[HeldOutPredictionSpec],
) -> ScienceProgressDiagnostic:
    repetition = _repetition_score([message.content for message in messages])
    failure_modes: list[str] = []
    if repetition > 0.55:
        failure_modes.append("agent_discussion_repeated_same_blocker")
    if not sketches:
        failure_modes.append("no_candidate_microscopic_model")
    if not tasks:
        failure_modes.append("no_verifier_tasks")
    if not predictions:
        failure_modes.append("no_preregisterable_held_out_predictions")
    if session.status != "completed":
        failure_modes.append(f"meeting_status_{session.status}")
    if not failure_modes and all(task.status == "queued" for task in tasks):
        outcome = "candidate_requires_verification"
    elif sketches and tasks:
        outcome = "candidate_requires_verification"
        if repetition > 0.55:
            failure_modes.append("science_gate_blocks_claim_of_new_knowledge")
    else:
        outcome = "no_new_scientific_knowledge"
    return ScienceProgressDiagnostic(
        meeting_id=session.meeting_id,
        outcome=outcome,  # type: ignore[arg-type]
        repetition_score=repetition,
        valid_model_sketch_count=len([sketch for sketch in sketches if sketch.status != "invalid"]),
        verifier_task_count=len(tasks),
        held_out_prediction_count=len(predictions),
        failure_modes=failure_modes,
        required_next_artifacts=(
            [
                "energy_decomposition_audit_tool/final_theory_outcome.json",
                "energy_decomposition_audit_tool/representation_counterexample.json",
                "energy_decomposition_audit_tool/observable_classification.json",
                "formal_invariant_theorem_obligations.md",
                "claim_dag_seed_tasks.jsonl",
            ]
            if _is_decomposition_uniqueness_question(session.research_question)
            else [
                "frozen_data_coverage_audit.json",
                "microscopic_derivation_review.md",
                "verifier_results.jsonl",
                "preregistered_held_out_predictions.jsonl",
                "competitor_model_checks.jsonl",
            ]
        ),
        concise_assessment=(
            "The meeting produced or cited a theory audit. Treat the finite counterexample as evidence of non-unique component partition, while keeping theorem obligations explicit."
            if _is_decomposition_uniqueness_question(session.research_question)
            else "The meeting produced a candidate model sketch and verifier queue, but it is not new scientific knowledge until "
            "the derivation, data coverage, competitor, and held-out gates pass."
            if sketches
            else "The meeting did not produce a checkable scientific model."
        ),
        created_at=datetime.now(UTC),
    )


def _call_record(
    meeting_id: str,
    agent: AgentDefinition,
    prompt_hash: str,
    *,
    parsing_result: str,
    permission_mode: str,
) -> ProviderCallRecordV23:
    return ProviderCallRecordV23(
        call_id=f"call-{meeting_id}-{agent.agent_id}-{prompt_hash[:8]}",
        meeting_id=meeting_id,
        agent_id=agent.agent_id,
        provider=agent.provider,
        model=agent.model,
        timestamp=datetime.now(UTC),
        prompt_hash=prompt_hash,
        parsing_result=parsing_result,  # type: ignore[arg-type]
        permission_mode=permission_mode,  # type: ignore[arg-type]
    )


def _blocked_call_records(session: MeetingSession, agents: list[AgentDefinition]) -> list[ProviderCallRecordV23]:
    return [
        _call_record(
            session.meeting_id,
            agent,
            hashlib.sha256(f"{session.meeting_id}:{agent.agent_id}:blocked".encode("utf-8")).hexdigest(),
            parsing_result="blocked",
            permission_mode="blocked",
        )
        for agent in agents
    ]


def _safe_status(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())[:240]


def _append_jsonl(path: Path, payload: object) -> None:
    rows = read_jsonl(path) if path.exists() else []
    write_jsonl(path, [*rows, payload])


def _write_repair_chain(root: Path, result: ClaimRepairResult) -> None:
    path = root / "artifact_version_chain.json"
    chain = read_json(path) if path.exists() else {"schema_version": "v23", "repairs": []}
    chain.setdefault("repairs", []).append(result.model_dump(mode="json"))
    write_json(path, chain)


def _meeting_rows(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "round_number": item.get("round_number"),
            "agent_id": item.get("agent_id"),
            "role": item.get("role"),
            "critic_influenced": item.get("critic_influenced"),
            "content": item.get("content"),
        }
        for item in messages
    ]


def _provider_rows(root: Path) -> list[dict[str, object]]:
    path = root / "provider_calls.jsonl"
    if not path.exists():
        return []
    return [
        {
            "call_id": item.get("call_id"),
            "agent_id": item.get("agent_id"),
            "provider": item.get("provider"),
            "model": item.get("model"),
            "permission_mode": item.get("permission_mode"),
            "parsing_result": item.get("parsing_result"),
            "input_tokens": item.get("input_tokens"),
            "output_tokens": item.get("output_tokens"),
        }
        for item in read_jsonl(path)
    ]


def _transcript_markdown(messages: list[dict[str, object]]) -> str:
    lines = ["# Live Agent Meeting Room", ""]
    for item in messages:
        lines.extend(
            [
                f"## Round {item.get('round_number')} - {item.get('role')}",
                "",
                str(item.get("content") or ""),
                "",
            ]
        )
    return "\n".join(lines).strip()


def _status_transcript(session: MeetingSession) -> str:
    reason = session.stopping_reason or "unknown"
    return (
        "# Live Agent Meeting Room\n\n"
        f"Status: `{session.status}`\n\n"
        f"Reason: `{reason}`\n\n"
        "No live agent messages were generated. This run did not fall back to mock reasoning."
    )


def _should_stop_no_progress(session: MeetingSession, round_number: int, root: Path) -> bool:
    if not _is_decomposition_uniqueness_question(session.research_question):
        return False
    if round_number < 2:
        return False
    action_rows = action_execution_rows(root)
    if action_rows and action_rows[-1].get("status") == "no_eligible_action":
        return True
    recent_actions = action_rows[-2:]
    recent_progress = [
        item for item in recent_actions
        if item.get("status") in {"succeeded", "inconclusive"} and item.get("claim_dag_diff")
    ]
    if recent_progress:
        return False
    repair_results = read_jsonl(root / "repair_results.jsonl") if (root / "repair_results.jsonl").exists() else []
    return not repair_results


def _live_credentials_available() -> bool:
    return bool((os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")) and _live_model_name())


def _live_model_name() -> str:
    return os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-4.1-mini"


def _is_bcs_question(question: str) -> bool:
    lowered = question.lower()
    return "bcs" in lowered or "superconduct" in lowered or "correlated hopping" in lowered or "phonon" in lowered


def _is_decomposition_uniqueness_question(question: str) -> bool:
    lowered = question.lower()
    return (
        "decomposition" in lowered
        or "condensation energy" in lowered
        or "gauge invariant" in lowered
        or "gauge invariance" in lowered
        or "hellmann" in lowered
        or "optical sum" in lowered
        or "current operator" in lowered
        or "unique" in lowered and "energy" in lowered
    )


def _fallback_hamiltonian_terms(question: str) -> list[str]:
    if not _is_bcs_question(question):
        return []
    if _is_decomposition_uniqueness_question(question):
        return [
            "H_t[A] = -sum_<ij>,sigma t_ij (exp(i phi_ij)c_i^dag c_j + h.c.)",
            "H_corr[A] = -Delta_t sum_<ij>,sigma (n_i_barsigma+n_j_barsigma)(exp(i phi_ij)c_i^dag c_j+h.c.)",
            "j_ij^p = (i q/hbar) [t_ij + Delta_t(n_i_barsigma+n_j_barsigma)](c_i^dag c_j - h.c.)",
            "K_ij = (q^2/hbar^2)[t_ij + Delta_t(n_i_barsigma+n_j_barsigma)](c_i^dag c_j + h.c.)",
        ]
    return [
        "H0 = sum_{k,sigma} xi_k c^dagger_{k,sigma} c_{k,sigma}",
        "H_ph = -sum_{k,k'} V_ph(k,k') c^dagger_{k,up} c^dagger_{-k,down} c_{-k',down} c_{k',up}",
        "H_ch = sum_{<ij>,sigma} Delta t(n_{i,-sigma}+n_{j,-sigma}) c^dagger_{i,sigma} c_{j,sigma} + h.c.",
        "H_mix = H0 + H_ph + H_ch with the pairing vertex derived from the same Hamiltonian",
    ]


def _fallback_verifier_task_names(question: str) -> list[str]:
    if not _is_bcs_question(question):
        return []
    if _is_decomposition_uniqueness_question(question):
        return [
            "verify Peierls substitution in ordinary and correlated hopping terms",
            "derive paramagnetic current and diamagnetic kernel including correlated hopping",
            "verify continuity equation and restricted lattice optical sum rule",
            "run representation counterexample for component energy non-uniqueness",
            "classify total energy, optical response, component ledger entries, and Hellmann-Feynman derivatives",
            "formalize theorem obligations beyond finite counterexample",
        ]
    return [
        "derive mean-field decoupling of correlated hopping without inserting an effective kernel by hand",
        "solve gap equation and record stable superconducting solution bounds",
        "compute delta_E_kinetic, delta_E_phonon_channel, delta_E_correlated_hopping, and closure residual",
        "derive current operator from H_mix and test optical sum consistency",
        "compare against a non-Hirsch kinetic competitor before claiming specificity",
    ]


def _fallback_prediction_names(question: str) -> list[str]:
    if not _is_bcs_question(question):
        return []
    if _is_decomposition_uniqueness_question(question):
        return [
            "two equivalent representations keep total spectrum fixed while component ledger entries shift",
            "full current response uses H_t[A]+H_corr[A], not H_t[A] alone",
            "Hellmann-Feynman derivatives remain operational only after microscopic couplings are fixed",
        ]
    return [
        "preregister cutoff-dependent optical spectral-weight shift at a held-out doping",
        "preregister kinetic/interacting condensation-energy fraction before held-out reveal",
    ]


def _fallback_derivation_gaps(question: str) -> list[str]:
    if not _is_bcs_question(question):
        return ["no domain-specific derivation gap template is available"]
    if _is_decomposition_uniqueness_question(question):
        return [
            "formal theorem for allowed canonical/unitary transformations remains open",
            "finite-size counterexample must not be promoted to general proof",
            "finite-cutoff optical mapping requires explicit cutoff and effective-model assumptions",
        ]
    return [
        "correlated-hopping mean-field decoupling is not yet derived from H_ch",
        "fixed-N versus fixed-mu energy decomposition convention is not frozen",
        "current operator and optical sum are not yet derived from the same Hamiltonian",
        "non-Hirsch kinetic-energy-lowering competitors are not yet encoded",
        "material-family data coverage is not yet audited or frozen",
    ]


def _repetition_score(texts: list[str]) -> float:
    token_sets = [_tokens(text) for text in texts if text.strip()]
    if len(token_sets) < 2:
        return 0.0
    scores: list[float] = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            if not left or not right:
                continue
            scores.append(len(left & right) / len(left | right))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "a",
        "an",
        "in",
        "for",
        "with",
        "is",
        "are",
        "be",
        "must",
        "not",
        "that",
        "this",
        "within",
    }
    return {token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if len(token) > 3 and token not in stop}


def _visible_artifacts(question: str = "") -> list[str]:
    artifacts = ["claim_dag.json", "validation_blockers.jsonl", "candidate_archive.jsonl", "verifier_results.jsonl"]
    if _is_bcs_question(question):
        artifacts.extend(
            [
                "phase1_minimal_bcs_tool/hamiltonian_spec.json",
                "phase1_minimal_bcs_tool/base_solution.json",
                "phase1_minimal_bcs_tool/energy_ledger.json",
                "phase1_minimal_bcs_tool/minimal_bcs_verifier_results.jsonl",
                "phase1_minimal_bcs_tool/phase2_held_out_evaluation.json",
            ]
        )
    if _is_decomposition_uniqueness_question(question):
        artifacts.extend(
            [
                "energy_decomposition_audit_tool/research_objective.json",
                "energy_decomposition_audit_tool/gauge_coupled_hamiltonian.json",
                "energy_decomposition_audit_tool/electromagnetic_response_derivation.json",
                "energy_decomposition_audit_tool/representation_counterexample.json",
                "energy_decomposition_audit_tool/hellmann_feynman_diagnostics.json",
                "energy_decomposition_audit_tool/observable_classification.json",
                "energy_decomposition_audit_tool/energy_decomposition_verifier_results.jsonl",
                "energy_decomposition_audit_tool/final_theory_outcome.json",
            ]
        )
    return artifacts


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _mermaid_id(value: str) -> str:
    return "n_" + "".join(char if char.isalnum() else "_" for char in value)


def _mermaid_label(claim_id: str, statement: str) -> str:
    text = f"{claim_id}: {statement}".replace('"', "'")
    return text[:96] + ("..." if len(text) > 96 else "")


def _contains_secret_like_text(root: Path, names: list[str]) -> bool:
    text = "\n".join((root / name).read_text(encoding="utf-8", errors="ignore") for name in names if (root / name).exists()).lower()
    return "openai_api_key" in text or "openrouter_api_key" in text or "sk-" in text or "bearer " in text


def _validate_jsonl(path: Path, schema):
    return [schema.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
