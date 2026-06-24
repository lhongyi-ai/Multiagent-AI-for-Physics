from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.providers.base import ProviderError
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.schemas.v23 import (
    AgentDefinition,
    ClaimRepairRequest,
    ClaimRepairResult,
    MeetingAgentResponse,
    MeetingMessage,
    MeetingSession,
    ProviderCallRecordV23,
)


MEETING_ARTIFACTS = [
    "agent_definitions.jsonl",
    "meeting_session.json",
    "meeting_messages.jsonl",
    "provider_calls.jsonl",
    "meeting_checkpoints.jsonl",
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
    ]


def run_live_agent_meeting(
    run_dir: str | Path,
    research_question: str,
    *,
    live_model: bool = False,
    max_rounds: int = 2,
    force: bool = False,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
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
    messages: list[MeetingMessage] = []
    calls: list[ProviderCallRecordV23] = []
    checkpoints: list[dict[str, object]] = []
    live_available = live_model and _live_credentials_available()
    if live_model and not live_available:
        session.status = "live_connection_blocked"
        session.stopping_reason = "live_model_requested_but_openrouter_or_openai_credentials_missing"

    for round_number in range(1, max_rounds + 1):
        session.current_round = round_number
        critic_objections = [message.content for message in messages if message.role == "critic"][-2:]
        for agent in agents:
            prompt = _meeting_prompt(
                session,
                agent,
                round_number=round_number,
                previous_messages=messages[-8:],
                critic_objections=critic_objections,
            )
            response, call = _call_agent_or_fixture(
                session,
                agent,
                prompt,
                live_model=live_available,
                round_number=round_number,
                critic_objections=critic_objections,
            )
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
    session.status = "live_connection_blocked" if live_model and not live_available else "completed"
    session.stopping_reason = session.stopping_reason or "max_rounds_complete"
    session.updated_at = datetime.now(UTC)
    _write_meeting_artifacts(root, agents, session, messages, calls, checkpoints)
    return root


def stream_live_agent_meeting(
    run_dir: str | Path,
    research_question: str,
    *,
    live_model: bool = False,
    max_rounds: int = 2,
    force: bool = True,
) -> Iterator[tuple[str, list[dict[str, object]], list[dict[str, object]], dict[str, object]]]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
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
    messages: list[MeetingMessage] = []
    calls: list[ProviderCallRecordV23] = []
    checkpoints: list[dict[str, object]] = []
    live_available = live_model and _live_credentials_available()
    if live_model and not live_available:
        session.status = "live_connection_blocked"
        session.stopping_reason = "live_model_requested_but_openrouter_or_openai_credentials_missing"
    _write_meeting_artifacts(root, agents, session, messages, calls, checkpoints)
    for round_number in range(1, max_rounds + 1):
        session.current_round = round_number
        critic_objections = [message.content for message in messages if message.role == "critic"][-2:]
        for agent in agents:
            prompt = _meeting_prompt(
                session,
                agent,
                round_number=round_number,
                previous_messages=messages[-8:],
                critic_objections=critic_objections,
            )
            response, call = _call_agent_or_fixture(
                session,
                agent,
                prompt,
                live_model=live_available,
                round_number=round_number,
                critic_objections=critic_objections,
            )
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
            _write_meeting_artifacts(root, agents, session, messages, calls, checkpoints)
            message_rows = [item.model_dump(mode="json") for item in messages]
            yield _transcript_markdown(message_rows), _meeting_rows(message_rows), _provider_rows(root), read_json(root / "meeting_session.json")
    session.status = "live_connection_blocked" if live_model and not live_available else "completed"
    session.stopping_reason = session.stopping_reason or "max_rounds_complete"
    session.updated_at = datetime.now(UTC)
    _write_meeting_artifacts(root, agents, session, messages, calls, checkpoints)
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
        calls = _validate_jsonl(root / "provider_calls.jsonl", ProviderCallRecordV23)
    except Exception as exc:
        return [f"invalid meeting artifact: {exc}"]
    agent_ids = {agent.agent_id for agent in agents}
    if not messages:
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
    if _contains_secret_like_text(root, ["meeting_session.json", "meeting_messages.jsonl", "provider_calls.jsonl"]):
        errors.append("secret-like content appears in meeting artifacts")
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
        try:
            response, call = asyncio.run(_live_agent_call(agent, prompt, session.meeting_id, prompt_hash))
            return response, call
        except ProviderError as exc:
            return _fixture_response(agent, session, round_number, critic_objections, str(exc)), _call_record(
                session.meeting_id,
                agent,
                prompt_hash,
                parsing_result="failed",
                permission_mode="live_model",
            )
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
        message = f"{prefix}Keep two competing mechanistic branches alive until a verifier distinguishes energy lowering, parameter plausibility, or held-out predictions."
        objections = []
        repairs = ["repair the branch whose load-bearing claim lacks independent reproduction"]
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
        cited_artifact_ids=_visible_artifacts(),
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
) -> str:
    transcript = "\n".join(f"{item.role}: {item.content[:500]}" for item in previous_messages[-6:])
    objections = "\n".join(critic_objections[-3:])
    return (
        f"Research question:\n{session.research_question}\n\n"
        f"Round: {round_number}/{session.max_rounds}\n"
        f"Agent role: {agent.role}\n"
        f"Agent goal: {agent.goal}\n\n"
        "Visible artifacts may include claim_dag.json, validation_blockers.jsonl, verifier_results.jsonl, "
        "candidate_archive.jsonl, and report.md if present. Do not cite unavailable IDs.\n\n"
        f"Recent transcript:\n{transcript or 'none'}\n\n"
        f"Open critic objections:\n{objections or 'none'}\n\n"
        "Return concise JSON. Do not reveal chain of thought. Prefer actionable repair proposals over broad exposition."
    )


def _format_agent_response(response: MeetingAgentResponse) -> str:
    parts = [response.concise_message]
    if response.objections:
        parts.append("Objections: " + "; ".join(response.objections))
    if response.proposed_repairs:
        parts.append("Proposed repairs: " + "; ".join(response.proposed_repairs))
    return "\n".join(parts)


def _write_meeting_artifacts(
    root: Path,
    agents: list[AgentDefinition],
    session: MeetingSession,
    messages: list[MeetingMessage],
    calls: list[ProviderCallRecordV23],
    checkpoints: list[dict[str, object]],
) -> None:
    write_jsonl(root / "agent_definitions.jsonl", agents)
    write_json(root / "meeting_session.json", session)
    write_jsonl(root / "meeting_messages.jsonl", messages)
    write_jsonl(root / "provider_calls.jsonl", calls)
    write_jsonl(root / "meeting_checkpoints.jsonl", checkpoints)


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


def _live_credentials_available() -> bool:
    return bool((os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")) and _live_model_name())


def _live_model_name() -> str:
    return os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-4.1-mini"


def _visible_artifacts() -> list[str]:
    return ["claim_dag.json", "validation_blockers.jsonl", "candidate_archive.jsonl", "verifier_results.jsonl"]


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
