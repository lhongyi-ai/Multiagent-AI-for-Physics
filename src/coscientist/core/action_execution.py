from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from coscientist.claim_dag import create_claim_dag_artifacts_from_run, rebuild_claim_dag_database
from coscientist.core.optimizer_v2 import ExpectedInformationGainAction
from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.v23 import ClaimCheckRecord, ClaimContradiction, ValidationBlocker, ValidationVerdict
from coscientist.superconductivity.energy_decomposition import run_energy_decomposition_audit, validate_energy_decomposition_audit
from coscientist.superconductivity.minimal_model import run_minimal_mixed_bcs_project, validate_minimal_mixed_bcs_run
from coscientist.superconductivity.phase2_data import run_phase2_data_coverage_tool, validate_phase2_data_coverage_tool


ActionStatus = Literal[
    "queued",
    "selected",
    "policy_blocked",
    "permission_blocked",
    "running",
    "succeeded",
    "failed",
    "inconclusive",
    "skipped_duplicate",
    "not_executable",
]


class ScientificAction(BaseModel):
    schema_version: str = "v28-scientific-action"
    action_id: str
    hypothesis_id: str
    domain_id: str = "general"
    action_type: str
    tool_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    success_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_cost: float = Field(default=1.0, gt=0.0)
    priority: float = 0.0
    rationale: str = ""
    required_permissions: list[str] = Field(default_factory=list)
    expected_artifact_types: list[str] = Field(default_factory=list)
    target_claims: list[str] = Field(default_factory=list)
    target_blockers: list[str] = Field(default_factory=list)
    execution_status: ActionStatus = "queued"
    retry_count: int = 0
    provenance: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    executed_at: str | None = None
    idempotency_key: str


class ActionPolicyDecision(BaseModel):
    schema_version: str = "v28-action-policy-decision"
    action_id: str
    allowed: bool
    status: Literal["allowed", "policy_blocked", "permission_blocked", "not_executable", "skipped_duplicate"]
    reasons: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    execution_mode: Literal["deterministic", "live-model", "live-network"] = "deterministic"


class ToolInvocationRecord(BaseModel):
    schema_version: str = "v28-tool-invocation"
    action_id: str
    tool_id: str
    tool_version: str = "v28"
    execution_mode: Literal["deterministic", "live-model", "live-network"] = "deterministic"
    arguments: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    finished_at: str | None = None
    duration_ms: float = 0.0


class ActionExecutionResult(BaseModel):
    schema_version: str = "v28-action-execution-result"
    action_id: str
    hypothesis_id: str
    domain_id: str
    tool_id: str
    status: ActionStatus
    summary: str
    generated_artifacts: list[str] = Field(default_factory=list)
    verifier_results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str
    duration_ms: float


class ClaimDAGUpdateResult(BaseModel):
    schema_version: str = "v28-claim-dag-update"
    action_id: str
    status: Literal["updated", "not_updated", "failed"]
    created_claim_ids: list[str] = Field(default_factory=list)
    updated_claim_ids: list[str] = Field(default_factory=list)
    added_check_ids: list[str] = Field(default_factory=list)
    added_blocker_ids: list[str] = Field(default_factory=list)
    added_contradiction_ids: list[str] = Field(default_factory=list)
    resolved_blocker_ids: list[str] = Field(default_factory=list)
    invalidated_claim_ids: list[str] = Field(default_factory=list)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class OptimizerUpdateResult(BaseModel):
    schema_version: str = "v28-optimizer-update"
    action_id: str
    status: Literal["updated", "not_updated"]
    completed_action_ids: list[str] = Field(default_factory=list)
    downgraded_action_ids: list[str] = Field(default_factory=list)
    new_action_ids: list[str] = Field(default_factory=list)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ClosedLoopExecutionState(BaseModel):
    schema_version: str = "v28-closed-loop-state"
    run_id: str
    selected_action_id: str | None = None
    status: str
    reason: str = ""
    action: dict[str, Any] | None = None
    policy_decision: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    claim_dag_diff: dict[str, Any] | None = None
    optimizer_diff: dict[str, Any] | None = None
    next_eligible_actions: list[dict[str, Any]] = Field(default_factory=list)
    bundle_dir: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


ToolHandler = Callable[[Path, ScientificAction, dict[str, Any]], tuple[list[str], list[dict[str, Any]], str, list[str]]]


@dataclass
class ScientificToolRegistry:
    handlers: dict[str, ToolHandler]

    @classmethod
    def default(cls) -> "ScientificToolRegistry":
        registry = cls(handlers={})
        registry.register("phase1_minimal_mixed_bcs_solver", _run_phase1_tool)
        registry.register("phase2_data_coverage_tool", _run_phase2_coverage_tool)
        registry.register("phase2_lsco_acquisition", _run_phase2_coverage_tool)
        registry.register("energy_decomposition_audit_tool", _run_energy_decomposition_tool)
        registry.register("energy_decomposition_audit", _run_energy_decomposition_tool)
        registry.register("representation_counterexample", _run_energy_decomposition_tool)
        registry.register("hellmann_feynman_diagnostic", _run_energy_decomposition_tool)
        registry.register("targeted_repair_task", _run_targeted_repair_tool)
        return registry

    def register(self, tool_id: str, handler: ToolHandler) -> None:
        self.handlers[tool_id] = handler

    def has(self, tool_id: str) -> bool:
        return tool_id in self.handlers

    def run(self, bundle_dir: Path, action: ScientificAction, context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str, list[str]]:
        if action.tool_id not in self.handlers:
            raise KeyError(f"no registered scientific tool: {action.tool_id}")
        return self.handlers[action.tool_id](bundle_dir, action, context)


@dataclass
class ScientificActionExecutor:
    registry: ScientificToolRegistry | None = None

    def execute_next(
        self,
        run_dir: str | Path,
        *,
        round_number: int,
        live_model: bool = False,
        live_network: bool = False,
        phase2_data_path: str | Path | None = None,
    ) -> ClosedLoopExecutionState:
        return execute_next_scientific_action(
            run_dir,
            round_number=round_number,
            live_model=live_model,
            live_network=live_network,
            phase2_data_path=phase2_data_path,
            registry=self.registry,
        )


def execute_next_scientific_action(
    run_dir: str | Path,
    *,
    round_number: int,
    live_model: bool = False,
    live_network: bool = False,
    phase2_data_path: str | Path | None = None,
    registry: ScientificToolRegistry | None = None,
) -> ClosedLoopExecutionState:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    registry = registry or ScientificToolRegistry.default()
    queue = _load_action_queue(root)
    prior = _execution_history(root)
    eligible, ineligible_reasons = _eligible_actions(queue, prior, registry, live_model=live_model, live_network=live_network)
    if not eligible:
        state = ClosedLoopExecutionState(
            run_id=root.name,
            status="no_eligible_action",
            reason=_no_eligible_reason(ineligible_reasons),
            next_eligible_actions=[],
        )
        _persist_state(root, state)
        return state

    action = eligible[0].model_copy(update={"execution_status": "selected", "executed_at": datetime.now(UTC).isoformat()})
    bundle = root / "action_executions" / _safe_name(action.action_id)
    if bundle.exists() and (bundle / "execution_summary.json").exists():
        policy = ActionPolicyDecision(action_id=action.action_id, allowed=False, status="skipped_duplicate", reasons=["idempotent action bundle already exists"], required_permissions=action.required_permissions)
        result = ActionExecutionResult(
            action_id=action.action_id,
            hypothesis_id=action.hypothesis_id,
            domain_id=action.domain_id,
            tool_id=action.tool_id,
            status="skipped_duplicate",
            summary="Action was already executed with the same idempotency key.",
            started_at=datetime.now(UTC).isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            duration_ms=0.0,
        )
        write_json(bundle / "policy_decision.json", policy)
        write_json(bundle / "execution_result.json", result)
        state = ClosedLoopExecutionState(
            run_id=root.name,
            selected_action_id=action.action_id,
            status="skipped_duplicate",
            reason="idempotent duplicate",
            action=action.model_dump(mode="json"),
            policy_decision=policy.model_dump(mode="json"),
            execution_result=result.model_dump(mode="json"),
            next_eligible_actions=[item.model_dump(mode="json") for item in eligible[1:4]],
            bundle_dir=str(bundle.relative_to(root)),
        )
        _persist_state(root, state)
        return state

    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "action_request.json", action)
    policy = _policy_decision(action, registry, live_model=live_model, live_network=live_network)
    write_json(bundle / "policy_decision.json", policy)
    if not policy.allowed:
        result = ActionExecutionResult(
            action_id=action.action_id,
            hypothesis_id=action.hypothesis_id,
            domain_id=action.domain_id,
            tool_id=action.tool_id,
            status=policy.status,
            summary="; ".join(policy.reasons),
            errors=policy.reasons,
            started_at=datetime.now(UTC).isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            duration_ms=0.0,
        )
        write_json(bundle / "execution_result.json", result)
        write_json(bundle / "execution_summary.json", _summary_payload(action, policy, result, None, None))
        state = _state_from_parts(root, action, policy, result, None, None, eligible[1:4], bundle)
        _persist_state(root, state)
        return state

    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    invocation = ToolInvocationRecord(action_id=action.action_id, tool_id=action.tool_id, arguments=action.arguments, started_at=started_at, execution_mode=policy.execution_mode)
    write_json(bundle / "tool_invocation.json", invocation)
    try:
        generated, verifier_rows, summary, errors = registry.run(bundle, action, {"root": root, "phase2_data_path": phase2_data_path})
        status: ActionStatus = "succeeded" if not errors else "inconclusive"
    except Exception as exc:  # pragma: no cover - exercised through failure status in integration.
        generated, verifier_rows, summary, errors = [], [], f"Tool execution failed: {exc}", [str(exc)]
        status = "failed"
    finished_at = datetime.now(UTC).isoformat()
    duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
    invocation = invocation.model_copy(update={"finished_at": finished_at, "duration_ms": duration_ms})
    write_json(bundle / "tool_invocation.json", invocation)
    result = ActionExecutionResult(
        action_id=action.action_id,
        hypothesis_id=action.hypothesis_id,
        domain_id=action.domain_id,
        tool_id=action.tool_id,
        status=status,
        summary=summary,
        generated_artifacts=generated,
        verifier_results=verifier_rows,
        errors=errors,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
    )
    write_json(bundle / "execution_result.json", result)
    write_json(bundle / "generated_artifacts.json", {"schema_version": "v28-generated-artifacts", "action_id": action.action_id, "artifacts": generated})
    write_jsonl(bundle / "verifier_results.jsonl", verifier_rows)
    claim_diff = _update_claim_dag(root, action, result)
    optimizer_diff = _update_optimizer(root, action, result, queue)
    write_json(bundle / "claim_dag_diff.json", claim_diff)
    write_json(bundle / "optimizer_diff.json", optimizer_diff)
    write_json(bundle / "execution_summary.json", _summary_payload(action, policy, result, claim_diff, optimizer_diff))
    state = _state_from_parts(root, action, policy, result, claim_diff, optimizer_diff, eligible[1:4], bundle)
    _persist_state(root, state)
    return state


def validate_action_execution_bundle(bundle_dir: str | Path) -> list[str]:
    root = Path(bundle_dir)
    required = [
        "action_request.json",
        "policy_decision.json",
        "tool_invocation.json",
        "execution_result.json",
        "generated_artifacts.json",
        "verifier_results.jsonl",
        "claim_dag_diff.json",
        "optimizer_diff.json",
        "execution_summary.json",
    ]
    errors = [f"missing action artifact: {name}" for name in required if not (root / name).exists()]
    if errors:
        return errors
    try:
        ScientificAction.model_validate(read_json(root / "action_request.json"))
        ActionPolicyDecision.model_validate(read_json(root / "policy_decision.json"))
        ToolInvocationRecord.model_validate(read_json(root / "tool_invocation.json"))
        ActionExecutionResult.model_validate(read_json(root / "execution_result.json"))
        ClaimDAGUpdateResult.model_validate(read_json(root / "claim_dag_diff.json"))
        OptimizerUpdateResult.model_validate(read_json(root / "optimizer_diff.json"))
    except Exception as exc:
        errors.append(f"invalid action artifact: {exc}")
    if _contains_secret_like_text(root):
        errors.append("secret-like content appears in action execution bundle")
    return errors


def action_execution_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "closed_loop_action_executions.jsonl"
    return read_jsonl(path) if path.exists() else []


def latest_action_state(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "closed_loop_action_state.json"
    return read_json(path) if path.exists() else {}


def _load_action_queue(root: Path) -> list[ScientificAction]:
    path = root / "optimizer_v2_tool" / "information_gain_queue.jsonl"
    if not path.exists():
        return []
    actions = []
    for row in read_jsonl(path):
        if "tool_id" not in row or not row.get("tool_id"):
            row["tool_id"] = _infer_tool_id(row)
        row.setdefault("domain_id", "superconductivity_lsco")
        row.setdefault("arguments", {})
        row.setdefault("estimated_cost", row.get("cost", 1.0))
        row.setdefault("target_blockers", row.get("blockers", []))
        row.setdefault("idempotency_key", _digest(f"{row.get('domain_id')}:{row.get('hypothesis_id')}:{row.get('tool_id')}:{row.get('action_type')}"))
        actions.append(ScientificAction.model_validate(row))
    return sorted(actions, key=lambda item: item.priority, reverse=True)


def _execution_history(root: Path) -> list[dict[str, Any]]:
    path = root / "closed_loop_action_executions.jsonl"
    return read_jsonl(path) if path.exists() else []


def _eligible_actions(
    queue: list[ScientificAction],
    prior: list[dict[str, Any]],
    registry: ScientificToolRegistry,
    *,
    live_model: bool,
    live_network: bool,
) -> tuple[list[ScientificAction], dict[str, list[str]]]:
    completed_keys = {str(item.get("idempotency_key") or item.get("action", {}).get("idempotency_key")) for item in prior if item.get("status") in {"succeeded", "inconclusive", "skipped_duplicate"}}
    completed_ids = {str(item.get("selected_action_id")) for item in prior if item.get("status") in {"succeeded", "inconclusive", "skipped_duplicate"}}
    completed_tool_keys = {
        _tool_duplicate_key(item.get("action", {}))
        for item in prior
        if item.get("status") in {"succeeded", "inconclusive", "skipped_duplicate"} and isinstance(item.get("action"), dict)
    }
    seen_keys: set[str] = set()
    seen_tool_keys: set[str] = set()
    eligible: list[ScientificAction] = []
    reasons: dict[str, list[str]] = {}
    for action in queue:
        blocked: list[str] = []
        if action.action_id in completed_ids or action.idempotency_key in completed_keys:
            blocked.append("already completed")
        if action.idempotency_key in seen_keys:
            blocked.append("duplicate action")
        tool_key = _tool_duplicate_key(action.model_dump(mode="json"))
        if tool_key in completed_tool_keys and action.tool_id != "targeted_repair_task":
            blocked.append("duplicate deterministic tool already executed")
        if tool_key in seen_tool_keys and action.tool_id != "targeted_repair_task":
            blocked.append("duplicate tool candidate")
        if not registry.has(action.tool_id):
            blocked.append("missing tool")
        if "live_network" in action.required_permissions and not live_network:
            blocked.append("permission required: live_network")
        if "live_model" in action.required_permissions and not live_model:
            blocked.append("permission required: live_model")
        if action.execution_status in {"succeeded", "failed", "policy_blocked", "permission_blocked"}:
            blocked.append(f"action status is {action.execution_status}")
        if blocked:
            reasons[action.action_id] = blocked
        else:
            eligible.append(action)
        seen_keys.add(action.idempotency_key)
        seen_tool_keys.add(tool_key)
    return eligible, reasons


def _policy_decision(action: ScientificAction, registry: ScientificToolRegistry, *, live_model: bool, live_network: bool) -> ActionPolicyDecision:
    reasons: list[str] = []
    status = "allowed"
    if not registry.has(action.tool_id):
        reasons.append(f"missing registered executor: {action.tool_id}")
        status = "not_executable"
    if "live_network" in action.required_permissions and not live_network:
        reasons.append("live network permission was not enabled")
        status = "permission_blocked"
    if "live_model" in action.required_permissions and not live_model:
        reasons.append("live model permission was not enabled")
        status = "permission_blocked"
    if action.domain_id == "superconductivity_lsco" and action.action_type == "promote_tier_c":
        reasons.append("Tier A/B LSCO readiness cannot be promoted to Tier C without complete reviewed evidence")
        status = "policy_blocked"
    mode = "live-network" if "live_network" in action.required_permissions else "live-model" if "live_model" in action.required_permissions else "deterministic"
    return ActionPolicyDecision(
        action_id=action.action_id,
        allowed=not reasons,
        status=status,  # type: ignore[arg-type]
        reasons=reasons,
        required_permissions=action.required_permissions,
        execution_mode=mode,  # type: ignore[arg-type]
    )


def _run_phase1_tool(bundle: Path, action: ScientificAction, context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str, list[str]]:
    output = run_minimal_mixed_bcs_project(runs_dir=bundle / "tool_outputs", run_id="phase1_minimal_mixed_bcs_solver", force=True)
    errors = validate_minimal_mixed_bcs_run(output)
    rels = _relative_artifacts(bundle, output)
    verifiers = read_jsonl(output / "minimal_bcs_verifier_results.jsonl")
    base = read_json(output / "base_solution.json")
    summary = f"Phase-1 mixed BCS solver executed: stable={base.get('stable_superconducting_solution')}, gap_ev={base.get('gap_ev')}."
    return rels, verifiers, summary, errors


def _run_phase2_coverage_tool(bundle: Path, action: ScientificAction, context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str, list[str]]:
    output = run_phase2_data_coverage_tool(bundle / "tool_outputs" / "phase2_data_coverage_tool", source_path=context.get("phase2_data_path"))
    errors = validate_phase2_data_coverage_tool(output)
    rels = _relative_artifacts(bundle, output)
    evaluation = read_json(output / "phase2_data_tool_evaluation.json")
    coverage = read_json(output / "phase2_data_coverage.json")
    tier_c = coverage.get("tier_c", {})
    verifier = {
        "schema_version": "v28-action-verifier",
        "verifier_id": "phase2_tiered_readiness_gate",
        "verdict": "pass" if evaluation.get("status", "").startswith("partial") else "inconclusive",
        "basis_artifact_ids": [item for item in rels if item.endswith("phase2_data_coverage.json")],
        "summary": evaluation.get("conclusion"),
        "tier_c_status": tier_c.get("status") if isinstance(tier_c, dict) else None,
    }
    summary = f"Phase-2 readiness executed: status={evaluation.get('status')}; Tier C remains {tier_c.get('status') if isinstance(tier_c, dict) else 'unknown'}."
    return rels, [verifier], summary, errors


def _run_energy_decomposition_tool(bundle: Path, action: ScientificAction, context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str, list[str]]:
    output = run_energy_decomposition_audit(bundle / "tool_outputs" / "energy_decomposition_audit_tool")
    errors = validate_energy_decomposition_audit(output)
    rels = _relative_artifacts(bundle, output)
    verifiers = read_jsonl(output / "energy_decomposition_verifier_results.jsonl")
    outcome = read_json(output / "final_theory_outcome.json")
    summary = f"Energy-decomposition audit executed: status={outcome.get('status')}; valid_outcome={outcome.get('valid_outcome_type')}."
    return rels, verifiers, summary, errors


def _run_targeted_repair_tool(bundle: Path, action: ScientificAction, context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str, list[str]]:
    repair = {
        "schema_version": "v28-targeted-repair-task",
        "action_id": action.action_id,
        "hypothesis_id": action.hypothesis_id,
        "repair": "Create a narrower action with a registered verifier and explicit evidence target.",
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(bundle / "targeted_repair_task.json", repair)
    verifier = {
        "schema_version": "v28-action-verifier",
        "verifier_id": "targeted_repair_task_created",
        "verdict": "pass",
        "basis_artifact_ids": ["targeted_repair_task.json"],
        "summary": "Repair task persisted for human or optimizer follow-up.",
    }
    return ["targeted_repair_task.json"], [verifier], "Targeted repair task generated.", []


def _update_claim_dag(root: Path, action: ScientificAction, result: ActionExecutionResult) -> ClaimDAGUpdateResult:
    before = _claim_dag_snapshot(root)
    if not (root / "formal_claim.json").exists():
        create_claim_dag_artifacts_from_run(root, candidate_id=action.hypothesis_id, force=True)
    formal = read_json(root / "formal_claim.json")
    claim_id = str(formal.get("claim_id") or f"claim-main-{action.hypothesis_id}")
    check_id = f"check-{_digest(action.action_id + result.status)}"
    verdict = "pass" if result.status == "succeeded" else "uncertain" if result.status == "inconclusive" else "fail"
    check = ClaimCheckRecord(
        check_id=check_id,
        claim_id=claim_id,
        verdict=verdict,  # type: ignore[arg-type]
        severity="load_bearing",
        confidence=0.85 if verdict == "pass" else 0.55,
        basis_artifact_ids=[f"action_executions/{_safe_name(action.action_id)}/execution_result.json", *[f"action_executions/{_safe_name(action.action_id)}/{item}" for item in result.generated_artifacts[:6]]],
        concise_justification=result.summary[:500],
        verifier_name=f"scientific_action_executor:{action.tool_id}",
        verifier_version="v28",
        cached_or_recomputed="recomputed",
    )
    checks = read_jsonl(root / "claim_checks.jsonl") if (root / "claim_checks.jsonl").exists() else []
    checks.append(check.model_dump(mode="json"))
    write_jsonl(root / "claim_checks.jsonl", checks)
    added_blockers: list[str] = []
    added_contradictions: list[str] = []
    invalidated: list[str] = []
    if action.tool_id in {"energy_decomposition_audit_tool", "energy_decomposition_audit", "representation_counterexample", "hellmann_feynman_diagnostic"}:
        contradiction_id = f"contradiction-{_digest(action.action_id + 'unique-percentages')}"
        contradiction = ClaimContradiction(
            contradiction_id=contradiction_id,
            claim_id=claim_id,
            source_artifact_id=f"action_executions/{_safe_name(action.action_id)}/execution_result.json",
            severity="major",
            description="Counterexample shows raw Hamiltonian component percentages are representation dependent; total energy and full response remain the safer physical quantities.",
        )
        contradictions = read_jsonl(root / "claim_contradictions.jsonl") if (root / "claim_contradictions.jsonl").exists() else []
        contradictions.append(contradiction.model_dump(mode="json"))
        write_jsonl(root / "claim_contradictions.jsonl", contradictions)
        blocker_id = f"blocker-{_digest(action.action_id + 'representation-dependent')}"
        blocker = ValidationBlocker(
            blocker_id=blocker_id,
            claim_id=claim_id,
            blocker_type="verifier_gap",
            description="Mechanism percentages must be labeled representation/model dependent unless an observable mapping is established.",
            terminal_impact="insufficient_evidence",
        )
        blockers = read_jsonl(root / "validation_blockers.jsonl") if (root / "validation_blockers.jsonl").exists() else []
        blockers.append(blocker.model_dump(mode="json"))
        write_jsonl(root / "validation_blockers.jsonl", blockers)
        added_contradictions.append(contradiction_id)
        added_blockers.append(blocker_id)
        invalidated.append("unique_phonon_vs_kinetic_percentage_claim")
        _write_total_gate(root, formal.get("candidate_id", action.hypothesis_id), [blocker_id], "insufficient_evidence", "energy_decomposition_guardrail")
    elif action.tool_id in {"phase2_data_coverage_tool", "phase2_lsco_acquisition"}:
        coverage_status = _phase2_tier_c_status(result)
        if coverage_status != "pass":
            blocker_id = f"blocker-{_digest(action.action_id + 'tier-c')}"
            blocker = ValidationBlocker(
                blocker_id=blocker_id,
                claim_id=claim_id,
                blocker_type="missing_evidence",
                description="LSCO data are at partial Tier A/B readiness; Tier C quantitative separation remains blocked.",
                terminal_impact="insufficient_evidence",
            )
            blockers = read_jsonl(root / "validation_blockers.jsonl") if (root / "validation_blockers.jsonl").exists() else []
            blockers.append(blocker.model_dump(mode="json"))
            write_jsonl(root / "validation_blockers.jsonl", blockers)
            added_blockers.append(blocker_id)
            invalidated.append("material_level_tier_c_quantitative_separation_claim")
            _write_total_gate(root, formal.get("candidate_id", action.hypothesis_id), [blocker_id], "insufficient_evidence", "phase2_tiered_readiness_guardrail")
    try:
        rebuild_claim_dag_database(root)
    except Exception:
        pass
    after = _claim_dag_snapshot(root)
    return ClaimDAGUpdateResult(
        action_id=action.action_id,
        status="updated",
        updated_claim_ids=[claim_id],
        added_check_ids=[check_id],
        added_blocker_ids=added_blockers,
        added_contradiction_ids=added_contradictions,
        invalidated_claim_ids=invalidated,
        before=before,
        after=after,
        rationale="Action result was attached to the Claim DAG with guardrail-aware checks and blockers.",
    )


def _update_optimizer(root: Path, action: ScientificAction, result: ActionExecutionResult, queue: list[ScientificAction]) -> OptimizerUpdateResult:
    before = {"queued_count": len(queue), "top_action_id": queue[0].action_id if queue else None}
    updated_rows: list[dict[str, Any]] = []
    downgraded: list[str] = []
    new_actions: list[str] = []
    for item in queue:
        row = item.model_dump(mode="json")
        if item.action_id == action.action_id:
            row["execution_status"] = result.status
            row["executed_at"] = result.finished_at
        elif result.status in {"succeeded", "inconclusive"} and item.tool_id == action.tool_id:
            row["priority"] = round(float(row.get("priority", 0.0)) * 0.5, 6)
            downgraded.append(item.action_id)
        updated_rows.append(row)
    if result.status in {"succeeded", "inconclusive"} and action.tool_id in {"energy_decomposition_audit_tool", "energy_decomposition_audit"}:
        repair_id = f"eig-repair-{_digest(action.action_id)}"
        if not any(row.get("action_id") == repair_id for row in updated_rows):
            updated_rows.append(
                ScientificAction(
                    action_id=repair_id,
                    hypothesis_id=action.hypothesis_id,
                    domain_id=action.domain_id,
                    action_type="generate_targeted_repair_task",
                    tool_id="targeted_repair_task",
                    expected_information_gain=0.45,
                    success_probability=0.9,
                    feasibility=0.9,
                    estimated_cost=0.5,
                    priority=0.81,
                    rationale="Repair any claim that converts representation-dependent component ledgers into unique physical percentages.",
                    expected_artifact_types=["repair_task"],
                    target_claims=action.target_claims,
                    provenance=["scientific_action_executor"],
                    idempotency_key=_digest(f"repair:{action.action_id}"),
                ).model_dump(mode="json")
            )
            new_actions.append(repair_id)
    path = root / "optimizer_v2_tool" / "information_gain_queue.jsonl"
    if path.exists():
        write_jsonl(path, updated_rows)
    after = {
        "queued_count": len(updated_rows),
        "completed_action_ids": [action.action_id],
        "top_action_id": next((row["action_id"] for row in sorted(updated_rows, key=lambda row: float(row.get("priority", 0.0)), reverse=True) if row.get("execution_status") == "queued"), None),
    }
    return OptimizerUpdateResult(
        action_id=action.action_id,
        status="updated",
        completed_action_ids=[action.action_id],
        downgraded_action_ids=downgraded,
        new_action_ids=new_actions,
        before=before,
        after=after,
        rationale="Executed action was marked complete, related duplicate tool actions were downgraded, and repair actions were added when guardrails fired.",
    )


def _state_from_parts(
    root: Path,
    action: ScientificAction,
    policy: ActionPolicyDecision,
    result: ActionExecutionResult,
    claim_diff: ClaimDAGUpdateResult | None,
    optimizer_diff: OptimizerUpdateResult | None,
    next_actions: list[ScientificAction],
    bundle: Path,
) -> ClosedLoopExecutionState:
    return ClosedLoopExecutionState(
        run_id=root.name,
        selected_action_id=action.action_id,
        status=result.status,
        reason=result.summary,
        action=action.model_dump(mode="json"),
        policy_decision=policy.model_dump(mode="json"),
        execution_result=result.model_dump(mode="json"),
        claim_dag_diff=claim_diff.model_dump(mode="json") if claim_diff else None,
        optimizer_diff=optimizer_diff.model_dump(mode="json") if optimizer_diff else None,
        next_eligible_actions=[item.model_dump(mode="json") for item in next_actions],
        bundle_dir=str(bundle.relative_to(root)),
    )


def _persist_state(root: Path, state: ClosedLoopExecutionState) -> None:
    write_json(root / "closed_loop_action_state.json", state)
    rows = _execution_history(root)
    payload = state.model_dump(mode="json")
    if state.action:
        payload["idempotency_key"] = state.action.get("idempotency_key")
    rows.append(payload)
    write_jsonl(root / "closed_loop_action_executions.jsonl", rows)


def _summary_payload(
    action: ScientificAction,
    policy: ActionPolicyDecision,
    result: ActionExecutionResult,
    claim_diff: ClaimDAGUpdateResult | None,
    optimizer_diff: OptimizerUpdateResult | None,
) -> dict[str, Any]:
    return {
        "schema_version": "v28-action-execution-summary",
        "action_id": action.action_id,
        "hypothesis_id": action.hypothesis_id,
        "domain_id": action.domain_id,
        "tool_id": action.tool_id,
        "status": result.status,
        "execution_mode": policy.execution_mode,
        "summary": result.summary,
        "generated_artifacts": result.generated_artifacts,
        "verifier_result_count": len(result.verifier_results),
        "claim_dag_status": claim_diff.status if claim_diff else "not_updated",
        "optimizer_status": optimizer_diff.status if optimizer_diff else "not_updated",
        "guardrails": [
            "component energy percentages remain representation/model dependent",
            "LSCO Tier A/B partial readiness is not Tier C quantitative separation",
            "deterministic execution is not live validation",
        ],
    }


def _claim_dag_snapshot(root: Path) -> dict[str, Any]:
    return {
        "claim_count": len(read_jsonl(root / "atomic_claims.jsonl")) if (root / "atomic_claims.jsonl").exists() else 0,
        "check_count": len(read_jsonl(root / "claim_checks.jsonl")) if (root / "claim_checks.jsonl").exists() else 0,
        "blocker_count": len(read_jsonl(root / "validation_blockers.jsonl")) if (root / "validation_blockers.jsonl").exists() else 0,
        "contradiction_count": len(read_jsonl(root / "claim_contradictions.jsonl")) if (root / "claim_contradictions.jsonl").exists() else 0,
        "total_gate": read_json(root / "total_gate_result.json").get("terminal_status") if (root / "total_gate_result.json").exists() else None,
    }


def _write_total_gate(root: Path, candidate_id: str, blocker_ids: list[str], status: str, selected_rule: str) -> None:
    verdict = ValidationVerdict(
        candidate_id=candidate_id,
        terminal_status=status,  # type: ignore[arg-type]
        selected_rule=selected_rule,
        blocking_claim_ids=[],
        blocker_ids=blocker_ids,
        rule_trace=["v28 closed-loop action hard gate"],
        arbiter_summary="Closed-loop action execution preserved scientific guardrails.",
    )
    write_json(root / "total_gate_result.json", verdict)


def _relative_artifacts(bundle: Path, output: Path) -> list[str]:
    return [str(path.relative_to(bundle)) for path in sorted(output.rglob("*")) if path.is_file()]


def _phase2_tier_c_status(result: ActionExecutionResult) -> str:
    for row in result.verifier_results:
        if row.get("tier_c_status"):
            return str(row["tier_c_status"])
    return "unknown"


def _infer_tool_id(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ["action_type", "hypothesis_id", "rationale"]).lower()
    if "decomposition" in text or "nonunique" in text or "hellmann" in text:
        return "energy_decomposition_audit_tool"
    if "phase2" in text or "data" in text or "lsco" in text:
        return "phase2_data_coverage_tool"
    if "bcs" in text or "mixed" in text:
        return "phase1_minimal_mixed_bcs_solver"
    return "targeted_repair_task"


def _tool_duplicate_key(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    tool_id = str(action.get("tool_id") or "")
    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    scoped = "" if tool_id in {"phase1_minimal_mixed_bcs_solver", "phase2_data_coverage_tool", "energy_decomposition_audit_tool", "energy_decomposition_audit"} else str(args)
    return _digest(f"{tool_id}:{scoped}")


def _no_eligible_reason(reasons: dict[str, list[str]]) -> str:
    if not reasons:
        return "no actions in optimizer queue"
    flattened = [reason for items in reasons.values() for reason in items]
    if any("permission required" in item for item in flattened):
        return "all remaining actions require unavailable permission"
    if any("missing tool" in item for item in flattened):
        return "remaining actions have no registered executor"
    if all("already completed" in item for item in flattened):
        return "all actions already attempted"
    return "; ".join(flattened[:4])


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[:120]


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _contains_secret_like_text(root: Path) -> bool:
    needles = ["OPENAI_API_KEY=", "OPENROUTER_API_KEY=", "Bearer "]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(needle in text for needle in needles):
            return True
        if any(token.startswith("sk-") and len(token) > 24 for token in text.replace('"', " ").replace("'", " ").split()):
            return True
    return False
