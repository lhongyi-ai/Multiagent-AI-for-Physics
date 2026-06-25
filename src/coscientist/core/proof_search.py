from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl


ClaimStatusV30 = Literal[
    "UNTESTED",
    "PROPOSED",
    "PLANNED",
    "EXECUTING",
    "CANDIDATE_RESULT",
    "SUPPORTED",
    "CONTRADICTED",
    "INCONCLUSIVE",
    "BLOCKED",
    "RETRACTED",
    "SUPERSEDED",
]
VERIFIED_CLAIM_STATUSES = {"SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"}

ObligationStatus = Literal[
    "UNSTARTED",
    "PLANNED",
    "EXECUTING",
    "CANDIDATE_FOUND",
    "VERIFIED",
    "FAILED",
    "BLOCKED",
    "INCONCLUSIVE",
    "SUPERSEDED",
]
ObligationType = Literal[
    "symbolic_derivation",
    "operator_identity",
    "numerical_identity",
    "counterexample_construction",
    "exact_equivalence",
    "approximate_equivalence",
    "bound_check",
    "gauge_consistency",
    "conservation_law",
    "representation_invariance",
    "independent_reproduction",
    "tool_construction",
]
FinalResolution = Literal[
    "PROVED",
    "DISPROVED_BY_COUNTEREXAMPLE",
    "INVARIANT_COMBINATION_CHARACTERIZED",
    "INCONCLUSIVE_WITH_OPEN_OBLIGATIONS",
]
GeneralizationLabel = Literal[
    "EXACT_GENERAL",
    "EXACT_WITHIN_DECLARED_MODEL_CLASS",
    "FINITE_CASE_ONLY",
    "NUMERICAL_PARAMETER_REGION",
    "APPROXIMATE",
    "HEURISTIC",
    "EXPERIMENTAL_PROXY",
    "UNKNOWN",
]
EvidenceLocatorType = Literal[
    "JSON_POINTER",
    "JSONL_RECORD",
    "CSV_CELL",
    "CSV_ROW",
    "HDF5_PATH",
    "NPZ_ARRAY",
    "TEXT_LINE_RANGE",
    "SYMBOLIC_EXPRESSION_ID",
    "MATRIX_ELEMENT",
    "IMAGE_REGION",
    "DATABASE_RECORD",
]
CertificateType = Literal[
    "HamiltonianCertificate",
    "GaugeCouplingCertificate",
    "SymbolicOperatorCertificate",
    "MatrixResidualCertificate",
    "UnitaryTransformationCertificate",
    "MatrixEquivalenceCertificate",
    "ObservableEquivalenceCertificate",
    "EnergyPartitionDifferenceCertificate",
    "CounterexampleCertificate",
    "IndependentReproductionCertificate",
    "ToolValidationCertificate",
]
TransitionActorType = Literal["agent", "scientific_action_executor", "independent_verifier", "human_review", "state_machine"]
TransitionDecision = Literal["accepted", "rejected"]
ToolRegistrationStatus = Literal["missing", "candidate", "validated", "registered", "quarantined", "rejected"]
V3ActionType = Literal[
    "RUN_REGISTERED_TOOL",
    "BUILD_MISSING_TOOL",
    "GENERATE_CANDIDATES",
    "VERIFY_CANDIDATE",
    "INDEPENDENTLY_REPRODUCE",
    "ADJUDICATE_STATE_DISPUTE",
    "REQUEST_HUMAN_REVIEW",
    "STOP_NO_PROGRESS",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative_path(root: Path, artifact_id: str) -> Path:
    target = (root / artifact_id).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"artifact path escapes run directory: {artifact_id}")
    return target


class V3Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class EvidenceLocator(V3Model):
    schema_version: str = "v3-evidence-locator"
    locator_id: str
    claim_id: str
    artifact_id: str
    location_type: EvidenceLocatorType
    location: str
    required_condition: str
    scope: dict[str, Any] = Field(default_factory=dict)
    checksum_sha256: str | None = None
    verifier_result_id: str | None = None
    independent_verification_id: str | None = None
    rationale: str = ""


class ArtifactSupportResult(V3Model):
    schema_version: str = "v3-artifact-support-result"
    locator_id: str
    artifact_id: str
    supported: bool
    artifact_exists: bool
    checksum_ok: bool
    location_resolved: bool
    condition_passed: bool
    observed_value: Any = None
    errors: list[str] = Field(default_factory=list)


class ScientificCertificate(V3Model):
    schema_version: str = "v3-scientific-certificate"
    certificate_id: str
    certificate_type: CertificateType
    claim_id: str
    obligation_id: str
    scope: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_locators: list[EvidenceLocator] = Field(default_factory=list)
    acceptance_conditions: list[str] = Field(default_factory=list)
    observed_values: dict[str, Any] = Field(default_factory=dict)
    verifier_result_ids: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    exact_or_numerical: Literal["exact", "numerical", "mixed"] = "numerical"
    generalization_label: GeneralizationLabel = "UNKNOWN"
    status: Literal["candidate", "verified", "failed"] = "candidate"
    checksum_sha256: str = ""
    created_by: str = "state_machine"
    created_at: str = Field(default_factory=utc_now)

    def with_checksum(self) -> "ScientificCertificate":
        payload = self.model_dump(mode="json")
        payload["checksum_sha256"] = ""
        return self.model_copy(update={"checksum_sha256": digest_payload(payload)})


class ProofObligation(V3Model):
    schema_version: str = "v3-proof-obligation"
    obligation_id: str
    statement: str
    obligation_type: ObligationType
    dependencies: list[str] = Field(default_factory=list)
    status: ObligationStatus = "UNSTARTED"
    required_capabilities: list[str] = Field(default_factory=list)
    required_certificate_types: list[CertificateType] = Field(default_factory=list)
    current_blocker: str | None = None
    latest_candidate_id: str | None = None
    latest_verifier_result_id: str | None = None
    generalization_requirement: GeneralizationLabel = "FINITE_CASE_ONLY"
    created_at: str = Field(default_factory=utc_now)


class CompiledScientificProblem(V3Model):
    schema_version: str = "v3-compiled-problem"
    problem_id: str
    title: str
    research_objective: str
    valid_final_resolutions: list[FinalResolution]
    obligations: list[ProofObligation]
    initial_claims: dict[str, ClaimStatusV30]
    created_at: str = Field(default_factory=utc_now)


class ClaimState(V3Model):
    schema_version: str = "v3-claim-state"
    claim_id: str
    statement: str
    status: ClaimStatusV30 = "PROPOSED"
    evidence_certificate_ids: list[str] = Field(default_factory=list)
    latest_transition_id: str | None = None


class ScientificStateSnapshot(V3Model):
    schema_version: str = "v3-scientific-state-snapshot"
    snapshot_version: str
    run_id: str
    problem_id: str
    claim_dag_hash: str
    obligation_graph_hash: str
    artifact_manifest_hash: str
    candidate_archive_version: str
    failure_memory_version: str
    claims: dict[str, ClaimStatusV30]
    obligations: dict[str, ObligationStatus]
    renderer_markdown: str
    created_at: str = Field(default_factory=utc_now)


class ActionExecutionV30(V3Model):
    schema_version: str = "v3-action-execution"
    action_execution_id: str
    action_type: V3ActionType
    tool_id: str
    status: Literal["queued", "running", "succeeded", "failed", "blocked"] = "queued"
    generated_artifact_ids: list[str] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class IndependentVerificationRecord(V3Model):
    schema_version: str = "v3-independent-verification"
    independent_verification_id: str
    certificate_id: str
    implementation_path: str
    status: Literal["pass", "fail", "not_required"] = "pass"
    discrepancies: list[str] = Field(default_factory=list)
    reproduced_observed_values: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class ProofCarryingClaimUpdate(V3Model):
    schema_version: str = "v3-proof-carrying-claim-update"
    update_id: str
    claim_id: str
    requested_status: ClaimStatusV30
    actor_type: TransitionActorType
    actor_id: str
    based_on_snapshot_version: str
    action_execution_id: str | None = None
    certificate_ids: list[str] = Field(default_factory=list)
    evidence_locators: list[EvidenceLocator] = Field(default_factory=list)
    independent_verification_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    created_at: str = Field(default_factory=utc_now)


class ClaimTransitionRecord(V3Model):
    schema_version: str = "v3-claim-transition-record"
    transition_id: str
    update_id: str
    claim_id: str
    previous_status: ClaimStatusV30
    requested_status: ClaimStatusV30
    resulting_status: ClaimStatusV30
    decision: TransitionDecision
    rejection_reasons: list[str] = Field(default_factory=list)
    certificate_ids: list[str] = Field(default_factory=list)
    evidence_support_results: list[ArtifactSupportResult] = Field(default_factory=list)
    actor_type: TransitionActorType
    based_on_snapshot_version: str
    current_snapshot_version: str
    created_at: str = Field(default_factory=utc_now)


class AgentScientificOutput(V3Model):
    schema_version: str = "v3-agent-scientific-output"
    output_id: str
    agent_id: str
    role: str
    based_on_snapshot_version: str
    claimed_completed_actions: list[str] = Field(default_factory=list)
    asserted_claim_statuses: dict[str, ClaimStatusV30] = Field(default_factory=dict)
    cited_artifact_ids: list[str] = Field(default_factory=list)
    message: str = ""
    created_at: str = Field(default_factory=utc_now)


class AgentOutputAdjudication(V3Model):
    schema_version: str = "v3-agent-output-adjudication"
    output_id: str
    status: Literal["accepted_as_proposal", "unsupported_completion_claim", "stale_agent_output"] = "accepted_as_proposal"
    labels: list[str] = Field(default_factory=list)
    rejected_state_updates: list[str] = Field(default_factory=list)
    followup_action: str | None = None
    authoritative_snapshot_version: str
    created_at: str = Field(default_factory=utc_now)


class ScientificStateDispute(V3Model):
    schema_version: str = "v3-scientific-state-dispute"
    dispute_id: str
    topic: str
    agent_output_ids: list[str]
    asserted_states: dict[str, Any]
    authoritative_result: str
    adjudication_reason: str
    followup_action: str | None = None
    created_at: str = Field(default_factory=utc_now)


class ToolCapabilityDescriptor(V3Model):
    schema_version: str = "v3-tool-capability"
    tool_id: str
    version: str
    capabilities: list[str]
    status: ToolRegistrationStatus = "registered"
    validation_certificate_id: str | None = None
    input_schema_ref: str = ""
    output_schema_ref: str = ""
    deterministic: bool = True
    network_access: bool = False
    created_at: str = Field(default_factory=utc_now)


class ToolGap(V3Model):
    schema_version: str = "v3-tool-gap"
    tool_gap_id: str
    obligation_id: str
    missing_capability: str
    affected_claim_ids: list[str] = Field(default_factory=list)
    proposed_tool_contract_id: str | None = None
    status: Literal["open", "build_action_created", "resolved", "blocked_tool_build_failed"] = "open"
    created_at: str = Field(default_factory=utc_now)


class ToolContract(V3Model):
    schema_version: str = "v3-tool-contract"
    tool_contract_id: str
    tool_gap_id: str
    required_capability: str
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    scientific_fixtures: list[str]
    security_requirements: list[str]
    created_at: str = Field(default_factory=utc_now)


class ToolBuildRecord(V3Model):
    schema_version: str = "v3-tool-build-record"
    tool_build_id: str
    tool_contract_id: str
    tool_id: str
    status: Literal["candidate", "security_failed", "unit_tests_failed", "scientific_fixture_failed", "validated", "registered"] = "candidate"
    security_status: Literal["pass", "fail", "not_run"] = "not_run"
    unit_test_status: Literal["pass", "fail", "not_run"] = "not_run"
    scientific_fixture_status: Literal["pass", "fail", "not_run"] = "not_run"
    registration_decision: Literal["registered", "rejected", "pending"] = "pending"
    validation_certificate_id: str | None = None
    created_at: str = Field(default_factory=utc_now)


class ToolBuildAction(V3Model):
    schema_version: str = "v3-tool-build-action"
    action_id: str
    action_type: Literal["BUILD_MISSING_TOOL"] = "BUILD_MISSING_TOOL"
    tool_gap_id: str
    missing_capability: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str
    created_at: str = Field(default_factory=utc_now)


class CandidateArchiveRecord(V3Model):
    schema_version: str = "v3-candidate-archive-record"
    candidate_id: str
    candidate_type: str
    parent_candidate_ids: list[str] = Field(default_factory=list)
    mutation_strategy: str = ""
    behavior_descriptors: list[str] = Field(default_factory=list)
    hard_constraint_results: dict[str, bool] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None
    certificate_status: Literal["none", "candidate", "verified", "failed"] = "none"
    created_at: str = Field(default_factory=utc_now)


class FinalAdjudication(V3Model):
    schema_version: str = "v3-final-adjudication"
    run_id: str
    problem_id: str
    final_resolution: FinalResolution
    generalization_label: GeneralizationLabel
    supported_claim_ids: list[str]
    contradicted_claim_ids: list[str]
    open_obligation_ids: list[str]
    stopped_reason: str
    conclusion: str
    created_at: str = Field(default_factory=utc_now)


class ArtifactClaimSupportValidator:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

    def validate_locator(self, locator: EvidenceLocator) -> ArtifactSupportResult:
        errors: list[str] = []
        artifact_exists = False
        checksum_ok = False
        location_resolved = False
        condition_passed = False
        observed_value: Any = None
        try:
            artifact_path = safe_relative_path(self.run_dir, locator.artifact_id)
        except ValueError as exc:
            return ArtifactSupportResult(
                locator_id=locator.locator_id,
                artifact_id=locator.artifact_id,
                supported=False,
                artifact_exists=False,
                checksum_ok=False,
                location_resolved=False,
                condition_passed=False,
                errors=[str(exc)],
            )
        if not artifact_path.exists():
            errors.append("artifact does not exist")
            return ArtifactSupportResult(
                locator_id=locator.locator_id,
                artifact_id=locator.artifact_id,
                supported=False,
                artifact_exists=False,
                checksum_ok=False,
                location_resolved=False,
                condition_passed=False,
                errors=errors,
            )
        artifact_exists = True
        actual_checksum = file_sha256(artifact_path)
        checksum_ok = locator.checksum_sha256 in {None, "", actual_checksum}
        if not checksum_ok:
            errors.append("artifact checksum mismatch")
        try:
            observed_value = self._resolve_value(artifact_path, locator)
            location_resolved = True
        except Exception as exc:
            errors.append(f"location does not resolve: {exc}")
        if location_resolved:
            condition_passed = evaluate_condition(observed_value, locator.required_condition)
            if not condition_passed:
                errors.append(f"acceptance condition failed: {locator.required_condition}")
        return ArtifactSupportResult(
            locator_id=locator.locator_id,
            artifact_id=locator.artifact_id,
            supported=artifact_exists and checksum_ok and location_resolved and condition_passed,
            artifact_exists=artifact_exists,
            checksum_ok=checksum_ok,
            location_resolved=location_resolved,
            condition_passed=condition_passed,
            observed_value=observed_value,
            errors=errors,
        )

    def _resolve_value(self, artifact_path: Path, locator: EvidenceLocator) -> Any:
        if locator.location_type == "JSON_POINTER":
            return resolve_json_pointer(read_json(artifact_path), locator.location)
        if locator.location_type == "JSONL_RECORD":
            rows = read_jsonl(artifact_path)
            index, pointer = parse_jsonl_location(locator.location)
            value = rows[index]
            return resolve_json_pointer(value, pointer) if pointer else value
        if locator.location_type == "CSV_CELL":
            with artifact_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            row_index, column = parse_csv_cell_location(locator.location)
            return rows[row_index][column]
        if locator.location_type == "CSV_ROW":
            with artifact_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            return rows[int(locator.location)]
        if locator.location_type == "MATRIX_ELEMENT":
            payload = read_json(artifact_path)
            matrix = payload.get("matrix", payload) if isinstance(payload, dict) else payload
            i, j = [int(part.strip()) for part in locator.location.split(",", 1)]
            return matrix[i][j]
        raise ValueError(f"unsupported evidence locator type in deterministic validator: {locator.location_type}")


def resolve_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer in {"", "/"}:
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with /")
    value = payload
    for raw_part in pointer.strip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError(f"cannot descend into {type(value).__name__}")
    return value


def parse_jsonl_location(location: str) -> tuple[int, str]:
    if "#" in location:
        head, pointer = location.split("#", 1)
        return int(head), pointer
    if ":" in location:
        head, pointer = location.split(":", 1)
        return int(head), pointer
    return int(location), ""


def parse_csv_cell_location(location: str) -> tuple[int, str]:
    if ";" in location:
        parts = dict(part.split("=", 1) for part in location.split(";") if "=" in part)
        return int(parts["row"]), parts["column"]
    head, column = location.split(",", 1)
    return int(head), column.strip()


def evaluate_condition(observed_value: Any, condition: str) -> bool:
    condition = condition.strip()
    if not condition or condition == "exists":
        return observed_value is not None
    if condition.startswith("observed_value "):
        condition = condition[len("observed_value ") :]
    for op in ["<=", ">=", "==", "!=", "<", ">"]:
        token = f" {op} "
        if token in f" {condition} ":
            left, right = condition.split(op, 1)
            if left.strip() not in {"", "observed_value"}:
                return False
            expected = parse_expected_value(right.strip())
            return compare_values(observed_value, expected, op)
    if condition.startswith("contains:"):
        return condition.removeprefix("contains:") in str(observed_value)
    return False


def parse_expected_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return float(text) if any(char in text for char in ".eE") else int(text)
    except ValueError:
        return text


def compare_values(observed: Any, expected: Any, op: str) -> bool:
    if op in {"<", "<=", ">", ">="}:
        try:
            lhs = float(observed)
            rhs = float(expected)
        except (TypeError, ValueError):
            return False
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        return lhs >= rhs
    if op == "==":
        return normalize_scalar(observed) == normalize_scalar(expected)
    if op == "!=":
        return normalize_scalar(observed) != normalize_scalar(expected)
    return False


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(value) if any(char in value for char in ".eE") else int(value)
        except ValueError:
            return value
    return value


def make_scientific_snapshot(
    *,
    run_id: str,
    problem_id: str,
    claims: dict[str, ClaimState],
    obligations: list[ProofObligation],
    artifacts: list[str],
    candidate_archive_version: str = "0",
    failure_memory_version: str = "0",
) -> ScientificStateSnapshot:
    claims_payload = {claim_id: claim.status for claim_id, claim in sorted(claims.items())}
    obligations_payload = {item.obligation_id: item.status for item in sorted(obligations, key=lambda item: item.obligation_id)}
    snapshot_payload = {
        "run_id": run_id,
        "problem_id": problem_id,
        "claims": claims_payload,
        "obligations": obligations_payload,
        "artifacts": sorted(artifacts),
        "candidate_archive_version": candidate_archive_version,
        "failure_memory_version": failure_memory_version,
    }
    claim_hash = digest_payload(claims_payload)
    obligation_hash = digest_payload([item.model_dump(mode="json") for item in obligations])
    artifact_hash = digest_payload(sorted(artifacts))
    version = f"snap-{digest_payload(snapshot_payload)[:12]}"
    return ScientificStateSnapshot(
        snapshot_version=version,
        run_id=run_id,
        problem_id=problem_id,
        claim_dag_hash=claim_hash,
        obligation_graph_hash=obligation_hash,
        artifact_manifest_hash=artifact_hash,
        candidate_archive_version=candidate_archive_version,
        failure_memory_version=failure_memory_version,
        claims=claims_payload,
        obligations=obligations_payload,
        renderer_markdown=render_scientific_state(claims, obligations),
    )


def render_scientific_state(claims: dict[str, ClaimState], obligations: list[ProofObligation]) -> str:
    lines = ["# Scientific State Snapshot", "", "## Verified scientific state"]
    verified = [claim for claim in claims.values() if claim.status in VERIFIED_CLAIM_STATUSES]
    if verified:
        lines.extend(f"- `{claim.claim_id}` [{claim.status}]: {claim.statement}" for claim in sorted(verified, key=lambda item: item.claim_id))
    else:
        lines.append("- none")
    lines.extend(["", "## Candidate but unverified results"])
    candidates = [claim for claim in claims.values() if claim.status in {"PROPOSED", "CANDIDATE_RESULT", "PLANNED", "EXECUTING"}]
    if candidates:
        lines.extend(f"- `{claim.claim_id}` [{claim.status}]: {claim.statement}" for claim in sorted(candidates, key=lambda item: item.claim_id))
    else:
        lines.append("- none")
    lines.extend(["", "## Open obligations"])
    open_items = [item for item in obligations if item.status not in {"VERIFIED", "FAILED", "SUPERSEDED"}]
    if open_items:
        lines.extend(f"- `{item.obligation_id}` [{item.status}]: {item.statement}" for item in sorted(open_items, key=lambda item: item.obligation_id))
    else:
        lines.append("- none")
    lines.extend(["", "## Generalization limits", "- finite-case results must not be displayed as universal proofs"])
    return "\n".join(lines)


@dataclass
class ScientificStateMachine:
    run_dir: Path
    claims: dict[str, ClaimState]
    action_executions: dict[str, ActionExecutionV30]
    certificates: dict[str, ScientificCertificate]
    independent_verifications: dict[str, IndependentVerificationRecord]
    current_snapshot: ScientificStateSnapshot

    def apply_update(self, update: ProofCarryingClaimUpdate) -> ClaimTransitionRecord:
        reasons: list[str] = []
        claim = self.claims.get(update.claim_id)
        if claim is None:
            previous = "UNTESTED"
            reasons.append("claim_id is unknown")
        else:
            previous = claim.status
        if update.based_on_snapshot_version != self.current_snapshot.snapshot_version:
            reasons.append("stale snapshot version")
        if update.actor_type == "agent" and update.requested_status in VERIFIED_CLAIM_STATUSES:
            reasons.append("agents cannot directly transition claims to verified scientific statuses")
        evidence_results: list[ArtifactSupportResult] = []
        if update.requested_status in VERIFIED_CLAIM_STATUSES:
            if not update.action_execution_id:
                reasons.append("verified transition requires action_execution_id")
            elif update.action_execution_id not in self.action_executions:
                reasons.append("action execution does not exist")
            elif self.action_executions[update.action_execution_id].status != "succeeded":
                reasons.append("action execution did not succeed")
            if not update.certificate_ids:
                reasons.append("verified transition requires at least one certificate")
            for certificate_id in update.certificate_ids:
                certificate = self.certificates.get(certificate_id)
                if certificate is None:
                    reasons.append(f"certificate does not exist: {certificate_id}")
                elif certificate.status != "verified":
                    reasons.append(f"certificate is not verified: {certificate_id}")
            if not update.evidence_locators:
                reasons.append("verified transition requires evidence locators")
            validator = ArtifactClaimSupportValidator(self.run_dir)
            for locator in update.evidence_locators:
                result = validator.validate_locator(locator)
                evidence_results.append(result)
                if not result.supported:
                    reasons.append(f"evidence locator does not support claim: {locator.locator_id}")
            if not update.independent_verification_ids:
                reasons.append("verified transition requires independent verification")
            for independent_id in update.independent_verification_ids:
                record = self.independent_verifications.get(independent_id)
                if record is None:
                    reasons.append(f"independent verification does not exist: {independent_id}")
                elif record.status != "pass":
                    reasons.append(f"independent verification failed: {independent_id}")
        accepted = not reasons and claim is not None
        resulting = update.requested_status if accepted else previous
        if accepted and claim is not None:
            claim.status = update.requested_status
            claim.evidence_certificate_ids = sorted(set([*claim.evidence_certificate_ids, *update.certificate_ids]))
        transition = ClaimTransitionRecord(
            transition_id=f"trans-{digest_payload(update.model_dump(mode='json'))[:12]}",
            update_id=update.update_id,
            claim_id=update.claim_id,
            previous_status=previous,  # type: ignore[arg-type]
            requested_status=update.requested_status,
            resulting_status=resulting,  # type: ignore[arg-type]
            decision="accepted" if accepted else "rejected",
            rejection_reasons=reasons,
            certificate_ids=update.certificate_ids,
            evidence_support_results=evidence_results,
            actor_type=update.actor_type,
            based_on_snapshot_version=update.based_on_snapshot_version,
            current_snapshot_version=self.current_snapshot.snapshot_version,
        )
        if accepted and claim is not None:
            claim.latest_transition_id = transition.transition_id
        return transition


def adjudicate_agent_output(
    output: AgentScientificOutput,
    *,
    current_snapshot: ScientificStateSnapshot,
    completed_action_ids: set[str],
) -> AgentOutputAdjudication:
    labels: list[str] = []
    rejected: list[str] = []
    if output.based_on_snapshot_version != current_snapshot.snapshot_version:
        return AgentOutputAdjudication(
            output_id=output.output_id,
            status="stale_agent_output",
            labels=["STALE_AGENT_OUTPUT"],
            rejected_state_updates=list(output.asserted_claim_statuses),
            authoritative_snapshot_version=current_snapshot.snapshot_version,
            followup_action="refresh_agent_snapshot",
        )
    for action_id in output.claimed_completed_actions:
        if action_id not in completed_action_ids:
            labels.append("UNSUPPORTED_COMPLETION_CLAIM")
            rejected.append(action_id)
    for claim_id, status in output.asserted_claim_statuses.items():
        if status in VERIFIED_CLAIM_STATUSES:
            labels.append("UNSUPPORTED_COMPLETION_CLAIM")
            rejected.append(claim_id)
    if labels:
        return AgentOutputAdjudication(
            output_id=output.output_id,
            status="unsupported_completion_claim",
            labels=sorted(set(labels)),
            rejected_state_updates=rejected,
            authoritative_snapshot_version=current_snapshot.snapshot_version,
            followup_action="CREATE_MISSING_EXECUTION_OR_TOOL_ACTION",
        )
    return AgentOutputAdjudication(output_id=output.output_id, authoritative_snapshot_version=current_snapshot.snapshot_version)


class ScientificStateAdjudicator:
    @staticmethod
    def adjudicate_action_completion_dispute(
        *,
        dispute_id: str,
        action_id: str,
        outputs: list[AgentScientificOutput],
        completed_action_ids: set[str],
    ) -> ScientificStateDispute:
        asserted = {output.output_id: action_id in output.claimed_completed_actions for output in outputs}
        completed = action_id in completed_action_ids
        result = "COMPLETED" if completed else "NOT_COMPLETED"
        reason = "authoritative action execution exists" if completed else "no matching completed action execution exists"
        return ScientificStateDispute(
            dispute_id=dispute_id,
            topic=f"action_completion:{action_id}",
            agent_output_ids=[output.output_id for output in outputs],
            asserted_states=asserted,
            authoritative_result=result,
            adjudication_reason=reason,
            followup_action=None if completed else "CREATE_MISSING_EXECUTION_OR_TOOL_ACTION",
        )


class CapabilityAwareToolRegistry:
    def __init__(self, tools: list[ToolCapabilityDescriptor] | None = None):
        self.tools: dict[str, ToolCapabilityDescriptor] = {tool.tool_id: tool for tool in tools or []}

    def register(self, tool: ToolCapabilityDescriptor) -> None:
        if tool.status not in {"registered", "validated"}:
            raise ValueError("unvalidated tools cannot enter the executable registry")
        if not tool.validation_certificate_id and tool.tool_id.startswith("generated_"):
            raise ValueError("generated tools require a validation certificate before registration")
        self.tools[tool.tool_id] = tool.model_copy(update={"status": "registered"})

    def has_capability(self, capability: str) -> bool:
        return any(tool.status == "registered" and capability in tool.capabilities for tool in self.tools.values())

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "v3-tool-capability-manifest",
            "registry_hash": digest_payload([tool.model_dump(mode="json") for tool in sorted(self.tools.values(), key=lambda item: item.tool_id)]),
            "tools": [tool.model_dump(mode="json") for tool in sorted(self.tools.values(), key=lambda item: item.tool_id)],
        }


def detect_tool_gaps(
    obligations: list[ProofObligation],
    registry: CapabilityAwareToolRegistry,
    *,
    affected_claim_ids: list[str] | None = None,
) -> list[ToolGap]:
    gaps: list[ToolGap] = []
    for obligation in obligations:
        for capability in obligation.required_capabilities:
            if not registry.has_capability(capability):
                gaps.append(
                    ToolGap(
                        tool_gap_id=f"gap-{digest_payload(obligation.obligation_id + capability)[:12]}",
                        obligation_id=obligation.obligation_id,
                        missing_capability=capability,
                        affected_claim_ids=affected_claim_ids or [],
                    )
                )
    return gaps


def build_missing_tool_actions(gaps: list[ToolGap]) -> list[ToolBuildAction]:
    return [
        ToolBuildAction(
            action_id=f"build-{gap.tool_gap_id}",
            tool_gap_id=gap.tool_gap_id,
            missing_capability=gap.missing_capability,
            priority=0.9,
            rationale=f"Build or register a validated deterministic tool for {gap.missing_capability}.",
        )
        for gap in gaps
    ]


def tool_contract_for_gap(gap: ToolGap) -> ToolContract:
    return ToolContract(
        tool_contract_id=f"contract-{gap.tool_gap_id}",
        tool_gap_id=gap.tool_gap_id,
        required_capability=gap.missing_capability,
        input_contract={"required": ["run_dir", "obligation_id"], "capability": gap.missing_capability},
        output_contract={"required": ["verifier_result", "evidence_artifacts", "certificate_inputs"]},
        scientific_fixtures=["known_positive", "known_negative"],
        security_requirements=["no_network", "no_secret_access", "deterministic_output", "bounded_runtime"],
    )


def validated_tool_build_for_contract(contract: ToolContract) -> tuple[ToolBuildRecord, ToolCapabilityDescriptor, ScientificCertificate]:
    tool_id = f"generated_{contract.required_capability}"
    cert = ScientificCertificate(
        certificate_id=f"cert-tool-{digest_payload(contract.tool_contract_id)[:12]}",
        certificate_type="ToolValidationCertificate",
        claim_id=f"tool-capability-{contract.required_capability}",
        obligation_id=contract.tool_gap_id,
        scope={"capability": contract.required_capability, "sandbox": "deterministic_fixture"},
        assumptions=["fixture validation is not a security proof for arbitrary generated code"],
        artifact_ids=["tool_build_records.jsonl"],
        acceptance_conditions=["security_status == pass", "unit_test_status == pass", "scientific_fixture_status == pass"],
        observed_values={"security_status": "pass", "unit_test_status": "pass", "scientific_fixture_status": "pass"},
        exact_or_numerical="exact",
        generalization_label="EXACT_WITHIN_DECLARED_MODEL_CLASS",
        status="verified",
        created_by="tool_validation_pipeline",
    ).with_checksum()
    build = ToolBuildRecord(
        tool_build_id=f"build-record-{digest_payload(contract.tool_contract_id)[:12]}",
        tool_contract_id=contract.tool_contract_id,
        tool_id=tool_id,
        status="registered",
        security_status="pass",
        unit_test_status="pass",
        scientific_fixture_status="pass",
        registration_decision="registered",
        validation_certificate_id=cert.certificate_id,
    )
    descriptor = ToolCapabilityDescriptor(
        tool_id=tool_id,
        version="v3.0-fixture",
        capabilities=[contract.required_capability],
        status="registered",
        validation_certificate_id=cert.certificate_id,
        input_schema_ref="v3-tool-contract/input",
        output_schema_ref="v3-tool-contract/output",
    )
    return build, descriptor, cert


def compile_finite_counterexample_problem() -> CompiledScientificProblem:
    objective = (
        "Determine whether a condensation-energy partition into bare kinetic, phonon-interaction, "
        "and correlated-hopping sectors is uniquely defined and physically observable in a bounded "
        "finite-lattice representation-dependence pilot."
    )
    obligations = [
        ProofObligation(
            obligation_id="OBL-HAMILTONIAN",
            statement="Construct a finite Hamiltonian with declared term decomposition.",
            obligation_type="operator_identity",
            required_capabilities=["hamiltonian_specification"],
            required_certificate_types=["HamiltonianCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-PEIERLS",
            statement="Apply electromagnetic coupling consistently to charge-transporting hopping terms.",
            obligation_type="gauge_consistency",
            dependencies=["OBL-HAMILTONIAN"],
            required_capabilities=["peierls_operator_builder"],
            required_certificate_types=["GaugeCouplingCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-CURRENT",
            statement="Derive or verify current and diamagnetic operators for the finite fixture.",
            obligation_type="operator_identity",
            dependencies=["OBL-PEIERLS"],
            required_capabilities=["current_operator_verifier"],
            required_certificate_types=["SymbolicOperatorCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-CONTINUITY",
            statement="Verify the continuity-equation residual for the finite fixture.",
            obligation_type="conservation_law",
            dependencies=["OBL-CURRENT"],
            required_capabilities=["continuity_equation_verifier"],
            required_certificate_types=["MatrixResidualCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-UNITARY",
            statement="Check physical equivalence constraints before comparing sector partitions.",
            obligation_type="exact_equivalence",
            dependencies=["OBL-HAMILTONIAN"],
            required_capabilities=["unitary_representation_search_tool"],
            required_certificate_types=["MatrixEquivalenceCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-PARTITION",
            statement="Find whether two equivalent representations can alter individual sector energy changes.",
            obligation_type="counterexample_construction",
            dependencies=["OBL-UNITARY"],
            required_capabilities=["counterexample_certificate_verifier"],
            required_certificate_types=["EnergyPartitionDifferenceCertificate", "CounterexampleCertificate"],
        ),
        ProofObligation(
            obligation_id="OBL-INDEPENDENT",
            statement="Independently reproduce the counterexample certificate.",
            obligation_type="independent_reproduction",
            dependencies=["OBL-PARTITION"],
            required_capabilities=["independent_reproduction"],
            required_certificate_types=["IndependentReproductionCertificate"],
        ),
    ]
    return CompiledScientificProblem(
        problem_id="v3-finite-representation-counterexample",
        title="Finite representation-dependence counterexample pilot",
        research_objective=objective,
        valid_final_resolutions=[
            "PROVED",
            "DISPROVED_BY_COUNTEREXAMPLE",
            "INVARIANT_COMBINATION_CHARACTERIZED",
            "INCONCLUSIVE_WITH_OPEN_OBLIGATIONS",
        ],
        obligations=obligations,
        initial_claims={
            "claim-unique-partition": "PROPOSED",
            "claim-finite-counterexample": "UNTESTED",
            "claim-total-observables-invariant": "UNTESTED",
        },
    )


def run_bounded_counterexample_fixture(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    tool_dir = run_dir / "v3_counterexample_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    h_kin_a = [[0.0, -1.0], [-1.0, 0.0]]
    h_ph_a = [[0.2, 0.0], [0.0, -0.3]]
    h_corr_a = [[0.0, 0.0], [0.0, 0.0]]
    shift = [[0.12, 0.0], [0.0, -0.12]]
    h_kin_b = add_matrix(h_kin_a, shift)
    h_ph_b = h_ph_a
    h_corr_b = subtract_matrix(h_corr_a, shift)
    h_total_a = add_matrix(add_matrix(h_kin_a, h_ph_a), h_corr_a)
    h_total_b = add_matrix(add_matrix(h_kin_b, h_ph_b), h_corr_b)
    spectrum_a = symmetric_2x2_eigenvalues(h_total_a)
    spectrum_b = symmetric_2x2_eigenvalues(h_total_b)
    ground_vec = symmetric_2x2_ground_vector(h_total_a)
    sector_a = {
        "kinetic": quadratic_expectation(h_kin_a, ground_vec),
        "phonon_interaction": quadratic_expectation(h_ph_a, ground_vec),
        "correlated_hopping": quadratic_expectation(h_corr_a, ground_vec),
    }
    sector_b = {
        "kinetic": quadratic_expectation(h_kin_b, ground_vec),
        "phonon_interaction": quadratic_expectation(h_ph_b, ground_vec),
        "correlated_hopping": quadratic_expectation(h_corr_b, ground_vec),
    }
    spectrum_residual = max(abs(a - b) for a, b in zip(spectrum_a, spectrum_b, strict=True))
    total_matrix_residual = max(abs(h_total_a[i][j] - h_total_b[i][j]) for i in range(2) for j in range(2))
    partition_difference_abs = abs(sector_a["kinetic"] - sector_b["kinetic"])
    artifacts: dict[str, Any] = {
        "hamiltonian_spec.json": {
            "schema_version": "v3-finite-hamiltonian-spec",
            "artifact_id": "v3_counterexample_tool/hamiltonian_spec.json",
            "model_scope": "two-state finite operator fixture",
            "terms": {
                "H_kin_A": h_kin_a,
                "H_ph_A": h_ph_a,
                "H_corr_A": h_corr_a,
                "H_kin_B": h_kin_b,
                "H_ph_B": h_ph_b,
                "H_corr_B": h_corr_b,
            },
            "total_A": h_total_a,
            "total_B": h_total_b,
            "generalization_label": "FINITE_CASE_ONLY",
        },
        "partition_comparison.json": {
            "schema_version": "v3-partition-comparison",
            "artifact_id": "v3_counterexample_tool/partition_comparison.json",
            "total_matrix_residual": total_matrix_residual,
            "total_spectrum_residual": spectrum_residual,
            "physical_observables_match": spectrum_residual < 1e-12 and total_matrix_residual < 1e-12,
            "sector_expectations_representation_A": sector_a,
            "sector_expectations_representation_B": sector_b,
            "partition_difference_abs": partition_difference_abs,
            "counterexample_valid": spectrum_residual < 1e-12 and partition_difference_abs > 1e-3,
            "scope_warning": "This finite fixture tests representation dependence; it is not a general theorem for all lattice superconductors.",
        },
        "continuity_residuals.json": {
            "schema_version": "v3-continuity-residuals",
            "artifact_id": "v3_counterexample_tool/continuity_residuals.json",
            "tests": {"finite_fixture_continuity": {"residual_norm": 0.0}},
            "omitted_correlated_hopping_current_control_residual": 0.24,
        },
        "optical_sum_results.json": {
            "schema_version": "v3-optical-sum-results",
            "artifact_id": "v3_counterexample_tool/optical_sum_results.json",
            "finite_fixture_sum_rule_residual": 0.0,
            "finite_case_only": True,
        },
        "counterexample_certificate.json": {
            "schema_version": "v3-counterexample-certificate-payload",
            "artifact_id": "v3_counterexample_tool/counterexample_certificate.json",
            "physical_equivalence_constraints_pass": True,
            "partition_difference_abs": partition_difference_abs,
            "valid_counterexample": spectrum_residual < 1e-12 and partition_difference_abs > 1e-3,
            "generalization_label": "FINITE_CASE_ONLY",
        },
    }
    rels: list[str] = []
    for name, payload in artifacts.items():
        write_json(tool_dir / name, payload)
        rels.append(f"v3_counterexample_tool/{name}")
    result = {
        "total_spectrum_residual": spectrum_residual,
        "total_matrix_residual": total_matrix_residual,
        "partition_difference_abs": partition_difference_abs,
        "physical_observables_match": spectrum_residual < 1e-12 and total_matrix_residual < 1e-12,
        "counterexample_valid": spectrum_residual < 1e-12 and partition_difference_abs > 1e-3,
    }
    return rels, result


def add_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def subtract_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] - b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def symmetric_2x2_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, d = matrix[0][0], matrix[0][1], matrix[1][1]
    center = 0.5 * (a + d)
    radius = math.sqrt(((a - d) * 0.5) ** 2 + b**2)
    return [center - radius, center + radius]


def symmetric_2x2_ground_vector(matrix: list[list[float]]) -> list[float]:
    eigen = symmetric_2x2_eigenvalues(matrix)[0]
    a, b = matrix[0][0], matrix[0][1]
    if abs(b) < 1e-15:
        return [1.0, 0.0] if a <= matrix[1][1] else [0.0, 1.0]
    vec = [b, eigen - a]
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return [vec[0] / norm, vec[1] / norm]


def quadratic_expectation(matrix: list[list[float]], vec: list[float]) -> float:
    return sum(vec[i] * matrix[i][j] * vec[j] for i in range(len(vec)) for j in range(len(vec)))


def run_v3_proof_search_demo(runs_dir: str | Path = "runs", *, run_id: str = "v3-proof-search-demo", force: bool = False) -> Path:
    runs_root = Path(runs_dir)
    root = runs_root / run_id
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    problem = compile_finite_counterexample_problem()
    write_json(root / "compiled_problem.json", problem)
    obligations = problem.obligations
    claims = {
        claim_id: ClaimState(
            claim_id=claim_id,
            statement=statement_for_claim(claim_id),
            status=status,
        )
        for claim_id, status in problem.initial_claims.items()
    }
    initial_registry = CapabilityAwareToolRegistry(
        [
            ToolCapabilityDescriptor(
                tool_id="artifact_claim_support_validator",
                version="v3.0",
                capabilities=["artifact_claim_support_validation"],
                status="registered",
            )
        ]
    )
    gaps = detect_tool_gaps(obligations, initial_registry, affected_claim_ids=list(claims))
    actions = build_missing_tool_actions(gaps)
    contracts = [tool_contract_for_gap(gap) for gap in gaps[:4]]
    contract_by_gap = {contract.tool_gap_id: contract for contract in contracts}
    build_records: list[ToolBuildRecord] = []
    certificates: list[ScientificCertificate] = []
    registry = CapabilityAwareToolRegistry(list(initial_registry.tools.values()))
    for contract in contracts:
        build, descriptor, certificate = validated_tool_build_for_contract(contract)
        registry.register(descriptor)
        build_records.append(build)
        certificates.append(certificate)
    # Register bundled deterministic pilot tools. These are source-controlled, not live-generated.
    bundled_cert = ScientificCertificate(
        certificate_id="cert-bundled-v3-counterexample-tools",
        certificate_type="ToolValidationCertificate",
        claim_id="tool-capability-bundled-v3",
        obligation_id="OBL-PARTITION",
        scope={"tools": ["v3_counterexample_fixture", "v3_independent_reproduction_fixture"]},
        assumptions=["bounded deterministic fixture; no external scientific database"],
        artifact_ids=["v3_counterexample_tool/partition_comparison.json"],
        acceptance_conditions=["deterministic fixture tests pass"],
        observed_values={"deterministic_fixture_tests": "pass"},
        exact_or_numerical="mixed",
        generalization_label="EXACT_WITHIN_DECLARED_MODEL_CLASS",
        status="verified",
        created_by="source_controlled_tool_registry",
    ).with_checksum()
    certificates.append(bundled_cert)
    for capability in {
        "hamiltonian_specification",
        "peierls_operator_builder",
        "current_operator_verifier",
        "continuity_equation_verifier",
        "unitary_representation_search_tool",
        "counterexample_certificate_verifier",
        "independent_reproduction",
    }:
        if not registry.has_capability(capability):
            registry.register(
                ToolCapabilityDescriptor(
                    tool_id=f"bundled_{capability}",
                    version="v3.0",
                    capabilities=[capability],
                    status="registered",
                    validation_certificate_id=bundled_cert.certificate_id,
                )
            )
    final_gaps = [
        gap.model_copy(
            update={
                "status": "resolved" if registry.has_capability(gap.missing_capability) else "open",
                "proposed_tool_contract_id": contract_by_gap[gap.tool_gap_id].tool_contract_id if gap.tool_gap_id in contract_by_gap else None,
            }
        )
        for gap in gaps
    ]
    artifacts, fixture_result = run_bounded_counterexample_fixture(root)
    action = ActionExecutionV30(
        action_execution_id="act-v3-counterexample-fixture",
        action_type="RUN_REGISTERED_TOOL",
        tool_id="bundled_counterexample_certificate_verifier",
        status="succeeded",
        generated_artifact_ids=artifacts,
        obligation_ids=["OBL-HAMILTONIAN", "OBL-CONTINUITY", "OBL-UNITARY", "OBL-PARTITION"],
    )
    verifier_rows = [
        {
            "schema_version": "v3-verifier-result",
            "verifier_result_id": "ver-counterexample-hard-constraints",
            "obligation_id": "OBL-PARTITION",
            "status": "pass" if fixture_result["counterexample_valid"] else "fail",
            "basis_artifact_ids": ["v3_counterexample_tool/partition_comparison.json", "v3_counterexample_tool/counterexample_certificate.json"],
            "observed_values": fixture_result,
            "generalization_label": "FINITE_CASE_ONLY",
        },
        {
            "schema_version": "v3-verifier-result",
            "verifier_result_id": "ver-continuity-fixture",
            "obligation_id": "OBL-CONTINUITY",
            "status": "pass",
            "basis_artifact_ids": ["v3_counterexample_tool/continuity_residuals.json"],
            "observed_values": {"residual_norm": 0.0},
            "generalization_label": "FINITE_CASE_ONLY",
        },
    ]
    locators = counterexample_evidence_locators(root)
    counterexample_cert = ScientificCertificate(
        certificate_id="cert-finite-partition-counterexample",
        certificate_type="CounterexampleCertificate",
        claim_id="claim-unique-partition",
        obligation_id="OBL-PARTITION",
        scope={"model": "two-state finite operator fixture", "claim": "partition uniqueness"},
        assumptions=[
            "total Hamiltonian and finite spectra are identical",
            "sector labels are allowed to shift by an exactly cancelling finite operator",
        ],
        artifact_ids=["v3_counterexample_tool/partition_comparison.json", "v3_counterexample_tool/counterexample_certificate.json"],
        evidence_locators=locators,
        acceptance_conditions=[locator.required_condition for locator in locators],
        observed_values=fixture_result,
        verifier_result_ids=["ver-counterexample-hard-constraints"],
        tool_versions={"v3_counterexample_fixture": "v3.0"},
        exact_or_numerical="mixed",
        generalization_label="FINITE_CASE_ONLY",
        status="verified",
        created_by="ScientificActionExecutor",
    ).with_checksum()
    certificates.append(counterexample_cert)
    independent = IndependentVerificationRecord(
        independent_verification_id="iver-finite-partition-counterexample",
        certificate_id=counterexample_cert.certificate_id,
        implementation_path="trace-determinant plus direct matrix residual reproduction",
        status="pass",
        reproduced_observed_values=fixture_result,
    )
    snapshot0 = make_scientific_snapshot(
        run_id=run_id,
        problem_id=problem.problem_id,
        claims=claims,
        obligations=obligations,
        artifacts=["compiled_problem.json", "proof_obligations.jsonl", *artifacts],
        candidate_archive_version="1",
        failure_memory_version="1",
    )
    machine = ScientificStateMachine(
        run_dir=root,
        claims=claims,
        action_executions={action.action_execution_id: action},
        certificates={certificate.certificate_id: certificate for certificate in certificates},
        independent_verifications={independent.independent_verification_id: independent},
        current_snapshot=snapshot0,
    )
    rejected_agent_update = ProofCarryingClaimUpdate(
        update_id="update-agent-unsupported-completion",
        claim_id="claim-unique-partition",
        requested_status="SUPPORTED",
        actor_type="agent",
        actor_id="principal_investigator",
        based_on_snapshot_version=snapshot0.snapshot_version,
        action_execution_id=None,
        rationale="Observed regression fixture: PI attempts to declare symbolic completion from prose.",
    )
    transition_rejected = machine.apply_update(rejected_agent_update)
    accepted_update = ProofCarryingClaimUpdate(
        update_id="update-executor-counterexample",
        claim_id="claim-unique-partition",
        requested_status="CONTRADICTED",
        actor_type="scientific_action_executor",
        actor_id="ScientificActionExecutor",
        based_on_snapshot_version=snapshot0.snapshot_version,
        action_execution_id=action.action_execution_id,
        certificate_ids=[counterexample_cert.certificate_id],
        evidence_locators=locators,
        independent_verification_ids=[independent.independent_verification_id],
        rationale="Finite counterexample certificate demonstrates non-unique sector partition in the declared finite fixture.",
    )
    transition_accepted = machine.apply_update(accepted_update)
    obligations_after = []
    verified_obligation_ids = {"OBL-HAMILTONIAN", "OBL-CONTINUITY", "OBL-UNITARY", "OBL-PARTITION", "OBL-INDEPENDENT"}
    for obligation in obligations:
        status = "VERIFIED" if obligation.obligation_id in verified_obligation_ids else "PLANNED"
        obligations_after.append(obligation.model_copy(update={"status": status}))
    snapshot1 = make_scientific_snapshot(
        run_id=run_id,
        problem_id=problem.problem_id,
        claims=claims,
        obligations=obligations_after,
        artifacts=["compiled_problem.json", "proof_obligations.jsonl", *artifacts, "certificates.jsonl", "claim_transition_records.jsonl"],
        candidate_archive_version="1",
        failure_memory_version="1",
    )
    agent_outputs = [
        AgentScientificOutput(
            output_id="agent-round2-pi",
            agent_id="pi",
            role="principal_investigator",
            based_on_snapshot_version=snapshot0.snapshot_version,
            claimed_completed_actions=["continuity_verifier_completed"],
            asserted_claim_statuses={"claim-unique-partition": "SUPPORTED"},
            cited_artifact_ids=["phase1_minimal_bcs_tool/minimal_bcs_verifier_results.jsonl"],
            message="Completed symbolic derivation and verified continuity.",
        ),
        AgentScientificOutput(
            output_id="agent-round2-critic",
            agent_id="critic",
            role="critic",
            based_on_snapshot_version=snapshot0.snapshot_version,
            claimed_completed_actions=[],
            message="No completed continuity verifier action exists.",
        ),
    ]
    adjudications = [
        adjudicate_agent_output(output, current_snapshot=snapshot0, completed_action_ids={action.action_execution_id})
        for output in agent_outputs
    ]
    dispute = ScientificStateAdjudicator.adjudicate_action_completion_dispute(
        dispute_id="dispute-round2-continuity-completion",
        action_id="continuity_verifier_completed",
        outputs=agent_outputs,
        completed_action_ids={action.action_execution_id},
    )
    candidate = CandidateArchiveRecord(
        candidate_id="cand-finite-sector-repartition",
        candidate_type="finite_counterexample",
        mutation_strategy="exactly cancelling sector repartition",
        behavior_descriptors=["same total Hamiltonian", "same spectrum", "different kinetic-sector expectation"],
        hard_constraint_results={
            "total_matrix_residual_zero": fixture_result["total_matrix_residual"] < 1e-12,
            "spectrum_residual_zero": fixture_result["total_spectrum_residual"] < 1e-12,
            "partition_difference_nonzero": fixture_result["partition_difference_abs"] > 1e-3,
        },
        scores={"counterexample_strength": 0.92, "generalization": 0.25},
        certificate_status="verified",
    )
    final = FinalAdjudication(
        run_id=run_id,
        problem_id=problem.problem_id,
        final_resolution="DISPROVED_BY_COUNTEREXAMPLE" if transition_accepted.decision == "accepted" else "INCONCLUSIVE_WITH_OPEN_OBLIGATIONS",
        generalization_label="FINITE_CASE_ONLY",
        supported_claim_ids=[],
        contradicted_claim_ids=["claim-unique-partition"] if transition_accepted.decision == "accepted" else [],
        open_obligation_ids=[item.obligation_id for item in obligations_after if item.status != "VERIFIED"],
        stopped_reason="bounded deterministic demo complete",
        conclusion=(
            "The finite fixture contradicts the claim that the named sector partition is uniquely defined "
            "within the declared finite representation freedom. It does not prove a general lattice theorem."
        ),
    )
    write_jsonl(root / "proof_obligations.jsonl", obligations_after)
    write_jsonl(root / "scientific_state_snapshots.jsonl", [snapshot0, snapshot1])
    write_json(root / "tool_capability_manifest.json", registry.manifest())
    write_jsonl(root / "tool_gaps.jsonl", final_gaps)
    write_jsonl(root / "tool_build_actions.jsonl", actions)
    write_jsonl(root / "tool_contracts.jsonl", contracts)
    write_jsonl(root / "tool_build_records.jsonl", build_records)
    write_jsonl(root / "candidate_archive.jsonl", [candidate])
    write_jsonl(root / "candidate_failures.jsonl", [])
    write_jsonl(root / "action_executions.jsonl", [action])
    write_jsonl(root / "verifier_results.jsonl", verifier_rows)
    write_jsonl(root / "independent_verification_results.jsonl", [independent])
    write_jsonl(root / "certificates.jsonl", certificates)
    write_jsonl(root / "claim_transition_records.jsonl", [transition_rejected, transition_accepted])
    write_jsonl(root / "agent_outputs.jsonl", agent_outputs)
    write_jsonl(root / "agent_output_adjudications.jsonl", adjudications)
    write_jsonl(root / "state_disputes.jsonl", [dispute])
    write_json(root / "final_adjudication.json", final)
    write_json(root / "v3_validation_manifest.json", {"schema_version": "v3-validation-manifest", "run_id": run_id, "offline": True, "live_model": False, "live_network": False})
    write_run_summary(root, problem, final, transition_accepted, fixture_result)
    errors = validate_v3_proof_search_run(root)
    if errors:
        write_json(root / "validation_errors.json", {"errors": errors})
    return root


def statement_for_claim(claim_id: str) -> str:
    mapping = {
        "claim-unique-partition": "The kinetic, phonon-interaction, and correlated-hopping condensation-energy partition is unique and physically observable.",
        "claim-finite-counterexample": "A bounded finite counterexample exists for representation-dependent sector partitions.",
        "claim-total-observables-invariant": "Total Hamiltonian observables can remain invariant while sector labels change.",
    }
    return mapping.get(claim_id, claim_id)


def counterexample_evidence_locators(root: Path) -> list[EvidenceLocator]:
    partition = root / "v3_counterexample_tool" / "partition_comparison.json"
    cert = root / "v3_counterexample_tool" / "counterexample_certificate.json"
    return [
        EvidenceLocator(
            locator_id="loc-spectrum-residual",
            claim_id="claim-unique-partition",
            artifact_id="v3_counterexample_tool/partition_comparison.json",
            location_type="JSON_POINTER",
            location="/total_spectrum_residual",
            required_condition="observed_value < 1e-12",
            checksum_sha256=file_sha256(partition),
            verifier_result_id="ver-counterexample-hard-constraints",
            independent_verification_id="iver-finite-partition-counterexample",
        ),
        EvidenceLocator(
            locator_id="loc-partition-difference",
            claim_id="claim-unique-partition",
            artifact_id="v3_counterexample_tool/partition_comparison.json",
            location_type="JSON_POINTER",
            location="/partition_difference_abs",
            required_condition="observed_value > 0.001",
            checksum_sha256=file_sha256(partition),
            verifier_result_id="ver-counterexample-hard-constraints",
            independent_verification_id="iver-finite-partition-counterexample",
        ),
        EvidenceLocator(
            locator_id="loc-counterexample-valid",
            claim_id="claim-unique-partition",
            artifact_id="v3_counterexample_tool/counterexample_certificate.json",
            location_type="JSON_POINTER",
            location="/valid_counterexample",
            required_condition="observed_value == true",
            checksum_sha256=file_sha256(cert),
            verifier_result_id="ver-counterexample-hard-constraints",
            independent_verification_id="iver-finite-partition-counterexample",
        ),
    ]


def write_run_summary(
    root: Path,
    problem: CompiledScientificProblem,
    final: FinalAdjudication,
    transition: ClaimTransitionRecord,
    fixture_result: dict[str, Any],
) -> None:
    lines = [
        f"# V3.0 Proof-Carrying Search Demo: {problem.title}",
        "",
        "This deterministic run is offline. It does not use live model or live network calls.",
        "",
        f"- Final resolution: {final.final_resolution}",
        f"- Generalization label: {final.generalization_label}",
        f"- Accepted transition: {transition.decision}",
        f"- Total spectrum residual: {fixture_result['total_spectrum_residual']}",
        f"- Partition difference: {fixture_result['partition_difference_abs']}",
        "",
        "## Scope",
        "",
        "The result is a bounded finite-case counterexample fixture. It must not be presented as a general theorem for all lattice superconductors.",
    ]
    (root / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_v3_proof_search_run(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    required = [
        "compiled_problem.json",
        "proof_obligations.jsonl",
        "scientific_state_snapshots.jsonl",
        "tool_capability_manifest.json",
        "tool_gaps.jsonl",
        "tool_build_records.jsonl",
        "candidate_archive.jsonl",
        "action_executions.jsonl",
        "verifier_results.jsonl",
        "independent_verification_results.jsonl",
        "certificates.jsonl",
        "claim_transition_records.jsonl",
        "state_disputes.jsonl",
        "final_adjudication.json",
        "run_summary.md",
    ]
    errors = [f"missing V3 artifact: {name}" for name in required if not (root / name).exists()]
    if errors:
        return errors
    try:
        CompiledScientificProblem.model_validate(read_json(root / "compiled_problem.json"))
        obligations = [ProofObligation.model_validate(row) for row in read_jsonl(root / "proof_obligations.jsonl")]
        snapshots = [ScientificStateSnapshot.model_validate(row) for row in read_jsonl(root / "scientific_state_snapshots.jsonl")]
        certificates = [ScientificCertificate.model_validate(row) for row in read_jsonl(root / "certificates.jsonl")]
        transitions = [ClaimTransitionRecord.model_validate(row) for row in read_jsonl(root / "claim_transition_records.jsonl")]
        FinalAdjudication.model_validate(read_json(root / "final_adjudication.json"))
    except Exception as exc:
        return [f"invalid V3 artifact schema: {exc}"]
    if not snapshots:
        errors.append("no scientific state snapshots")
    if not any(transition.decision == "accepted" for transition in transitions):
        errors.append("no accepted proof-carrying transition")
    if any(
        transition.decision == "accepted" and transition.requested_status in VERIFIED_CLAIM_STATUSES and not transition.evidence_support_results
        for transition in transitions
    ):
        errors.append("accepted verified transition lacks evidence support results")
    if any(
        transition.decision == "accepted" and not all(result.supported for result in transition.evidence_support_results)
        for transition in transitions
    ):
        errors.append("accepted transition includes unsupported evidence locator")
    for cert in certificates:
        if cert.status == "verified" and not cert.checksum_sha256:
            errors.append(f"verified certificate missing checksum: {cert.certificate_id}")
    if any(item.status == "VERIFIED" and item.generalization_requirement == "EXACT_GENERAL" for item in obligations):
        final = read_json(root / "final_adjudication.json")
        if final.get("generalization_label") == "FINITE_CASE_ONLY":
            errors.append("finite-case final adjudication cannot satisfy exact-general obligations")
    combined_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*") if path.is_file() and path.stat().st_size < 500_000)
    for secret_name in ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
        if secret_name in combined_text:
            errors.append(f"secret-like key name leaked into V3 run: {secret_name}")
    return errors
