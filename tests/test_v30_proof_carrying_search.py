from __future__ import annotations

from pathlib import Path

from coscientist.core.proof_search import (
    ActionExecutionV30,
    AgentScientificOutput,
    ArtifactClaimSupportValidator,
    CapabilityAwareToolRegistry,
    ClaimState,
    EvidenceLocator,
    IndependentVerificationRecord,
    ProofCarryingClaimUpdate,
    ScientificCertificate,
    ScientificStateAdjudicator,
    ScientificStateMachine,
    ToolCapabilityDescriptor,
    adjudicate_agent_output,
    build_missing_tool_actions,
    compile_finite_counterexample_problem,
    counterexample_evidence_locators,
    detect_tool_gaps,
    file_sha256,
    make_scientific_snapshot,
    run_bounded_counterexample_fixture,
    run_v3_proof_search_demo,
    tool_contract_for_gap,
    validate_v3_proof_search_run,
    validated_tool_build_for_contract,
)
from coscientist.pilot.artifacts import write_json


def test_artifact_claim_support_requires_exact_location_and_condition(tmp_path: Path) -> None:
    artifact = tmp_path / "continuity.json"
    write_json(artifact, {"tests": {"continuity_equation": {"residual_norm": 2.3e-14}}})
    validator = ArtifactClaimSupportValidator(tmp_path)

    valid = EvidenceLocator(
        locator_id="loc-valid",
        claim_id="claim-continuity",
        artifact_id="continuity.json",
        location_type="JSON_POINTER",
        location="/tests/continuity_equation/residual_norm",
        required_condition="observed_value < 1e-12",
        checksum_sha256=file_sha256(artifact),
    )
    assert validator.validate_locator(valid).supported

    missing_pointer = valid.model_copy(update={"locator_id": "loc-missing", "location": "/tests/gauge/residual_norm"})
    result = validator.validate_locator(missing_pointer)
    assert not result.supported
    assert "location does not resolve" in result.errors[0]

    failed_condition = valid.model_copy(update={"locator_id": "loc-fail", "required_condition": "observed_value < 1e-16"})
    assert not validator.validate_locator(failed_condition).supported

    bad_checksum = valid.model_copy(update={"locator_id": "loc-checksum", "checksum_sha256": "deadbeef"})
    assert not validator.validate_locator(bad_checksum).supported


def test_phase1_artifact_cannot_support_unrelated_continuity_claim(tmp_path: Path) -> None:
    artifact = tmp_path / "phase1_minimal_bcs_tool"
    artifact.mkdir()
    write_json(artifact / "minimal_bcs_verifier_results.jsonl", {"verifier_id": "gap_equation_verifier", "verdict": "pass"})
    locator = EvidenceLocator(
        locator_id="loc-wrong-artifact",
        claim_id="claim-continuity",
        artifact_id="phase1_minimal_bcs_tool/minimal_bcs_verifier_results.jsonl",
        location_type="JSON_POINTER",
        location="/tests/continuity_equation/residual_norm",
        required_condition="observed_value < 1e-12",
    )
    result = ArtifactClaimSupportValidator(tmp_path).validate_locator(locator)
    assert not result.supported
    assert result.artifact_exists
    assert not result.location_resolved


def test_agent_cannot_directly_transition_claim_to_supported(tmp_path: Path) -> None:
    artifacts, _ = run_bounded_counterexample_fixture(tmp_path)
    claims = {"claim-unique-partition": ClaimState(claim_id="claim-unique-partition", statement="unique partition", status="PROPOSED")}
    problem = compile_finite_counterexample_problem()
    snapshot = make_scientific_snapshot(
        run_id="test-run",
        problem_id=problem.problem_id,
        claims=claims,
        obligations=problem.obligations,
        artifacts=artifacts,
    )
    machine = ScientificStateMachine(
        run_dir=tmp_path,
        claims=claims,
        action_executions={},
        certificates={},
        independent_verifications={},
        current_snapshot=snapshot,
    )
    update = ProofCarryingClaimUpdate(
        update_id="u-agent",
        claim_id="claim-unique-partition",
        requested_status="SUPPORTED",
        actor_type="agent",
        actor_id="pi",
        based_on_snapshot_version=snapshot.snapshot_version,
    )
    transition = machine.apply_update(update)
    assert transition.decision == "rejected"
    assert claims["claim-unique-partition"].status == "PROPOSED"
    assert any("agents cannot directly transition" in reason for reason in transition.rejection_reasons)


def test_executor_accepts_proof_carrying_transition_with_verified_evidence(tmp_path: Path) -> None:
    artifacts, fixture_result = run_bounded_counterexample_fixture(tmp_path)
    locators = counterexample_evidence_locators(tmp_path)
    cert = ScientificCertificate(
        certificate_id="cert-finite",
        certificate_type="CounterexampleCertificate",
        claim_id="claim-unique-partition",
        obligation_id="OBL-PARTITION",
        artifact_ids=["v3_counterexample_tool/partition_comparison.json"],
        evidence_locators=locators,
        acceptance_conditions=[locator.required_condition for locator in locators],
        observed_values=fixture_result,
        status="verified",
        generalization_label="FINITE_CASE_ONLY",
    ).with_checksum()
    independent = IndependentVerificationRecord(
        independent_verification_id="iver-finite",
        certificate_id=cert.certificate_id,
        implementation_path="test path",
        status="pass",
    )
    action = ActionExecutionV30(
        action_execution_id="act-finite",
        action_type="RUN_REGISTERED_TOOL",
        tool_id="counterexample_fixture",
        status="succeeded",
        generated_artifact_ids=artifacts,
    )
    claims = {"claim-unique-partition": ClaimState(claim_id="claim-unique-partition", statement="unique partition", status="PROPOSED")}
    problem = compile_finite_counterexample_problem()
    snapshot = make_scientific_snapshot(
        run_id="test-run",
        problem_id=problem.problem_id,
        claims=claims,
        obligations=problem.obligations,
        artifacts=artifacts,
    )
    machine = ScientificStateMachine(
        run_dir=tmp_path,
        claims=claims,
        action_executions={action.action_execution_id: action},
        certificates={cert.certificate_id: cert},
        independent_verifications={independent.independent_verification_id: independent},
        current_snapshot=snapshot,
    )
    update = ProofCarryingClaimUpdate(
        update_id="u-executor",
        claim_id="claim-unique-partition",
        requested_status="CONTRADICTED",
        actor_type="scientific_action_executor",
        actor_id="executor",
        based_on_snapshot_version=snapshot.snapshot_version,
        action_execution_id=action.action_execution_id,
        certificate_ids=[cert.certificate_id],
        evidence_locators=locators,
        independent_verification_ids=[independent.independent_verification_id],
    )
    transition = machine.apply_update(update)
    assert transition.decision == "accepted"
    assert claims["claim-unique-partition"].status == "CONTRADICTED"
    assert all(result.supported for result in transition.evidence_support_results)


def test_stale_snapshot_rejects_transition(tmp_path: Path) -> None:
    claims = {"claim-x": ClaimState(claim_id="claim-x", statement="x", status="PROPOSED")}
    problem = compile_finite_counterexample_problem()
    snapshot = make_scientific_snapshot(run_id="run", problem_id=problem.problem_id, claims=claims, obligations=problem.obligations, artifacts=[])
    machine = ScientificStateMachine(
        run_dir=tmp_path,
        claims=claims,
        action_executions={},
        certificates={},
        independent_verifications={},
        current_snapshot=snapshot,
    )
    transition = machine.apply_update(
        ProofCarryingClaimUpdate(
            update_id="u-stale",
            claim_id="claim-x",
            requested_status="PLANNED",
            actor_type="state_machine",
            actor_id="scheduler",
            based_on_snapshot_version="snap-old",
        )
    )
    assert transition.decision == "rejected"
    assert "stale snapshot version" in transition.rejection_reasons


def test_tool_gap_creates_build_action_and_unvalidated_tool_is_rejected() -> None:
    problem = compile_finite_counterexample_problem()
    registry = CapabilityAwareToolRegistry([])
    gaps = detect_tool_gaps(problem.obligations, registry)
    assert gaps
    actions = build_missing_tool_actions(gaps)
    assert actions[0].action_type == "BUILD_MISSING_TOOL"

    unvalidated = ToolCapabilityDescriptor(
        tool_id="generated_unsafe",
        version="v0",
        capabilities=[gaps[0].missing_capability],
        status="candidate",
    )
    try:
        registry.register(unvalidated)
    except ValueError as exc:
        assert "unvalidated tools" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unvalidated generated tool should not register")

    contract = tool_contract_for_gap(gaps[0])
    build, descriptor, cert = validated_tool_build_for_contract(contract)
    assert build.status == "registered"
    assert cert.status == "verified"
    registry.register(descriptor)
    assert registry.has_capability(gaps[0].missing_capability)


def test_agent_output_and_dispute_regression_cases(tmp_path: Path) -> None:
    claims = {"claim-x": ClaimState(claim_id="claim-x", statement="x", status="PROPOSED")}
    problem = compile_finite_counterexample_problem()
    snapshot = make_scientific_snapshot(run_id="run", problem_id=problem.problem_id, claims=claims, obligations=problem.obligations, artifacts=[])
    pi = AgentScientificOutput(
        output_id="pi-round2",
        agent_id="pi",
        role="principal_investigator",
        based_on_snapshot_version=snapshot.snapshot_version,
        claimed_completed_actions=["continuity_verifier_completed"],
        asserted_claim_statuses={"claim-x": "SUPPORTED"},
    )
    critic = AgentScientificOutput(
        output_id="critic-round2",
        agent_id="critic",
        role="critic",
        based_on_snapshot_version=snapshot.snapshot_version,
    )
    adjudication = adjudicate_agent_output(pi, current_snapshot=snapshot, completed_action_ids=set())
    assert adjudication.status == "unsupported_completion_claim"
    assert "UNSUPPORTED_COMPLETION_CLAIM" in adjudication.labels
    dispute = ScientificStateAdjudicator.adjudicate_action_completion_dispute(
        dispute_id="dispute-1",
        action_id="continuity_verifier_completed",
        outputs=[pi, critic],
        completed_action_ids=set(),
    )
    assert dispute.authoritative_result == "NOT_COMPLETED"
    assert dispute.followup_action == "CREATE_MISSING_EXECUTION_OR_TOOL_ACTION"


def test_v3_demo_runs_and_validates(tmp_path: Path) -> None:
    run_dir = run_v3_proof_search_demo(tmp_path, run_id="v3-demo", force=True)
    assert not validate_v3_proof_search_run(run_dir)
    final = (run_dir / "final_adjudication.json").read_text(encoding="utf-8")
    assert "DISPROVED_BY_COUNTEREXAMPLE" in final
    transitions = (run_dir / "claim_transition_records.jsonl").read_text(encoding="utf-8")
    assert "update-agent-unsupported-completion" in transitions
    assert "accepted" in transitions
    assert "rejected" in transitions


def test_live_agent_meeting_includes_v3_proof_context(tmp_path: Path) -> None:
    from coscientist.live_agents import meeting_tool_context, meeting_transcript, run_live_agent_meeting, validate_meeting_artifacts

    question = (
        "Determine whether the decomposition of superconducting condensation energy into bare kinetic-energy, "
        "phonon-interaction-energy, and correlated-hopping-energy contributions is uniquely defined, gauge invariant, "
        "and physically observable."
    )
    run_live_agent_meeting(tmp_path, question, live_model=False, max_rounds=1, force=True, phase2_data_path="data/phase2_lsco.csv")
    assert not validate_meeting_artifacts(tmp_path)
    context = meeting_tool_context(tmp_path)
    assert context["v3_proof_search"]["validation"] == "valid"
    assert context["v3_proof_search"]["final_adjudication"]["generalization_label"] == "FINITE_CASE_ONLY"
    transcript = meeting_transcript(tmp_path)
    assert "Tool Call: v3_proof_carrying_scientific_search" in transcript


def test_frontend_v3_adapter_reads_backend_artifacts(tmp_path: Path) -> None:
    from coscientist.frontend import OfflineDiscoveryFrontend

    facade = OfflineDiscoveryFrontend(runs_dir=tmp_path)
    run_dir = Path(facade.run_v3_proof_search_demo(run_id="frontend-v3", force=True))
    assert not facade.validate_v3_proof_search(run_dir)
    assert facade.v3_snapshot_rows(run_dir)
    assert facade.v3_obligation_rows(run_dir)
    assert facade.v3_claim_transition_rows(run_dir)
    assert facade.v3_certificate_rows(run_dir)
    assert facade.v3_final_adjudication(run_dir)["generalization_label"] == "FINITE_CASE_ONLY"
