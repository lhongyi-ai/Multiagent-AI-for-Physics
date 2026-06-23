from __future__ import annotations

import asyncio
from pathlib import Path

from coscientist.agents.proximity import ProximityAgent
from coscientist.feedback import RecommendationExecutor, RecommendationValidator, build_next_round_plan
from coscientist.pilot.artifacts import read_json, write_json
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.feedback_ab import compare_feedback_project_sync, validate_feedback_ab_artifacts
from coscientist.pilot.project_io import load_fixture_corpus, load_project_spec
from coscientist.providers.mock import MockProvider
from coscientist.schemas.hypothesis import HypothesisBatch
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.v15c import ControlledFeedbackConfig, MetaReviewRecommendation


PROJECT = "examples/materials_synthesis_grounded_pilot/project.yaml"


def _hypotheses(count: int = 4):
    provider = MockProvider()
    return asyncio.run(provider.generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "mechanistic", "count": count},
    )).hypotheses


def _context():
    project = load_project_spec(PROJECT)
    corpus = load_fixture_corpus(Path("research-projects/code_assistant_fixture/corpus.jsonl"))
    hypotheses = attach_fixture_evidence(_hypotheses(4), corpus, "round_0")
    verifications = verify_hypothesis_evidence(hypotheses, corpus)
    proximity = ProximityAgent().analyze(
        project_id=project.project_id,
        run_id="r",
        round_label="round_0",
        round_number=0,
        hypotheses=hypotheses,
        rankings=[],
        verifications=verifications,
        config=project.v15b.proximity,
        model_mode="mock",
        literature_mode="fixture",
    )
    return project, hypotheses, verifications, proximity


def _rec(
    rec_id: str,
    action_type: str,
    target_ids: list[str],
    *,
    target_type: str = "hypothesis",
    requested_change: str = "repair grounded claim",
    source_cluster_ids: list[str] | None = None,
    source_evidence_ids: list[str] | None = None,
    source_verification_ids: list[str] | None = None,
    source_metric_names: list[str] | None = None,
):
    return MetaReviewRecommendation.model_validate({
        "recommendation_id": rec_id,
        "round_number": 0,
        "action_type": action_type,
        "target_type": target_type,
        "target_ids": target_ids,
        "requested_change": requested_change,
        "rationale": "test",
        "source_hypothesis_ids": target_ids if target_type == "hypothesis" else [],
        "source_cluster_ids": source_cluster_ids or (target_ids if target_type == "cluster" else []),
        "source_evidence_ids": source_evidence_ids or [],
        "source_verification_ids": source_verification_ids or [],
        "source_metric_names": source_metric_names or ["grounding_coverage_score"],
        "expected_effect": "test effect",
        "confidence": 0.8,
        "constraints": [],
    })


def test_recommendation_validator_accepts_valid_and_rejects_invalid_references_conflicts_and_budgets() -> None:
    project, hypotheses, verifications, proximity = _context()
    h0, h1 = hypotheses[0].id, hypotheses[1].id
    cluster_id = proximity.clusters[0].cluster_id
    validator = RecommendationValidator(ControlledFeedbackConfig(enabled=True, max_repairs=1, max_generator_reallocation=1))
    unsupported = MetaReviewRecommendation.model_construct(
        recommendation_id="rec-unsupported",
        round_number=0,
        action_type="rewrite_world",
        target_type="project",
        target_ids=[],
        requested_change="unsupported",
        rationale="test",
        source_hypothesis_ids=[],
        source_cluster_ids=[],
        source_evidence_ids=[],
        source_verification_ids=[],
        source_metric_names=[],
        expected_effect="none",
        confidence=0.5,
        constraints=[],
        status="proposed",
        schema_version="v15c",
    )
    recommendations = [
        _rec("rec-valid", "repair_hypothesis", [h0]),
        _rec("rec-missing-hyp", "repair_hypothesis", ["missing"]),
        _rec("rec-missing-cluster", "explore_underrepresented_cluster", ["missing-cluster"], target_type="cluster"),
        _rec("rec-valid", "repair_hypothesis", [h1]),
        _rec("rec-conflict", "branch_hypothesis", [h0]),
        _rec("rec-budget", "repair_hypothesis", [h1]),
        _rec("rec-combine-invalid", "combine_hypotheses", [h0, h0]),
        _rec("rec-permission", "increase_generator_strategy", ["contrarian"], target_type="strategy", requested_change="turn on live_model and use API key"),
        _rec("rec-cluster", "explore_underrepresented_cluster", [cluster_id], target_type="cluster"),
        unsupported,
    ]
    decisions = validator.validate(
        recommendations=recommendations,
        hypotheses=hypotheses,
        proximity=proximity,
        verifications=verifications,
        project=project,
        mode="controlled_feedback",
        live_model_enabled=False,
        live_network_enabled=False,
    )
    by_id = {decision.recommendation_id: decision.decision for decision in decisions}
    valid_decisions = [decision.decision for decision in decisions if decision.recommendation_id == "rec-valid"]
    assert "accepted" in valid_decisions
    assert "rejected_duplicate" in valid_decisions
    assert by_id["rec-missing-hyp"] == "rejected_invalid_reference"
    assert by_id["rec-missing-cluster"] == "rejected_invalid_reference"
    assert by_id["rec-conflict"] == "rejected_conflict"
    assert by_id["rec-budget"] == "rejected_budget"
    assert by_id["rec-combine-invalid"] == "rejected_invalid_reference"
    assert by_id["rec-permission"] == "rejected_constraint_violation"
    assert by_id["rec-cluster"] == "accepted"
    assert by_id["rec-unsupported"] == "rejected_constraint_violation"


def test_advisory_mode_never_creates_executable_plan() -> None:
    project, hypotheses, verifications, proximity = _context()
    recommendation = _rec("rec-valid", "repair_hypothesis", [hypotheses[0].id])
    decisions = RecommendationValidator(ControlledFeedbackConfig(enabled=False)).validate(
        recommendations=[recommendation],
        hypotheses=hypotheses,
        proximity=proximity,
        verifications=verifications,
        project=project,
        mode="advisory",
        live_model_enabled=False,
        live_network_enabled=False,
    )
    plan = build_next_round_plan(
        project_id=project.project_id,
        run_id="r",
        branch="control",
        source_round=0,
        mode="advisory",
        recommendations=[recommendation],
        decisions=decisions,
        config=ControlledFeedbackConfig(enabled=False),
    )
    assert decisions[0].decision == "advisory_only"
    assert plan.accepted_recommendation_ids == []
    assert plan.repair_hypothesis_ids == []


def test_recommendation_executor_applies_only_accepted_actions_and_preserves_lineage() -> None:
    project, hypotheses, verifications, proximity = _context()
    h0, h1 = hypotheses[0].id, hypotheses[1].id
    config = ControlledFeedbackConfig(enabled=True, max_repairs=1, max_branches=1, max_combinations=1, max_holds=1, max_generator_reallocation=1, max_targeted_search_queries=1)
    recommendations = [
        _rec("rec-repair", "repair_hypothesis", [h0]),
        _rec("rec-branch", "branch_hypothesis", [h1]),
        _rec("rec-combine", "combine_hypotheses", [h0, h1]),
        _rec("rec-hold", "hold_hypothesis", [hypotheses[2].id]),
        _rec("rec-query", "add_targeted_search_query", [], target_type="project", requested_change="bounded offline query"),
        _rec("rec-strategy", "increase_generator_strategy", ["contrarian"], target_type="strategy"),
    ]
    decisions = RecommendationValidator(config).validate(
        recommendations=recommendations,
        hypotheses=hypotheses,
        proximity=proximity,
        verifications=verifications,
        project=project,
        mode="controlled_feedback",
        live_model_enabled=False,
        live_network_enabled=False,
    )
    plan = build_next_round_plan(
        project_id=project.project_id,
        run_id="r",
        branch="treatment",
        source_round=0,
        mode="controlled_feedback",
        recommendations=recommendations,
        decisions=decisions,
        config=config,
    )
    provider = MockProvider()
    goal = ResearchGoal(
        id=project.project_id,
        title=project.title,
        question=project.research_question,
        background=project.background,
        constraints=project.constraints,
        desired_attributes=project.evaluation_criteria,
        evaluation_criteria=project.evaluation_criteria,
        prohibited_methods=project.excluded_directions,
        max_rounds=project.maximum_evolution_rounds,
    )
    generated, records = asyncio.run(RecommendationExecutor(provider).execute(
        plan=plan,
        goal=goal,
        hypotheses=hypotheses,
        reviews=[],
        maximum_llm_calls=project.maximum_model_call_budget,
    ))
    executed_ids = {record.recommendation_id for record in records}
    assert executed_ids.issubset(set(plan.accepted_recommendation_ids))
    assert any(record.planned_action == "repair_hypothesis" and record.execution_status == "executed" for record in records)
    assert any(h0 in child.parent_ids for child in generated if child.status == "repaired")
    assert any(record.planned_action == "add_targeted_search_query" and record.execution_status == "recorded_noop" for record in records)
    assert len(plan.generator_allocation) <= 1


def test_feedback_ab_runner_is_deterministic_offline_and_validates(tmp_path: Path) -> None:
    run_dir = compare_feedback_project_sync(PROJECT, runs_dir=tmp_path, experiment_id="ab-one")
    run_dir_2 = compare_feedback_project_sync(PROJECT, runs_dir=tmp_path, experiment_id="ab-two")
    assert validate_feedback_ab_artifacts(run_dir) == []
    assert validate_feedback_ab_artifacts(run_dir_2) == []
    manifest = read_json(run_dir / "feedback_ab_manifest.json")
    comparison = read_json(run_dir / "feedback_ab_comparison.json")
    comparison_2 = read_json(run_dir_2 / "feedback_ab_comparison.json")
    control_execution = read_json(run_dir / "control_advisory" / "feedback_execution_round_1.json")
    treatment_plan = read_json(run_dir / "treatment_controlled_feedback" / "next_round_plan_round_1.json")
    assert manifest["model_mode"] == "mock"
    assert "no live network" in manifest["permission_guarantees"]
    assert control_execution == []
    assert treatment_plan["mode"] == "controlled_feedback"
    assert comparison["outcome_label"] in {"improved", "mixed", "no_material_change", "regressed", "insufficient_evidence"}
    assert comparison["deltas"] == comparison_2["deltas"]


def test_feedback_ab_validation_detects_rejected_recommendation_execution(tmp_path: Path) -> None:
    run_dir = compare_feedback_project_sync(PROJECT, runs_dir=tmp_path, experiment_id="ab-invalid")
    branch = run_dir / "treatment_controlled_feedback"
    executions = read_json(branch / "feedback_execution_round_1.json")
    executions[0]["recommendation_id"] = "not-accepted"
    write_json(branch / "feedback_execution_round_1.json", executions)
    errors = validate_feedback_ab_artifacts(run_dir)
    assert any("rejected recommendation executed" in error for error in errors)
