from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from coscientist.agents.evolution import EvolutionAgent
from coscientist.agents.generator import GeneratorAgent
from coscientist.agents.grounding import GroundingAgent
from coscientist.agents.meta_review import MetaReviewAgent
from coscientist.agents.proximity import ProximityAgent
from coscientist.agents.ranker import RankerAgent
from coscientist.agents.reviewer import ReviewerAgent
from coscientist.agents.supervisor import Supervisor
from coscientist.config import DEFAULT_STRATEGIES, WorkflowConfig
from coscientist.feedback import (
    RecommendationExecutor,
    RecommendationValidator,
    build_next_round_plan,
    recommendations_from_meta_review,
)
from coscientist.literature.scholarly import ScholarlyLiteratureOrchestrator
from coscientist.pilot.artifacts import read_json, write_json, write_jsonl
from coscientist.pilot.evaluation import compare_rounds, evaluate_round
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.project_io import load_project_spec
from coscientist.providers.mock import MockProvider
from coscientist.providers.usage import summarize_model_usage
from coscientist.schemas.evaluation import RoundComparison, RoundEvaluation
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.review import Review
from coscientist.schemas.run_state import RunState
from coscientist.schemas.scholarly import ProjectLiteratureConfig
from coscientist.schemas.v15b import GroundingDiagnostics, MetaReview, ProximityAnalysis
from coscientist.schemas.v15c import (
    ControlledFeedbackConfig,
    FeedbackABComparison,
    FeedbackABManifest,
    FeedbackBranchSummary,
    FeedbackExecutionRecord,
    MetaReviewRecommendation,
    NextRoundPlan,
    RecommendationDecision,
)


async def compare_feedback_project(
    project_path: str | Path,
    *,
    runs_dir: str | Path = "runs",
    experiment_id: str | None = None,
    force: bool = False,
) -> Path:
    project_file = Path(project_path)
    project = load_project_spec(project_file)
    if project.literature.mode == "live":
        raise ValueError("compare-feedback defaults to offline execution and does not permit live literature mode")
    experiment_id = experiment_id or f"{project.project_id}-feedback-ab"
    experiment_dir = Path(runs_dir) / experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not force:
        raise ValueError(f"feedback experiment artifacts are immutable; use a new experiment id or --force: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    corpus_result = await ScholarlyLiteratureOrchestrator(
        project=project,
        literature_config=project.literature,
        allow_live_network=False,
    ).acquire(
        fixture_path=_resolve_project_path(project_file, project.literature_fixture_path),
        existing_corpus_path=_resolve_project_path(project_file, project.literature.existing_corpus_path),
    )
    corpus = corpus_result.papers
    workflow_config = _workflow_config(project)
    goal = _goal(project)
    baseline = await _round_zero(project, goal, workflow_config)
    branches = []
    for designation, branch_name, controlled in [
        ("control", project.v15c.ab_experiment.control_label, False),
        ("treatment", project.v15c.ab_experiment.treatment_label, True),
    ]:
        branch_dir = experiment_dir / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch = await _run_branch(
            project=project,
            goal=goal,
            corpus=corpus,
            workflow_config=workflow_config,
            baseline=baseline,
            branch_dir=branch_dir,
            branch=branch_name,
            designation=designation,
            controlled=controlled,
        )
        branches.append(branch)
    control, treatment = branches
    manifest = FeedbackABManifest(
        experiment_id=experiment_id,
        project_id=project.project_id,
        project_title=project.title,
        created_at=datetime.now(UTC),
        model_mode="mock",
        literature_mode=project.literature.mode,
        random_seed=project.v15c.ab_experiment.random_seed,
        control_run_id=control["run_id"],
        treatment_run_id=treatment["run_id"],
        control_branch=project.v15c.ab_experiment.control_label,
        treatment_branch=project.v15c.ab_experiment.treatment_label,
        shared_research_question=project.research_question,
        shared_initial_hypothesis_ids=[hypothesis.id for hypothesis in baseline["hypotheses"]],
        shared_budget={
            "maximum_model_call_budget": project.maximum_model_call_budget,
            "maximum_evolution_rounds": project.maximum_evolution_rounds,
            "v15c_target_rounds": project.v15c.ab_experiment.max_evolution_rounds,
        },
        permission_guarantees=[
            "mock model provider only",
            "no live network",
            "no live model",
            "feedback cannot mutate API credentials or permission gates",
        ],
        configuration_hash=_hash_json({
            "project": project.model_dump(mode="json"),
            "workflow_config": workflow_config.model_dump(mode="json"),
        }),
        validation_status="validated",
    )
    comparison = _compare_branches(
        experiment_id=experiment_id,
        project=project,
        control=control["summary"],
        treatment=treatment["summary"],
    )
    write_json(experiment_dir / "feedback_ab_manifest.json", manifest)
    write_json(experiment_dir / "feedback_ab_comparison.json", comparison)
    summary = _summary_markdown(project, manifest, comparison, control["summary"], treatment["summary"], perf_counter() - started)
    (experiment_dir / "feedback_ab_summary.md").write_text(summary, encoding="utf-8")
    (experiment_dir / "report.md").write_text(summary, encoding="utf-8")
    (experiment_dir / "human_review.md").write_text(_human_review_markdown(project, comparison), encoding="utf-8")
    return experiment_dir


def compare_feedback_project_sync(*args, **kwargs) -> Path:
    return asyncio.run(compare_feedback_project(*args, **kwargs))


def validate_feedback_ab_artifacts(experiment_dir: str | Path) -> list[str]:
    path = Path(experiment_dir)
    errors: list[str] = []
    for name in ["feedback_ab_manifest.json", "feedback_ab_comparison.json", "feedback_ab_summary.md", "report.md", "human_review.md"]:
        if not (path / name).exists():
            errors.append(f"missing feedback A/B artifact: {name}")
    if errors:
        return errors
    manifest = read_json(path / "feedback_ab_manifest.json")
    comparison = read_json(path / "feedback_ab_comparison.json")
    for branch_key in ["control_branch", "treatment_branch"]:
        branch = manifest[branch_key]
        branch_dir = path / branch
        if not branch_dir.exists():
            errors.append(f"missing branch directory: {branch}")
            continue
        errors.extend(_validate_branch(branch_dir, branch == manifest["control_branch"]))
    if comparison.get("experiment_id") != manifest.get("experiment_id"):
        errors.append("comparison experiment_id does not match manifest")
    if comparison.get("project_id") != manifest.get("project_id"):
        errors.append("comparison project_id does not match manifest")
    if manifest.get("model_mode") != "mock":
        errors.append("feedback A/B manifest must remain mock-mode by default")
    for artifact in path.rglob("*.json"):
        text = artifact.read_text(encoding="utf-8").lower()
        if "openai_api_key" in text or "sk-" in text or "bearer " in text:
            errors.append(f"secret-like content appears in {artifact.relative_to(path)}")
    return errors


async def _round_zero(project: ResearchProjectSpec, goal: ResearchGoal, config: WorkflowConfig) -> dict[str, Any]:
    provider = MockProvider()
    generator = GeneratorAgent(provider)
    reviewer = ReviewerAgent(provider)
    ranker = RankerAgent(provider)
    supervisor = Supervisor(RunState(run_id=f"{project.project_id}-round0", phase="created", round_number=0, maximum_llm_calls=config.max_llm_calls))
    strategies = DEFAULT_STRATEGIES[: config.generators]
    supervisor.reserve(len(strategies), "ab_initial_generation")
    generated_groups = []
    for strategy in strategies:
        generated_groups.append(await generator.generate(goal, strategy, config.hypotheses_per_generator))
    hypotheses = [hypothesis for group in generated_groups for hypothesis in group]
    supervisor.reserve(1, "ab_review_round_0")
    reviews = await reviewer.review(hypotheses)
    supervisor.reserve(2, "ab_rank_round_0")
    rankings = await ranker.rank(hypotheses, reviews, config, round_number=0, seed=project.v15c.ab_experiment.random_seed)
    selected = _select(hypotheses, rankings, config.top_k_after_review)
    return {
        "provider": provider,
        "hypotheses": hypotheses,
        "reviews": reviews,
        "rankings": rankings,
        "selected": selected,
        "llm_call_count": supervisor.state.llm_call_count,
    }


async def _run_branch(
    *,
    project: ResearchProjectSpec,
    goal: ResearchGoal,
    corpus,
    workflow_config: WorkflowConfig,
    baseline: dict[str, Any],
    branch_dir: Path,
    branch: str,
    designation: str,
    controlled: bool,
) -> dict[str, Any]:
    run_id = f"{project.project_id}-{branch}"
    hypotheses_round_0 = attach_fixture_evidence(list(baseline["hypotheses"]), corpus, "round_0")
    selected = attach_fixture_evidence(list(baseline["selected"]), corpus, "round_0_selected")
    verifications_round_0 = verify_hypothesis_evidence(hypotheses_round_0, corpus)
    eval_round_0 = evaluate_round(hypotheses_round_0, verifications_round_0, "initial")
    source_comparison = RoundComparison(project_id=project.project_id, evaluator_self_preference_note="V1.5C source-round comparison placeholder.", generated_at=datetime.now(UTC))
    proximity = ProximityAgent().analyze(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_0",
        round_number=0,
        hypotheses=hypotheses_round_0,
        rankings=list(baseline["rankings"]),
        verifications=verifications_round_0,
        config=project.v15b.proximity,
        model_mode="mock",
        literature_mode=project.literature.mode,
    )
    grounding_agent = GroundingAgent()
    packet = grounding_agent.build_packet(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_0",
        corpus=corpus,
        verifications=verifications_round_0,
        config=project.v15b.grounding,
    )
    grounding = grounding_agent.diagnostics(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_0",
        hypotheses=hypotheses_round_0,
        verifications=verifications_round_0,
        packet=packet,
        config=project.v15b.grounding,
    )
    meta_config = project.v15b.meta_review.model_copy(update={
        "feedback_mode": "controlled_feedback" if controlled else "advisory",
        "feed_into_next_round": controlled,
    })
    meta_review = MetaReviewAgent().review(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_0",
        round_number=0,
        hypotheses=hypotheses_round_0,
        rankings=list(baseline["rankings"]),
        verifications=verifications_round_0,
        evaluations=[eval_round_0],
        comparison=source_comparison,
        proximity=proximity,
        grounding=grounding,
        config=meta_config,
        model_mode="mock",
        literature_mode=project.literature.mode,
    )
    feedback_config = project.v15c.controlled_feedback.model_copy(update={"enabled": controlled})
    recommendations = recommendations_from_meta_review(
        meta_review,
        proximity,
        grounding,
        max_recommendations=feedback_config.max_recommendations,
    )
    decisions = RecommendationValidator(feedback_config).validate(
        recommendations=recommendations,
        hypotheses=hypotheses_round_0,
        proximity=proximity,
        verifications=verifications_round_0,
        project=project,
        mode="controlled_feedback" if controlled else "advisory",
        live_model_enabled=False,
        live_network_enabled=False,
    )
    plan = build_next_round_plan(
        project_id=project.project_id,
        run_id=run_id,
        branch=branch,
        source_round=0,
        mode="controlled_feedback" if controlled else "advisory",
        recommendations=recommendations,
        decisions=decisions,
        config=feedback_config,
    )
    provider = MockProvider()
    if controlled:
        round_1, executions = await RecommendationExecutor(provider).execute(
            plan=plan,
            goal=goal,
            hypotheses=hypotheses_round_0,
            reviews=list(baseline["reviews"]),
            maximum_llm_calls=project.maximum_model_call_budget,
        )
    else:
        evolution = EvolutionAgent(provider)
        supervisor = Supervisor(RunState(run_id=run_id, phase="control_evolution", round_number=1, maximum_llm_calls=project.maximum_model_call_budget))
        calls = len(selected) * min(workflow_config.children_per_selected_hypothesis, 2)
        if selected and workflow_config.children_per_selected_hypothesis > 1:
            calls += 1
        supervisor.reserve(calls, "control_evolution_round_1")
        round_1 = await evolution.evolve(selected, list(baseline["reviews"]), round_number=1, children_per_selected=workflow_config.children_per_selected_hypothesis)
        executions = []
    round_1 = attach_fixture_evidence(round_1, corpus, "round_1")
    verifications_round_1 = verify_hypothesis_evidence(round_1, corpus)
    eval_round_1 = evaluate_round(round_1, verifications_round_1, "final")
    rounds = {"initial": hypotheses_round_0, "final": round_1}
    verifications = {"initial": verifications_round_0, "final": verifications_round_1}
    evaluations = [eval_round_0, eval_round_1]
    comparison = compare_rounds(project, evaluations, rounds, verifications)
    final_proximity = ProximityAgent().analyze(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_1",
        round_number=1,
        hypotheses=round_1,
        rankings=[],
        verifications=verifications_round_1,
        config=project.v15b.proximity,
        model_mode="mock",
        literature_mode=project.literature.mode,
    )
    final_packet = grounding_agent.build_packet(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_1",
        corpus=corpus,
        verifications=verifications_round_1,
        config=project.v15b.grounding,
    )
    final_grounding = grounding_agent.diagnostics(
        project_id=project.project_id,
        run_id=run_id,
        round_label="round_1",
        hypotheses=round_1,
        verifications=verifications_round_1,
        packet=final_packet,
        config=project.v15b.grounding,
    )
    model_usage = summarize_model_usage("mock", "mock", list(provider.call_records))
    metrics = _metrics(round_1, final_proximity, final_grounding, eval_round_1, comparison, recommendations, decisions, executions, model_usage)
    summary = FeedbackBranchSummary(
        project_id=project.project_id,
        run_id=run_id,
        branch=branch,
        designation=designation,  # type: ignore[arg-type]
        source_round=0,
        target_round=1,
        model_mode="mock",
        literature_mode=project.literature.mode,
        random_seed=project.v15c.ab_experiment.random_seed,
        recommendation_count=len(recommendations),
        accepted_recommendation_count=sum(1 for decision in decisions if decision.decision in {"accepted", "accepted_with_modification"}),
        executed_action_count=sum(1 for record in executions if record.execution_status == "executed"),
        metrics=metrics,
        validation_status="validated",
        created_at=datetime.now(UTC),
    )
    _write_branch(
        branch_dir,
        project,
        hypotheses_round_0,
        round_1,
        list(baseline["reviews"]),
        list(baseline["rankings"]),
        evaluations,
        comparison,
        proximity,
        grounding,
        meta_review,
        recommendations,
        decisions,
        plan,
        executions,
        summary,
    )
    return {"run_id": run_id, "summary": summary}


def _write_branch(
    branch_dir: Path,
    project: ResearchProjectSpec,
    hypotheses_round_0: list[Hypothesis],
    round_1: list[Hypothesis],
    reviews: list[Review],
    rankings,
    evaluations: list[RoundEvaluation],
    comparison: RoundComparison,
    proximity: ProximityAnalysis,
    grounding: GroundingDiagnostics,
    meta_review: MetaReview,
    recommendations: list[MetaReviewRecommendation],
    decisions: list[RecommendationDecision],
    plan: NextRoundPlan,
    executions: list[FeedbackExecutionRecord],
    summary: FeedbackBranchSummary,
) -> None:
    write_json(branch_dir / "project_snapshot.json", project)
    write_json(branch_dir / "hypotheses_round_0.json", hypotheses_round_0)
    write_json(branch_dir / "hypotheses_round_1.json", round_1)
    write_json(branch_dir / "reviews_round_0.json", reviews)
    write_json(branch_dir / "rankings_round_0.json", rankings)
    write_json(branch_dir / "evaluation_by_round.json", evaluations)
    write_json(branch_dir / "round_comparison.json", comparison)
    write_json(branch_dir / "proximity_round_0.json", proximity)
    write_json(branch_dir / "grounding_diagnostics_round_0.json", grounding)
    write_json(branch_dir / "meta_review_round_0.json", meta_review)
    write_json(branch_dir / "meta_review_recommendations_round_0.json", recommendations)
    write_json(branch_dir / "recommendation_decisions_round_0.json", decisions)
    write_json(branch_dir / "next_round_plan_round_1.json", plan)
    write_json(branch_dir / "feedback_execution_round_1.json", executions)
    write_json(branch_dir / "feedback_branch_summary.json", summary)
    write_jsonl(branch_dir / "feedback_execution_round_1.jsonl", executions)
    (branch_dir / "report.md").write_text(_branch_report(summary, plan, decisions, executions), encoding="utf-8")


def _metrics(
    hypotheses: list[Hypothesis],
    proximity: ProximityAnalysis,
    grounding: GroundingDiagnostics,
    evaluation: RoundEvaluation,
    comparison: RoundComparison,
    recommendations: list[MetaReviewRecommendation],
    decisions: list[RecommendationDecision],
    executions: list[FeedbackExecutionRecord],
    model_usage,
) -> dict[str, float | int | str]:
    accepted = [decision for decision in decisions if decision.decision in {"accepted", "accepted_with_modification"}]
    return {
        "total_hypothesis_count": len(hypotheses),
        "unique_cluster_count": proximity.search_space_coverage.unique_cluster_count,
        "effective_hypothesis_count": round(proximity.search_space_coverage.effective_hypothesis_count, 3),
        "largest_cluster_fraction": proximity.search_space_coverage.largest_cluster_fraction,
        "mean_pairwise_similarity": proximity.search_space_coverage.mean_pairwise_similarity,
        "median_pairwise_similarity": proximity.search_space_coverage.median_pairwise_similarity,
        "duplicate_rate": round(len(proximity.search_space_coverage.duplicate_groups) / max(1, len(hypotheses)), 3),
        "isolated_hypothesis_count": proximity.search_space_coverage.isolated_hypothesis_count,
        "generation_strategy_coverage": len(proximity.search_space_coverage.generation_strategy_coverage),
        "mechanism_family_coverage": proximity.search_space_coverage.mechanism_family_coverage,
        "underexplored_region_coverage": len(proximity.search_space_coverage.underexplored_regions),
        "collapse_risk": proximity.search_space_coverage.collapse_risk,
        "diversity_score": proximity.search_space_coverage.diversity_score,
        "supported_claim_count": grounding.supported_claim_count,
        "unsupported_claim_count": grounding.unsupported_claim_count,
        "supported_claim_fraction": round(grounding.supported_claim_count / max(1, grounding.supported_claim_count + grounding.unsupported_claim_count), 3),
        "verified_evidence_coverage": grounding.grounding_coverage_score,
        "missing_evidence_reference_count": grounding.claims_citing_missing_evidence_ids,
        "invalid_evidence_reference_count": grounding.claims_citing_missing_evidence_ids,
        "citation_hallucination_count": grounding.citation_hallucination_count,
        "metadata_only_misuse_count": grounding.metadata_only_misuse_count,
        "contradiction_evidence_coverage": grounding.final_hypotheses_with_contradicting_evidence_fraction,
        "evidence_source_concentration": grounding.evidence_reuse_concentration,
        "exact_source_location_fraction": grounding.grounding_coverage_score,
        "quality_mean": round(sum(evaluation.mean_scores.values()) / max(1, len(evaluation.mean_scores)), 3),
        "testability": evaluation.mean_scores.get("testability", 0.0),
        "falsifiability": evaluation.mean_scores.get("falsifiability", 0.0),
        "feasibility": evaluation.mean_scores.get("feasibility", 0.0),
        "explanatory_power": evaluation.mean_scores.get("explanatory_power", 0.0),
        "evidence_quality": evaluation.mean_scores.get("evidence_grounding", 0.0),
        "llm_call_count": model_usage.call_count,
        "successful_call_count": model_usage.successful_call_count,
        "failed_call_count": model_usage.failed_call_count,
        "structured_output_failures": model_usage.structured_output_failures,
        "repair_attempts": model_usage.repair_attempts,
        "total_tokens": model_usage.total_tokens or 0,
        "literature_query_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "recommendation_count": len(recommendations),
        "acceptance_rate": round(len(accepted) / max(1, len(recommendations)), 3),
        "execution_success_rate": round(sum(1 for record in executions if record.execution_status == "executed") / max(1, len(executions)), 3),
        "citation_coverage_final": comparison.citation_coverage.get("final", 0.0),
    }


def _compare_branches(
    *,
    experiment_id: str,
    project: ResearchProjectSpec,
    control: FeedbackBranchSummary,
    treatment: FeedbackBranchSummary,
) -> FeedbackABComparison:
    control_metrics = control.metrics
    treatment_metrics = treatment.metrics
    keys = sorted(set(control_metrics) | set(treatment_metrics))
    metrics = {
        "control": control_metrics,
        "treatment": treatment_metrics,
    }
    deltas: dict[str, float | int] = {}
    for key in keys:
        left = control_metrics.get(key)
        right = treatment_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = round(float(right) - float(left), 3)
    outcome, rationale = _outcome(deltas)
    return FeedbackABComparison(
        experiment_id=experiment_id,
        project_id=project.project_id,
        control_run_id=control.run_id,
        treatment_run_id=treatment.run_id,
        metrics=metrics,
        deltas=deltas,
        outcome_label=outcome,
        outcome_rationale=rationale,
        limitations=[
            "This is a deterministic offline process comparison, not evidence of scientific truth.",
            "Mock-model outputs are suitable for workflow validation, not domain claims.",
            "A live-model comparison may be less reproducible and must be explicitly authorized later.",
        ],
        generated_at=datetime.now(UTC),
        validation_status="validated",
    )


def _outcome(deltas: dict[str, float | int]) -> tuple[str, str]:
    if not deltas:
        return "insufficient_evidence", "No numeric comparison metrics were available."
    diversity = float(deltas.get("diversity_score", 0.0))
    grounding = float(deltas.get("verified_evidence_coverage", 0.0))
    quality = float(deltas.get("quality_mean", 0.0))
    cost = float(deltas.get("llm_call_count", 0.0))
    positives = sum(value > 0.05 for value in [diversity, grounding, quality])
    negatives = sum(value < -0.05 for value in [diversity, grounding, quality]) + (1 if cost > 2 else 0)
    if positives >= 2 and negatives == 0:
        return "improved", "Treatment improved multiple proxy metrics without a material cost or quality regression."
    if negatives >= 2:
        return "regressed", "Treatment regressed on multiple proxy metrics or cost dimensions."
    if positives and negatives:
        return "mixed", "Treatment improved some proxy metrics while regressing on others."
    if positives:
        return "mixed", "Treatment shows limited positive movement but not enough evidence for a broad improvement claim."
    return "no_material_change", "Treatment did not materially change the bounded comparison metrics."


def _summary_markdown(
    project: ResearchProjectSpec,
    manifest: FeedbackABManifest,
    comparison: FeedbackABComparison,
    control: FeedbackBranchSummary,
    treatment: FeedbackBranchSummary,
    runtime_seconds: float,
) -> str:
    return "\n".join([
        f"# V1.5C Feedback A/B Summary: {project.title}",
        "",
        "> Offline deterministic comparison. This does not prove scientific correctness.",
        "",
        f"- Experiment ID: `{manifest.experiment_id}`",
        f"- Control: `{manifest.control_branch}` advisory only",
        f"- Treatment: `{manifest.treatment_branch}` controlled feedback",
        f"- Model mode: {manifest.model_mode}",
        f"- Literature mode: {manifest.literature_mode}",
        f"- Runtime seconds: {runtime_seconds:.3f}",
        "",
        "## Outcome",
        "",
        f"- Label: `{comparison.outcome_label}`",
        f"- Rationale: {comparison.outcome_rationale}",
        "",
        "## Key Metrics",
        "",
        f"- Diversity score: control {control.metrics.get('diversity_score')}, treatment {treatment.metrics.get('diversity_score')}, delta {comparison.deltas.get('diversity_score', 0)}",
        f"- Grounding coverage: control {control.metrics.get('verified_evidence_coverage')}, treatment {treatment.metrics.get('verified_evidence_coverage')}, delta {comparison.deltas.get('verified_evidence_coverage', 0)}",
        f"- Quality mean: control {control.metrics.get('quality_mean')}, treatment {treatment.metrics.get('quality_mean')}, delta {comparison.deltas.get('quality_mean', 0)}",
        f"- LLM calls: control {control.metrics.get('llm_call_count')}, treatment {treatment.metrics.get('llm_call_count')}, delta {comparison.deltas.get('llm_call_count', 0)}",
        "",
        "## Feedback Execution",
        "",
        f"- Control accepted recommendations: {control.accepted_recommendation_count}",
        f"- Control executed actions: {control.executed_action_count}",
        f"- Treatment accepted recommendations: {treatment.accepted_recommendation_count}",
        f"- Treatment executed actions: {treatment.executed_action_count}",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in comparison.limitations],
        "",
        "Machine-readable artifacts: `feedback_ab_manifest.json`, `feedback_ab_comparison.json`, branch `meta_review_recommendations_round_0.json`, `recommendation_decisions_round_0.json`, `next_round_plan_round_1.json`, and `feedback_execution_round_1.json`.",
        "",
    ])


def _human_review_markdown(project: ResearchProjectSpec, comparison: FeedbackABComparison) -> str:
    questions = [
        "Were the MetaReview recommendations scientifically sensible?",
        "Were any valid recommendations incorrectly rejected?",
        "Were any unsafe recommendations incorrectly accepted?",
        "Did controlled feedback increase real conceptual diversity or only wording diversity?",
        "Did grounding improve?",
        "Did useful hypotheses disappear because a cluster was suppressed?",
        "Were underexplored directions scientifically plausible?",
        "Were suggested search queries relevant?",
        "Were repair, branch, and combine actions appropriate?",
        "Was any apparent improvement worth the added cost?",
        "Should controlled feedback remain disabled, be enabled selectively, or be rejected?",
    ]
    return "\n".join([
        f"# V1.5C Human Review: {project.title}",
        "",
        f"Outcome label: `{comparison.outcome_label}`",
        "",
        "## Review Questions",
        "",
        *[f"- {item}" for item in questions],
        "",
    ])


def _branch_report(
    summary: FeedbackBranchSummary,
    plan: NextRoundPlan,
    decisions: list[RecommendationDecision],
    executions: list[FeedbackExecutionRecord],
) -> str:
    accepted = [item for item in decisions if item.decision in {"accepted", "accepted_with_modification"}]
    rejected = [item for item in decisions if item.decision not in {"accepted", "accepted_with_modification", "advisory_only"}]
    return "\n".join([
        f"# V1.5C Branch Report: {summary.branch}",
        "",
        f"- Designation: {summary.designation}",
        f"- Recommendations: {summary.recommendation_count}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        f"- Plan hash: `{plan.plan_hash}`",
        f"- Executed actions: {summary.executed_action_count}",
        "",
        "## Next-Round Plan",
        "",
        f"- Mode: {plan.mode}",
        f"- Generator allocation: {plan.generator_allocation}",
        f"- Targeted queries: {plan.targeted_search_queries}",
        f"- Repairs: {plan.repair_hypothesis_ids}",
        f"- Branches: {plan.branch_hypothesis_ids}",
        f"- Combine pairs: {plan.combine_pairs}",
        f"- Holds: {plan.hold_hypothesis_ids}",
        "",
        "## Actual Feedback Actions",
        "",
        *[f"- {record.planned_action}: {record.execution_status} ({', '.join(record.affected_hypothesis_ids) or 'no hypothesis'})" for record in executions],
        "",
    ])


def _validate_branch(branch_dir: Path, is_control: bool) -> list[str]:
    errors: list[str] = []
    required = [
        "hypotheses_round_0.json",
        "hypotheses_round_1.json",
        "meta_review_recommendations_round_0.json",
        "recommendation_decisions_round_0.json",
        "next_round_plan_round_1.json",
        "feedback_execution_round_1.json",
        "feedback_branch_summary.json",
        "report.md",
    ]
    for name in required:
        if not (branch_dir / name).exists():
            errors.append(f"{branch_dir.name}: missing {name}")
    if errors:
        return errors
    hypotheses = {item["id"] for item in read_json(branch_dir / "hypotheses_round_0.json")}
    next_round_hypotheses = read_json(branch_dir / "hypotheses_round_1.json")
    next_round_ids = {item["id"] for item in next_round_hypotheses}
    recommendations = read_json(branch_dir / "meta_review_recommendations_round_0.json")
    decisions = read_json(branch_dir / "recommendation_decisions_round_0.json")
    plan = read_json(branch_dir / "next_round_plan_round_1.json")
    executions = read_json(branch_dir / "feedback_execution_round_1.json")
    rec_ids = [item["recommendation_id"] for item in recommendations]
    if len(rec_ids) != len(set(rec_ids)):
        errors.append(f"{branch_dir.name}: duplicate recommendation_id")
    decision_ids = {item["recommendation_id"] for item in decisions}
    for rec_id in rec_ids:
        if rec_id not in decision_ids:
            errors.append(f"{branch_dir.name}: recommendation lacks decision: {rec_id}")
    accepted = {item["recommendation_id"] for item in decisions if item["decision"] in {"accepted", "accepted_with_modification"}}
    if set(plan.get("accepted_recommendation_ids", [])) - accepted:
        errors.append(f"{branch_dir.name}: plan includes non-accepted recommendation")
    executed_ids = {item["recommendation_id"] for item in executions}
    if executed_ids - accepted:
        errors.append(f"{branch_dir.name}: rejected recommendation executed")
    if is_control and executions:
        errors.append(f"{branch_dir.name}: control branch contains executed feedback")
    for pair in plan.get("combine_pairs", []):
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(hypotheses):
            errors.append(f"{branch_dir.name}: invalid combine pair")
    for field in ["repair_hypothesis_ids", "branch_hypothesis_ids", "hold_hypothesis_ids"]:
        missing = set(plan.get(field, [])) - hypotheses
        if missing:
            errors.append(f"{branch_dir.name}: {field} references missing hypotheses: {sorted(missing)}")
    for hypothesis in next_round_hypotheses:
        missing_parents = set(hypothesis.get("parent_ids", [])) - hypotheses - next_round_ids
        if missing_parents:
            errors.append(f"{branch_dir.name}: hypothesis {hypothesis.get('id')} has missing parent ids: {sorted(missing_parents)}")
    if sum(plan.get("generator_allocation", {}).values()) > plan.get("budget", {}).get("max_generator_reallocation", 0):
        errors.append(f"{branch_dir.name}: generator allocation exceeds budget")
    return errors


def _workflow_config(project: ResearchProjectSpec) -> WorkflowConfig:
    return WorkflowConfig(
        max_llm_calls=project.maximum_model_call_budget,
        evolution_rounds=1,
        generators=4,
        hypotheses_per_generator=3,
        top_k_after_review=6,
        children_per_selected_hypothesis=2,
        final_top_k=3,
    )


def _goal(project: ResearchProjectSpec) -> ResearchGoal:
    return ResearchGoal(
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


def _select(hypotheses, rankings, count: int) -> list[Hypothesis]:
    by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
    ordered = sorted(
        rankings,
        key=lambda item: (item.weighted_total, item.pairwise_wins, -item.pairwise_losses, item.hypothesis_id),
        reverse=True,
    )
    return [by_id[ranking.hypothesis_id] for ranking in ordered[:count] if ranking.hypothesis_id in by_id]


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _resolve_project_path(project_file: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return project_file.parent / path
