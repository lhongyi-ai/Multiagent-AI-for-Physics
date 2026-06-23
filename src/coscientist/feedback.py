from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Iterable

from coscientist.agents.evolution import EvolutionAgent
from coscientist.agents.generator import GeneratorAgent
from coscientist.agents.supervisor import BudgetExhausted, Supervisor
from coscientist.config import DEFAULT_STRATEGIES
from coscientist.providers.base import StructuredLLMProvider
from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.review import Review
from coscientist.schemas.run_state import RunState
from coscientist.schemas.v15b import GroundingDiagnostics, MetaReview, ProximityAnalysis
from coscientist.schemas.v15c import (
    ControlledFeedbackConfig,
    FeedbackExecutionRecord,
    MetaReviewRecommendation,
    NextRoundPlan,
    RecommendationDecision,
)


SUPPORTED_ACTIONS = {
    "increase_generator_strategy",
    "decrease_generator_strategy",
    "explore_underrepresented_cluster",
    "suppress_duplicate_cluster",
    "add_targeted_search_query",
    "repair_hypothesis",
    "branch_hypothesis",
    "combine_hypotheses",
    "hold_hypothesis",
    "request_more_evidence",
    "request_human_review",
    "preserve_current_strategy",
}
EXECUTABLE_ACTIONS = {
    "increase_generator_strategy",
    "repair_hypothesis",
    "branch_hypothesis",
    "combine_hypotheses",
    "hold_hypothesis",
    "add_targeted_search_query",
    "request_more_evidence",
    "request_human_review",
}
KNOWN_METRICS = {
    "diversity_score",
    "collapse_risk",
    "unique_cluster_count",
    "largest_cluster_fraction",
    "mean_pairwise_similarity",
    "median_pairwise_similarity",
    "effective_hypothesis_count",
    "supported_claim_count",
    "unsupported_claim_count",
    "grounding_coverage_score",
    "citation_hallucination_count",
    "metadata_only_misuse_count",
    "evidence_reuse_concentration",
}
INCOMPATIBLE_HYPOTHESIS_ACTIONS = {"repair_hypothesis", "branch_hypothesis", "hold_hypothesis"}
SECRET_OR_PERMISSION_TERMS = {
    "openai_api_key",
    "api key",
    "bearer ",
    "sk-",
    "live_model",
    "live model",
    "live_network",
    "live network",
    "max_model_call_budget",
    "maximum_model_call_budget",
}


def recommendations_from_meta_review(
    review: MetaReview,
    proximity: ProximityAnalysis,
    grounding: GroundingDiagnostics,
    *,
    max_recommendations: int,
) -> list[MetaReviewRecommendation]:
    recommendations: list[MetaReviewRecommendation] = []

    for index, strategy in enumerate(review.recommended_generation_strategies):
        recommendations.append(_recommendation(
            review,
            f"strategy-{index}-{strategy}",
            "increase_generator_strategy",
            "strategy",
            [strategy],
            f"Allocate one controlled-feedback generation slot to {strategy}.",
            "Meta-review identified this strategy as useful for the next round.",
            ["diversity_score", "collapse_risk"],
            "Increase conceptual diversity while preserving grounding constraints.",
        ))

    for index, query in enumerate(review.recommended_search_queries):
        recommendations.append(_recommendation(
            review,
            f"query-{index}",
            "add_targeted_search_query",
            "project",
            [],
            query,
            "Meta-review identified an evidence gap that can be represented as a bounded offline search request.",
            ["grounding_coverage_score", "unsupported_claim_count"],
            "Improve evidence coverage if later run in a permitted literature mode.",
        ))

    for cluster_id in proximity.search_space_coverage.underexplored_regions:
        if not cluster_id.startswith("cluster:"):
            continue
        target = cluster_id.split(":", 1)[1]
        recommendations.append(_recommendation(
            review,
            f"cluster-explore-{target}",
            "explore_underrepresented_cluster",
            "cluster",
            [target],
            f"Explore underrepresented cluster {target}.",
            "Coverage metrics marked this cluster as underrepresented.",
            ["unique_cluster_count", "effective_hypothesis_count"],
            "Reduce search-space collapse risk.",
        ))

    for cluster in proximity.clusters:
        if len(cluster.member_ids) < 2:
            continue
        recommendations.append(_recommendation(
            review,
            f"cluster-suppress-{cluster.cluster_id}",
            "suppress_duplicate_cluster",
            "cluster",
            [cluster.cluster_id],
            f"Reduce new generation from duplicate-heavy cluster {cluster.cluster_id}.",
            "Proximity analysis found multiple similar hypotheses in the same cluster.",
            ["largest_cluster_fraction", "mean_pairwise_similarity"],
            "Reduce duplicate hypotheses in the next round.",
            source_hypothesis_ids=cluster.member_ids,
            source_cluster_ids=[cluster.cluster_id],
        ))

    for hyp_id in review.recommended_hypothesis_repairs:
        recommendations.append(_recommendation(
            review,
            f"repair-{hyp_id}",
            "repair_hypothesis",
            "hypothesis",
            [hyp_id],
            f"Repair unsupported or weakly grounded claims in {hyp_id}.",
            "Evidence verification found unsupported or invalid references.",
            ["unsupported_claim_count", "grounding_coverage_score"],
            "Improve grounding and falsifiability for a promising hypothesis.",
            source_hypothesis_ids=[hyp_id],
        ))

    for hyp_id in review.recommended_hypothesis_branches:
        recommendations.append(_recommendation(
            review,
            f"branch-{hyp_id}",
            "branch_hypothesis",
            "hypothesis",
            [hyp_id],
            f"Branch {hyp_id} toward a more discriminative mechanism.",
            "Meta-review selected this hypothesis for targeted exploration.",
            ["diversity_score", "testability"],
            "Increase mechanistic alternatives while preserving parent lineage.",
            source_hypothesis_ids=[hyp_id],
        ))

    for pair in review.recommended_hypothesis_merges:
        if len(pair) >= 2:
            left, right = pair[:2]
            recommendations.append(_recommendation(
                review,
                f"combine-{left}-{right}",
                "combine_hypotheses",
                "hypothesis",
                [left, right],
                f"Combine complementary mechanisms from {left} and {right}.",
                "Meta-review identified overlapping or complementary hypotheses.",
                ["mean_pairwise_similarity", "explanatory_power"],
                "Consolidate duplicate structure while preserving both parents.",
                source_hypothesis_ids=[left, right],
            ))

    for hyp_id in review.recommended_hypotheses_to_hold:
        recommendations.append(_recommendation(
            review,
            f"hold-{hyp_id}",
            "hold_hypothesis",
            "hypothesis",
            [hyp_id],
            f"Hold {hyp_id} without mutation.",
            "Top-ranked hypothesis should remain available for comparison.",
            ["evidence_quality", "weighted_total"],
            "Avoid losing a strong baseline hypothesis.",
            source_hypothesis_ids=[hyp_id],
        ))

    for index, gap in enumerate(review.evidence_gaps):
        recommendations.append(_recommendation(
            review,
            f"evidence-{index}",
            "request_more_evidence",
            "evidence",
            [],
            gap,
            "Meta-review identified an evidence gap.",
            ["grounding_coverage_score"],
            "Flag evidence needs without fabricating sources.",
        ))

    return _dedupe_recommendations(recommendations)[:max_recommendations]


class RecommendationValidator:
    def __init__(self, config: ControlledFeedbackConfig) -> None:
        self.config = config

    def validate(
        self,
        *,
        recommendations: list[MetaReviewRecommendation],
        hypotheses: list[Hypothesis],
        proximity: ProximityAnalysis,
        verifications: list[EvidenceVerificationRecord],
        project: ResearchProjectSpec,
        mode: str,
        live_model_enabled: bool,
        live_network_enabled: bool,
    ) -> list[RecommendationDecision]:
        now = datetime.now(UTC)
        hypothesis_ids = {hypothesis.id for hypothesis in hypotheses}
        cluster_ids = {cluster.cluster_id for cluster in proximity.clusters}
        verification_ids = {record.claim_id for record in verifications}
        evidence_ids = {paper_id for record in verifications for paper_id in record.existing_paper_ids}
        seen: set[str] = set()
        used_hypothesis_actions: dict[str, str] = {}
        accepted_counts = Counter()
        decisions: list[RecommendationDecision] = []

        if mode == "advisory" or not self.config.enabled:
            return [
                RecommendationDecision(
                    recommendation_id=item.recommendation_id,
                    decision="advisory_only",
                    reason="Advisory mode records recommendations but does not permit workflow mutation.",
                    validator_checks=["advisory_mode"],
                    decided_at=now,
                )
                for item in recommendations
            ]

        for index, recommendation in enumerate(recommendations):
            checks: list[str] = []
            conflicts: list[str] = []
            normalized = _normalize_action(recommendation)
            budget_effect = _budget_effect(recommendation)
            decision = "accepted"
            reason = "Recommendation passed deterministic validation."

            if index >= self.config.max_recommendations:
                decision, reason = "rejected_budget", "Recommendation count exceeds controlled-feedback maximum."
            elif recommendation.recommendation_id in seen:
                decision, reason = "rejected_duplicate", "Duplicate recommendation_id."
            else:
                seen.add(recommendation.recommendation_id)
                checks.append("unique_recommendation_id")

            if decision == "accepted" and recommendation.action_type not in SUPPORTED_ACTIONS:
                decision, reason = "rejected_constraint_violation", "Unsupported action type."
            elif decision == "accepted":
                checks.append("supported_action_type")

            if decision == "accepted":
                missing_hypotheses = sorted((set(recommendation.target_ids) | set(recommendation.source_hypothesis_ids)) - hypothesis_ids)
                if recommendation.target_type == "hypothesis" and missing_hypotheses:
                    decision, reason = "rejected_invalid_reference", f"Missing hypothesis ids: {missing_hypotheses}"
                elif set(recommendation.source_cluster_ids) - cluster_ids:
                    decision, reason = "rejected_invalid_reference", "Recommendation references a missing cluster id."
                elif recommendation.target_type == "cluster" and set(recommendation.target_ids) - cluster_ids:
                    decision, reason = "rejected_invalid_reference", "Recommendation targets a missing cluster id."
                elif set(recommendation.source_verification_ids) - verification_ids:
                    decision, reason = "rejected_invalid_reference", "Recommendation references a missing verification id."
                elif set(recommendation.source_evidence_ids) - evidence_ids:
                    decision, reason = "rejected_invalid_reference", "Recommendation references a missing evidence id."
                elif set(recommendation.source_metric_names) - KNOWN_METRICS:
                    decision, reason = "rejected_invalid_reference", "Recommendation references an unknown metric."
                else:
                    checks.append("artifact_references_valid")

            if decision == "accepted" and recommendation.action_type == "combine_hypotheses":
                if len(recommendation.target_ids) != 2 or recommendation.target_ids[0] == recommendation.target_ids[1]:
                    decision, reason = "rejected_invalid_reference", "Combine action requires two distinct hypotheses."
                else:
                    checks.append("combine_pair_valid")

            if decision == "accepted" and recommendation.action_type in INCOMPATIBLE_HYPOTHESIS_ACTIONS:
                for hyp_id in recommendation.target_ids:
                    previous = used_hypothesis_actions.get(hyp_id)
                    if previous and previous != recommendation.action_type:
                        conflicts.append(f"{hyp_id} already assigned {previous}")
                        decision, reason = "rejected_conflict", "Hypothesis has incompatible controlled-feedback actions."
                        break
                if decision == "accepted":
                    for hyp_id in recommendation.target_ids:
                        used_hypothesis_actions[hyp_id] = recommendation.action_type
                    checks.append("hypothesis_action_conflicts_checked")

            if decision == "accepted":
                accepted_counts.update(budget_effect)
                if accepted_counts["generator_reallocation"] > self.config.max_generator_reallocation:
                    decision, reason = "rejected_budget", "Generator reallocation exceeds configured maximum."
                elif accepted_counts["targeted_search_queries"] > self.config.max_targeted_search_queries:
                    decision, reason = "rejected_budget", "Targeted search queries exceed configured maximum."
                elif accepted_counts["repairs"] > self.config.max_repairs:
                    decision, reason = "rejected_budget", "Repairs exceed configured maximum."
                elif accepted_counts["branches"] > self.config.max_branches:
                    decision, reason = "rejected_budget", "Branches exceed configured maximum."
                elif accepted_counts["combinations"] > self.config.max_combinations:
                    decision, reason = "rejected_budget", "Combinations exceed configured maximum."
                elif accepted_counts["holds"] > self.config.max_holds:
                    decision, reason = "rejected_budget", "Holds exceed configured maximum."
                if decision != "accepted":
                    accepted_counts.subtract(budget_effect)
                else:
                    checks.append("budget_limits_checked")

            if decision == "accepted":
                text = " ".join([recommendation.requested_change, *recommendation.constraints]).lower()
                if any(term in text for term in SECRET_OR_PERMISSION_TERMS):
                    decision, reason = "rejected_constraint_violation", "Feedback cannot change credentials, live permissions, or project budget ceilings."
                elif any(item.lower() in text for item in project.excluded_directions):
                    decision, reason = "rejected_constraint_violation", "Recommendation conflicts with excluded project direction."
                elif live_model_enabled or live_network_enabled:
                    checks.append("live_flags_observed_without_mutation")
                else:
                    checks.append("permission_gates_preserved")

            decisions.append(RecommendationDecision(
                recommendation_id=recommendation.recommendation_id,
                decision=decision,  # type: ignore[arg-type]
                reason=reason,
                validator_checks=checks,
                normalized_action=normalized if decision in {"accepted", "accepted_with_modification"} else {},
                conflicts=conflicts,
                budget_effect=budget_effect,
                decided_at=now,
            ))
        return decisions


def build_next_round_plan(
    *,
    project_id: str,
    run_id: str,
    branch: str,
    source_round: int,
    mode: str,
    recommendations: list[MetaReviewRecommendation],
    decisions: list[RecommendationDecision],
    config: ControlledFeedbackConfig,
) -> NextRoundPlan:
    by_id = {item.recommendation_id: item for item in recommendations}
    accepted = [decision for decision in decisions if decision.decision in {"accepted", "accepted_with_modification"}]
    generator_allocation: dict[str, int] = {}
    target_clusters: list[str] = []
    suppressed_clusters: list[str] = []
    targeted_search_queries: list[str] = []
    repair_ids: list[str] = []
    branch_ids: list[str] = []
    combine_pairs: list[list[str]] = []
    hold_ids: list[str] = []
    evidence_requests: list[str] = []
    human_review_requests: list[str] = []

    if mode == "controlled_feedback" and config.enabled:
        for decision in accepted:
            recommendation = by_id[decision.recommendation_id]
            action = recommendation.action_type
            if action == "increase_generator_strategy":
                strategy = recommendation.target_ids[0] if recommendation.target_ids else "contrarian"
                generator_allocation[strategy] = generator_allocation.get(strategy, 0) + 1
            elif action == "explore_underrepresented_cluster":
                target_clusters.extend(recommendation.target_ids)
            elif action == "suppress_duplicate_cluster":
                suppressed_clusters.extend(recommendation.target_ids)
            elif action == "add_targeted_search_query":
                targeted_search_queries.append(recommendation.requested_change)
            elif action == "repair_hypothesis":
                repair_ids.extend(recommendation.target_ids)
            elif action == "branch_hypothesis":
                branch_ids.extend(recommendation.target_ids)
            elif action == "combine_hypotheses":
                combine_pairs.append(recommendation.target_ids[:2])
            elif action == "hold_hypothesis":
                hold_ids.extend(recommendation.target_ids)
            elif action == "request_more_evidence":
                evidence_requests.append(recommendation.requested_change)
            elif action == "request_human_review":
                human_review_requests.append(recommendation.requested_change)

    generator_allocation = _bounded_counts(generator_allocation, config.max_generator_reallocation)
    targeted_search_queries = _stable_unique(targeted_search_queries)[: config.max_targeted_search_queries]
    repair_ids = _stable_unique(repair_ids)[: config.max_repairs]
    branch_ids = _stable_unique(branch_ids)[: config.max_branches]
    combine_pairs = _stable_unique_pairs(combine_pairs)[: config.max_combinations]
    hold_ids = _stable_unique(hold_ids)[: config.max_holds]
    plan_payload = {
        "accepted": [item.recommendation_id for item in accepted],
        "generator_allocation": generator_allocation,
        "target_clusters": _stable_unique(target_clusters),
        "suppressed_clusters": _stable_unique(suppressed_clusters),
        "targeted_search_queries": targeted_search_queries,
        "repair_hypothesis_ids": repair_ids,
        "branch_hypothesis_ids": branch_ids,
        "combine_pairs": combine_pairs,
        "hold_hypothesis_ids": hold_ids,
    }
    plan_hash = hashlib.sha256(json.dumps(plan_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return NextRoundPlan(
        project_id=project_id,
        run_id=run_id,
        branch=branch,
        source_round=source_round,
        target_round=source_round + 1,
        mode=mode,  # type: ignore[arg-type]
        accepted_recommendation_ids=plan_payload["accepted"],
        generator_allocation=generator_allocation,
        target_clusters=plan_payload["target_clusters"],
        suppressed_clusters=plan_payload["suppressed_clusters"],
        targeted_search_queries=targeted_search_queries,
        repair_hypothesis_ids=repair_ids,
        branch_hypothesis_ids=branch_ids,
        combine_pairs=combine_pairs,
        hold_hypothesis_ids=hold_ids,
        evidence_requests=_stable_unique(evidence_requests),
        human_review_requests=_stable_unique(human_review_requests),
        budget={
            "max_generator_reallocation": config.max_generator_reallocation,
            "max_targeted_search_queries": config.max_targeted_search_queries,
            "max_repairs": config.max_repairs,
            "max_branches": config.max_branches,
            "max_combinations": config.max_combinations,
            "max_holds": config.max_holds,
        },
        safeguards=[
            "no_live_model_or_network_flags_modified",
            "no_credentials_modified",
            "project_budget_ceilings_preserved",
            "only_accepted_recommendations_executable",
        ],
        plan_hash=plan_hash,
        validation_status="validated" if mode == "controlled_feedback" and config.enabled else "validated",
    )


class RecommendationExecutor:
    def __init__(self, provider: StructuredLLMProvider) -> None:
        self.provider = provider
        self.generator = GeneratorAgent(provider)
        self.evolution = EvolutionAgent(provider)

    async def execute(
        self,
        *,
        plan: NextRoundPlan,
        goal: ResearchGoal,
        hypotheses: list[Hypothesis],
        reviews: list[Review],
        maximum_llm_calls: int,
    ) -> tuple[list[Hypothesis], list[FeedbackExecutionRecord]]:
        if plan.mode != "controlled_feedback" or not plan.accepted_recommendation_ids:
            return list(hypotheses), []
        supervisor = Supervisor(RunState(
            run_id=plan.run_id,
            phase="feedback_execution",
            round_number=plan.target_round,
            maximum_llm_calls=maximum_llm_calls,
        ))
        by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        review_map = {review.hypothesis_id: review for review in reviews}
        next_round = [hypothesis for hypothesis in hypotheses if hypothesis.id in plan.hold_hypothesis_ids]
        records: list[FeedbackExecutionRecord] = []

        query_ids = [item for item in plan.accepted_recommendation_ids if "-query-" in item]
        for index, query in enumerate(plan.targeted_search_queries):
            records.append(_record(plan, "add_targeted_search_query", "literature_planner", [], "recorded_noop", recommendation_id=_at(query_ids, index), after={"query": query}, notes="Recorded bounded targeted query; no network call was made."))
        evidence_ids = [item for item in plan.accepted_recommendation_ids if "-evidence-" in item]
        for index, request in enumerate(plan.evidence_requests):
            records.append(_record(plan, "request_more_evidence", "grounding", [], "recorded_noop", recommendation_id=_at(evidence_ids, index), after={"request": request}, notes="Evidence request recorded without fabricating evidence."))
        human_ids = [item for item in plan.accepted_recommendation_ids if "-human-" in item]
        for index, request in enumerate(plan.human_review_requests):
            records.append(_record(plan, "request_human_review", "human_review", [], "recorded_noop", recommendation_id=_at(human_ids, index), after={"request": request}, notes="Human-review request recorded but not auto-approved."))
        for hyp_id in plan.hold_hypothesis_ids:
            records.append(_record(plan, "hold_hypothesis", "supervisor", [hyp_id], "recorded_noop", recommendation_id=_find_rec(plan, "hold", hyp_id), notes="Hypothesis held unchanged for comparison."))

        for hyp_id in plan.repair_hypothesis_ids:
            hypothesis = by_id.get(hyp_id)
            if not hypothesis:
                records.append(_record(plan, "repair_hypothesis", "evolution", [hyp_id], "failed", recommendation_id=_find_rec(plan, "repair", hyp_id), notes="Missing hypothesis at execution time."))
                continue
            try:
                supervisor.reserve(1, "feedback_repair")
                children = await self.evolution.repair(hypothesis, _reviews(review_map, hyp_id), plan.target_round)
                next_round.extend(children)
                records.append(_record(plan, "repair_hypothesis", "evolution", [hyp_id, *[child.id for child in children]], "executed", recommendation_id=_find_rec(plan, "repair", hyp_id), before={"parent": hyp_id}, after={"children": [child.id for child in children]}))
            except BudgetExhausted as exc:
                records.append(_record(plan, "repair_hypothesis", "evolution", [hyp_id], "failed", recommendation_id=_find_rec(plan, "repair", hyp_id), notes=str(exc)))

        for hyp_id in plan.branch_hypothesis_ids:
            hypothesis = by_id.get(hyp_id)
            if not hypothesis:
                records.append(_record(plan, "branch_hypothesis", "evolution", [hyp_id], "failed", recommendation_id=_find_rec(plan, "branch", hyp_id), notes="Missing hypothesis at execution time."))
                continue
            try:
                supervisor.reserve(1, "feedback_branch")
                children = await self.evolution.branch(hypothesis, _reviews(review_map, hyp_id), plan.target_round)
                next_round.extend(children)
                records.append(_record(plan, "branch_hypothesis", "evolution", [hyp_id, *[child.id for child in children]], "executed", recommendation_id=_find_rec(plan, "branch", hyp_id), before={"parent": hyp_id}, after={"children": [child.id for child in children]}))
            except BudgetExhausted as exc:
                records.append(_record(plan, "branch_hypothesis", "evolution", [hyp_id], "failed", recommendation_id=_find_rec(plan, "branch", hyp_id), notes=str(exc)))

        for pair in plan.combine_pairs:
            parents = [by_id[item] for item in pair if item in by_id]
            if len(parents) != 2:
                records.append(_record(plan, "combine_hypotheses", "evolution", pair, "failed", recommendation_id=_find_rec(plan, "combine", "-".join(pair)), notes="Missing combine parent at execution time."))
                continue
            try:
                supervisor.reserve(1, "feedback_combine")
                children = await self.evolution.combine(parents, plan.target_round, count=1)
                next_round.extend(children)
                records.append(_record(plan, "combine_hypotheses", "evolution", [*pair, *[child.id for child in children]], "executed", recommendation_id=_find_rec(plan, "combine", "-".join(pair)), before={"parents": pair}, after={"children": [child.id for child in children]}))
            except BudgetExhausted as exc:
                records.append(_record(plan, "combine_hypotheses", "evolution", pair, "failed", recommendation_id=_find_rec(plan, "combine", "-".join(pair)), notes=str(exc)))

        for strategy, count in plan.generator_allocation.items():
            if strategy not in DEFAULT_STRATEGIES or count <= 0:
                continue
            try:
                supervisor.reserve(1, f"feedback_generate_{strategy}")
                generated = await self.generator.generate(goal, strategy, count)
                next_round.extend(generated)
                records.append(_record(plan, "increase_generator_strategy", "generator", [item.id for item in generated], "executed", recommendation_id=_find_rec(plan, "strategy", strategy), before={"strategy": strategy}, after={"generated": [item.id for item in generated]}))
            except BudgetExhausted as exc:
                records.append(_record(plan, "increase_generator_strategy", "generator", [], "failed", recommendation_id=_find_rec(plan, "strategy", strategy), before={"strategy": strategy}, notes=str(exc)))

        return _dedupe_hypotheses(next_round), records


def _recommendation(
    review: MetaReview,
    suffix: str,
    action_type: str,
    target_type: str,
    target_ids: list[str],
    requested_change: str,
    rationale: str,
    source_metric_names: list[str],
    expected_effect: str,
    *,
    source_hypothesis_ids: list[str] | None = None,
    source_cluster_ids: list[str] | None = None,
) -> MetaReviewRecommendation:
    return MetaReviewRecommendation(
        recommendation_id=f"rec-r{review.round_number}-{_slug(suffix)}",
        round_number=review.round_number,
        action_type=action_type,  # type: ignore[arg-type]
        target_type=target_type,  # type: ignore[arg-type]
        target_ids=target_ids,
        requested_change=requested_change,
        rationale=rationale,
        source_hypothesis_ids=source_hypothesis_ids or target_ids if target_type == "hypothesis" else source_hypothesis_ids or [],
        source_cluster_ids=source_cluster_ids or target_ids if target_type == "cluster" else source_cluster_ids or [],
        source_metric_names=source_metric_names,
        expected_effect=expected_effect,
        confidence=review.confidence,
        constraints=["preserve live-model and live-network permission gates", "do not increase project budget ceilings"],
    )


def _normalize_action(recommendation: MetaReviewRecommendation) -> dict[str, str | list[str] | list[list[str]]]:
    return {
        "action_type": recommendation.action_type,
        "target_type": recommendation.target_type,
        "target_ids": recommendation.target_ids,
        "requested_change": recommendation.requested_change,
    }


def _budget_effect(recommendation: MetaReviewRecommendation) -> dict[str, int]:
    mapping = {
        "increase_generator_strategy": "generator_reallocation",
        "add_targeted_search_query": "targeted_search_queries",
        "repair_hypothesis": "repairs",
        "branch_hypothesis": "branches",
        "combine_hypotheses": "combinations",
        "hold_hypothesis": "holds",
    }
    key = mapping.get(recommendation.action_type)
    return {key: 1} if key else {}


def _record(
    plan: NextRoundPlan,
    action: str,
    agent: str,
    affected: list[str],
    status: str,
    *,
    recommendation_id: str | None = None,
    before: dict[str, str | int | float | list[str]] | None = None,
    after: dict[str, str | int | float | list[str]] | None = None,
    notes: str = "",
) -> FeedbackExecutionRecord:
    return FeedbackExecutionRecord(
        target_round=plan.target_round,
        recommendation_id=recommendation_id or (plan.accepted_recommendation_ids[0] if plan.accepted_recommendation_ids else "none"),
        planned_action=action,
        actual_action=action,
        affected_agent=agent,
        affected_hypothesis_ids=affected,
        before_state=before or {},
        after_state=after or {},
        execution_status=status,  # type: ignore[arg-type]
        notes=notes,
    )


def _dedupe_recommendations(recommendations: Iterable[MetaReviewRecommendation]) -> list[MetaReviewRecommendation]:
    seen: set[str] = set()
    deduped: list[MetaReviewRecommendation] = []
    for item in recommendations:
        if item.recommendation_id in seen:
            continue
        seen.add(item.recommendation_id)
        deduped.append(item)
    return deduped


def _dedupe_hypotheses(hypotheses: Iterable[Hypothesis]) -> list[Hypothesis]:
    seen: set[str] = set()
    deduped: list[Hypothesis] = []
    for item in hypotheses:
        if item.id in seen:
            continue
        seen.add(item.id)
        deduped.append(item)
    return deduped


def _stable_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _stable_unique_pairs(items: Iterable[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, str]] = set()
    result: list[list[str]] = []
    for pair in items:
        if len(pair) != 2 or pair[0] == pair[1]:
            continue
        key = tuple(pair)
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
    return result


def _bounded_counts(counts: dict[str, int], limit: int) -> dict[str, int]:
    remaining = limit
    bounded: dict[str, int] = {}
    for key in sorted(counts):
        if remaining <= 0:
            break
        value = min(counts[key], remaining)
        if value > 0:
            bounded[key] = value
            remaining -= value
    return bounded


def _reviews(review_map: dict[str, Review], hyp_id: str) -> list[Review]:
    return [review_map[hyp_id]] if hyp_id in review_map else []


def _find_rec(plan: NextRoundPlan, kind: str, token: str) -> str | None:
    return next((item for item in plan.accepted_recommendation_ids if kind in item and token in item), None)


def _at(items: list[str], index: int) -> str | None:
    return items[index] if index < len(items) else None


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")[:80]
