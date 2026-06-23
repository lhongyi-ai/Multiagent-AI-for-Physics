from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from coscientist.schemas.evaluation import RoundComparison, RoundEvaluation
from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.ranking import HypothesisRanking
from coscientist.schemas.v15b import (
    GroundingDiagnostics,
    MetaReview,
    MetaReviewConfig,
    MetaReviewDecision,
    ProximityAnalysis,
)


class MetaReviewAgent:
    def review(
        self,
        *,
        project_id: str,
        run_id: str,
        round_label: str,
        round_number: int,
        hypotheses: list[Hypothesis],
        rankings: list[HypothesisRanking],
        verifications: list[EvidenceVerificationRecord],
        evaluations: list[RoundEvaluation],
        comparison: RoundComparison,
        proximity: ProximityAnalysis,
        grounding: GroundingDiagnostics,
        config: MetaReviewConfig,
        model_mode: str,
        literature_mode: str,
    ) -> MetaReview:
        top_ids = [ranking.hypothesis_id for ranking in sorted(rankings, key=lambda item: item.weighted_total, reverse=True)[:3]]
        unsupported = [record for record in verifications if record.status in {"unsupported", "invalid_reference"}]
        conflicting = [record for record in verifications if record.status == "conflicting"]
        duplicate_clusters = [cluster.cluster_id for cluster in proximity.clusters if len(cluster.member_ids) > 1]
        recurring_weaknesses = _weaknesses(hypotheses, unsupported, grounding)
        stopping = "request_more_evidence" if unsupported or grounding.grounding_coverage_score < 0.5 else "continue_targeted_revision"
        if proximity.search_space_coverage.collapse_risk == "high":
            stopping = "continue_exploration"
        search_queries = _search_queries(proximity, grounding, config.max_recommendations)
        repair_ids = [record.hypothesis_id for record in unsupported[: config.max_recommendations]]
        branch_ids = proximity.search_space_coverage.isolated_hypotheses[: config.max_recommendations]
        hold_ids = top_ids[:2]
        return MetaReview(
            project_id=project_id,
            run_id=run_id,
            round_label=round_label,
            round_number=round_number,
            created_at=datetime.now(UTC),
            model_mode=model_mode,
            literature_mode=literature_mode,
            feedback_mode=config.feedback_mode,
            executive_summary=(
                f"Final round contains {len(hypotheses)} hypotheses across "
                f"{proximity.search_space_coverage.unique_cluster_count} clusters with "
                f"{proximity.search_space_coverage.collapse_risk} collapse risk."
            ),
            strongest_hypotheses=top_ids,
            recurring_strengths=[
                "Hypotheses include explicit predictions and falsification criteria.",
                "Artifact validation preserves evidence and lineage for audit.",
            ],
            recurring_weaknesses=recurring_weaknesses,
            unsupported_claim_patterns=[f"{len(unsupported)} unsupported or invalid verification records."],
            evidence_gaps=[
                "Fixture corpus is intentionally small; independent literature support remains incomplete.",
                *[f"underexplored:{item}" for item in proximity.search_space_coverage.underexplored_regions[:3]],
            ],
            novelty_risks=["Novelty cannot be established from fixture evidence alone."],
            feasibility_risks=["Feasibility claims require external validation beyond automated ranking."],
            contradiction_patterns=[f"{len(conflicting)} claims have both supporting and contradicting fixture links."],
            duplicate_or_collapse_patterns=[f"Duplicate/collapsed clusters: {', '.join(duplicate_clusters) if duplicate_clusters else 'none'}"],
            underexplored_directions=proximity.search_space_coverage.underexplored_regions[: config.max_recommendations],
            ranking_concerns=["Ranking scores are decision aids and can reflect evaluator self-preference."],
            reviewer_concerns=recurring_weaknesses[:3],
            literature_coverage_concerns=["No claim should treat the fixture corpus as exhaustive."],
            next_round_strategy=_next_strategy(proximity, grounding),
            recommended_generation_strategies=_recommended_strategies(proximity)[: config.max_recommendations],
            recommended_search_queries=search_queries,
            recommended_hypothesis_merges=[cluster.member_ids for cluster in proximity.clusters if len(cluster.member_ids) > 1][: config.max_recommendations],
            recommended_hypothesis_branches=branch_ids,
            recommended_hypothesis_repairs=sorted(set(repair_ids))[: config.max_recommendations],
            recommended_hypotheses_to_hold=hold_ids,
            recommended_falsification_tests=_falsification_tests(hypotheses)[: config.max_recommendations],
            stopping_assessment=stopping,  # type: ignore[arg-type]
            stopping_reasons=[
                f"grounding_coverage_score={grounding.grounding_coverage_score}",
                f"collapse_risk={proximity.search_space_coverage.collapse_risk}",
            ],
            confidence=0.74,
            limitations=[
                "Meta-review is deterministic and artifact-based in offline mode.",
                "It does not establish scientific correctness or novelty.",
            ],
            referenced_cluster_ids=[cluster.cluster_id for cluster in proximity.clusters],
            referenced_evidence_ids=[],
            referenced_verification_ids=[record.claim_id for record in unsupported[: config.max_recommendations]],
            validation_status="validated",
        )

    def decide(self, *, project_id: str, run_id: str, round_label: str, review: MetaReview, config: MetaReviewConfig) -> MetaReviewDecision:
        if config.feedback_mode == "advisory" or not config.feed_into_next_round:
            return MetaReviewDecision(
                project_id=project_id,
                run_id=run_id,
                round_label=round_label,
                feedback_mode=config.feedback_mode,
                feed_into_next_round=False,
                rejected_recommendations=["controlled feedback disabled by project configuration"],
                decision_rationale="Advisory mode persists recommendations but does not alter the next round.",
            )
        return MetaReviewDecision(
            project_id=project_id,
            run_id=run_id,
            round_label=round_label,
            feedback_mode=config.feedback_mode,
            feed_into_next_round=True,
            accepted_generation_strategy_adjustments=review.recommended_generation_strategies,
            accepted_search_queries=review.recommended_search_queries,
            selected_repairs=review.recommended_hypothesis_repairs,
            selected_branches=review.recommended_hypothesis_branches,
            selected_combinations=review.recommended_hypothesis_merges,
            held_hypotheses=review.recommended_hypotheses_to_hold,
            decision_rationale="Controlled feedback accepted only structured validated recommendation fields.",
        )


def _weaknesses(hypotheses: list[Hypothesis], unsupported: list[EvidenceVerificationRecord], grounding: GroundingDiagnostics) -> list[str]:
    weaknesses = []
    if unsupported:
        weaknesses.append("Unsupported or invalid evidence references recur across hypotheses.")
    if grounding.evidence_reuse_concentration >= 0.5:
        weaknesses.append("Evidence support is concentrated in a small number of corpus records.")
    common_assumptions = Counter(assumption for hypothesis in hypotheses for assumption in hypothesis.assumptions)
    if common_assumptions:
        assumption, count = common_assumptions.most_common(1)[0]
        if count > 1:
            weaknesses.append(f"Repeated assumption pattern: {assumption}")
    return weaknesses or ["No dominant recurring weakness detected by deterministic meta-review."]


def _recommended_strategies(proximity: ProximityAnalysis) -> list[str]:
    if proximity.search_space_coverage.collapse_risk in {"medium", "high"}:
        return ["contrarian", "analogy"]
    if proximity.search_space_coverage.isolated_hypothesis_count:
        return ["branch", "repair"]
    return ["mechanistic", "minimal-explanation"]


def _search_queries(proximity: ProximityAnalysis, grounding: GroundingDiagnostics, limit: int) -> list[str]:
    queries = [item.replace("strategy:", "").replace("isolated:", "") for item in proximity.search_space_coverage.underexplored_regions]
    if grounding.grounding_coverage_score < 0.5:
        queries.append("independent evidence for top hypothesis mechanisms")
    return [query for query in queries if query][:limit]


def _next_strategy(proximity: ProximityAnalysis, grounding: GroundingDiagnostics) -> str:
    if grounding.grounding_coverage_score < 0.5:
        return "Prioritize evidence acquisition and repair unsupported claims before expanding conclusions."
    if proximity.search_space_coverage.collapse_risk != "low":
        return "Broaden generation toward underrepresented clusters before additional ranking."
    return "Continue targeted revision of top-ranked hypotheses while preserving cluster diversity."


def _falsification_tests(hypotheses: list[Hypothesis]) -> list[str]:
    tests = []
    for hypothesis in hypotheses:
        for item in hypothesis.falsification_criteria:
            tests.append(f"{hypothesis.id}: {item}")
    return tests
