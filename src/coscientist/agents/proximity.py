from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations

from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.ranking import HypothesisRanking
from coscientist.schemas.v15b import (
    HypothesisCluster,
    HypothesisGraphEdge,
    HypothesisGraphNode,
    HypothesisSimilarity,
    ProximityAnalysis,
    ProximityConfig,
    SearchSpaceCoverage,
)


class ProximityAgent:
    def analyze(
        self,
        *,
        project_id: str,
        run_id: str,
        round_label: str,
        round_number: int,
        hypotheses: list[Hypothesis],
        rankings: list[HypothesisRanking],
        verifications: list[EvidenceVerificationRecord],
        config: ProximityConfig,
        model_mode: str,
        literature_mode: str,
    ) -> ProximityAnalysis:
        ranking_by_id = {ranking.hypothesis_id: ranking for ranking in rankings}
        verification_by_hypothesis: dict[str, list[EvidenceVerificationRecord]] = defaultdict(list)
        for record in verifications:
            verification_by_hypothesis[record.hypothesis_id].append(record)
        similarities = [
            self._similarity(left, right)
            for left, right in combinations(hypotheses, 2)
        ]
        clusters = self._clusters(hypotheses, similarities, config.similarity_threshold)
        cluster_by_member = {member: cluster.cluster_id for cluster in clusters for member in cluster.member_ids}
        graph_nodes = [
            HypothesisGraphNode(
                hypothesis_id=hypothesis.id,
                title=hypothesis.title,
                round_label=round_label,
                generation_strategy=hypothesis.generation_strategy,
                rank=_rank_position(hypothesis.id, rankings),
                score=(ranking_by_id[hypothesis.id].weighted_total if hypothesis.id in ranking_by_id else None),
                cluster_id=cluster_by_member.get(hypothesis.id),
                parent_ids=hypothesis.parent_ids,
                status=hypothesis.status,
                evidence_count=len(hypothesis.evidence_links),
                verified_evidence_count=sum(1 for record in verification_by_hypothesis[hypothesis.id] if record.status in {"verified", "partially_verified", "conflicting"}),
            )
            for hypothesis in hypotheses
        ]
        graph_edges = self._edges(hypotheses, similarities, config, config.similarity_threshold)
        coverage = self._coverage(hypotheses, similarities, clusters, config.duplicate_threshold)
        return ProximityAnalysis(
            project_id=project_id,
            run_id=run_id,
            round_label=round_label,
            round_number=round_number,
            created_at=datetime.now(UTC),
            model_mode=model_mode,
            literature_mode=literature_mode,
            pairwise_similarities=similarities,
            clusters=clusters,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            search_space_coverage=coverage,
            method_metadata={
                "method": config.method,
                "similarity_threshold": config.similarity_threshold,
                "duplicate_threshold": config.duplicate_threshold,
            },
            validation_status="validated",
        )

    def _similarity(self, left: Hypothesis, right: Hypothesis) -> HypothesisSimilarity:
        claim = _jaccard(_tokens(left.core_claim), _tokens(right.core_claim))
        mechanism = _jaccard(_tokens(left.mechanism), _tokens(right.mechanism))
        assumptions = _list_similarity(left.assumptions, right.assumptions)
        predictions = _list_similarity(left.testable_predictions, right.testable_predictions)
        experiments = _list_similarity(left.proposed_experiments, right.proposed_experiments)
        evidence = _jaccard(_evidence_ids(left), _evidence_ids(right))
        lineage = bool(set(left.parent_ids).intersection(right.parent_ids) or left.id in right.parent_ids or right.id in left.parent_ids)
        overall = round((claim * 0.24) + (mechanism * 0.28) + (assumptions * 0.14) + (predictions * 0.14) + (experiments * 0.10) + (evidence * 0.06) + (0.04 if lineage else 0.0), 3)
        notes = []
        if overall >= 0.9:
            notes.append("Potential duplicate or superficial paraphrase.")
        elif mechanism >= 0.75 and predictions < 0.5:
            notes.append("Similar mechanism but different predicted behavior.")
        elif assumptions >= 0.7:
            notes.append("Shared assumption family detected.")
        return HypothesisSimilarity(
            hypothesis_a_id=left.id,
            hypothesis_b_id=right.id,
            claim_similarity=round(claim, 3),
            mechanism_similarity=round(mechanism, 3),
            assumption_similarity=round(assumptions, 3),
            prediction_similarity=round(predictions, 3),
            experiment_similarity=round(experiments, 3),
            evidence_overlap=round(evidence, 3),
            lineage_related=lineage,
            overall_similarity=overall,
            notes=notes,
        )

    def _clusters(self, hypotheses: list[Hypothesis], similarities: list[HypothesisSimilarity], threshold: float) -> list[HypothesisCluster]:
        ids = [hypothesis.id for hypothesis in hypotheses]
        parent = {hypothesis_id: hypothesis_id for hypothesis_id in ids}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

        for item in similarities:
            if item.overall_similarity >= threshold:
                union(item.hypothesis_a_id, item.hypothesis_b_id)
        grouped: dict[str, list[Hypothesis]] = defaultdict(list)
        by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        for hypothesis_id in ids:
            grouped[find(hypothesis_id)].append(by_id[hypothesis_id])
        clusters = []
        for index, members in enumerate(sorted(grouped.values(), key=lambda group: sorted(item.id for item in group)[0]), start=1):
            member_ids = sorted(item.id for item in members)
            cluster_id = "cluster-" + hashlib.sha1("|".join(member_ids).encode("utf-8")).hexdigest()[:10]
            representative = sorted(members, key=lambda item: (-len(item.evidence_links), item.id))[0]
            clusters.append(HypothesisCluster(
                cluster_id=cluster_id,
                member_ids=member_ids,
                representative_hypothesis_id=representative.id,
                shared_claims=_top_terms([item.core_claim for item in members], 5),
                shared_mechanisms=_top_terms([item.mechanism for item in members], 5),
                shared_assumptions=_top_terms([assumption for item in members for assumption in item.assumptions], 5),
                distinguishing_features=[item.generation_strategy for item in members],
                evidence_sources=sorted({paper_id for item in members for paper_id in _evidence_ids(item)}),
                cluster_confidence=round(min(1.0, 0.45 + 0.12 * len(members)), 3),
            ))
        return clusters

    def _edges(self, hypotheses: list[Hypothesis], similarities: list[HypothesisSimilarity], config: ProximityConfig, threshold: float) -> list[HypothesisGraphEdge]:
        edges = []
        for item in similarities:
            if item.overall_similarity >= threshold:
                edges.append(HypothesisGraphEdge(source_id=item.hypothesis_a_id, target_id=item.hypothesis_b_id, edge_type="similarity", weight=item.overall_similarity))
            if item.evidence_overlap > 0 and config.include_evidence_overlap:
                edges.append(HypothesisGraphEdge(source_id=item.hypothesis_a_id, target_id=item.hypothesis_b_id, edge_type="shared_evidence", weight=item.evidence_overlap))
            if item.assumption_similarity >= 0.5:
                edges.append(HypothesisGraphEdge(source_id=item.hypothesis_a_id, target_id=item.hypothesis_b_id, edge_type="shared_assumption", weight=item.assumption_similarity))
        if config.include_lineage_edges:
            ids = {item.id for item in hypotheses}
            for hypothesis in hypotheses:
                for parent_id in hypothesis.parent_ids:
                    if parent_id in ids:
                        edges.append(HypothesisGraphEdge(source_id=parent_id, target_id=hypothesis.id, edge_type="parent_child", weight=1.0))
        return edges

    def _coverage(self, hypotheses: list[Hypothesis], similarities: list[HypothesisSimilarity], clusters: list[HypothesisCluster], duplicate_threshold: float) -> SearchSpaceCoverage:
        n = len(hypotheses)
        values = sorted(item.overall_similarity for item in similarities)
        mean = round(sum(values) / len(values), 3) if values else 0.0
        median = values[len(values) // 2] if values else 0.0
        largest = max((len(cluster.member_ids) for cluster in clusters), default=0)
        largest_fraction = round(largest / n, 3) if n else 0.0
        isolated = [cluster.member_ids[0] for cluster in clusters if len(cluster.member_ids) == 1]
        duplicate_groups = [[item.hypothesis_a_id, item.hypothesis_b_id] for item in similarities if item.overall_similarity >= duplicate_threshold]
        diversity = round((len(clusters) / n) * (1 - (mean * 0.35)), 3) if n else 0.0
        collapse = "high" if largest_fraction >= 0.75 or mean >= 0.75 else "medium" if largest_fraction >= 0.5 or mean >= 0.55 else "low"
        evidence_counts = Counter(paper_id for hypothesis in hypotheses for paper_id in _evidence_ids(hypothesis))
        total_evidence = sum(evidence_counts.values())
        evidence_concentration = round(max(evidence_counts.values()) / total_evidence, 3) if total_evidence else 0.0
        return SearchSpaceCoverage(
            represented_mechanism_families=_top_terms([hypothesis.mechanism for hypothesis in hypotheses], 8),
            represented_assumption_families=_top_terms([assumption for hypothesis in hypotheses for assumption in hypothesis.assumptions], 8),
            underexplored_regions=_underexplored(hypotheses, clusters),
            overrepresented_regions=[cluster.cluster_id for cluster in clusters if n and len(cluster.member_ids) / n >= 0.5],
            isolated_hypotheses=isolated,
            duplicate_groups=duplicate_groups,
            diversity_score=max(0.0, min(1.0, diversity)),
            collapse_risk=collapse,
            unique_cluster_count=len(clusters),
            largest_cluster_fraction=largest_fraction,
            mean_pairwise_similarity=mean,
            median_pairwise_similarity=round(median, 3),
            isolated_hypothesis_count=len(isolated),
            effective_hypothesis_count=round(math.exp(-sum((len(c.member_ids) / n) * math.log(len(c.member_ids) / n) for c in clusters if n)), 3) if n else 0.0,
            generation_strategy_coverage=sorted({hypothesis.generation_strategy for hypothesis in hypotheses}),
            mechanism_family_coverage=len(_top_terms([hypothesis.mechanism for hypothesis in hypotheses], 20)),
            evidence_source_concentration=evidence_concentration,
            notes=["Lexical structured similarity is deterministic and may miss deep semantic equivalence."],
        )


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "or", "of", "to", "a", "in", "for", "with", "by", "is", "are", "can", "may", "this", "that", "from", "through"}
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()) if token not in stop and len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _list_similarity(left: list[str], right: list[str]) -> float:
    return _jaccard(set().union(*(_tokens(item) for item in left)) if left else set(), set().union(*(_tokens(item) for item in right)) if right else set())


def _evidence_ids(hypothesis: Hypothesis) -> set[str]:
    return {paper_id for link in hypothesis.evidence_links for paper_id in [*link.supporting_paper_ids, *link.contradicting_paper_ids]}


def _top_terms(texts: list[str], limit: int) -> list[str]:
    counts = Counter(token for text in texts for token in _tokens(text))
    return [term for term, _ in counts.most_common(limit)]


def _underexplored(hypotheses: list[Hypothesis], clusters: list[HypothesisCluster]) -> list[str]:
    strategies = {hypothesis.generation_strategy for hypothesis in hypotheses}
    missing = [item for item in ["mechanistic", "analogy", "contrarian", "minimal-explanation", "repair", "branch", "combine"] if item not in strategies]
    isolated = [cluster.cluster_id for cluster in clusters if len(cluster.member_ids) == 1]
    return [*(f"strategy:{item}" for item in missing[:4]), *(f"isolated:{item}" for item in isolated[:4])]


def _rank_position(hypothesis_id: str, rankings: list[HypothesisRanking]) -> int | None:
    ordered = sorted(rankings, key=lambda item: (item.weighted_total, item.pairwise_wins, -item.pairwise_losses, item.hypothesis_id), reverse=True)
    for index, ranking in enumerate(ordered, start=1):
        if ranking.hypothesis_id == hypothesis_id:
            return index
    return None
