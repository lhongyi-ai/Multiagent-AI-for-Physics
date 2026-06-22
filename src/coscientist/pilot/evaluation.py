from __future__ import annotations

from datetime import UTC, datetime
from itertools import combinations

from coscientist.schemas.evaluation import EvaluationRecord, RoundComparison, RoundEvaluation, RubricScore
from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.project import ResearchProjectSpec


DIMENSIONS = [
    "relevance",
    "plausibility",
    "evidence_grounding",
    "novelty",
    "testability",
    "falsifiability",
    "explanatory_power",
    "clarity",
    "diversity",
    "feasibility",
    "uncertainty_calibration",
]


def evaluate_round(
    hypotheses: list[Hypothesis],
    verifications: list[EvidenceVerificationRecord],
    round_label: str,
    sequence_start: int = 0,
) -> RoundEvaluation:
    records = [
        evaluate_hypothesis(hypothesis, verifications, round_label, sequence_start + index)
        for index, hypothesis in enumerate(hypotheses)
    ]
    mean_scores = {}
    for dimension in DIMENSIONS:
        values = [
            score.score
            for record in records
            for score in record.scores
            if score.dimension == dimension
        ]
        mean_scores[dimension] = round(sum(values) / len(values), 3) if values else 0.0
    return RoundEvaluation(round_label=round_label, records=records, mean_scores=mean_scores)  # type: ignore[arg-type]


def evaluate_hypothesis(
    hypothesis: Hypothesis,
    verifications: list[EvidenceVerificationRecord],
    round_label: str,
    sequence_index: int,
) -> EvaluationRecord:
    linked = [record for record in verifications if record.hypothesis_id == hypothesis.id]
    verified = sum(1 for record in linked if record.status in {"verified", "partially_verified", "conflicting"})
    unsupported = sum(1 for record in linked if record.status in {"unsupported", "invalid_reference"})
    conflicts = sum(1 for record in linked if record.status == "conflicting")
    citation_coverage = verified / len(linked) if linked else 0.0
    prediction_specificity = min(10.0, 4.0 + len(" ".join(hypothesis.testable_predictions).split()) / 6)
    falsification_quality = min(10.0, 4.5 + len(hypothesis.falsification_criteria) * 1.1)
    evidence_grounding = max(0.0, min(10.0, 3.0 + citation_coverage * 5.0 - unsupported * 1.5 - conflicts * 0.5))
    values = {
        "relevance": 7.0,
        "plausibility": max(0.0, 7.0 - hypothesis.uncertainty * 2.0),
        "evidence_grounding": evidence_grounding,
        "novelty": 6.5 if hypothesis.parent_ids else 6.0,
        "testability": min(10.0, 5.0 + len(hypothesis.testable_predictions) * 1.2),
        "falsifiability": falsification_quality,
        "explanatory_power": min(10.0, 5.0 + len(hypothesis.mechanism.split()) / 12),
        "clarity": min(10.0, 6.0 + (1.0 if len(hypothesis.core_claim) < 220 else 0.0)),
        "diversity": 6.0 + min(2.0, len(set(hypothesis.parent_ids)) * 0.5),
        "feasibility": min(10.0, 5.5 + len(hypothesis.proposed_experiments) * 0.7),
        "uncertainty_calibration": max(0.0, 8.0 - abs(hypothesis.uncertainty - 0.5) * 5.0),
    }
    scores = [
        RubricScore(
            dimension=dimension,
            score=round(score, 3),
            rationale=f"Deterministic pilot rubric score for {dimension}; not a scientific truth claim.",
            evidence_used=[record.claim_id for record in linked],
        )
        for dimension, score in values.items()
    ]
    aggregate = round(sum(values.values()) / len(values), 3)
    return EvaluationRecord(
        hypothesis_id=hypothesis.id,
        round_label=round_label,  # type: ignore[arg-type]
        scores=scores,
        aggregate_score=aggregate,
        rationale="Rule-based offline evaluation for comparing workflow stages; possible evaluator self-preference is exposed in comparison.",
        sequence_index=sequence_index,
        model_metadata={"provider": "deterministic_offline"},
        evaluated_at=datetime.now(UTC),
    )


def compare_rounds(
    project: ResearchProjectSpec,
    evaluations: list[RoundEvaluation],
    rounds: dict[str, list[Hypothesis]],
    verifications: dict[str, list[EvidenceVerificationRecord]],
) -> RoundComparison:
    by_label = {item.round_label: item for item in evaluations}
    initial = by_label.get("initial")
    final = by_label.get("final")
    changes = {}
    if initial and final:
        for dimension, final_value in final.mean_scores.items():
            changes[dimension] = round(final_value - initial.mean_scores.get(dimension, 0.0), 3)
    citation_coverage = {
        label: _citation_coverage(records)
        for label, records in verifications.items()
    }
    unsupported = {
        label: sum(1 for record in records if record.status in {"unsupported", "invalid_reference"})
        for label, records in verifications.items()
    }
    diversity = {label: _diversity(hypotheses) for label, hypotheses in rounds.items()}
    return RoundComparison(
        project_id=project.project_id,
        score_changes_by_dimension=changes,
        citation_coverage=citation_coverage,
        unsupported_claim_count=unsupported,
        hypothesis_diversity=diversity,
        duplicate_hypotheses=_duplicates(rounds.get("final", [])),
        prediction_specificity={label: _prediction_specificity(hypotheses) for label, hypotheses in rounds.items()},
        falsification_plan_quality={label: _falsification_quality(hypotheses) for label, hypotheses in rounds.items()},
        final_lineage={hypothesis.id: hypothesis.parent_ids for hypothesis in rounds.get("final", [])},
        repaired_or_rejected={
            label: [hypothesis.id for hypothesis in hypotheses if hypothesis.status in {"repaired", "rejected"}]
            for label, hypotheses in rounds.items()
        },
        evaluator_self_preference_note=(
            "This comparison uses the same deterministic evaluator before and after evolution; "
            "higher scores may reflect rubric self-preference and require human review."
        ),
        generated_at=datetime.now(UTC),
    )


def _citation_coverage(records: list[EvidenceVerificationRecord]) -> float:
    if not records:
        return 0.0
    covered = sum(1 for record in records if record.status in {"verified", "partially_verified", "conflicting"})
    return round(covered / len(records), 3)


def _diversity(hypotheses: list[Hypothesis]) -> float:
    if len(hypotheses) < 2:
        return 1.0 if hypotheses else 0.0
    pairs = list(combinations(hypotheses, 2))
    distinct = sum(1 for left, right in pairs if left.generation_strategy != right.generation_strategy or left.parent_ids != right.parent_ids)
    return round(distinct / len(pairs), 3)


def _duplicates(hypotheses: list[Hypothesis]) -> list[list[str]]:
    seen: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        key = " ".join(hypothesis.core_claim.lower().split()[:12])
        seen.setdefault(key, []).append(hypothesis.id)
    return [ids for ids in seen.values() if len(ids) > 1]


def _prediction_specificity(hypotheses: list[Hypothesis]) -> float:
    if not hypotheses:
        return 0.0
    values = [min(10.0, 4.0 + len(" ".join(item.testable_predictions).split()) / 6) for item in hypotheses]
    return round(sum(values) / len(values), 3)


def _falsification_quality(hypotheses: list[Hypothesis]) -> float:
    if not hypotheses:
        return 0.0
    values = [min(10.0, 4.5 + len(item.falsification_criteria) * 1.1) for item in hypotheses]
    return round(sum(values) / len(values), 3)
