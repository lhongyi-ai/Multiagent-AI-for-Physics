from __future__ import annotations

from datetime import UTC, datetime

from coscientist.literature.identifiers import normalize_doi
from coscientist.schemas.evidence import ClaimEvidenceLink, EvidenceExcerpt, EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import Paper


def attach_fixture_evidence(hypotheses: list[Hypothesis], corpus: list[Paper], round_label: str) -> list[Hypothesis]:
    if not corpus:
        return hypotheses
    grounded = []
    for index, hypothesis in enumerate(hypotheses):
        support = corpus[index % len(corpus)]
        conflict = corpus[(index + 1) % len(corpus)] if len(corpus) > 1 else None
        excerpt = _best_excerpt(support)
        evidence = [
            EvidenceExcerpt(
                paper_id=support.id,
                excerpt=excerpt,
                evidence_type="fixture_excerpt",
                source_field="source_metadata.fixture_excerpts",
            )
        ]
        link = ClaimEvidenceLink(
            claim_id=f"{hypothesis.id}:claim:core",
            claim_text=hypothesis.core_claim,
            claim_kind="agent_inference",
            supporting_paper_ids=[support.id],
            contradicting_paper_ids=[conflict.id] if conflict else [],
            evidence=evidence,
            evidence_type="fixture_excerpt",
            confidence=0.55,
            unresolved_conflict=conflict is not None,
            provenance={
                "grounding": "deterministic_fixture_mapping",
                "round": round_label,
                "note": "Fixture evidence grounds claim context but does not prove the hypothesis.",
            },
        )
        prediction_link = ClaimEvidenceLink(
            claim_id=f"{hypothesis.id}:claim:prediction",
            claim_text=hypothesis.testable_predictions[0],
            claim_kind="prediction",
            supporting_paper_ids=[],
            contradicting_paper_ids=[],
            evidence=[],
            evidence_type="prediction",
            confidence=0.35,
            unresolved_conflict=False,
            provenance={"round": round_label, "grounding": "hypothesis_prediction"},
        )
        grounded.append(hypothesis.model_copy(update={"evidence_links": [link, prediction_link]}))
    return grounded


def verify_hypothesis_evidence(hypotheses: list[Hypothesis], corpus: list[Paper]) -> list[EvidenceVerificationRecord]:
    paper_by_id = {paper.id: paper for paper in corpus}
    canonical_to_ids: dict[tuple[str, str], list[str]] = {}
    for paper in corpus:
        for identifier in paper.identifiers:
            canonical_to_ids.setdefault((identifier.scheme, identifier.canonical_value), []).append(paper.id)
        doi = normalize_doi(paper.doi)
        if doi:
            canonical_to_ids.setdefault(("doi", doi), []).append(paper.id)

    records = []
    seq = 0
    for hypothesis in hypotheses:
        for link in hypothesis.evidence_links:
            cited_ids = [*link.supporting_paper_ids, *link.contradicting_paper_ids]
            missing = [paper_id for paper_id in cited_ids if paper_id not in paper_by_id]
            existing = [paper_id for paper_id in cited_ids if paper_id in paper_by_id]
            duplicate_ids = sorted({
                paper_id
                for ids in canonical_to_ids.values()
                if len(ids) > 1
                for paper_id in ids
                if paper_id in cited_ids
            })
            support_count = sum(1 for item in link.evidence if item.paper_id in paper_by_id and item.excerpt.strip())
            contradiction_count = len([paper_id for paper_id in link.contradicting_paper_ids if paper_id in paper_by_id])
            overstatement = link.claim_kind in {"agent_inference", "speculation"} and link.confidence > 0.8
            if missing:
                status = "invalid_reference"
                rationale = "One or more cited paper IDs are absent from the corpus."
            elif not link.supporting_paper_ids and link.claim_kind != "prediction":
                status = "unsupported"
                rationale = "The claim has no supporting paper IDs."
            elif link.claim_kind == "prediction" and not link.supporting_paper_ids:
                status = "unsupported"
                rationale = "Prediction is not source-backed yet and needs validation."
            elif contradiction_count and support_count:
                status = "conflicting"
                rationale = "Supporting and contradicting corpus records are both linked."
            elif support_count:
                status = "verified"
                rationale = "Cited paper IDs exist and at least one associated excerpt is present."
            else:
                status = "partially_verified"
                rationale = "Cited paper IDs exist but no excerpt was attached."
            records.append(EvidenceVerificationRecord(
                claim_id=link.claim_id,
                hypothesis_id=hypothesis.id,
                status=status,  # type: ignore[arg-type]
                existing_paper_ids=existing,
                missing_paper_ids=missing,
                normalized_duplicate_ids=duplicate_ids,
                supporting_evidence_count=support_count,
                contradicting_evidence_count=contradiction_count,
                overstatement=overstatement,
                rationale=rationale,
                sequence_index=seq,
                verified_at=datetime.now(UTC),
            ))
            seq += 1
    return records


def _best_excerpt(paper: Paper) -> str:
    excerpts = paper.source_metadata.get("fixture_excerpts")
    if isinstance(excerpts, list) and excerpts:
        return str(excerpts[0])
    if paper.abstract:
        return paper.abstract
    return f"Metadata-only fixture record: {paper.title}"
