from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import Paper
from coscientist.schemas.v15b import GroundingConfig, GroundingDiagnostics, GroundingEvidenceItem, GroundingPacket


class GroundingAgent:
    def build_packet(
        self,
        *,
        project_id: str,
        run_id: str,
        round_label: str,
        corpus: list[Paper],
        verifications: list[EvidenceVerificationRecord],
        config: GroundingConfig,
    ) -> GroundingPacket:
        status_by_paper: dict[str, str] = {}
        for record in verifications:
            for paper_id in record.existing_paper_ids:
                status_by_paper.setdefault(paper_id, record.status)
        items = []
        chars = 0
        truncated = False
        for paper in corpus:
            excerpt = _best_excerpt(paper)
            evidence_type = "fixture_excerpt" if _has_fixture_excerpt(paper) else "metadata"
            if evidence_type == "metadata" and not config.include_metadata_only_records:
                continue
            item = GroundingEvidenceItem(
                evidence_id=f"evidence:{paper.id}",
                paper_id=paper.id,
                title=paper.title,
                evidence_type=evidence_type,
                excerpt=excerpt,
                verification_status=status_by_paper.get(paper.id),
                source_field="source_metadata.fixture_excerpts" if evidence_type == "fixture_excerpt" else "metadata",
            )
            item_chars = len(item.model_dump_json())
            if chars + item_chars > config.max_context_characters:
                truncated = True
                break
            chars += item_chars
            items.append(item)
        return GroundingPacket(
            project_id=project_id,
            run_id=run_id,
            round_label=round_label,
            created_at=datetime.now(UTC),
            mode=config.mode,
            evidence_items=items,
            source_limitations=[
                "Fixture corpus is incomplete and cannot establish scientific truth.",
                "Metadata-only records cannot support strict scientific claims.",
            ],
            context_character_count=chars,
            truncated=truncated,
            validation_status="validated",
        )

    def diagnostics(
        self,
        *,
        project_id: str,
        run_id: str,
        round_label: str,
        hypotheses: list[Hypothesis],
        verifications: list[EvidenceVerificationRecord],
        packet: GroundingPacket,
        config: GroundingConfig,
    ) -> GroundingDiagnostics:
        packet_papers = {item.paper_id: item for item in packet.evidence_items}
        existing_evidence_ids = {item.evidence_id for item in packet.evidence_items}
        supported = 0
        unsupported = 0
        verified_ids = 0
        missing_ids = 0
        hallucinations = 0
        metadata_misuse = 0
        inference_as_fact = 0
        contradicting_hypotheses = 0
        cited_counts: Counter[str] = Counter()
        verification_by_claim = {record.claim_id: record for record in verifications}
        for hypothesis in hypotheses:
            has_contradiction = False
            for link in hypothesis.evidence_links:
                cited = [*link.supporting_paper_ids, *link.contradicting_paper_ids]
                cited_counts.update(cited)
                missing = [paper_id for paper_id in cited if paper_id not in packet_papers]
                if missing:
                    missing_ids += len(missing)
                    hallucinations += len(missing)
                record = verification_by_claim.get(link.claim_id)
                if record and record.status in {"verified", "partially_verified", "conflicting"}:
                    verified_ids += 1
                if link.supporting_paper_ids and not missing:
                    supported += 1
                else:
                    unsupported += 1
                if any(packet_papers.get(paper_id) and packet_papers[paper_id].evidence_type == "metadata" for paper_id in link.supporting_paper_ids):
                    metadata_misuse += 1
                if link.claim_kind == "source_observation" and "may" in link.claim_text.lower():
                    inference_as_fact += 1
                if link.contradicting_paper_ids:
                    has_contradiction = True
            if has_contradiction:
                contradicting_hypotheses += 1
        total_citations = sum(cited_counts.values())
        concentration = round(max(cited_counts.values()) / total_citations, 3) if total_citations else 0.0
        total_claims = supported + unsupported
        coverage = round(supported / total_claims, 3) if total_claims else 0.0
        warnings = []
        if config.mode == "strict" and hallucinations:
            warnings.append("Strict grounding found citations not present in the grounding packet.")
        if concentration >= 0.5 and total_citations:
            warnings.append("Evidence reuse is concentrated in one source.")
        return GroundingDiagnostics(
            project_id=project_id,
            run_id=run_id,
            round_label=round_label,
            created_at=datetime.now(UTC),
            grounding_mode=config.mode,
            supported_claim_count=supported,
            unsupported_claim_count=unsupported,
            claims_with_verified_evidence_ids=verified_ids,
            claims_citing_missing_evidence_ids=missing_ids,
            citation_hallucination_count=hallucinations,
            metadata_only_misuse_count=metadata_misuse,
            inference_as_source_fact_count=inference_as_fact,
            evidence_reuse_concentration=concentration,
            final_hypotheses_with_contradicting_evidence_fraction=round(contradicting_hypotheses / len(hypotheses), 3) if hypotheses else 0.0,
            grounding_coverage_score=coverage,
            warnings=warnings,
            validation_status="validated",
        )


def _has_fixture_excerpt(paper: Paper) -> bool:
    excerpts = paper.source_metadata.get("fixture_excerpts")
    return isinstance(excerpts, list) and bool(excerpts)


def _best_excerpt(paper: Paper) -> str:
    excerpts = paper.source_metadata.get("fixture_excerpts")
    if isinstance(excerpts, list) and excerpts:
        return str(excerpts[0])
    if paper.abstract:
        return paper.abstract
    return f"Metadata-only record: {paper.title}"
