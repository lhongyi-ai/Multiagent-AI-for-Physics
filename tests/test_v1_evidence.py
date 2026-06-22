from __future__ import annotations

from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.schemas.evidence import ClaimEvidenceLink, EvidenceExcerpt
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import ExternalIdentifier, Paper


def _paper(paper_id: str, doi: str = "10.1/x") -> Paper:
    return Paper(
        id=paper_id,
        title=f"Paper {paper_id}",
        doi=doi,
        identifiers=[ExternalIdentifier(scheme="doi", value=doi, canonical_value=doi.lower(), source="test")],
        source_provider="test",
        source_metadata={"fixture_excerpts": ["Evidence excerpt"]},
    )


def _hypothesis(link: ClaimEvidenceLink | None = None) -> Hypothesis:
    return Hypothesis(
        id="h",
        title="H",
        core_claim="Claim",
        mechanism="Mechanism",
        assumptions=["a"],
        supporting_evidence=[],
        contradicting_evidence=[],
        novelty_statement="Novel",
        testable_predictions=["Prediction"],
        falsification_criteria=["Falsify"],
        proposed_experiments=["Experiment"],
        uncertainty=0.5,
        generation_strategy="mechanistic",
        parent_ids=[],
        version=1,
        evidence_links=[link] if link else [],
    )


def test_attach_fixture_evidence_adds_structured_links() -> None:
    grounded = attach_fixture_evidence([_hypothesis()], [_paper("p1"), _paper("p2", "10.1/y")], "initial")
    assert grounded[0].evidence_links[0].supporting_paper_ids == ["p1"]
    assert grounded[0].evidence_links[0].contradicting_paper_ids == ["p2"]


def test_unknown_citation_id_is_invalid_reference() -> None:
    link = ClaimEvidenceLink(
        claim_id="c",
        claim_text="Claim",
        claim_kind="agent_inference",
        supporting_paper_ids=["missing"],
        contradicting_paper_ids=[],
        evidence=[EvidenceExcerpt(paper_id="missing", excerpt="x", evidence_type="fixture_excerpt")],
        evidence_type="fixture_excerpt",
        confidence=0.5,
    )
    records = verify_hypothesis_evidence([_hypothesis(link)], [_paper("p1")])
    assert records[0].status == "invalid_reference"


def test_duplicate_citations_are_normalized() -> None:
    link = ClaimEvidenceLink(
        claim_id="c",
        claim_text="Claim",
        claim_kind="agent_inference",
        supporting_paper_ids=["p1", "p2"],
        evidence=[EvidenceExcerpt(paper_id="p1", excerpt="x", evidence_type="fixture_excerpt")],
        evidence_type="fixture_excerpt",
        confidence=0.5,
    )
    records = verify_hypothesis_evidence([_hypothesis(link)], [_paper("p1", "10.1/x"), _paper("p2", "10.1/x")])
    assert records[0].normalized_duplicate_ids == ["p1", "p2"]


def test_unsupported_claim_is_detected() -> None:
    link = ClaimEvidenceLink(
        claim_id="c",
        claim_text="Claim",
        claim_kind="agent_inference",
        supporting_paper_ids=[],
        evidence=[],
        evidence_type="reasoning",
        confidence=0.5,
    )
    records = verify_hypothesis_evidence([_hypothesis(link)], [_paper("p1")])
    assert records[0].status == "unsupported"


def test_conflicting_evidence_is_detected() -> None:
    link = ClaimEvidenceLink(
        claim_id="c",
        claim_text="Claim",
        claim_kind="agent_inference",
        supporting_paper_ids=["p1"],
        contradicting_paper_ids=["p2"],
        evidence=[EvidenceExcerpt(paper_id="p1", excerpt="x", evidence_type="fixture_excerpt")],
        evidence_type="fixture_excerpt",
        confidence=0.5,
    )
    records = verify_hypothesis_evidence([_hypothesis(link)], [_paper("p1"), _paper("p2", "10.1/y")])
    assert records[0].status == "conflicting"
