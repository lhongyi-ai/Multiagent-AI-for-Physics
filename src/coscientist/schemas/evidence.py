from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ClaimKind = Literal["source_observation", "agent_inference", "speculation", "prediction"]
EvidenceType = Literal["abstract", "metadata", "fixture_excerpt", "reasoning", "prediction"]
EvidenceVerificationStatus = Literal[
    "verified",
    "partially_verified",
    "unsupported",
    "conflicting",
    "unresolved",
    "invalid_reference",
]


class EvidenceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    evidence_type: EvidenceType
    source_field: str | None = None


class ClaimEvidenceLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    claim_kind: ClaimKind
    supporting_paper_ids: list[str] = Field(default_factory=list)
    contradicting_paper_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    evidence_type: EvidenceType
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_conflict: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)


class EvidenceVerificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    status: EvidenceVerificationStatus
    existing_paper_ids: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    normalized_duplicate_ids: list[str] = Field(default_factory=list)
    supporting_evidence_count: int = Field(default=0, ge=0)
    contradicting_evidence_count: int = Field(default=0, ge=0)
    overstatement: bool = False
    rationale: str = Field(min_length=1)
    verifier_name: str = "deterministic-evidence-verifier"
    sequence_index: int = Field(ge=0)
    verified_at: datetime
