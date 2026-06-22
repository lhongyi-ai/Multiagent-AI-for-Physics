from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


IdentifierScheme = Literal["doi", "openalex", "arxiv", "pmid", "pmcid", "other"]
ResolutionStatus = Literal[
    "resolved",
    "partially_resolved",
    "conflicting",
    "not_found",
    "invalid_identifier",
    "provider_error",
]
FullTextAccessStatus = Literal[
    "full_text_retrieved",
    "abstract_only",
    "metadata_only",
    "open_location_found",
    "no_legal_open_copy",
]
ProviderRequestStatus = Literal["success", "cache_hit", "network_disabled", "provider_error"]


class ExternalIdentifier(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scheme: IdentifierScheme
    value: str = Field(min_length=1)
    canonical_value: str = Field(min_length=1)
    source: str = Field(min_length=1)


class PaperAuthor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    orcid: str | None = None
    institutions: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[PaperAuthor] = Field(default_factory=list)
    abstract: str | None = None
    venue: str | None = None
    publication_year: int | None = Field(default=None, ge=0)
    publication_date: str | None = None
    publication_type: str | None = None
    doi: str | None = None
    identifiers: list[ExternalIdentifier] = Field(default_factory=list)
    source_provider: str = Field(min_length=1)
    provider_record_id: str | None = None
    cited_by_count: int | None = Field(default=None, ge=0)
    referenced_work_ids: list[str] = Field(default_factory=list)
    open_access: dict[str, Any] = Field(default_factory=dict)
    retraction_status: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1)
    from_year: int | None = Field(default=None, ge=0)
    to_year: int | None = Field(default=None, ge=0)
    limit: int = Field(default=10, ge=1, le=100)


class MetadataResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    doi: str | None = None
    title: str | None = None
    paper: Paper | None = None


class MetadataFieldProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field_name: str = Field(min_length=1)
    value: str
    provider: str = Field(min_length=1)
    retrieved_at: datetime
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class MetadataConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field_name: str = Field(min_length=1)
    existing_value: str | None = None
    incoming_value: str | None = None
    existing_provider: str | None = None
    incoming_provider: str = Field(min_length=1)
    resolution_status: str = Field(default="unresolved")
    notes: list[str] = Field(default_factory=list)


class MetadataResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str | None = None
    resolved_identifiers: list[ExternalIdentifier] = Field(default_factory=list)
    normalized_metadata: Paper | None = None
    provenance: list[MetadataFieldProvenance] = Field(default_factory=list)
    conflicts: list[MetadataConflict] = Field(default_factory=list)
    status: ResolutionStatus
    resolver_name: str = Field(min_length=1)
    retrieved_at: datetime


class FullTextLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    landing_page_url: str | None = None
    document_url: str | None = None
    content_type: str | None = None
    access_status: FullTextAccessStatus
    host_type: str | None = None
    version: str | None = None
    license: str | None = None
    is_best: bool = False
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime


class ProviderErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    retryable: bool
    status_code: int | None = None
    message: str = Field(min_length=1)
    retry_after_seconds: float | None = None
    occurred_at: datetime


class ProviderRequestLog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    normalized_query: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    cache_hit: bool
    status: ProviderRequestStatus
    result_count: int = Field(default=0, ge=0)
    latency_ms: float | None = None


class RetrievedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    location_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    content_type: str
    text: str | None = None
    local_path: str | None = None
    retrieved_at: datetime


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    passage: str | None = None
    source_kind: Literal["source_observation", "source_interpretation", "system_inference"]
    verification_status: str = Field(default="unverified")


class CitationVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    passage: str | None = None
    supports_claim: bool
    verifier_name: str
    notes: list[str] = Field(default_factory=list)
