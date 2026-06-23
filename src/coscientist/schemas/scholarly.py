from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


LiteratureMode = Literal["fixture", "live", "existing"]
QueryOrigin = Literal["user", "deterministic_rule", "model"]
QueryStatus = Literal["planned", "success", "partial", "failed", "skipped"]
CachePolicy = Literal["reuse", "refresh", "offline-only"]
RunStatus = Literal["complete", "complete_with_warnings", "partial", "failed", "aborted_by_budget"]

SEARCH_PROVIDERS = {"openalex", "arxiv", "mock"}
ENRICHMENT_PROVIDERS = {"crossref", "unpaywall", "mock"}
ALL_LITERATURE_PROVIDERS = SEARCH_PROVIDERS | ENRICHMENT_PROVIDERS


class LiteratureQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    providers: list[str] = Field(default_factory=list)
    origin: QueryOrigin = "user"
    filters: dict[str, Any] = Field(default_factory=dict)


class ProjectLiteratureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: LiteratureMode = "fixture"
    query_generation: Literal["explicit", "deterministic"] = "explicit"
    queries: list[LiteratureQuerySpec] = Field(default_factory=list)
    search_providers: list[str] = Field(default_factory=lambda: ["mock"])
    enrichment_providers: list[str] = Field(default_factory=lambda: ["mock"])
    max_queries: int = Field(default=5, ge=1, le=50)
    max_results_per_query: int = Field(default=20, ge=1, le=200)
    max_results_per_provider: int = Field(default=40, ge=1, le=500)
    max_total_results: int = Field(default=80, ge=1, le=1000)
    max_total_requests: int = Field(default=40, ge=1, le=1000)
    max_requests_per_provider: int = Field(default=20, ge=1, le=500)
    date_from: int | None = Field(default=None, ge=0)
    date_to: int | None = Field(default=None, ge=0)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    require_open_access: bool = False
    locate_open_access: bool = True
    retrieve_full_text: bool = False
    minimum_corpus_size: int = Field(default=1, ge=0)
    cache_policy: CachePolicy = "reuse"
    existing_corpus_path: str | None = None

    @model_validator(mode="after")
    def validate_literature_config(self) -> ProjectLiteratureConfig:
        unknown_search = set(self.search_providers) - SEARCH_PROVIDERS
        unknown_enrich = set(self.enrichment_providers) - ENRICHMENT_PROVIDERS
        if unknown_search:
            raise ValueError(f"unknown search provider(s): {sorted(unknown_search)}")
        if unknown_enrich:
            raise ValueError(f"unknown enrichment provider(s): {sorted(unknown_enrich)}")
        bad_search_roles = set(self.search_providers).intersection({"crossref", "unpaywall"})
        if bad_search_roles:
            raise ValueError(f"provider(s) are enrichment-only, not search providers: {sorted(bad_search_roles)}")
        bad_enrichment_roles = set(self.enrichment_providers).intersection({"openalex", "arxiv"})
        if bad_enrichment_roles:
            raise ValueError(f"provider(s) are search-only, not enrichment providers: {sorted(bad_enrichment_roles)}")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be after date_to")
        if self.query_generation == "explicit" and self.mode == "live" and not self.queries:
            raise ValueError("live explicit query mode requires at least one query")
        if self.mode == "live" and ("mock" in self.search_providers or "mock" in self.enrichment_providers):
            raise ValueError("live literature mode cannot use mock providers")
        if self.mode == "existing" and not self.existing_corpus_path:
            raise ValueError("existing literature mode requires existing_corpus_path or --corpus")
        if self.minimum_corpus_size > self.max_total_results:
            raise ValueError("minimum_corpus_size cannot exceed max_total_results")
        for query in self.queries:
            providers = query.providers or self.search_providers
            unknown = set(providers) - SEARCH_PROVIDERS
            if unknown:
                raise ValueError(f"query uses unknown search provider(s): {sorted(unknown)}")
            if self.mode == "live" and "mock" in providers:
                raise ValueError("live literature queries cannot use mock providers")
        return self


class LiteratureQueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query_id: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_text: str = Field(min_length=1)
    origin: QueryOrigin
    filters: dict[str, Any] = Field(default_factory=dict)
    date_from: int | None = None
    date_to: int | None = None
    created_at: datetime
    execution_status: QueryStatus = "planned"
    result_count: int = Field(default=0, ge=0)
    failure_details: str | None = None


class ProviderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(min_length=1)
    role: Literal["search", "metadata_enrichment", "open_access_enrichment", "full_text_location"]
    configured: bool
    enabled: bool
    status: RunStatus | Literal["not_configured", "skipped"]
    message: str | None = None
    secret_configured: bool = False


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(min_length=1)
    request_count: int = Field(default=0, ge=0)
    result_count: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    throttling_waits: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    partial_failures: int = Field(default=0, ge=0)
    rate_limit_metadata: dict[str, Any] = Field(default_factory=dict)
    network_mode: Literal["fixture", "live", "existing", "mocked"] = "fixture"


class DeduplicationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input_record_count: int = Field(ge=0)
    output_paper_count: int = Field(ge=0)
    exact_merges: int = Field(default=0, ge=0)
    probable_merges: int = Field(default=0, ge=0)
    unresolved_candidate_duplicates: list[list[str]] = Field(default_factory=list)
    conflict_count: int = Field(default=0, ge=0)
    merge_rationale: list[str] = Field(default_factory=list)
    deduplication_version: str = "v1"


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    corpus_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    mode: LiteratureMode
    paper_count: int = Field(ge=0)
    corpus_hash: str = Field(min_length=1)
    normalization_version: str = "v1"
    deduplication_version: str = "v1"
    enrichment_version: str = "v1"
    providers: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime
