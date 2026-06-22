from __future__ import annotations

from datetime import UTC, datetime

from coscientist.schemas.literature import (
    ExternalIdentifier,
    FullTextLocation,
    MetadataFieldProvenance,
    MetadataResolution,
    MetadataResolveRequest,
    Paper,
    PaperAuthor,
    ProviderRequestLog,
    SearchQuery,
)


class MockLiteratureSearch:
    provider_name = "mock"

    def __init__(self) -> None:
        self.request_logs: list[ProviderRequestLog] = []

    async def search(self, query: SearchQuery) -> list[Paper]:
        papers = []
        for index in range(query.limit):
            doi = f"10.0000/mock-{index}"
            identifier = ExternalIdentifier(scheme="doi", value=doi, canonical_value=doi, source="mock")
            papers.append(Paper(
                id=f"paper-doi-mock-{index}",
                title=f"Mock literature record {index + 1} for {query.query}",
                authors=[PaperAuthor(name="Mock Author", institutions=["Synthetic Institute"])],
                abstract="Synthetic abstract for offline deterministic testing.",
                venue="Journal of Mock Results",
                publication_year=query.from_year or 2024,
                publication_date=f"{query.from_year or 2024}-01-01",
                publication_type="journal-article",
                doi=doi,
                identifiers=[identifier],
                source_provider="mock",
                provider_record_id=f"mock-{index}",
                source_metadata={"fixture": True},
            ))
        self.request_logs.append(ProviderRequestLog(
            provider="mock",
            operation="search",
            normalized_query=query.model_dump(),
            timestamp=datetime.now(UTC),
            cache_hit=False,
            status="success",
            result_count=len(papers),
            latency_ms=0.0,
        ))
        return papers


class MockMetadataResolver:
    provider_name = "mock"

    def __init__(self) -> None:
        self.request_logs: list[ProviderRequestLog] = []

    async def resolve(self, request: MetadataResolveRequest) -> MetadataResolution:
        if request.paper:
            paper = request.paper
        elif request.doi:
            identifier = ExternalIdentifier(scheme="doi", value=request.doi, canonical_value=request.doi.lower(), source="mock")
            paper = Paper(
                id=f"paper-doi-{request.doi.lower().replace('/', '-')}",
                title=f"Mock metadata for {request.doi}",
                authors=[PaperAuthor(name="Mock Author", institutions=["Synthetic Institute"])],
                venue="Journal of Mock Results",
                publication_year=2024,
                publication_date="2024-01-01",
                publication_type="journal-article",
                doi=request.doi.lower(),
                identifiers=[identifier],
                source_provider="mock",
                provider_record_id=request.doi,
                source_metadata={"fixture": True},
            )
        else:
            paper = (await MockLiteratureSearch().search(SearchQuery(query=request.title or "mock", limit=1)))[0]
        self.request_logs.append(ProviderRequestLog(
            provider="mock",
            operation="resolve",
            normalized_query={"doi": request.doi, "title": request.title, "paper_id": request.paper.id if request.paper else None},
            timestamp=datetime.now(UTC),
            cache_hit=False,
            status="success",
            result_count=1,
            latency_ms=0.0,
        ))
        return MetadataResolution(
            paper_id=paper.id,
            resolved_identifiers=paper.identifiers,
            normalized_metadata=paper,
            provenance=[
                MetadataFieldProvenance(field_name="title", value=paper.title, provider="mock", retrieved_at=datetime.now(UTC)),
            ],
            status="resolved",
            resolver_name="mock",
            retrieved_at=datetime.now(UTC),
        )


class MockFullTextLocator:
    provider_name = "mock"

    def __init__(self) -> None:
        self.request_logs: list[ProviderRequestLog] = []

    async def locate(self, paper: Paper) -> list[FullTextLocation]:
        self.request_logs.append(ProviderRequestLog(
            provider="mock",
            operation="locate",
            normalized_query={"paper_id": paper.id, "doi": paper.doi},
            timestamp=datetime.now(UTC),
            cache_hit=False,
            status="success",
            result_count=1,
            latency_ms=0.0,
        ))
        return [FullTextLocation(
            id=f"loc-mock-{paper.id}",
            paper_id=paper.id,
            provider="mock",
            landing_page_url="https://example.test/mock-paper",
            document_url=None,
            content_type="text/plain",
            access_status="abstract_only",
            host_type="repository",
            version="mock",
            is_best=True,
            retrieved_at=datetime.now(UTC),
        )]
