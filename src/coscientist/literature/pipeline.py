from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from coscientist.config import LiteratureConfig
from coscientist.literature.cache import ProviderResponseCache
from coscientist.literature.http import AsyncHttpTransport
from coscientist.literature.interfaces import FullTextLocator, LiteratureSearchTool, MetadataResolver
from coscientist.literature.normalization import PaperDeduplicator
from coscientist.literature.providers import (
    ArxivFullTextLocator,
    ArxivLiteratureSearch,
    CrossrefMetadataResolver,
    MockFullTextLocator,
    MockLiteratureSearch,
    MockMetadataResolver,
    OpenAlexLiteratureSearch,
    UnpaywallFullTextLocator,
)
from coscientist.schemas.literature import (
    CitationVerification,
    EvidenceClaim,
    FullTextLocation,
    MetadataConflict,
    MetadataResolution,
    MetadataResolveRequest,
    Paper,
    ProviderRequestLog,
    SearchQuery,
)


@dataclass
class LiteratureAcquisitionResult:
    raw_papers: list[Paper]
    normalized_papers: list[Paper]
    metadata_resolutions: list[MetadataResolution]
    metadata_conflicts: list[MetadataConflict]
    full_text_locations: list[FullTextLocation]
    provider_requests: list[ProviderRequestLog]
    evidence_claims: list[EvidenceClaim]
    citation_verifications: list[CitationVerification]


class LiteratureAcquisitionPipeline:
    def __init__(
        self,
        search_tools: list[LiteratureSearchTool],
        metadata_resolvers: list[MetadataResolver],
        full_text_locators: list[FullTextLocator],
        config: LiteratureConfig,
    ) -> None:
        self.search_tools = search_tools
        self.metadata_resolvers = metadata_resolvers
        self.full_text_locators = full_text_locators
        self.config = config
        self.deduplicator = PaperDeduplicator()

    async def acquire(self, query_text: str) -> LiteratureAcquisitionResult:
        query = SearchQuery(query=query_text, limit=self.config.max_results_per_provider)
        raw: list[Paper] = []
        provider_logs: list[ProviderRequestLog] = []
        for tool in self.search_tools:
            papers = await tool.search(query)
            raw.extend(papers[: self.config.max_results_per_provider])
            provider_logs.extend(_drain_logs(tool))
        deduped = self.deduplicator.merge(raw[: self.config.max_total_results])
        normalized = deduped.papers[: self.config.max_total_results]
        resolutions: list[MetadataResolution] = []
        conflicts: list[MetadataConflict] = list(deduped.conflicts)
        for paper in normalized:
            for resolver in self.metadata_resolvers:
                resolution = await resolver.resolve(MetadataResolveRequest(doi=paper.doi, title=paper.title, paper=paper))
                resolutions.append(resolution)
                conflicts.extend(resolution.conflicts)
                provider_logs.extend(_drain_logs(resolver))
        locations: list[FullTextLocation] = []
        for paper in normalized:
            for locator in self.full_text_locators:
                located = await locator.locate(paper)
                locations.extend(located)
                provider_logs.extend(_drain_logs(locator))
        return LiteratureAcquisitionResult(
            raw_papers=raw,
            normalized_papers=normalized,
            metadata_resolutions=resolutions,
            metadata_conflicts=conflicts,
            full_text_locations=locations,
            provider_requests=provider_logs,
            evidence_claims=[
                EvidenceClaim(
                    paper_id=paper.id,
                    claim="Metadata-only record requires document retrieval and citation verification before use as evidence.",
                    source_kind="system_inference",
                    verification_status="unverified",
                )
                for paper in normalized
            ],
            citation_verifications=[
                CitationVerification(
                    paper_id=paper.id,
                    claim="No exact passage was verified in the acquisition stage.",
                    passage=None,
                    supports_claim=False,
                    verifier_name="not-run",
                    notes=["Finding a source is separate from verifying a citation."],
                )
                for paper in normalized
            ],
        )


def build_literature_pipeline(config: LiteratureConfig) -> LiteratureAcquisitionPipeline:
    cache = ProviderResponseCache(config.cache_dir, config.cache_ttl_hours, config.cache_enabled)
    transport = AsyncHttpTransport(
        allow_live_network=config.allow_live_network,
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        user_agent=config.user_agent,
        concurrency_limit=4,
    )
    searches = [_build_search_provider(name, config, transport, cache) for name in config.search_providers]
    resolvers = [_build_resolver(name, config, transport, cache) for name in config.metadata_resolvers]
    locators = [_build_locator(name, config, transport, cache) for name in config.full_text_locators]
    return LiteratureAcquisitionPipeline(searches, resolvers, locators, config)


def _build_search_provider(name: str, config: LiteratureConfig, transport: AsyncHttpTransport, cache: ProviderResponseCache):
    if name == "mock":
        return MockLiteratureSearch()
    if name == "openalex":
        return OpenAlexLiteratureSearch(transport, cache, force_refresh=config.force_refresh)
    if name == "arxiv":
        transport.pace_seconds = max(transport.pace_seconds, 3.0)
        return ArxivLiteratureSearch(transport, cache, force_refresh=config.force_refresh)
    raise ValueError(f"Unknown literature search provider: {name}")


def _build_resolver(name: str, config: LiteratureConfig, transport: AsyncHttpTransport, cache: ProviderResponseCache):
    if name == "mock":
        return MockMetadataResolver()
    if name == "crossref":
        return CrossrefMetadataResolver(transport, cache, force_refresh=config.force_refresh, user_agent=config.user_agent)
    raise ValueError(f"Unknown metadata resolver: {name}")


def _build_locator(name: str, config: LiteratureConfig, transport: AsyncHttpTransport, cache: ProviderResponseCache):
    if name == "mock":
        return MockFullTextLocator()
    if name == "unpaywall":
        return UnpaywallFullTextLocator(transport, cache, force_refresh=config.force_refresh)
    if name == "arxiv":
        return ArxivFullTextLocator()
    raise ValueError(f"Unknown full-text locator: {name}")


def no_literature_result() -> LiteratureAcquisitionResult:
    return LiteratureAcquisitionResult([], [], [], [], [], [], [], [])


def _drain_logs(provider: object) -> list[ProviderRequestLog]:
    logs = list(getattr(provider, "request_logs", []))
    if hasattr(provider, "request_logs"):
        provider.request_logs.clear()  # type: ignore[attr-defined]
    return logs
