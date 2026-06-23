from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coscientist.literature.cache import ProviderResponseCache
from coscientist.literature.http import AsyncHttpTransport, NetworkDisabledError, ProviderTransportError
from coscientist.literature.normalization import PaperDeduplicator
from coscientist.literature.providers.arxiv import ArxivLiteratureSearch
from coscientist.literature.providers.crossref import CrossrefMetadataResolver
from coscientist.literature.providers.openalex import OpenAlexLiteratureSearch
from coscientist.literature.providers.unpaywall import UnpaywallFullTextLocator
from coscientist.literature.query_planner import plan_literature_queries
from coscientist.pilot.project_io import load_fixture_corpus
from coscientist.schemas.literature import FullTextLocation, MetadataConflict, MetadataResolution, MetadataResolveRequest, Paper, ProviderRequestLog, SearchQuery
from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.scholarly import (
    CorpusManifest,
    DeduplicationReport,
    LiteratureQueryRecord,
    ProjectLiteratureConfig,
    ProviderStatus,
    ProviderUsage,
)


@dataclass
class ScholarlyCorpusResult:
    papers: list[Paper]
    raw_records: dict[str, list[dict[str, Any]]]
    query_records: list[LiteratureQueryRecord]
    provider_events: list[ProviderRequestLog]
    provider_status: list[ProviderStatus]
    provider_usage: dict[str, ProviderUsage]
    metadata_resolutions: list[MetadataResolution]
    metadata_conflicts: list[MetadataConflict]
    full_text_locations: list[FullTextLocation]
    deduplication_report: DeduplicationReport
    corpus_manifest: CorpusManifest
    warnings: list[str]


class ScholarlyLiteratureOrchestrator:
    def __init__(
        self,
        *,
        project: ResearchProjectSpec,
        literature_config: ProjectLiteratureConfig,
        cache_dir: str | Path = ".coscientist_cache/provider_responses",
        allow_live_network: bool = False,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        user_agent: str | None = None,
        transport: AsyncHttpTransport | None = None,
    ) -> None:
        self.project = project
        self.config = literature_config
        self.allow_live_network = allow_live_network
        cache_enabled = self.config.cache_policy != "offline-only"
        self.cache = ProviderResponseCache(cache_dir, enabled=cache_enabled)
        self.transport = transport or AsyncHttpTransport(
            allow_live_network=allow_live_network,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            user_agent=user_agent or os.getenv("COSCIENTIST_USER_AGENT", "coscientist-mvp/0.1"),
            concurrency_limit=2,
        )
        self.raw_records: dict[str, list[dict[str, Any]]] = {"openalex": [], "arxiv": [], "crossref": [], "unpaywall": []}
        self.events: list[ProviderRequestLog] = []
        self.statuses: list[ProviderStatus] = []
        self.warnings: list[str] = []

    async def acquire(self, fixture_path: Path | None = None, existing_corpus_path: Path | None = None) -> ScholarlyCorpusResult:
        if self.config.mode == "fixture":
            if fixture_path is None:
                raise ValueError("fixture literature mode requires a fixture corpus path")
            papers = load_fixture_corpus(fixture_path)
            return self._result(papers, [], [], [], self._dedup_report(len(papers), len(papers), 0), "fixture")
        if self.config.mode == "existing":
            if existing_corpus_path is None and self.config.existing_corpus_path:
                existing_corpus_path = Path(self.config.existing_corpus_path)
            if existing_corpus_path is None:
                raise ValueError("existing literature mode requires a corpus path")
            papers = load_fixture_corpus(existing_corpus_path)
            return self._result(papers, [], [], [], self._dedup_report(len(papers), len(papers), 0), "existing")
        if self.config.mode == "live" and not self.allow_live_network:
            raise NetworkDisabledError("live literature mode requires explicit --live-network")
        return await self._acquire_live()

    def dry_run(self) -> ScholarlyCorpusResult:
        queries = plan_literature_queries(self.project, self.config)
        return self._result([], queries, [], [], self._dedup_report(0, 0, 0), self.config.mode, warnings=["dry-run: no network calls were made"])

    async def _acquire_live(self) -> ScholarlyCorpusResult:
        queries = plan_literature_queries(self.project, self.config)
        raw_papers: list[Paper] = []
        requests_by_provider: dict[str, int] = {}
        for query_record in queries:
            provider = query_record.provider
            if requests_by_provider.get(provider, 0) >= self.config.max_requests_per_provider:
                query_record.execution_status = "skipped"
                query_record.failure_details = "provider request budget exhausted"
                continue
            if sum(requests_by_provider.values()) >= self.config.max_total_requests:
                query_record.execution_status = "skipped"
                query_record.failure_details = "total request budget exhausted"
                self.warnings.append("total literature request budget exhausted")
                break
            try:
                papers = await self._search(provider, query_record)
            except (NetworkDisabledError, ProviderTransportError, ValueError) as exc:
                query_record.execution_status = "failed"
                query_record.failure_details = str(exc)
                self.warnings.append(f"{provider} search failed: {exc}")
                continue
            requests_by_provider[provider] = requests_by_provider.get(provider, 0) + 1
            query_record.execution_status = "success"
            query_record.result_count = len(papers)
            raw_papers.extend(papers)
            if len(raw_papers) >= self.config.max_total_results:
                break
        dedup = PaperDeduplicator().merge(raw_papers[: self.config.max_total_results])
        papers = dedup.papers[: self.config.max_total_results]
        resolutions, enrichment_conflicts = await self._crossref_enrich(papers)
        locations = await self._unpaywall_enrich(papers)
        conflicts = [*dedup.conflicts, *enrichment_conflicts]
        if self.config.require_open_access:
            oa_ids = {location.paper_id for location in locations if location.access_status == "open_location_found"}
            papers = [paper for paper in papers if paper.id in oa_ids]
        if len(papers) < self.config.minimum_corpus_size:
            raise ValueError(f"corpus below minimum size: {len(papers)} < {self.config.minimum_corpus_size}")
        report = self._dedup_report(len(raw_papers), len(papers), len(conflicts))
        return self._result(papers, queries, resolutions, locations, report, "live", conflicts)

    async def _search(self, provider: str, query_record: LiteratureQueryRecord) -> list[Paper]:
        query = SearchQuery(
            query=query_record.provider_text,
            from_year=query_record.date_from,
            to_year=query_record.date_to,
            limit=min(self.config.max_results_per_query, self.config.max_results_per_provider),
        )
        if provider == "openalex":
            tool = OpenAlexLiteratureSearch(self.transport, self.cache, force_refresh=self.config.cache_policy == "refresh")
        elif provider == "arxiv":
            if self.transport.client is None:
                self.transport.pace_seconds = max(self.transport.pace_seconds, 3.0)
            tool = ArxivLiteratureSearch(self.transport, self.cache, force_refresh=self.config.cache_policy == "refresh")
        else:
            raise ValueError(f"unsupported search provider for live acquisition: {provider}")
        papers = await tool.search(query)
        self.raw_records.setdefault(provider, []).extend(getattr(tool, "last_raw_records", []))
        self.events.extend(_drain_logs(tool))
        self._status(provider, "search", "complete", f"{len(papers)} result(s)")
        return papers

    async def _crossref_enrich(self, papers: list[Paper]) -> tuple[list[MetadataResolution], list[MetadataConflict]]:
        if "crossref" not in self.config.enrichment_providers:
            self._status("crossref", "metadata_enrichment", "skipped", "not enabled")
            return [], []
        resolver = CrossrefMetadataResolver(self.transport, self.cache, force_refresh=self.config.cache_policy == "refresh")
        resolutions: list[MetadataResolution] = []
        conflicts: list[MetadataConflict] = []
        for paper in papers:
            if not paper.doi:
                continue
            try:
                resolution = await resolver.resolve(MetadataResolveRequest(doi=paper.doi, title=paper.title, paper=paper))
            except Exception as exc:
                self.warnings.append(f"crossref enrichment failed for {paper.id}: {exc}")
                continue
            resolutions.append(resolution)
            conflicts.extend(resolution.conflicts)
            if getattr(resolver, "last_raw_record", None):
                self.raw_records["crossref"].append(resolver.last_raw_record)  # type: ignore[arg-type]
            self.events.extend(_drain_logs(resolver))
        self._status("crossref", "metadata_enrichment", "complete_with_warnings" if self.warnings else "complete", f"{len(resolutions)} resolution(s)")
        return resolutions, conflicts

    async def _unpaywall_enrich(self, papers: list[Paper]) -> list[FullTextLocation]:
        if "unpaywall" not in self.config.enrichment_providers:
            self._status("unpaywall", "open_access_enrichment", "skipped", "not enabled")
            return []
        email = os.getenv("UNPAYWALL_EMAIL")
        if not email:
            message = "UNPAYWALL_EMAIL is not configured; skipping Unpaywall enrichment"
            self.warnings.append(message)
            self._status("unpaywall", "open_access_enrichment", "not_configured", message, secret_configured=False)
            return []
        locator = UnpaywallFullTextLocator(self.transport, self.cache, email=email, require_email=False, force_refresh=self.config.cache_policy == "refresh")
        locations: list[FullTextLocation] = []
        for paper in papers:
            if not paper.doi:
                continue
            try:
                found = await locator.locate(paper)
            except Exception as exc:
                self.warnings.append(f"unpaywall enrichment failed for {paper.id}: {exc}")
                continue
            locations.extend(found)
            if getattr(locator, "last_raw_record", None):
                self.raw_records["unpaywall"].append(locator.last_raw_record)  # type: ignore[arg-type]
            self.events.extend(_drain_logs(locator))
        self._status("unpaywall", "open_access_enrichment", "complete_with_warnings" if self.warnings else "complete", f"{len(locations)} location(s)", secret_configured=True)
        return locations

    def _result(
        self,
        papers: list[Paper],
        queries: list[LiteratureQueryRecord],
        resolutions: list[MetadataResolution],
        locations: list[FullTextLocation],
        dedup_report: DeduplicationReport,
        mode: str,
        conflicts: list[MetadataConflict] | None = None,
        warnings: list[str] | None = None,
    ) -> ScholarlyCorpusResult:
        all_warnings = [*self.warnings, *(warnings or [])]
        corpus_manifest = CorpusManifest(
            corpus_id=f"corpus-{self.project.project_id}-{hash_papers(papers)[:10]}",
            project_id=self.project.project_id,
            mode=mode,  # type: ignore[arg-type]
            paper_count=len(papers),
            corpus_hash=hash_papers(papers),
            providers=sorted(set(self.config.search_providers + self.config.enrichment_providers)),
            limitations=all_warnings or ["Corpus is bounded by configured provider/query limits."],
            generated_at=datetime.now(UTC),
        )
        usage = summarize_usage(self.events, mode)
        return ScholarlyCorpusResult(
            papers=papers,
            raw_records=self.raw_records,
            query_records=queries,
            provider_events=self.events,
            provider_status=self.statuses,
            provider_usage=usage,
            metadata_resolutions=resolutions,
            metadata_conflicts=conflicts or [],
            full_text_locations=locations,
            deduplication_report=dedup_report,
            corpus_manifest=corpus_manifest,
            warnings=all_warnings,
        )

    def _dedup_report(self, input_count: int, output_count: int, conflict_count: int) -> DeduplicationReport:
        return DeduplicationReport(
            input_record_count=input_count,
            output_paper_count=output_count,
            exact_merges=max(0, input_count - output_count),
            conflict_count=conflict_count,
            merge_rationale=["Strong identifiers were preferred: DOI, arXiv ID, OpenAlex ID, then conservative title-author-year."],
        )

    def _status(self, provider: str, role: str, status: str, message: str | None = None, secret_configured: bool = False) -> None:
        self.statuses.append(ProviderStatus(
            provider=provider,
            role=role,  # type: ignore[arg-type]
            configured=True,
            enabled=status not in {"skipped", "not_configured"},
            status=status,  # type: ignore[arg-type]
            message=message,
            secret_configured=secret_configured,
        ))


def summarize_usage(events: list[ProviderRequestLog], mode: str) -> dict[str, ProviderUsage]:
    usage: dict[str, ProviderUsage] = {}
    for event in events:
        item = usage.setdefault(event.provider, ProviderUsage(provider=event.provider, network_mode=mode))  # type: ignore[arg-type]
        item.request_count += 1
        item.result_count += event.result_count
        if event.cache_hit:
            item.cache_hits += 1
        else:
            item.cache_misses += 1
        if event.status == "provider_error":
            item.failures += 1
        if event.status == "network_disabled":
            item.failures += 1
    return usage


def hash_papers(papers: list[Paper]) -> str:
    payload = json.dumps([paper.model_dump(mode="json") for paper in papers], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _drain_logs(provider: object) -> list[ProviderRequestLog]:
    logs = list(getattr(provider, "request_logs", []))
    if hasattr(provider, "request_logs"):
        provider.request_logs.clear()  # type: ignore[attr-defined]
    return logs
