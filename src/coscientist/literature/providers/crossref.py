from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from coscientist.literature.identifiers import normalize_doi
from coscientist.literature.normalization import paper_id_from_identifiers
from coscientist.literature.providers.base import CachedProvider
from coscientist.schemas.literature import (
    ExternalIdentifier,
    MetadataConflict,
    MetadataFieldProvenance,
    MetadataResolution,
    MetadataResolveRequest,
    Paper,
    PaperAuthor,
)


class CrossrefMetadataResolver(CachedProvider):
    provider_name = "crossref"
    base_url = "https://api.crossref.org/v1"

    def __init__(self, *args, mailto: str | None = None, user_agent: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mailto = mailto or os.getenv("CROSSREF_MAILTO")
        self.user_agent = user_agent or os.getenv("COSCIENTIST_USER_AGENT", "coscientist-mvp/0.1")
        self.last_raw_record: dict[str, Any] | None = None

    async def resolve(self, request: MetadataResolveRequest) -> MetadataResolution:
        doi = normalize_doi(request.doi or (request.paper.doi if request.paper else None))
        if not doi and request.title:
            return await self.search_bibliographic(request)
        if not doi:
            return self._empty("invalid_identifier", request.paper.id if request.paper else None)
        params = {"mailto": self.mailto} if self.mailto else None
        try:
            payload = await self.get_json_cached("resolve_doi", f"{self.base_url}/works/{doi}", params=params, normalized_query={"doi": doi})
        except Exception:
            return self._empty("provider_error", request.paper.id if request.paper else None)
        message = payload.get("message") or {}
        self.last_raw_record = message
        paper = self.normalize_work(message)
        conflicts = self.compare(request.paper, paper) if request.paper else []
        return MetadataResolution(
            paper_id=(request.paper.id if request.paper else paper.id),
            resolved_identifiers=paper.identifiers,
            normalized_metadata=paper,
            provenance=self._provenance(paper),
            conflicts=conflicts,
            status="conflicting" if conflicts else "resolved",
            resolver_name=self.provider_name,
            retrieved_at=datetime.now(UTC),
        )

    async def search_bibliographic(self, request: MetadataResolveRequest) -> MetadataResolution:
        params = {"query.bibliographic": request.title, "rows": 1}
        if self.mailto:
            params["mailto"] = self.mailto
        payload = await self.get_json_cached("search_bibliographic", f"{self.base_url}/works", params=params, normalized_query={"title": request.title})
        items = (payload.get("message") or {}).get("items") or []
        if not items:
            return self._empty("not_found", request.paper.id if request.paper else None)
        self.last_raw_record = items[0]
        paper = self.normalize_work(items[0])
        conflicts = self.compare(request.paper, paper) if request.paper else []
        return MetadataResolution(
            paper_id=(request.paper.id if request.paper else paper.id),
            resolved_identifiers=paper.identifiers,
            normalized_metadata=paper,
            provenance=self._provenance(paper),
            conflicts=conflicts,
            status="conflicting" if conflicts else "partially_resolved",
            resolver_name=self.provider_name,
            retrieved_at=datetime.now(UTC),
        )

    def normalize_work(self, record: dict[str, Any]) -> Paper:
        doi = normalize_doi(record.get("DOI"))
        identifiers = []
        if doi:
            identifiers.append(ExternalIdentifier(scheme="doi", value=str(record.get("DOI")), canonical_value=doi, source=self.provider_name))
        title = self._first(record.get("title")) or "Untitled Crossref work"
        authors = []
        for author in record.get("author") or []:
            parts = [author.get("given"), author.get("family")]
            name = " ".join(part for part in parts if part).strip()
            if name:
                authors.append(PaperAuthor(name=name, orcid=author.get("ORCID")))
        year, date = self._publication_date(record)
        identifiers_for_id = identifiers
        paper_id = paper_id_from_identifiers(self.provider_name, title, identifiers_for_id, year)
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            venue=self._first(record.get("container-title")),
            publication_year=year,
            publication_date=date,
            publication_type=record.get("type"),
            doi=doi,
            identifiers=identifiers,
            source_provider=self.provider_name,
            provider_record_id=doi,
            referenced_work_ids=[str(ref.get("DOI")) for ref in record.get("reference") or [] if ref.get("DOI")],
            source_metadata={
                "publisher": record.get("publisher"),
                "license": record.get("license") or [],
                "member": record.get("member"),
                "prefix": record.get("prefix"),
                "update_policy": record.get("update-policy"),
                "relation": record.get("relation"),
            },
        )

    def compare(self, existing: Paper | None, incoming: Paper) -> list[MetadataConflict]:
        if not existing:
            return []
        conflicts = []
        for field in ("title", "doi", "venue", "publication_year", "publication_type"):
            left = getattr(existing, field)
            right = getattr(incoming, field)
            if left and right and str(left).strip().lower() != str(right).strip().lower():
                conflicts.append(MetadataConflict(
                    field_name=field,
                    existing_value=str(left),
                    incoming_value=str(right),
                    existing_provider=existing.source_provider,
                    incoming_provider=self.provider_name,
                    notes=["Crossref metadata differs from existing paper record."],
                ))
        return conflicts

    def _provenance(self, paper: Paper) -> list[MetadataFieldProvenance]:
        now = datetime.now(UTC)
        return [
            MetadataFieldProvenance(field_name="title", value=paper.title, provider=self.provider_name, retrieved_at=now),
            MetadataFieldProvenance(field_name="doi", value=paper.doi or "", provider=self.provider_name, retrieved_at=now),
            MetadataFieldProvenance(field_name="venue", value=paper.venue or "", provider=self.provider_name, retrieved_at=now),
        ]

    @staticmethod
    def _publication_date(record: dict[str, Any]) -> tuple[int | None, str | None]:
        for key in ("published-print", "published-online", "published", "issued"):
            parts = ((record.get(key) or {}).get("date-parts") or [[]])[0]
            if parts:
                year = int(parts[0])
                date = "-".join(f"{int(part):02d}" for part in parts)
                return year, date
        return None, None

    @staticmethod
    def _first(value: Any) -> str | None:
        if isinstance(value, list) and value:
            return str(value[0])
        if value:
            return str(value)
        return None

    def _empty(self, status: str, paper_id: str | None) -> MetadataResolution:
        return MetadataResolution(
            paper_id=paper_id,
            status=status,  # type: ignore[arg-type]
            resolver_name=self.provider_name,
            retrieved_at=datetime.now(UTC),
        )
