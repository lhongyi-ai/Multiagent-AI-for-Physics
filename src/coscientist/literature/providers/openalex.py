from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from coscientist.literature.identifiers import normalize_doi
from coscientist.literature.normalization import paper_id_from_identifiers
from coscientist.literature.providers.base import CachedProvider, ProviderConfigurationError
from coscientist.schemas.literature import ExternalIdentifier, Paper, PaperAuthor, SearchQuery


class OpenAlexLiteratureSearch(CachedProvider):
    provider_name = "openalex"
    base_url = "https://api.openalex.org"

    def __init__(self, *args, api_key: str | None = None, require_api_key: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        if require_api_key and not self.api_key:
            raise ProviderConfigurationError("OPENALEX_API_KEY is required for live OpenAlex mode.")

    async def search(self, query: SearchQuery) -> list[Paper]:
        params: dict[str, Any] = {"search": query.query, "per_page": min(query.limit, 100)}
        filters = []
        if query.from_year or query.to_year:
            start = query.from_year or 0
            end = query.to_year or datetime.now(UTC).year
            filters.append(f"publication_year:{start}-{end}")
        if filters:
            params["filter"] = ",".join(filters)
        if self.api_key:
            params["api_key"] = self.api_key
        payload = await self.get_json_cached("search_works", f"{self.base_url}/works", params=params, normalized_query=query.model_dump())
        return [self.normalize_work(item) for item in payload.get("results", [])[: query.limit]]

    async def get_work(self, openalex_id: str) -> Paper | None:
        params = {"api_key": self.api_key} if self.api_key else None
        payload = await self.get_json_cached("get_work", f"{self.base_url}/works/{openalex_id}", params=params, normalized_query={"id": openalex_id})
        return self.normalize_work(payload) if payload else None

    async def get_work_by_doi(self, doi: str) -> Paper | None:
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        params = {"api_key": self.api_key} if self.api_key else None
        payload = await self.get_json_cached("get_work_by_doi", f"{self.base_url}/works/doi:{normalized}", params=params, normalized_query={"doi": normalized})
        return self.normalize_work(payload) if payload else None

    def normalize_work(self, record: dict[str, Any]) -> Paper:
        identifiers = []
        openalex_id = record.get("id")
        if openalex_id:
            identifiers.append(ExternalIdentifier(scheme="openalex", value=str(openalex_id), canonical_value=str(openalex_id).rsplit("/", 1)[-1], source=self.provider_name))
        doi = normalize_doi(record.get("doi"))
        if doi:
            identifiers.append(ExternalIdentifier(scheme="doi", value=str(record.get("doi")), canonical_value=doi, source=self.provider_name))
        title = str(record.get("title") or record.get("display_name") or "Untitled OpenAlex work")
        authors = []
        for authorship in record.get("authorships") or []:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if not name:
                continue
            institutions = [
                inst.get("display_name")
                for inst in authorship.get("institutions") or []
                if inst.get("display_name")
            ]
            authors.append(PaperAuthor(name=str(name), orcid=author.get("orcid"), institutions=institutions))
        primary = record.get("primary_location") or {}
        source = primary.get("source") or {}
        paper_id = paper_id_from_identifiers(self.provider_name, title, identifiers, record.get("publication_year"))
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=record.get("abstract"),
            venue=source.get("display_name"),
            publication_year=record.get("publication_year"),
            publication_date=record.get("publication_date"),
            publication_type=record.get("type"),
            doi=doi,
            identifiers=identifiers,
            source_provider=self.provider_name,
            provider_record_id=str(openalex_id) if openalex_id else None,
            cited_by_count=record.get("cited_by_count"),
            referenced_work_ids=[str(item) for item in record.get("referenced_works") or []],
            open_access=record.get("open_access") or {},
            retraction_status=self._retraction_status(record),
            source_metadata={
                "primary_location": primary,
                "locations_count": record.get("locations_count"),
                "updated_date": record.get("updated_date"),
            },
        )

    @staticmethod
    def _retraction_status(record: dict[str, Any]) -> str | None:
        if record.get("is_retracted"):
            return "retracted"
        if record.get("retraction") or record.get("correction"):
            return "has_update_metadata"
        return None
