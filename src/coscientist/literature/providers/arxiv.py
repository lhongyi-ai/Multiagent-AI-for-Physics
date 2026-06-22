from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any

from coscientist.literature.identifiers import normalize_arxiv_id, normalize_doi
from coscientist.literature.normalization import paper_id_from_identifiers
from coscientist.literature.providers.base import CachedProvider
from coscientist.schemas.literature import ExternalIdentifier, FullTextLocation, Paper, PaperAuthor, SearchQuery


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivLiteratureSearch(CachedProvider):
    provider_name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    async def search(self, query: SearchQuery) -> list[Paper]:
        params = {
            "search_query": f"all:{query.query}",
            "start": 0,
            "max_results": query.limit,
        }
        text = await self._get_atom("search", params, query.model_dump())
        return parse_arxiv_feed(text)[: query.limit]

    async def get_record(self, arxiv_id: str) -> Paper | None:
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized:
            return None
        text = await self._get_atom("get_by_id", {"id_list": normalized, "max_results": 1}, {"arxiv_id": normalized})
        papers = parse_arxiv_feed(text)
        return papers[0] if papers else None

    async def _get_atom(self, operation: str, params: dict[str, Any], normalized_query: dict[str, Any]) -> str:
        request = {"url": self.base_url, "params": params}
        cache_key = self.cache.key(self.provider_name, operation, request, "atom-v1") if self.cache else ""
        if self.cache and not self.force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return str(cached)
        result = await self.transport.get_text(self.provider_name, operation, self.base_url, params=params, normalized_query=normalized_query)
        self.request_logs.append(result.request_log)
        if self.cache:
            self.cache.set(cache_key, result.payload)
        return str(result.payload)


class ArxivFullTextLocator:
    provider_name = "arxiv"

    async def locate(self, paper: Paper) -> list[FullTextLocation]:
        arxiv_id = None
        for identifier in paper.identifiers:
            if identifier.scheme == "arxiv":
                arxiv_id = identifier.canonical_value
                break
        if not arxiv_id:
            return []
        work_id = normalize_arxiv_id(arxiv_id)
        if not work_id:
            return []
        abstract_url = f"https://arxiv.org/abs/{work_id}"
        pdf_url = f"https://arxiv.org/pdf/{work_id}.pdf"
        return [FullTextLocation(
            id="loc-arxiv-" + hashlib.sha1(work_id.encode("utf-8")).hexdigest()[:12],
            paper_id=paper.id,
            provider=self.provider_name,
            landing_page_url=abstract_url,
            document_url=pdf_url,
            content_type="application/pdf",
            access_status="open_location_found",
            host_type="repository",
            version=_version(work_id),
            license=None,
            is_best=True,
            retrieved_at=datetime.now(UTC),
        )]


def parse_arxiv_feed(text: str) -> list[Paper]:
    root = ET.fromstring(text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        title = _clean(entry.findtext(f"{ATOM}title") or "Untitled arXiv record")
        summary = _clean(entry.findtext(f"{ATOM}summary") or "")
        entry_id = entry.findtext(f"{ATOM}id") or ""
        arxiv_id = normalize_arxiv_id(entry_id)
        identifiers: list[ExternalIdentifier] = []
        if arxiv_id:
            identifiers.append(ExternalIdentifier(scheme="arxiv", value=entry_id, canonical_value=arxiv_id, source="arxiv"))
        doi_el = entry.find(f"{ARXIV}doi")
        doi = normalize_doi(doi_el.text if doi_el is not None else None)
        if doi:
            identifiers.append(ExternalIdentifier(scheme="doi", value=doi, canonical_value=doi, source="arxiv"))
        authors = [
            PaperAuthor(name=_clean(author.findtext(f"{ATOM}name") or "Unknown arXiv author"))
            for author in entry.findall(f"{ATOM}author")
        ]
        categories = [cat.attrib.get("term") for cat in entry.findall(f"{ATOM}category") if cat.attrib.get("term")]
        journal_ref = entry.find(f"{ARXIV}journal_ref")
        published = entry.findtext(f"{ATOM}published")
        year = int(published[:4]) if published and published[:4].isdigit() else None
        paper_id = paper_id_from_identifiers("arxiv", title, identifiers, year)
        papers.append(Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=summary,
            venue=journal_ref.text if journal_ref is not None else "arXiv",
            publication_year=year,
            publication_date=published[:10] if published else None,
            publication_type="preprint",
            doi=doi,
            identifiers=identifiers,
            source_provider="arxiv",
            provider_record_id=arxiv_id,
            source_metadata={
                "updated": entry.findtext(f"{ATOM}updated"),
                "categories": categories,
                "journal_ref": journal_ref.text if journal_ref is not None else None,
                "version": _version(arxiv_id or ""),
            },
        ))
    return papers


def _clean(value: str) -> str:
    return " ".join(value.split())


def _version(value: str) -> str | None:
    match = re.search(r"(v\d+)$", value)
    return match.group(1) if match else None
