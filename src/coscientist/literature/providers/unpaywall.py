from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from typing import Any

from coscientist.literature.identifiers import normalize_doi
from coscientist.literature.providers.base import CachedProvider, ProviderConfigurationError
from coscientist.schemas.literature import FullTextLocation, Paper


class UnpaywallFullTextLocator(CachedProvider):
    provider_name = "unpaywall"
    base_url = "https://api.unpaywall.org/v2"

    def __init__(self, *args, email: str | None = None, require_email: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.email = email or os.getenv("UNPAYWALL_EMAIL")
        if require_email and not self.email:
            raise ProviderConfigurationError("UNPAYWALL_EMAIL is required for live Unpaywall mode.")

    async def locate(self, paper: Paper) -> list[FullTextLocation]:
        doi = normalize_doi(paper.doi)
        if not doi:
            return []
        payload = await self.get_json_cached(
            "locate_by_doi",
            f"{self.base_url}/{doi}",
            params={"email": self.email},
            normalized_query={"doi": doi},
        )
        return self.normalize_locations(paper, payload)

    def normalize_locations(self, paper: Paper, record: dict[str, Any]) -> list[FullTextLocation]:
        locations = []
        raw_locations = record.get("oa_locations") or []
        best = record.get("best_oa_location")
        if best:
            raw_locations = [best, *[item for item in raw_locations if item != best]]
        if not record.get("is_oa") or not raw_locations:
            return [FullTextLocation(
                id=self._location_id(paper.id, "closed"),
                paper_id=paper.id,
                provider=self.provider_name,
                access_status="no_legal_open_copy",
                source_metadata={"oa_status": record.get("oa_status")},
                retrieved_at=datetime.now(UTC),
            )]
        seen: set[str] = set()
        for index, item in enumerate(raw_locations):
            url = item.get("url_for_pdf") or item.get("url_for_landing_page")
            marker = str(url or index)
            if marker in seen:
                continue
            seen.add(marker)
            locations.append(FullTextLocation(
                id=self._location_id(paper.id, marker),
                paper_id=paper.id,
                provider=self.provider_name,
                landing_page_url=item.get("url_for_landing_page"),
                document_url=item.get("url_for_pdf"),
                content_type="application/pdf" if item.get("url_for_pdf") else "text/html",
                access_status="open_location_found",
                host_type=item.get("host_type"),
                version=item.get("version"),
                license=item.get("license"),
                is_best=index == 0,
                source_metadata={
                    "oa_status": record.get("oa_status"),
                    "repository_institution": item.get("repository_institution"),
                    "evidence": item.get("evidence"),
                },
                retrieved_at=datetime.now(UTC),
            ))
        return locations

    @staticmethod
    def _location_id(paper_id: str, value: str) -> str:
        return "loc-unpaywall-" + hashlib.sha1(f"{paper_id}:{value}".encode("utf-8")).hexdigest()[:12]
