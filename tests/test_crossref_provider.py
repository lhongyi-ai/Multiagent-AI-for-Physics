from __future__ import annotations

import asyncio

import httpx

from coscientist.literature.http import AsyncHttpTransport
from coscientist.literature.providers.crossref import CrossrefMetadataResolver
from coscientist.schemas.literature import MetadataResolveRequest, Paper


def _resolver(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, client=client, max_retries=0)
    return CrossrefMetadataResolver(transport, mailto="test@example.com", user_agent="coscientist-test")


def test_crossref_doi_resolution_and_normalization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "mailto=test%40example.com" in str(request.url)
        return httpx.Response(200, json={"message": {
            "DOI": "10.1234/ABC",
            "title": ["A Crossref Paper"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "container-title": ["Journal"],
            "published-online": {"date-parts": [[2024, 5, 1]]},
            "type": "journal-article",
            "license": [{"URL": "https://license.test"}],
            "reference": [{"DOI": "10.1/ref"}],
        }})

    resolution = asyncio.run(_resolver(handler).resolve(MetadataResolveRequest(doi="https://doi.org/10.1234/ABC")))
    assert resolution.status == "resolved"
    assert resolution.normalized_metadata.doi == "10.1234/abc"  # type: ignore[union-attr]
    assert resolution.normalized_metadata.publication_date == "2024-05-01"  # type: ignore[union-attr]


def test_crossref_bibliographic_search_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"items": [{"DOI": "10.1/x", "title": ["Found"]}]}})

    resolution = asyncio.run(_resolver(handler).resolve(MetadataResolveRequest(title="Found")))
    assert resolution.status == "partially_resolved"


def test_crossref_metadata_conflict_reporting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"DOI": "10.1/x", "title": ["Incoming"]}})

    existing = Paper(id="p", title="Existing", doi="10.1/x", source_provider="openalex")
    resolution = asyncio.run(_resolver(handler).resolve(MetadataResolveRequest(doi="10.1/x", paper=existing)))
    assert resolution.status == "conflicting"
    assert resolution.conflicts[0].field_name == "title"
