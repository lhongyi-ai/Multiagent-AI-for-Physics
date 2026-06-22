from __future__ import annotations

import asyncio

import httpx
import pytest

from coscientist.literature.http import AsyncHttpTransport, ProviderTransportError
from coscientist.literature.providers.base import ProviderConfigurationError
from coscientist.literature.providers.openalex import OpenAlexLiteratureSearch
from coscientist.schemas.literature import SearchQuery


def _provider(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, client=client, max_retries=0)
    return OpenAlexLiteratureSearch(transport, api_key="test-key")


def test_openalex_requires_api_key_for_live_mode() -> None:
    with pytest.raises(ProviderConfigurationError):
        OpenAlexLiteratureSearch(AsyncHttpTransport(allow_live_network=True), api_key=None)


def test_openalex_query_construction_year_filter_and_limit() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"results": []})

    provider = _provider(handler)
    result = asyncio.run(provider.search(SearchQuery(query="hydride", from_year=2020, to_year=2024, limit=5)))
    assert result == []
    assert "search=hydride" in captured["url"]
    assert "per_page=5" in captured["url"]
    assert "publication_year%3A2020-2024" in captured["url"]


def test_openalex_response_normalization_missing_optional_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1", "title": "Paper", "publication_year": 2024}]})

    papers = asyncio.run(_provider(handler).search(SearchQuery(query="paper", limit=1)))
    assert papers[0].provider_record_id == "https://openalex.org/W1"
    assert papers[0].publication_year == 2024


def test_openalex_rate_limit_response_raises_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow"})

    with pytest.raises(ProviderTransportError):
        asyncio.run(_provider(handler).search(SearchQuery(query="x", limit=1)))
