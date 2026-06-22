from __future__ import annotations

import asyncio

import httpx
import pytest

from coscientist.literature.http import AsyncHttpTransport
from coscientist.literature.providers.base import ProviderConfigurationError
from coscientist.literature.providers.unpaywall import UnpaywallFullTextLocator
from coscientist.schemas.literature import Paper


def _locator(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, client=client, max_retries=0)
    return UnpaywallFullTextLocator(transport, email="test@example.com")


def test_unpaywall_requires_email_for_live_mode() -> None:
    with pytest.raises(ProviderConfigurationError):
        UnpaywallFullTextLocator(AsyncHttpTransport(allow_live_network=True), email=None)


def test_unpaywall_open_access_best_and_multiple_locations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": {
                "url_for_landing_page": "https://repo.test/paper",
                "url_for_pdf": "https://repo.test/paper.pdf",
                "host_type": "repository",
                "version": "acceptedVersion",
                "license": "cc-by",
            },
            "oa_locations": [
                {"url_for_landing_page": "https://publisher.test/paper", "host_type": "publisher", "version": "publishedVersion"}
            ],
        })

    locations = asyncio.run(_locator(handler).locate(Paper(id="p", title="P", doi="10.1/x", source_provider="test")))
    assert len(locations) == 2
    assert locations[0].is_best
    assert locations[0].document_url == "https://repo.test/paper.pdf"
    assert locations[1].host_type == "publisher"


def test_unpaywall_closed_article_returns_clean_no_copy_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"is_oa": False, "oa_status": "closed"})

    locations = asyncio.run(_locator(handler).locate(Paper(id="p", title="P", doi="10.1/x", source_provider="test")))
    assert locations[0].access_status == "no_legal_open_copy"
