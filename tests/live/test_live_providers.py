from __future__ import annotations

import asyncio
import os

import pytest

from coscientist.literature.http import AsyncHttpTransport
from coscientist.literature.providers.arxiv import ArxivLiteratureSearch
from coscientist.literature.providers.crossref import CrossrefMetadataResolver
from coscientist.literature.providers.openalex import OpenAlexLiteratureSearch
from coscientist.literature.providers.unpaywall import UnpaywallFullTextLocator
from coscientist.schemas.literature import MetadataResolveRequest, Paper, SearchQuery


pytestmark = pytest.mark.live


def _live_transport() -> AsyncHttpTransport:
    return AsyncHttpTransport(allow_live_network=True, timeout_seconds=15, max_retries=1, user_agent="coscientist-live-test/0.1")


@pytest.mark.skipif(os.getenv("RUN_LIVE_API_TESTS") != "1" or not os.getenv("OPENALEX_API_KEY"), reason="requires explicit live opt-in and OPENALEX_API_KEY")
def test_live_openalex_search_smoke() -> None:
    provider = OpenAlexLiteratureSearch(_live_transport())
    papers = asyncio.run(provider.search(SearchQuery(query="superconductivity", limit=1)))
    assert len(papers) <= 1


@pytest.mark.skipif(os.getenv("RUN_LIVE_API_TESTS") != "1", reason="requires explicit live opt-in")
def test_live_crossref_resolve_smoke() -> None:
    provider = CrossrefMetadataResolver(_live_transport(), mailto=os.getenv("CROSSREF_MAILTO"))
    resolution = asyncio.run(provider.resolve(MetadataResolveRequest(doi="10.1038/nature12373")))
    assert resolution.status in {"resolved", "partially_resolved", "conflicting", "not_found", "provider_error"}


@pytest.mark.skipif(os.getenv("RUN_LIVE_API_TESTS") != "1" or not os.getenv("UNPAYWALL_EMAIL"), reason="requires explicit live opt-in and UNPAYWALL_EMAIL")
def test_live_unpaywall_locate_smoke() -> None:
    provider = UnpaywallFullTextLocator(_live_transport())
    locations = asyncio.run(provider.locate(Paper(id="p", title="P", doi="10.1038/nature12373", source_provider="live-test")))
    assert isinstance(locations, list)


@pytest.mark.skipif(os.getenv("RUN_LIVE_API_TESTS") != "1", reason="requires explicit live opt-in")
def test_live_arxiv_search_smoke() -> None:
    provider = ArxivLiteratureSearch(_live_transport())
    papers = asyncio.run(provider.search(SearchQuery(query="electron", limit=1)))
    assert len(papers) <= 1
