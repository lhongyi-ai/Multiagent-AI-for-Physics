from __future__ import annotations

import asyncio

import httpx

from coscientist.literature.http import AsyncHttpTransport
from coscientist.literature.providers.arxiv import ArxivFullTextLocator, ArxivLiteratureSearch
from coscientist.schemas.literature import SearchQuery


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.12345v2</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title> A Test Preprint </title>
    <summary> Abstract text. </summary>
    <author><name>Grace Hopper</name></author>
    <category term="cond-mat.supr-con"/>
    <arxiv:doi>10.1234/test</arxiv:doi>
    <arxiv:journal_ref>Journal Ref</arxiv:journal_ref>
  </entry>
</feed>
"""


def _search(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, client=client, max_retries=0)
    return ArxivLiteratureSearch(transport)


def test_arxiv_query_construction_and_result_parsing() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, text=ARXIV_XML)

    papers = asyncio.run(_search(handler).search(SearchQuery(query="electron", limit=1)))
    assert "search_query=all%3Aelectron" in captured["url"]
    assert papers[0].title == "A Test Preprint"
    assert papers[0].doi == "10.1234/test"
    assert papers[0].source_metadata["categories"] == ["cond-mat.supr-con"]


def test_arxiv_id_lookup_and_full_text_location() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "id_list=2101.12345v2" in str(request.url)
        return httpx.Response(200, text=ARXIV_XML)

    paper = asyncio.run(_search(handler).get_record("2101.12345v2"))
    locations = asyncio.run(ArxivFullTextLocator().locate(paper))
    assert locations[0].landing_page_url == "https://arxiv.org/abs/2101.12345v2"
    assert locations[0].document_url == "https://arxiv.org/pdf/2101.12345v2.pdf"


def test_arxiv_zero_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')

    assert asyncio.run(_search(handler).search(SearchQuery(query="none", limit=1))) == []
