from __future__ import annotations

from typing import Protocol

from coscientist.schemas.literature import (
    FullTextLocation,
    MetadataResolution,
    MetadataResolveRequest,
    Paper,
    RetrievedDocument,
    SearchQuery,
)


class LiteratureSearchTool(Protocol):
    async def search(self, query: SearchQuery) -> list[Paper]:
        ...


class MetadataResolver(Protocol):
    async def resolve(self, request: MetadataResolveRequest) -> MetadataResolution:
        ...


class FullTextLocator(Protocol):
    async def locate(self, paper: Paper) -> list[FullTextLocation]:
        ...


class DocumentRetriever(Protocol):
    async def retrieve(self, location: FullTextLocation) -> RetrievedDocument:
        ...
