from __future__ import annotations

import hashlib
from dataclasses import dataclass

from coscientist.literature.identifiers import arxiv_work_id, normalize_doi, normalize_title
from coscientist.schemas.literature import ExternalIdentifier, MetadataConflict, Paper


def paper_id_from_identifiers(provider: str, title: str, identifiers: list[ExternalIdentifier], year: int | None = None) -> str:
    for scheme in ("doi", "arxiv", "openalex"):
        for identifier in identifiers:
            if identifier.scheme == scheme:
                return f"paper-{scheme}-{_slug(identifier.canonical_value)}"
    seed = f"{provider}:{normalize_title(title)}:{year or ''}"
    return "paper-title-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


@dataclass
class DeduplicationResult:
    papers: list[Paper]
    conflicts: list[MetadataConflict]


class PaperDeduplicator:
    def merge(self, papers: list[Paper]) -> DeduplicationResult:
        merged: list[Paper] = []
        conflicts: list[MetadataConflict] = []
        groups: dict[str, Paper] = {}
        for paper in papers:
            key = self._key(paper)
            if key not in groups:
                groups[key] = paper
                merged.append(paper)
                continue
            existing = groups[key]
            combined, new_conflicts = self._merge_pair(existing, paper)
            groups[key] = combined
            merged[merged.index(existing)] = combined
            conflicts.extend(new_conflicts)
        return DeduplicationResult(merged, conflicts)

    def _key(self, paper: Paper) -> str:
        doi = normalize_doi(paper.doi)
        if doi:
            return f"doi:{doi}"
        for identifier in paper.identifiers:
            if identifier.scheme == "doi":
                return f"doi:{normalize_doi(identifier.canonical_value)}"
        for identifier in paper.identifiers:
            if identifier.scheme == "arxiv":
                return f"arxiv:{arxiv_work_id(identifier.canonical_value)}"
        for identifier in paper.identifiers:
            if identifier.scheme == "openalex":
                return f"openalex:{identifier.canonical_value}"
        if paper.publication_year and paper.authors:
            first_author = normalize_title(paper.authors[0].name)
            return f"title-author:{normalize_title(paper.title)}:{paper.publication_year}:{first_author}"
        return f"unique:{paper.id}"

    def _merge_pair(self, existing: Paper, incoming: Paper) -> tuple[Paper, list[MetadataConflict]]:
        conflicts: list[MetadataConflict] = []
        updates = {}
        for field in ("title", "venue", "publication_date", "publication_type", "doi"):
            existing_value = getattr(existing, field)
            incoming_value = getattr(incoming, field)
            if not existing_value and incoming_value:
                updates[field] = incoming_value
            elif existing_value and incoming_value and str(existing_value).strip() != str(incoming_value).strip():
                conflicts.append(MetadataConflict(
                    field_name=field,
                    existing_value=str(existing_value),
                    incoming_value=str(incoming_value),
                    existing_provider=existing.source_provider,
                    incoming_provider=incoming.source_provider,
                    notes=["Conservative merge retained existing value and recorded conflict."],
                ))
        identifiers = self._merge_identifiers(existing.identifiers, incoming.identifiers)
        authors = existing.authors or incoming.authors
        source_metadata = {
            **existing.source_metadata,
            f"{incoming.source_provider}_metadata": incoming.source_metadata,
        }
        return existing.model_copy(update={
            **updates,
            "authors": authors,
            "identifiers": identifiers,
            "referenced_work_ids": sorted(set(existing.referenced_work_ids + incoming.referenced_work_ids)),
            "source_metadata": source_metadata,
        }), conflicts

    @staticmethod
    def _merge_identifiers(left: list[ExternalIdentifier], right: list[ExternalIdentifier]) -> list[ExternalIdentifier]:
        seen = {(item.scheme, item.canonical_value) for item in left}
        merged = list(left)
        for item in right:
            key = (item.scheme, item.canonical_value)
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged


def _slug(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
