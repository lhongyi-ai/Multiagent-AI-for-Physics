from __future__ import annotations

from coscientist.literature.identifiers import arxiv_work_id, normalize_arxiv_id, normalize_doi
from coscientist.literature.normalization import PaperDeduplicator
from coscientist.schemas.literature import ExternalIdentifier, Paper


def test_doi_variants_normalize_to_one_identifier() -> None:
    assert normalize_doi("https://doi.org/10.1038/NATURE12373") == "10.1038/nature12373"
    assert normalize_doi("doi:10.1038/nature12373.") == "10.1038/nature12373"


def test_arxiv_versions_group_without_losing_exact_version() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2101.12345v2") == "2101.12345v2"
    assert arxiv_work_id("2101.12345v2") == "2101.12345"


def test_dedup_merges_doi_duplicates_and_preserves_conflict() -> None:
    ident = ExternalIdentifier(scheme="doi", value="10.1/ABC", canonical_value="10.1/abc", source="a")
    left = Paper(id="p1", title="A title", doi="10.1/abc", identifiers=[ident], source_provider="openalex")
    right = Paper(
        id="p2",
        title="Different title",
        doi="https://doi.org/10.1/ABC",
        identifiers=[ident.model_copy(update={"source": "crossref"})],
        source_provider="crossref",
    )
    result = PaperDeduplicator().merge([left, right])
    assert len(result.papers) == 1
    assert result.conflicts


def test_similar_titles_do_not_merge_without_identifier_or_author_year() -> None:
    first = Paper(id="p1", title="High pressure hydrides", source_provider="openalex")
    second = Paper(id="p2", title="High-pressure hydride", source_provider="arxiv")
    result = PaperDeduplicator().merge([first, second])
    assert len(result.papers) == 2
