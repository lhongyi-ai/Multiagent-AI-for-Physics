from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from coscientist.cli import main
from coscientist.literature.http import AsyncHttpTransport, NetworkDisabledError
from coscientist.literature.query_planner import plan_literature_queries
from coscientist.literature.scholarly import ScholarlyLiteratureOrchestrator
from coscientist.pilot.artifacts import validate_v1_artifacts
from coscientist.pilot.project_io import load_project_spec
from coscientist.pilot.runner import run_pilot_project_sync
from coscientist.schemas.scholarly import LiteratureQuerySpec, ProjectLiteratureConfig


PROJECT = "research-projects/interdisciplinary_fixture/project.yaml"

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-03T00:00:00Z</updated>
    <published>2024-01-02T00:00:00Z</published>
    <title>Mock arXiv retrieval paper</title>
    <summary>Mock preprint abstract.</summary>
    <author><name>Mock Author</name></author>
    <arxiv:doi>10.5555/arxiv-demo</arxiv:doi>
  </entry>
</feed>
"""


def test_project_literature_config_validation() -> None:
    with pytest.raises(ValidationError):
        ProjectLiteratureConfig(mode="live", search_providers=["crossref"])
    with pytest.raises(ValidationError):
        ProjectLiteratureConfig(mode="live", enrichment_providers=["openalex"])
    with pytest.raises(ValidationError):
        ProjectLiteratureConfig(mode="live", query_generation="explicit", queries=[])
    with pytest.raises(ValidationError):
        ProjectLiteratureConfig(mode="fixture", date_from=2025, date_to=2024)


def test_query_planner_expands_provider_specific_records() -> None:
    project = load_project_spec(PROJECT)
    config = ProjectLiteratureConfig(
        mode="live",
        queries=[LiteratureQuerySpec(text="calcium synthesis retrieval", providers=["openalex", "arxiv"])],
        search_providers=["openalex", "arxiv"],
        enrichment_providers=["crossref"],
        max_queries=2,
        date_from=2020,
        date_to=2024,
    )
    records = plan_literature_queries(project, config)
    assert [record.provider for record in records] == ["openalex", "arxiv"]
    assert records[1].provider_text == "calcium retrieval"
    assert records[0].date_from == 2020


def test_live_mode_requires_explicit_network_unless_dry_run(tmp_path: Path) -> None:
    project = load_project_spec(PROJECT)
    config = ProjectLiteratureConfig(
        mode="live",
        queries=[LiteratureQuerySpec(text="safe query", providers=["openalex"])],
        search_providers=["openalex"],
        enrichment_providers=[],
    )
    orchestrator = ScholarlyLiteratureOrchestrator(
        project=project,
        literature_config=config,
        cache_dir=tmp_path / "cache",
        allow_live_network=False,
    )
    with pytest.raises(NetworkDisabledError):
        asyncio.run(orchestrator.acquire())
    dry = orchestrator.dry_run()
    assert dry.query_records[0].provider == "openalex"
    assert dry.provider_events == []


def test_scholarly_orchestrator_uses_mock_transport_for_all_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNPAYWALL_EMAIL", "fixture@example.com")
    project = load_project_spec(PROJECT)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        url = str(request.url)
        if "api.openalex.org/works" in url:
            return httpx.Response(200, json={"results": [{
                "id": "https://openalex.org/W123",
                "title": "Mock OpenAlex paper",
                "publication_year": 2024,
                "doi": "https://doi.org/10.1234/openalex-demo",
                "authorships": [{"author": {"display_name": "Ada Mock"}}],
                "primary_location": {"source": {"display_name": "Mock Journal"}},
                "open_access": {"is_oa": True},
            }]})
        if "export.arxiv.org/api/query" in url:
            return httpx.Response(200, text=ARXIV_XML)
        if "api.crossref.org/v1/works" in url:
            doi = str(request.url.path).rsplit("/", 1)[-1]
            return httpx.Response(200, json={"message": {
                "DOI": doi,
                "title": ["Resolved " + doi],
                "container-title": ["Resolved Venue"],
                "published-online": {"date-parts": [[2024, 1, 1]]},
                "type": "journal-article",
            }})
        if "api.unpaywall.org/v2" in url:
            return httpx.Response(200, json={
                "is_oa": True,
                "oa_status": "green",
                "best_oa_location": {
                    "url_for_landing_page": "https://repo.example/paper",
                    "url_for_pdf": "https://repo.example/paper.pdf",
                    "host_type": "repository",
                    "version": "acceptedVersion",
                },
                "oa_locations": [],
            })
        raise AssertionError(f"unexpected URL: {url}")

    transport = AsyncHttpTransport(
        allow_live_network=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    config = ProjectLiteratureConfig(
        mode="live",
        queries=[LiteratureQuerySpec(text="mock retrieval paper", providers=["openalex", "arxiv"])],
        search_providers=["openalex", "arxiv"],
        enrichment_providers=["crossref", "unpaywall"],
        max_total_results=5,
        cache_policy="refresh",
    )
    result = asyncio.run(ScholarlyLiteratureOrchestrator(
        project=project,
        literature_config=config,
        cache_dir=tmp_path / "cache",
        allow_live_network=True,
        transport=transport,
    ).acquire())
    assert {paper.source_provider for paper in result.papers} == {"openalex", "arxiv"}
    assert len(result.metadata_resolutions) == 2
    assert len(result.full_text_locations) == 2
    assert result.raw_records["openalex"]
    assert result.raw_records["arxiv"]
    assert all("fixture@example.com" not in json.dumps(event.model_dump(mode="json")) for event in result.provider_events)
    assert len(calls) == 6


def test_run_project_writes_new_literature_artifacts(tmp_path: Path) -> None:
    run_dir = run_pilot_project_sync(PROJECT, runs_dir=tmp_path, run_id="pilot")
    assert validate_v1_artifacts(run_dir) == []
    assert (run_dir / "resolved_configuration.json").exists()
    assert (run_dir / "corpus_manifest.json").exists()
    assert json.loads((run_dir / "deduplication_report.json").read_text())["output_paper_count"] == 4


def test_acquire_literature_dry_run_cli_makes_no_network(tmp_path: Path) -> None:
    code = main([
        "acquire-literature",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--run-id",
        "dry",
        "--literature-mode",
        "live",
        "--search-providers",
        "openalex",
        "arxiv",
        "--enrichment-providers",
        "crossref",
        "--dry-run",
    ])
    assert code == 0
    assert (tmp_path / "dry" / "literature_search_events.jsonl").read_text() == ""
    assert json.loads((tmp_path / "dry" / "corpus_manifest.json").read_text())["paper_count"] == 0


def test_acquire_literature_existing_corpus_cli(tmp_path: Path) -> None:
    corpus = str(Path(PROJECT).parent / "corpus.jsonl")
    code = main([
        "acquire-literature",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--run-id",
        "existing",
        "--corpus",
        corpus,
    ])
    assert code == 0
    assert len((tmp_path / "existing" / "corpus.jsonl").read_text().splitlines()) == 4
