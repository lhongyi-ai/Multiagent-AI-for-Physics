from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha1

from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.scholarly import LiteratureQueryRecord, ProjectLiteratureConfig


def plan_literature_queries(project: ResearchProjectSpec, config: ProjectLiteratureConfig) -> list[LiteratureQueryRecord]:
    specs = list(config.queries)
    if not specs and config.query_generation == "deterministic":
        specs = _deterministic_queries(project, config)
    records: list[LiteratureQueryRecord] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs[: config.max_queries]:
        providers = spec.providers or config.search_providers
        for provider in providers:
            provider_text = _provider_query(spec.text, provider)
            key = (provider, provider_text.lower())
            if key in seen:
                continue
            seen.add(key)
            records.append(LiteratureQueryRecord(
                query_id=_query_id(provider, provider_text),
                original_text=spec.text,
                provider=provider,
                provider_text=provider_text,
                origin=spec.origin,
                filters=spec.filters,
                date_from=config.date_from,
                date_to=config.date_to,
                created_at=datetime.now(UTC),
            ))
    return records


def _deterministic_queries(project: ResearchProjectSpec, config: ProjectLiteratureConfig):
    from coscientist.schemas.scholarly import LiteratureQuerySpec

    base = project.research_question.strip().rstrip("?")
    terms = [project.title, base]
    if project.known_observations:
        terms.append(project.known_observations[0])
    return [
        LiteratureQuerySpec(text=item, providers=config.search_providers, origin="deterministic_rule")
        for item in terms[: config.max_queries]
    ]


def _provider_query(text: str, provider: str) -> str:
    cleaned = " ".join(text.split())
    if provider == "arxiv":
        return cleaned.replace(" synthesis ", " ")
    return cleaned


def _query_id(provider: str, text: str) -> str:
    return f"query-{provider}-{sha1(text.encode('utf-8')).hexdigest()[:10]}"
