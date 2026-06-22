from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from coscientist.config import load_config, load_research_goal
from coscientist.literature.pipeline import build_literature_pipeline
from coscientist.orchestration.workflow import run_workflow
from coscientist.providers.base import ProviderError
from coscientist.providers.mock import MockProvider
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.schemas.literature import ExternalIdentifier, MetadataResolveRequest, Paper, SearchQuery
from coscientist.storage.local_store import LocalStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coscientist")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run", help="Run the co-scientist MVP workflow.")
    run_parser.add_argument("goal", help="Path to a research goal YAML file.")
    run_parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    run_parser.add_argument("--config", default="config/default.yaml")
    run_parser.add_argument("--runs-dir", default="runs")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--literature-providers", nargs="*", default=None)
    run_parser.add_argument("--metadata-resolver", default=None)
    run_parser.add_argument("--full-text-locators", nargs="*", default=None)
    run_parser.add_argument("--live-network", action="store_true")
    run_parser.add_argument("--force-refresh", action="store_true")

    validate_parser = subcommands.add_parser("validate", help="Validate a research goal YAML file.")
    validate_parser.add_argument("goal", help="Path to a research goal YAML file.")

    search_parser = subcommands.add_parser("search-literature", help="Search configured literature providers.")
    search_parser.add_argument("query")
    search_parser.add_argument("--providers", nargs="+", default=["mock"])
    search_parser.add_argument("--config", default="config/default.yaml")
    search_parser.add_argument("--live-network", action="store_true")
    search_parser.add_argument("--limit", type=int, default=10)

    resolve_parser = subcommands.add_parser("resolve-doi", help="Resolve DOI metadata.")
    resolve_parser.add_argument("doi")
    resolve_parser.add_argument("--provider", default="mock")
    resolve_parser.add_argument("--config", default="config/default.yaml")
    resolve_parser.add_argument("--live-network", action="store_true")

    locate_parser = subcommands.add_parser("locate-full-text", help="Locate legal open full-text copies for a DOI.")
    locate_parser.add_argument("doi")
    locate_parser.add_argument("--providers", nargs="+", default=["mock"])
    locate_parser.add_argument("--config", default="config/default.yaml")
    locate_parser.add_argument("--live-network", action="store_true")
    return parser


def make_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown provider {name}")


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    literature = config.literature.model_copy(update={
        "enabled": bool(args.literature_providers or args.metadata_resolver or args.full_text_locators),
        "allow_live_network": args.live_network,
        "force_refresh": args.force_refresh,
        "search_providers": args.literature_providers or config.literature.search_providers,
        "metadata_resolvers": [args.metadata_resolver] if args.metadata_resolver else config.literature.metadata_resolvers,
        "full_text_locators": args.full_text_locators or config.literature.full_text_locators,
    })
    _guard_live(literature.search_providers + literature.metadata_resolvers + literature.full_text_locators, literature.allow_live_network)
    config = config.model_copy(update={"literature": literature})
    goal = load_research_goal(args.goal)
    provider = make_provider(args.provider)
    result = await run_workflow(
        goal,
        provider,
        config,
        store=LocalStore(args.runs_dir),
        run_id=args.run_id,
    )
    print(f"Run complete: {result.run_id}")
    print(f"Artifacts: {Path(result.run_dir).resolve()}")
    print(f"Final report: {(Path(result.run_dir) / 'final_report.md').resolve()}")
    return 0


async def _search_literature(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    literature = config.literature.model_copy(update={
        "enabled": True,
        "search_providers": args.providers,
        "metadata_resolvers": [],
        "full_text_locators": [],
        "allow_live_network": args.live_network,
        "max_results_per_provider": args.limit,
    })
    _guard_live(args.providers, args.live_network)
    pipeline = build_literature_pipeline(literature)
    result = await pipeline.acquire(args.query)
    for paper in result.normalized_papers:
        print(f"{paper.id}\t{paper.title}\t{paper.doi or ''}")
    return 0


async def _resolve_doi(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    literature = config.literature.model_copy(update={
        "enabled": True,
        "search_providers": [],
        "metadata_resolvers": [args.provider],
        "full_text_locators": [],
        "allow_live_network": args.live_network,
    })
    _guard_live([args.provider], args.live_network)
    pipeline = build_literature_pipeline(literature)
    resolver = pipeline.metadata_resolvers[0]
    resolution = await resolver.resolve(MetadataResolveRequest(doi=args.doi))
    print(resolution.model_dump_json(indent=2))
    return 0


async def _locate_full_text(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    literature = config.literature.model_copy(update={
        "enabled": True,
        "search_providers": [],
        "metadata_resolvers": [],
        "full_text_locators": args.providers,
        "allow_live_network": args.live_network,
    })
    _guard_live(args.providers, args.live_network)
    pipeline = build_literature_pipeline(literature)
    doi = args.doi.lower()
    paper = Paper(
        id=f"paper-doi-{doi.replace('/', '-')}",
        title=f"DOI {doi}",
        doi=doi,
        identifiers=[ExternalIdentifier(scheme="doi", value=doi, canonical_value=doi, source="cli")],
        source_provider="cli",
    )
    locations = []
    for locator in pipeline.full_text_locators:
        locations.extend(await locator.locate(paper))
    for location in locations:
        print(location.model_dump_json())
    return 0


def _guard_live(provider_names: list[str], live_network: bool) -> None:
    live_providers = {"openalex", "crossref", "unpaywall", "arxiv"}
    if any(name in live_providers for name in provider_names) and not live_network:
        raise ValueError("Live provider selected but live network is disabled. Re-run with --live-network.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            goal = load_research_goal(args.goal)
            print(f"Valid research goal: {goal.id}")
            return 0
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "search-literature":
            return asyncio.run(_search_literature(args))
        if args.command == "resolve-doi":
            return asyncio.run(_resolve_doi(args))
        if args.command == "locate-full-text":
            return asyncio.run(_locate_full_text(args))
    except (ValidationError, ValueError, ProviderError) as exc:
        print(f"Error: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
