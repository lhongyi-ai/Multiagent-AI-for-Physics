from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from coscientist.closed_question import (
    compare_closed_feedback,
    run_closed_question_project,
    validate_closed_feedback_artifacts,
    validate_closed_question_artifacts,
)
from coscientist.atomic.discovery import compare_atomic_verifiers, refresh_atomic_artifacts_if_present, run_atomic_discovery_project, validate_atomic_discovery_artifacts
from coscientist.atomic.campaign import (
    compare_atomic_campaign,
    curate_atomic_campaign_project,
    resume_atomic_campaign_checkpoint,
    run_atomic_campaign_project,
    validate_atomic_campaign_artifacts,
)
from coscientist.config import load_config, load_research_goal
from coscientist.discovery import run_discovery_project, resume_discovery_checkpoint, validate_discovery_artifacts
from coscientist.literature.http import NetworkDisabledError
from coscientist.literature.pipeline import build_literature_pipeline
from coscientist.orchestration.workflow import run_workflow
from coscientist.pilot.artifacts import read_json, read_jsonl, validate_v1_artifacts
from coscientist.pilot.evidence import verify_hypothesis_evidence
from coscientist.pilot.feedback_ab import compare_feedback_project, validate_feedback_ab_artifacts
from coscientist.pilot.model_comparison import compare_model_runs
from coscientist.pilot.project_io import load_project_spec
from coscientist.pilot.reports import build_human_review_package
from coscientist.pilot.runner import CompletedRunError, run_pilot_project
from coscientist.providers.base import ProviderError
from coscientist.providers.mock import MockProvider
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import ExternalIdentifier, MetadataResolveRequest, Paper, SearchQuery
from coscientist.storage.local_store import LocalStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coscientist")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run", help="Run the co-scientist MVP workflow.")
    run_parser.add_argument("goal", help="Path to a research goal YAML file.")
    run_parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    run_parser.add_argument("--live-model", action="store_true", help="Required for --provider openai; never inferred from environment variables.")
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

    project_show = subcommands.add_parser("project-show", help="Validate and display a research project specification.")
    project_show.add_argument("project", help="Path to a V1 project YAML or JSON file.")

    run_project = subcommands.add_parser("run-project", help="Run a V1 pilot project with deterministic model execution.")
    run_project.add_argument("project", help="Path to a V1 project YAML or JSON file.")
    run_project.add_argument("--provider", choices=["mock", "openai"], default="mock", help="Model provider. openai requires --live-model.")
    run_project.add_argument("--runs-dir", default="runs")
    run_project.add_argument("--run-id", default=None)
    run_project.add_argument("--force", action="store_true", help="Overwrite a completed pilot run directory.")
    run_project.add_argument("--literature-mode", choices=["fixture", "live", "existing"], default=None)
    run_project.add_argument("--search-providers", nargs="+", default=None)
    run_project.add_argument("--enrichment-providers", nargs="+", default=None)
    run_project.add_argument("--max-literature-results", type=int, default=None)
    run_project.add_argument("--max-model-calls", type=int, default=None)
    run_project.add_argument("--max-evolution-rounds", type=int, default=None)
    run_project.add_argument("--corpus", default=None, help="Existing normalized corpus JSONL path.")
    run_project.add_argument("--acquire-literature-only", action="store_true")
    run_project.add_argument("--smoke", action="store_true", help="Minimal live-model-compatible run: one generator, one hypothesis, no evolution.")
    run_project.add_argument("--dry-run", action="store_true", help="Plan literature acquisition without network calls.")
    run_project.add_argument("--live-network", action="store_true", help="Explicitly allow live scholarly provider HTTP calls.")
    run_project.add_argument("--live-model", action="store_true", help="Explicitly allow OpenAI-compatible live model calls.")

    acquire_lit = subcommands.add_parser("acquire-literature", help="Acquire or plan a project literature corpus.")
    acquire_lit.add_argument("project", help="Path to a V1 project YAML or JSON file.")
    acquire_lit.add_argument("--provider", choices=["mock", "openai"], default="mock")
    acquire_lit.add_argument("--runs-dir", default="runs")
    acquire_lit.add_argument("--run-id", default=None)
    acquire_lit.add_argument("--force", action="store_true", help="Overwrite a completed acquisition run directory.")
    acquire_lit.add_argument("--literature-mode", choices=["fixture", "live", "existing"], default=None)
    acquire_lit.add_argument("--search-providers", nargs="+", default=None)
    acquire_lit.add_argument("--enrichment-providers", nargs="+", default=None)
    acquire_lit.add_argument("--max-literature-results", type=int, default=None)
    acquire_lit.add_argument("--corpus", default=None, help="Existing normalized corpus JSONL path.")
    acquire_lit.add_argument("--dry-run", action="store_true", help="Plan literature acquisition without network calls.")
    acquire_lit.add_argument("--live-network", action="store_true", help="Explicitly allow live scholarly provider HTTP calls.")

    verify = subcommands.add_parser("verify-evidence", help="Verify V1 evidence links for a run directory.")
    verify.add_argument("run_dir")

    evaluate = subcommands.add_parser("evaluate-run", help="Inspect V1 evaluation artifacts for a run directory.")
    evaluate.add_argument("run_dir")

    compare = subcommands.add_parser("compare-rounds", help="Print baseline-versus-final V1 round comparison.")
    compare.add_argument("run_dir")

    review = subcommands.add_parser("build-review-package", help="Regenerate or print the V1 human-review package path.")
    review.add_argument("run_dir")

    validate_artifacts = subcommands.add_parser("validate-artifacts", help="Validate required V1 run artifacts.")
    validate_artifacts.add_argument("run_dir")

    compare_models = subcommands.add_parser("compare-model-runs", help="Compare a mock V1 run with a live-model candidate run.")
    compare_models.add_argument("mock_run_dir")
    compare_models.add_argument("candidate_run_dir")
    compare_models.add_argument("--output-dir", default=None)

    compare_feedback = subcommands.add_parser("compare-feedback", help="Run a deterministic advisory-vs-controlled-feedback V1.5C A/B experiment.")
    compare_feedback.add_argument("project", help="Path to a V1 project YAML or JSON file.")
    compare_feedback.add_argument("--runs-dir", default="runs")
    compare_feedback.add_argument("--experiment-id", default=None)
    compare_feedback.add_argument("--force", action="store_true", help="Overwrite a completed feedback A/B experiment directory.")

    validate_feedback = subcommands.add_parser("validate-feedback-ab", help="Validate V1.5C feedback A/B artifacts.")
    validate_feedback.add_argument("experiment_dir")

    run_closed = subcommands.add_parser("run-closed-question", help="Run an offline V1.6 closed-question benchmark or pilot.")
    run_closed.add_argument("project")
    run_closed.add_argument("--runs-dir", default="runs")
    run_closed.add_argument("--run-id", default=None)
    run_closed.add_argument("--force", action="store_true")

    evaluate_closed = subcommands.add_parser("evaluate-closed-question", help="Print V1.6 closed-question evaluation summary.")
    evaluate_closed.add_argument("run_dir")

    compare_closed = subcommands.add_parser("compare-closed-feedback", help="Run deterministic closed-question advisory-vs-feedback comparison.")
    compare_closed.add_argument("project")
    compare_closed.add_argument("--runs-dir", default="runs")
    compare_closed.add_argument("--experiment-id", default=None)
    compare_closed.add_argument("--force", action="store_true")

    validate_closed = subcommands.add_parser("validate-closed-question", help="Validate V1.6 closed-question artifacts.")
    validate_closed.add_argument("run_or_experiment_dir")

    run_discovery = subcommands.add_parser("run-discovery", help="Run a deterministic V1.7 scientific discovery search.")
    run_discovery.add_argument("project")
    run_discovery.add_argument("--runs-dir", default="runs")
    run_discovery.add_argument("--run-id", default=None)
    run_discovery.add_argument("--force", action="store_true")
    run_discovery.add_argument("--stop-after-tasks", type=int, default=None)

    resume_discovery = subcommands.add_parser("resume-discovery", help="Resume a deterministic V1.7 discovery checkpoint.")
    resume_discovery.add_argument("checkpoint")

    validate_discovery = subcommands.add_parser("validate-discovery", help="Validate V1.7 discovery artifacts.")
    validate_discovery.add_argument("run_dir")

    run_atomic = subcommands.add_parser("run-atomic-discovery", help="Run a deterministic V1.8 atomic/AMO discovery benchmark.")
    run_atomic.add_argument("project")
    run_atomic.add_argument("--runs-dir", default="runs")
    run_atomic.add_argument("--run-id", default=None)
    run_atomic.add_argument("--force", action="store_true")
    run_atomic.add_argument("--stop-after-tasks", type=int, default=None)

    validate_atomic = subcommands.add_parser("validate-atomic-discovery", help="Validate V1.8 atomic discovery artifacts.")
    validate_atomic.add_argument("run_dir")

    compare_atomic = subcommands.add_parser("compare-atomic-verifiers", help="Compare deterministic generic and atomic verifier baselines.")
    compare_atomic.add_argument("project")
    compare_atomic.add_argument("--runs-dir", default="runs")
    compare_atomic.add_argument("--experiment-id", default=None)
    compare_atomic.add_argument("--force", action="store_true")

    curate_campaign = subcommands.add_parser("curate-atomic-campaign", help="Curate a deterministic V1.9 real-data atomic campaign.")
    curate_campaign.add_argument("project")
    curate_campaign.add_argument("--runs-dir", default="runs")
    curate_campaign.add_argument("--run-id", default=None)
    curate_campaign.add_argument("--force", action="store_true")

    run_campaign = subcommands.add_parser("run-atomic-campaign", help="Run a deterministic V1.9 real-data atomic campaign.")
    run_campaign.add_argument("project")
    run_campaign.add_argument("--runs-dir", default="runs")
    run_campaign.add_argument("--run-id", default=None)
    run_campaign.add_argument("--force", action="store_true")
    run_campaign.add_argument("--stop-after-stage", choices=["curation"], default=None)

    validate_campaign = subcommands.add_parser("validate-atomic-campaign", help="Validate V1.9 atomic campaign artifacts.")
    validate_campaign.add_argument("run_dir")

    compare_campaign = subcommands.add_parser("compare-atomic-campaign", help="Compare deterministic V1.9 campaign baselines.")
    compare_campaign.add_argument("project")
    compare_campaign.add_argument("--runs-dir", default="runs")
    compare_campaign.add_argument("--experiment-id", default=None)
    compare_campaign.add_argument("--force", action="store_true")
    return parser


def make_provider(name: str):
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown provider {name}")


async def _run(args: argparse.Namespace) -> int:
    if args.provider == "openai" and not args.live_model:
        raise ValueError("--provider openai requires explicit --live-model.")
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


async def _run_project(args: argparse.Namespace) -> int:
    run_dir = await run_pilot_project(
        args.project,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        live_network=args.live_network,
        live_model=args.live_model,
        provider_name=args.provider,
        literature_mode=args.literature_mode,
        search_providers=args.search_providers,
        enrichment_providers=args.enrichment_providers,
        max_literature_results=args.max_literature_results,
        max_model_calls=args.max_model_calls,
        max_evolution_rounds=args.max_evolution_rounds,
        corpus_path=args.corpus,
        acquire_literature_only=args.acquire_literature_only,
        smoke=args.smoke,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"Pilot run complete: {Path(run_dir).resolve()}")
    if args.acquire_literature_only or args.dry_run:
        print(f"Corpus: {(Path(run_dir) / 'corpus.jsonl').resolve()}")
        print(f"Manifest: {(Path(run_dir) / 'corpus_manifest.json').resolve()}")
    else:
        print(f"Report: {(Path(run_dir) / 'report.md').resolve()}")
        print(f"Human review: {(Path(run_dir) / 'human_review.md').resolve()}")
    return 0


async def _acquire_literature(args: argparse.Namespace) -> int:
    run_dir = await run_pilot_project(
        args.project,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        live_network=args.live_network,
        live_model=False,
        provider_name=args.provider,
        literature_mode=args.literature_mode,
        search_providers=args.search_providers,
        enrichment_providers=args.enrichment_providers,
        max_literature_results=args.max_literature_results,
        corpus_path=args.corpus,
        acquire_literature_only=True,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"Literature artifacts: {Path(run_dir).resolve()}")
    print(f"Corpus: {(Path(run_dir) / 'corpus.jsonl').resolve()}")
    print(f"Manifest: {(Path(run_dir) / 'corpus_manifest.json').resolve()}")
    return 0


def _project_show(args: argparse.Namespace) -> int:
    project = load_project_spec(args.project)
    print(project.model_dump_json(indent=2))
    return 0


def _verify_evidence(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    project = load_project_spec(run_dir / "project_snapshot.json")
    corpus = [Paper.model_validate(item) for item in read_jsonl(run_dir / "corpus.jsonl")]
    hypotheses = [Hypothesis.model_validate(item) for item in read_json(run_dir / "hypotheses_final.json")]
    records = verify_hypothesis_evidence(hypotheses, corpus)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(f"Project: {project.project_id}")
    print(f"Evidence verification records: {len(records)}")
    print(counts)
    return 0


def _evaluate_run(args: argparse.Namespace) -> int:
    data = read_json(Path(args.run_dir) / "evaluation_by_round.json")
    print(f"Evaluation rounds: {len(data)}")
    for item in data:
        print(f"{item['round_label']}: {item['mean_scores']}")
    return 0


def _compare_rounds(args: argparse.Namespace) -> int:
    comparison = read_json(Path(args.run_dir) / "round_comparison.json")
    print("Score changes by dimension:")
    for dimension, value in comparison["score_changes_by_dimension"].items():
        print(f"- {dimension}: {value:+.3f}")
    print(f"Citation coverage: {comparison['citation_coverage']}")
    print(comparison["evaluator_self_preference_note"])
    return 0


def _build_review_package(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    project = load_project_spec(run_dir / "project_snapshot.json")
    corpus = [Paper.model_validate(item) for item in read_jsonl(run_dir / "corpus.jsonl")]
    final_hypotheses = [Hypothesis.model_validate(item) for item in read_json(run_dir / "hypotheses_final.json")]
    comparison = read_json(run_dir / "round_comparison.json")
    records = read_jsonl(run_dir / "evidence_verification.jsonl")
    from coscientist.schemas.evaluation import RoundComparison
    from coscientist.schemas.evidence import EvidenceVerificationRecord

    text = build_human_review_package(
        project,
        corpus,
        final_hypotheses,
        RoundComparison.model_validate_json(json.dumps(comparison)),
        [EvidenceVerificationRecord.model_validate_json(json.dumps(item)) for item in records],
    )
    output = run_dir / "human_review.md"
    output.write_text(text, encoding="utf-8")
    print(f"Human review package: {output.resolve()}")
    return 0


def _validate_artifacts(args: argparse.Namespace) -> int:
    errors = validate_v1_artifacts(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1 artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _compare_model_runs(args: argparse.Namespace) -> int:
    output = compare_model_runs(args.mock_run_dir, args.candidate_run_dir, args.output_dir)
    print(f"Model run comparison: {output.resolve()}")
    print(f"Markdown summary: {(output.parent / 'model_run_comparison.md').resolve()}")
    return 0


async def _compare_feedback(args: argparse.Namespace) -> int:
    output = await compare_feedback_project(
        args.project,
        runs_dir=args.runs_dir,
        experiment_id=args.experiment_id,
        force=args.force,
    )
    print(f"Feedback A/B experiment: {Path(output).resolve()}")
    print(f"Summary: {(Path(output) / 'feedback_ab_summary.md').resolve()}")
    print(f"Human review: {(Path(output) / 'human_review.md').resolve()}")
    return 0


def _validate_feedback_ab(args: argparse.Namespace) -> int:
    errors = validate_feedback_ab_artifacts(args.experiment_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1.5C feedback A/B artifacts: {Path(args.experiment_dir).resolve()}")
    return 0


def _run_closed_question(args: argparse.Namespace) -> int:
    output = run_closed_question_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force)
    print(f"Closed-question run complete: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'report.md').resolve()}")
    print(f"Human review: {(Path(output) / 'human_review.md').resolve()}")
    return 0


def _evaluate_closed_question(args: argparse.Namespace) -> int:
    evaluations = read_json(Path(args.run_dir) / "closed_question_evaluations.json")
    correct = sum(1 for item in evaluations if item.get("correct"))
    print(f"Closed questions: {len(evaluations)}")
    print(f"Correct: {correct}")
    for item in evaluations:
        print(f"{item['question_id']}: {item['outcome']} confidence={item['confidence']}")
    return 0


def _compare_closed_feedback(args: argparse.Namespace) -> int:
    output = compare_closed_feedback(args.project, runs_dir=args.runs_dir, experiment_id=args.experiment_id, force=args.force)
    print(f"Closed feedback comparison: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'report.md').resolve()}")
    return 0


def _validate_closed_question(args: argparse.Namespace) -> int:
    path = Path(args.run_or_experiment_dir)
    if (path / "closed_question_ab_comparison.json").exists():
        errors = validate_closed_feedback_artifacts(path)
    else:
        errors = validate_closed_question_artifacts(path)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1.6 closed-question artifacts: {path.resolve()}")
    return 0


def _run_discovery(args: argparse.Namespace) -> int:
    output = run_discovery_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force, stop_after_tasks=args.stop_after_tasks)
    print(f"Discovery run: {Path(output).resolve()}")
    print(f"Checkpoint: {(Path(output) / 'search_checkpoint.json').resolve()}")
    print(f"Report: {(Path(output) / 'discovery_report.md').resolve()}")
    return 0


def _resume_discovery(args: argparse.Namespace) -> int:
    if Path(args.checkpoint).name == "campaign_checkpoint.json":
        output = resume_atomic_campaign_checkpoint(args.checkpoint)
        print(f"Atomic campaign resumed: {Path(output).resolve()}")
        print(f"Report: {(Path(output) / 'campaign_report.md').resolve()}")
        return 0
    output = resume_discovery_checkpoint(args.checkpoint)
    refreshed_atomic = refresh_atomic_artifacts_if_present(output)
    print(f"Discovery resumed: {Path(output).resolve()}")
    if refreshed_atomic:
        print(f"Atomic artifacts refreshed: {(Path(output) / 'atomic_benchmark_metrics.json').resolve()}")
    print(f"Report: {(Path(output) / 'discovery_report.md').resolve()}")
    return 0


def _validate_discovery(args: argparse.Namespace) -> int:
    errors = validate_discovery_artifacts(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1.7 discovery artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _run_atomic_discovery(args: argparse.Namespace) -> int:
    output = run_atomic_discovery_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force, stop_after_tasks=args.stop_after_tasks)
    print(f"Atomic discovery run: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'atomic_discovery_report.md').resolve()}")
    return 0


def _validate_atomic_discovery(args: argparse.Namespace) -> int:
    errors = validate_atomic_discovery_artifacts(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1.8 atomic discovery artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _compare_atomic_verifiers(args: argparse.Namespace) -> int:
    output = compare_atomic_verifiers(args.project, runs_dir=args.runs_dir, experiment_id=args.experiment_id, force=args.force)
    print(f"Atomic verifier comparison: {Path(output).resolve()}")
    print(f"Summary: {(Path(output) / 'atomic_benchmark_summary.md').resolve()}")
    return 0


def _curate_atomic_campaign(args: argparse.Namespace) -> int:
    output = curate_atomic_campaign_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force)
    print(f"Atomic campaign curated: {Path(output).resolve()}")
    print(f"Dataset manifest: {(Path(output) / 'dataset_manifest.json').resolve()}")
    return 0


def _run_atomic_campaign(args: argparse.Namespace) -> int:
    output = run_atomic_campaign_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force, stop_after_stage=args.stop_after_stage)
    print(f"Atomic campaign run: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'campaign_report.md').resolve()}")
    return 0


def _validate_atomic_campaign(args: argparse.Namespace) -> int:
    errors = validate_atomic_campaign_artifacts(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V1.9 atomic campaign artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _compare_atomic_campaign(args: argparse.Namespace) -> int:
    output = compare_atomic_campaign(args.project, runs_dir=args.runs_dir, experiment_id=args.experiment_id, force=args.force)
    print(f"Atomic campaign comparison: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'campaign_report.md').resolve()}")
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
        if args.command == "project-show":
            return _project_show(args)
        if args.command == "run-project":
            return asyncio.run(_run_project(args))
        if args.command == "acquire-literature":
            return asyncio.run(_acquire_literature(args))
        if args.command == "verify-evidence":
            return _verify_evidence(args)
        if args.command == "evaluate-run":
            return _evaluate_run(args)
        if args.command == "compare-rounds":
            return _compare_rounds(args)
        if args.command == "build-review-package":
            return _build_review_package(args)
        if args.command == "validate-artifacts":
            return _validate_artifacts(args)
        if args.command == "compare-model-runs":
            return _compare_model_runs(args)
        if args.command == "compare-feedback":
            return asyncio.run(_compare_feedback(args))
        if args.command == "validate-feedback-ab":
            return _validate_feedback_ab(args)
        if args.command == "run-closed-question":
            return _run_closed_question(args)
        if args.command == "evaluate-closed-question":
            return _evaluate_closed_question(args)
        if args.command == "compare-closed-feedback":
            return _compare_closed_feedback(args)
        if args.command == "validate-closed-question":
            return _validate_closed_question(args)
        if args.command == "run-discovery":
            return _run_discovery(args)
        if args.command == "resume-discovery":
            return _resume_discovery(args)
        if args.command == "validate-discovery":
            return _validate_discovery(args)
        if args.command == "run-atomic-discovery":
            return _run_atomic_discovery(args)
        if args.command == "validate-atomic-discovery":
            return _validate_atomic_discovery(args)
        if args.command == "compare-atomic-verifiers":
            return _compare_atomic_verifiers(args)
        if args.command == "curate-atomic-campaign":
            return _curate_atomic_campaign(args)
        if args.command == "run-atomic-campaign":
            return _run_atomic_campaign(args)
        if args.command == "validate-atomic-campaign":
            return _validate_atomic_campaign(args)
        if args.command == "compare-atomic-campaign":
            return _compare_atomic_campaign(args)
    except (ValidationError, ValueError, ProviderError, CompletedRunError, NetworkDisabledError) as exc:
        print(f"Error: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
