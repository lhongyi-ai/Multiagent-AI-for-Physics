from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from coscientist.claim_dag import (
    create_claim_dag_artifacts_from_run,
    query_claim_dag_database,
    rebuild_claim_dag_database,
    validate_claim_dag_database,
)
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
from coscientist.core import GenericDataAcquisitionAgent, get_default_domain_registry, run_optimizer_v2
from coscientist.core.optimizer_v2 import mutation_operators
from coscientist.core.tasks import ScientificTaskType, TaskPolicyRegistry
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
from coscientist.superconductivity import (
    query_scientific_index,
    rebuild_scientific_index,
    import_digitized_points,
    CandidateReviewDecision,
    phase2_acquisition_gaps,
    phase2_candidate_rows,
    phase2_candidate_sources,
    phase2_digitization_queue,
    phase2_readiness,
    promote_reviewed_candidates,
    review_candidate_rows,
    run_phase2_acquisition,
    run_minimal_mixed_bcs_project,
    run_superconductivity_campaign,
    run_v22_campaign,
    test_data_connections,
    test_live_models,
    validate_scientific_index,
    validate_minimal_mixed_bcs_run,
    validate_phase2_acquisition_run,
    validate_superconductivity_campaign,
    validate_v22_campaign,
)


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

    run_sc = subcommands.add_parser("run-superconductivity-campaign", help="Run a deterministic V2.1 superconductivity domain campaign.")
    run_sc.add_argument("project")
    run_sc.add_argument("--runs-dir", default="runs")
    run_sc.add_argument("--run-id", default=None)
    run_sc.add_argument("--force", action="store_true")
    run_sc.add_argument("--live-network", action="store_true", help="Explicitly allow future live scientific database adapters.")

    validate_sc = subcommands.add_parser("validate-superconductivity-campaign", help="Validate V2.1 superconductivity campaign artifacts.")
    validate_sc.add_argument("run_dir")

    run_minimal_bcs = subcommands.add_parser("run-minimal-bcs", help="Run Phase-1 minimal mixed BCS theory scan.")
    run_minimal_bcs.add_argument("--project", default=None, help="Optional JSON project specification. Defaults to built-in minimal BCS scan.")
    run_minimal_bcs.add_argument("--runs-dir", default="runs")
    run_minimal_bcs.add_argument("--run-id", default="minimal-mixed-bcs")
    run_minimal_bcs.add_argument("--force", action="store_true")

    validate_minimal_bcs = subcommands.add_parser("validate-minimal-bcs", help="Validate Phase-1 minimal mixed BCS artifacts.")
    validate_minimal_bcs.add_argument("run_dir")

    phase2_acquire = subcommands.add_parser("phase2-acquire", help="Run Phase 2 LSCO data acquisition into staging artifacts.")
    phase2_acquire.add_argument("--material", default="lsco")
    phase2_acquire.add_argument("--mode", choices=["fixture", "existing", "live"], default="fixture")
    phase2_acquire.add_argument("--runs-dir", default="runs")
    phase2_acquire.add_argument("--run-id", default="phase2-lsco-acquisition")
    phase2_acquire.add_argument("--canonical-dataset", default="data/phase2_lsco.csv")
    phase2_acquire.add_argument("--live-network", action="store_true", help="Explicitly allow arXiv/OpenAlex/Crossref/Unpaywall network calls in live mode.")
    phase2_acquire.add_argument("--max-queries", type=int, default=8)
    phase2_acquire.add_argument("--max-results-per-query", type=int, default=5)
    phase2_acquire.add_argument("--auto-promote", action="store_true", help="Promote high-confidence validated rows. Disabled by default.")

    validate_phase2_acq = subcommands.add_parser("validate-phase2-acquisition", help="Validate Phase 2 acquisition artifacts.")
    validate_phase2_acq.add_argument("run_dir")

    phase2_review = subcommands.add_parser("phase2-review-staging", help="Print staged Phase 2 candidate rows.")
    phase2_review.add_argument("run_dir")

    phase2_promote = subcommands.add_parser("phase2-promote", help="Promote reviewed Phase 2 rows by rerunning fixture acquisition with auto-promotion.")
    phase2_promote.add_argument("--runs-dir", default="runs")
    phase2_promote.add_argument("--run-id", default="phase2-lsco-promotion")
    phase2_promote.add_argument("--canonical-dataset", default="data/phase2_lsco.csv")

    phase2_review_row = subcommands.add_parser("phase2-review-row", help="Approve, reject, or edit a staged Phase 2 candidate row.")
    phase2_review_row.add_argument("run_dir")
    phase2_review_row.add_argument("--candidate-row-id", required=True)
    phase2_review_row.add_argument("--decision", choices=["approve", "reject", "edit"], required=True)
    phase2_review_row.add_argument("--rationale", default="")
    phase2_review_row.add_argument("--reviewer", default="local-human")
    phase2_review_row.add_argument("--edited-value", default=None)
    phase2_review_row.add_argument("--edited-unit", default=None)

    phase2_promote_reviewed = subcommands.add_parser("phase2-promote-reviewed", help="Promote approved reviewed Phase 2 rows into a canonical CSV.")
    phase2_promote_reviewed.add_argument("run_dir")
    phase2_promote_reviewed.add_argument("--canonical-dataset", default="data/phase2_lsco.csv")

    phase2_digitization = subcommands.add_parser("phase2-digitization-queue", help="Print Phase 2 figure digitization queue.")
    phase2_digitization.add_argument("run_dir")

    phase2_import_digitized = subcommands.add_parser("phase2-import-digitized", help="Import reviewed digitized graph points into a run artifact.")
    phase2_import_digitized.add_argument("run_dir")
    phase2_import_digitized.add_argument("--task-id", required=True)
    phase2_import_digitized.add_argument("--csv", required=True)
    phase2_import_digitized.add_argument("--reviewer", default="local-human")

    phase2_coverage = subcommands.add_parser("phase2-coverage", help="Run Phase 2 LSCO coverage analysis.")
    phase2_coverage.add_argument("--canonical-dataset", default="data/phase2_lsco.csv")
    phase2_coverage.add_argument("--runs-dir", default="runs")
    phase2_coverage.add_argument("--run-id", default="phase2-lsco-coverage")

    phase2_compare = subcommands.add_parser("phase2-compare", help="Report Phase 2 comparison readiness from an acquisition run.")
    phase2_compare.add_argument("run_dir")

    test_live = subcommands.add_parser("test-live-models", help="Run permission-gated V2.2 live-model smoke tests.")
    test_live.add_argument("--live-model", action="store_true", help="Explicitly allow OpenAI-compatible live model calls.")
    test_live.add_argument("--runs-dir", default="runs")
    test_live.add_argument("--run-id", default="v22-live-model-smoke")
    test_live.add_argument("--force", action="store_true")

    test_connections = subcommands.add_parser("test-data-connections", help="Report V2.2 scholarly/material provider connection status.")
    test_connections.add_argument("--live-network", action="store_true", help="Explicitly allow live provider network checks when implemented.")
    test_connections.add_argument("--runs-dir", default="runs")
    test_connections.add_argument("--run-id", default="v22-data-connections")
    test_connections.add_argument("--force", action="store_true")

    run_v22 = subcommands.add_parser("run-v22-campaign", help="Run the V2.2 superconductivity theory-discrimination campaign.")
    run_v22.add_argument("project")
    run_v22.add_argument("--runs-dir", default="runs")
    run_v22.add_argument("--run-id", default=None)
    run_v22.add_argument("--force", action="store_true")
    run_v22.add_argument("--live-model", action="store_true", help="Explicitly allow live model calls where routed.")
    run_v22.add_argument("--live-network", action="store_true", help="Explicitly allow live database/network calls where implemented.")

    validate_v22 = subcommands.add_parser("validate-v22-campaign", help="Validate V2.2 superconductivity campaign artifacts.")
    validate_v22.add_argument("run_dir")

    rebuild_index = subcommands.add_parser("rebuild-index", help="Rebuild the optional local SQLite scientific artifact index.")
    rebuild_index.add_argument("run_or_project_path")

    validate_index = subcommands.add_parser("validate-index", help="Validate the optional local SQLite scientific artifact index.")
    validate_index.add_argument("run_or_project_path")

    query_index = subcommands.add_parser("query-index", help="Query the optional local SQLite scientific artifact index.")
    query_index.add_argument("run_or_project_path")
    query_index.add_argument("table")
    query_index.add_argument("--limit", type=int, default=20)

    build_claim_dag = subcommands.add_parser("build-claim-dag-db", help="Build claim DAG artifacts and SQLite database for a run directory.")
    build_claim_dag.add_argument("run_dir")
    build_claim_dag.add_argument("--candidate-id", default=None)
    build_claim_dag.add_argument("--force", action="store_true")

    validate_claim_dag = subcommands.add_parser("validate-claim-dag-db", help="Validate claim DAG artifacts and SQLite database for a run directory.")
    validate_claim_dag.add_argument("run_dir")

    query_claim_dag = subcommands.add_parser("query-claim-dag-db", help="Query the claim DAG SQLite database.")
    query_claim_dag.add_argument("run_dir")
    query_claim_dag.add_argument("table")
    query_claim_dag.add_argument("--limit", type=int, default=20)

    domains_list = subcommands.add_parser("domains-list", help="List registered scientific Domain Packs.")

    domains_inspect = subcommands.add_parser("domains-inspect", help="Inspect one scientific Domain Pack.")
    domains_inspect.add_argument("--domain", required=True)

    acquisition_run = subcommands.add_parser("acquisition-run", help="Run generic DomainPack-backed acquisition.")
    acquisition_run.add_argument("--domain", required=True)
    acquisition_run.add_argument("--question", default="")
    acquisition_run.add_argument("--task-type", choices=[item.value for item in ScientificTaskType], default=ScientificTaskType.DATA_EXTRACTION.value)
    acquisition_run.add_argument("--mode", choices=["fixture", "existing", "live"], default="fixture")
    acquisition_run.add_argument("--runs-dir", default="runs")
    acquisition_run.add_argument("--run-id", default=None)
    acquisition_run.add_argument("--live-network", action="store_true")
    acquisition_run.add_argument("--force", action="store_true")

    hypotheses_optimize = subcommands.add_parser("hypotheses-optimize", help="Run deterministic Hypothesis Optimizer V2 on fixture or run artifacts.")
    hypotheses_optimize.add_argument("--domain", default="general")
    hypotheses_optimize.add_argument("--runs-dir", default="runs")
    hypotheses_optimize.add_argument("--run-id", default="optimizer-v2")
    hypotheses_optimize.add_argument("--hypotheses-json", default=None, help="Optional JSON/JSONL hypotheses file. Uses a fixture set when omitted.")
    hypotheses_optimize.add_argument("--force", action="store_true")

    hypotheses_portfolio = subcommands.add_parser("hypotheses-portfolio", help="Print Optimizer V2 portfolio rows.")
    hypotheses_portfolio.add_argument("run_dir")

    failures_list = subcommands.add_parser("failures-list", help="Print Optimizer V2 failure memory.")
    failures_list.add_argument("run_dir")

    benchmark_run = subcommands.add_parser("benchmark-run", help="Run a deterministic DomainPack benchmark smoke.")
    benchmark_run.add_argument("--domain", required=True)
    benchmark_run.add_argument("--variant", default="optimizer_v2")
    benchmark_run.add_argument("--runs-dir", default="runs")
    benchmark_run.add_argument("--run-id", default=None)
    benchmark_run.add_argument("--force", action="store_true")
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


def _run_superconductivity_campaign(args: argparse.Namespace) -> int:
    output = run_superconductivity_campaign(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force, live_network=args.live_network)
    print(f"Superconductivity campaign run: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'superconductivity_report.md').resolve()}")
    print(f"Scientific index: {(Path(output) / 'scientific_index.sqlite').resolve()}")
    return 0


def _validate_superconductivity_campaign(args: argparse.Namespace) -> int:
    errors = validate_superconductivity_campaign(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V2.1 superconductivity campaign artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _run_minimal_bcs(args: argparse.Namespace) -> int:
    output = run_minimal_mixed_bcs_project(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force)
    print(f"Minimal mixed BCS run: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'minimal_bcs_report.md').resolve()}")
    print(f"Phase diagram: {(Path(output) / 'phase_diagram_points.jsonl').resolve()}")
    return 0


def _validate_minimal_bcs(args: argparse.Namespace) -> int:
    errors = validate_minimal_mixed_bcs_run(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid minimal mixed BCS artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _phase2_acquire(args: argparse.Namespace) -> int:
    if args.material.lower() != "lsco":
        raise ValueError("Only --material lsco is supported in this Phase 2 acquisition milestone.")
    output = run_phase2_acquisition(
        mode=args.mode,
        live_network=args.live_network,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        canonical_dataset=args.canonical_dataset,
        auto_promote=args.auto_promote,
        max_queries=args.max_queries,
        max_results_per_query=args.max_results_per_query,
    )
    summary = read_json(Path(output) / "acquisition_summary.json")
    print(f"Phase 2 acquisition run: {Path(output).resolve()}")
    print(f"Status: {summary['status']}")
    print(f"Candidate rows staged: {summary['candidate_rows_staged']}")
    print(f"Rows promoted: {summary['rows_promoted']}")
    print(f"Phase 2 status: {summary['final_phase2_status']}")
    return 0 if summary["status"] != "blocked_live_network_permission_required" else 2


def _validate_phase2_acquisition(args: argparse.Namespace) -> int:
    errors = validate_phase2_acquisition_run(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid Phase 2 acquisition artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _phase2_review_staging(args: argparse.Namespace) -> int:
    for row in phase2_candidate_rows(args.run_dir):
        print(json.dumps(row, sort_keys=True))
    return 0


def _phase2_promote(args: argparse.Namespace) -> int:
    output = run_phase2_acquisition(
        mode="fixture",
        live_network=False,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        canonical_dataset=args.canonical_dataset,
        auto_promote=True,
    )
    print(f"Phase 2 promotion run: {Path(output).resolve()}")
    print(f"Diff: {(Path(output) / 'canonical_dataset_diff.json').resolve()}")
    return 0


def _phase2_review_row(args: argparse.Namespace) -> int:
    value: float | str | None = args.edited_value
    if value is not None:
        try:
            value = float(value)
        except ValueError:
            pass
    output = review_candidate_rows(args.run_dir, [
        CandidateReviewDecision(
            candidate_row_id=args.candidate_row_id,
            decision=args.decision,
            reviewer=args.reviewer,
            rationale=args.rationale,
            edited_value=value,
            edited_unit=args.edited_unit,
        )
    ])
    print(f"Review decision recorded: {Path(output).resolve()}")
    return 0


def _phase2_promote_reviewed(args: argparse.Namespace) -> int:
    output = promote_reviewed_candidates(args.run_dir, args.canonical_dataset)
    print(f"Reviewed promotion report: {Path(output).resolve()}")
    return 0


def _phase2_digitization_queue(args: argparse.Namespace) -> int:
    for row in phase2_digitization_queue(args.run_dir):
        print(json.dumps(row, sort_keys=True))
    return 0


def _phase2_import_digitized(args: argparse.Namespace) -> int:
    output = import_digitized_points(args.run_dir, task_id=args.task_id, csv_path=args.csv, reviewer=args.reviewer)
    print(f"Digitized points imported: {Path(output).resolve()}")
    return 0


def _phase2_coverage(args: argparse.Namespace) -> int:
    from coscientist.superconductivity.phase2_data import run_phase2_data_coverage_tool

    output = run_phase2_data_coverage_tool(Path(args.runs_dir) / args.run_id, source_path=args.canonical_dataset)
    evaluation = read_json(Path(output) / "phase2_data_tool_evaluation.json")
    print(f"Phase 2 coverage: {Path(output).resolve()}")
    print(f"Status: {evaluation['status']}")
    return 0


def _phase2_compare(args: argparse.Namespace) -> int:
    readiness = phase2_readiness(args.run_dir)
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0


def _test_data_connections(args: argparse.Namespace) -> int:
    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise ValueError(f"data-connection artifacts are immutable; use a new run id or --force: {run_dir}")
    snapshots = run_dir / "provider_snapshots"
    results = test_data_connections(live_network=args.live_network, snapshots_dir=snapshots)
    run_dir.mkdir(parents=True, exist_ok=True)
    from coscientist.pilot.artifacts import write_json

    write_json(run_dir / "provider_connection_results.json", {"schema_version": "v22", "providers": results})
    for item in results:
        print(f"{item.provider}\tfixture={item.fixture_status}\tlive={item.live_status}\tauth={item.authentication_status}")
    return 0


async def _test_live_models(args: argparse.Namespace) -> int:
    output = await test_live_models(live_model=args.live_model, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force)
    print(f"Live model smoke artifacts: {Path(output).resolve()}")
    print(f"Results: {(Path(output) / 'live_model_smoke_results.jsonl').resolve()}")
    return 0


def _run_v22_campaign(args: argparse.Namespace) -> int:
    output = run_v22_campaign(args.project, runs_dir=args.runs_dir, run_id=args.run_id, force=args.force, live_model=args.live_model, live_network=args.live_network)
    print(f"V2.2 campaign run: {Path(output).resolve()}")
    print(f"Report: {(Path(output) / 'v22_scientific_report.md').resolve()}")
    print(f"Expert review: {(Path(output) / 'expert_review_package.md').resolve()}")
    return 0


def _validate_v22_campaign(args: argparse.Namespace) -> int:
    errors = validate_v22_campaign(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid V2.2 superconductivity campaign artifacts: {Path(args.run_dir).resolve()}")
    return 0


def _rebuild_index(args: argparse.Namespace) -> int:
    output = rebuild_scientific_index(args.run_or_project_path)
    print(f"Scientific index rebuilt: {Path(output).resolve()}")
    print(f"Manifest: {(Path(output).parent / 'scientific_index_manifest.json').resolve()}")
    return 0


def _validate_index(args: argparse.Namespace) -> int:
    errors = validate_scientific_index(args.run_or_project_path)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid scientific index: {Path(args.run_or_project_path).resolve()}")
    return 0


def _query_index(args: argparse.Namespace) -> int:
    for row in query_scientific_index(args.run_or_project_path, args.table, limit=args.limit):
        print(json.dumps(row, sort_keys=True))
    return 0


def _build_claim_dag_db(args: argparse.Namespace) -> int:
    create_claim_dag_artifacts_from_run(args.run_dir, candidate_id=args.candidate_id, force=args.force)
    output = rebuild_claim_dag_database(args.run_dir)
    print(f"Claim DAG database rebuilt: {Path(output).resolve()}")
    print(f"Validation: {'valid' if not validate_claim_dag_database(args.run_dir) else 'has errors'}")
    return 0


def _validate_claim_dag_db(args: argparse.Namespace) -> int:
    errors = validate_claim_dag_database(args.run_dir)
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 2
    print(f"Valid claim DAG database: {Path(args.run_dir).resolve()}")
    return 0


def _query_claim_dag_db(args: argparse.Namespace) -> int:
    for row in query_claim_dag_database(args.run_dir, args.table, limit=args.limit):
        print(json.dumps(row, sort_keys=True))
    return 0


def _domains_list(args: argparse.Namespace) -> int:
    registry = get_default_domain_registry()
    for item in registry.list():
        print(json.dumps(item.__dict__, sort_keys=True))
    return 0


def _domains_inspect(args: argparse.Namespace) -> int:
    pack = get_default_domain_registry().get(args.domain)
    payload = {
        "domain_id": pack.domain_id,
        "version": pack.version,
        "supported_task_types": sorted(item.value for item in pack.supported_task_types()),
        "tools": [item.model_dump(mode="json") for item in pack.tools()],
        "benchmark_cases": [item.model_dump(mode="json") for item in pack.benchmark_cases()],
        "readiness_gates": [item.model_dump(mode="json") for item in pack.readiness_gates()],
        "guardrails": pack.guardrails(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _generic_acquisition_run(args: argparse.Namespace) -> int:
    result = GenericDataAcquisitionAgent().run(
        domain_id=args.domain,
        question=args.question,
        task_type=args.task_type,
        mode=args.mode,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        live_network=args.live_network,
        force=args.force,
    )
    print(f"Generic acquisition run: {Path(result.run_dir).resolve()}")
    print(f"Status: {result.status}")
    print(f"Delegated to: {result.delegated_to or 'domain_pack_fixture'}")
    return 0 if result.status != "blocked_live_network_permission_required" else 2


def _hypotheses_optimize(args: argparse.Namespace) -> int:
    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise ValueError(f"optimizer artifacts are immutable; use --force or a new run id: {run_dir}")
    hypotheses = _load_optimizer_hypotheses(args.hypotheses_json)
    result = run_optimizer_v2(hypotheses, run_dir=run_dir, domain_id=args.domain, run_id=args.run_id)
    print(f"Optimizer V2 run: {run_dir.resolve()}")
    print(f"Hypotheses: {result.hypothesis_count}")
    print(f"Killed: {result.killed_count}")
    print(f"Pareto frontier: {', '.join(result.pareto_frontier_ids)}")
    return 0


def _hypotheses_portfolio(args: argparse.Namespace) -> int:
    for row in read_jsonl(Path(args.run_dir) / "hypothesis_portfolio_v2.jsonl"):
        print(json.dumps(row, sort_keys=True))
    return 0


def _failures_list(args: argparse.Namespace) -> int:
    path = Path(args.run_dir) / "failure_memory.jsonl"
    for row in read_jsonl(path) if path.exists() else []:
        print(json.dumps(row, sort_keys=True))
    return 0


def _benchmark_run(args: argparse.Namespace) -> int:
    pack = get_default_domain_registry().get(args.domain)
    cases = pack.benchmark_cases()
    result = GenericDataAcquisitionAgent().run(
        domain_id=args.domain,
        question=cases[0].prompt if cases else f"{args.domain} benchmark",
        task_type=cases[0].task_type if cases else ScientificTaskType.DATA_EXTRACTION,
        mode="fixture",
        runs_dir=args.runs_dir,
        run_id=args.run_id or f"benchmark-{args.domain}",
        force=args.force,
    )
    print(f"Benchmark run: {Path(result.run_dir).resolve()}")
    print(f"Domain: {args.domain}")
    print(f"Variant: {args.variant}")
    print(f"Status: {result.status}")
    return 0


def _load_optimizer_hypotheses(path: str | None) -> list[dict[str, object]]:
    if not path:
        return [
            {
                "id": "hyp-mixed-model",
                "title": "Mixed interaction model",
                "core_claim": "A mixed phonon and correlated-hopping model produces a stable superconducting branch in a bounded parameter region.",
                "assumptions": ["mean-field reduction is explicit", "parameters are bounded"],
                "testable_predictions": ["gap remains nonzero in mixed region", "energy ledger separates kinetic and interaction terms"],
                "falsification_criteria": ["no stable gap solution", "energy closure fails"],
                "supporting_evidence": ["phase1_minimal_bcs_tool/base_solution.json"],
                "novelty_statement": "combines two mechanisms with explicit energy ledger",
            },
            {
                "id": "hyp-unbounded",
                "title": "Unbounded universal model",
                "core_claim": "This mechanism can prove everything and is always true.",
                "assumptions": [],
                "testable_predictions": [],
                "falsification_criteria": [],
            },
            {
                "id": "hyp-contrarian",
                "title": "Contrarian single-channel branch",
                "core_claim": "A single-channel model may explain the same observables with fewer parameters.",
                "assumptions": ["model comparison uses held-out data"],
                "testable_predictions": ["mixed model does not improve held-out score"],
                "falsification_criteria": ["mixed model improves every observable group under grouped holdout"],
            },
        ]
    target = Path(path)
    if target.suffix == ".jsonl":
        return read_jsonl(target)
    payload = read_json(target)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "hypotheses" in payload:
        return payload["hypotheses"]
    raise ValueError("hypotheses file must be a JSON list, JSONL file, or {'hypotheses': [...]}")


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
        if args.command == "run-superconductivity-campaign":
            return _run_superconductivity_campaign(args)
        if args.command == "validate-superconductivity-campaign":
            return _validate_superconductivity_campaign(args)
        if args.command == "run-minimal-bcs":
            return _run_minimal_bcs(args)
        if args.command == "validate-minimal-bcs":
            return _validate_minimal_bcs(args)
        if args.command == "phase2-acquire":
            return _phase2_acquire(args)
        if args.command == "validate-phase2-acquisition":
            return _validate_phase2_acquisition(args)
        if args.command == "phase2-review-staging":
            return _phase2_review_staging(args)
        if args.command == "phase2-promote":
            return _phase2_promote(args)
        if args.command == "phase2-review-row":
            return _phase2_review_row(args)
        if args.command == "phase2-promote-reviewed":
            return _phase2_promote_reviewed(args)
        if args.command == "phase2-digitization-queue":
            return _phase2_digitization_queue(args)
        if args.command == "phase2-import-digitized":
            return _phase2_import_digitized(args)
        if args.command == "phase2-coverage":
            return _phase2_coverage(args)
        if args.command == "phase2-compare":
            return _phase2_compare(args)
        if args.command == "test-live-models":
            return asyncio.run(_test_live_models(args))
        if args.command == "test-data-connections":
            return _test_data_connections(args)
        if args.command == "run-v22-campaign":
            return _run_v22_campaign(args)
        if args.command == "validate-v22-campaign":
            return _validate_v22_campaign(args)
        if args.command == "rebuild-index":
            return _rebuild_index(args)
        if args.command == "validate-index":
            return _validate_index(args)
        if args.command == "query-index":
            return _query_index(args)
        if args.command == "build-claim-dag-db":
            return _build_claim_dag_db(args)
        if args.command == "validate-claim-dag-db":
            return _validate_claim_dag_db(args)
        if args.command == "query-claim-dag-db":
            return _query_claim_dag_db(args)
        if args.command == "domains-list":
            return _domains_list(args)
        if args.command == "domains-inspect":
            return _domains_inspect(args)
        if args.command == "acquisition-run":
            return _generic_acquisition_run(args)
        if args.command == "hypotheses-optimize":
            return _hypotheses_optimize(args)
        if args.command == "hypotheses-portfolio":
            return _hypotheses_portfolio(args)
        if args.command == "failures-list":
            return _failures_list(args)
        if args.command == "benchmark-run":
            return _benchmark_run(args)
    except (ValidationError, ValueError, ProviderError, CompletedRunError, NetworkDisabledError) as exc:
        print(f"Error: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
