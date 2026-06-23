from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from coscientist.config import WorkflowConfig
from coscientist.literature.http import NetworkDisabledError
from coscientist.literature.scholarly import ScholarlyCorpusResult, ScholarlyLiteratureOrchestrator
from coscientist.orchestration.workflow import run_workflow
from coscientist.pilot.artifacts import REQUIRED_V1_ARTIFACTS, read_json, write_json, write_jsonl
from coscientist.pilot.evaluation import compare_rounds, evaluate_round
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.project_io import load_project_spec
from coscientist.pilot.reports import build_human_review_package, build_pilot_report
from coscientist.providers.base import StructuredLLMProvider
from coscientist.providers.mock import MockProvider
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.providers.usage import provider_status, summarize_model_usage
from coscientist.schemas.evaluation import RunManifest
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.model_provider import ModelCallRecord
from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.scholarly import ProjectLiteratureConfig
from coscientist.storage.local_store import LocalStore


class CompletedRunError(RuntimeError):
    pass


async def run_pilot_project(
    project_path: str | Path,
    *,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    live_network: bool = False,
    live_model: bool = False,
    provider_name: str = "mock",
    literature_mode: str | None = None,
    search_providers: list[str] | None = None,
    enrichment_providers: list[str] | None = None,
    max_literature_results: int | None = None,
    max_model_calls: int | None = None,
    max_evolution_rounds: int | None = None,
    corpus_path: str | Path | None = None,
    acquire_literature_only: bool = False,
    smoke: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    if provider_name == "openai" and not live_model and not dry_run:
        raise ValueError("--provider openai requires explicit --live-model for run-project.")
    if live_model and provider_name == "mock":
        raise ValueError("--live-model requires --provider openai.")
    project_file = Path(project_path)
    project = load_project_spec(project_file)
    literature_config = _resolve_literature_config(
        project,
        literature_mode=literature_mode,
        search_providers=search_providers,
        enrichment_providers=enrichment_providers,
        max_literature_results=max_literature_results,
        corpus_path=corpus_path,
    )
    project = project.model_copy(update={"literature": literature_config})
    if literature_config.mode == "live" and not live_network and not dry_run:
        raise NetworkDisabledError("live literature mode requires explicit --live-network; use --dry-run to inspect the plan offline")
    run_id = run_id or f"{project.project_id}-pilot"
    run_dir = Path(runs_dir) / run_id
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        if manifest.get("completed_at"):
            raise CompletedRunError(f"completed run artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = ScholarlyLiteratureOrchestrator(
        project=project,
        literature_config=literature_config,
        allow_live_network=live_network,
    )
    fixture_path = _resolve_project_path(project_file, project.literature_fixture_path)
    if corpus_path:
        existing_path = Path(corpus_path)
    else:
        existing_path = _resolve_project_path(project_file, literature_config.existing_corpus_path)
    if dry_run:
        corpus_result = orchestrator.dry_run()
    else:
        corpus_result = await orchestrator.acquire(fixture_path=fixture_path, existing_corpus_path=existing_path)
    if acquire_literature_only or dry_run:
        return build_literature_artifacts(
            project,
            corpus_result,
            run_dir,
            run_id,
            live_network,
            live_model,
            provider_name=provider_name,
            dry_run=dry_run,
        )

    provider = _make_provider(provider_name, live_model)
    model_call_budget = _model_call_budget(project, max_model_calls, smoke)
    evolution_rounds = _evolution_rounds(project, max_evolution_rounds, smoke)
    config = WorkflowConfig(
        max_llm_calls=model_call_budget,
        evolution_rounds=evolution_rounds,
        generators=1 if smoke else 4,
        hypotheses_per_generator=1 if smoke else 3,
        top_k_after_review=1 if smoke else 6,
        children_per_selected_hypothesis=1 if smoke else 2,
        final_top_k=1 if smoke else 3,
    )
    goal = ResearchGoal(
        id=project.project_id,
        title=project.title,
        question=project.research_question,
        background=project.background,
        constraints=project.constraints,
        desired_attributes=project.evaluation_criteria,
        evaluation_criteria=project.evaluation_criteria,
        prohibited_methods=project.excluded_directions,
        max_rounds=project.maximum_evolution_rounds,
    )
    await run_workflow(goal, provider, config, LocalStore(runs_dir), run_id=run_id)
    return build_v1_artifacts(
        project,
        corpus_result,
        run_dir,
        run_id,
        live_network=live_network,
        live_model=live_model,
        provider=provider,
        model_call_budget=model_call_budget,
        run_status="complete",
    )


def build_v1_artifacts(
    project: ResearchProjectSpec,
    corpus_result: ScholarlyCorpusResult,
    run_dir: Path,
    run_id: str,
    *,
    live_network: bool = False,
    live_model: bool = False,
    provider: StructuredLLMProvider | None = None,
    model_call_budget: int | None = None,
    run_status: str = "complete",
) -> Path:
    started = datetime.now(UTC)
    corpus = corpus_result.papers
    initial = _load_hypotheses(run_dir / "hypotheses_initial.json")
    round_1 = _load_hypotheses(run_dir / "hypotheses_round_1.json") if (run_dir / "hypotheses_round_1.json").exists() else []
    round_2 = _load_hypotheses(run_dir / "hypotheses_round_2.json") if (run_dir / "hypotheses_round_2.json").exists() else []
    final_ids = set(read_json(run_dir / "run_state.json").get("active_hypothesis_ids", []))
    final_source = round_2 or round_1 or initial
    final = [hypothesis for hypothesis in final_source if hypothesis.id in final_ids] or final_source[:3]
    rounds = {
        "initial": attach_fixture_evidence(initial, corpus, "initial"),
        "reviewed": attach_fixture_evidence(initial, corpus, "reviewed"),
        "evolution_round_1": attach_fixture_evidence(round_1, corpus, "evolution_round_1"),
        "evolution_round_2": attach_fixture_evidence(round_2, corpus, "evolution_round_2"),
        "final": attach_fixture_evidence(final, corpus, "final"),
    }
    verifications_by_round = {
        label: verify_hypothesis_evidence(hypotheses, corpus)
        for label, hypotheses in rounds.items()
    }
    evaluations = [
        evaluate_round(hypotheses, verifications_by_round[label], label)
        for label, hypotheses in rounds.items()
    ]
    comparison = compare_rounds(project, evaluations, rounds, verifications_by_round)
    all_verifications = [record for records in verifications_by_round.values() for record in records]
    model_calls = _model_call_records(provider)
    model_usage = summarize_model_usage(provider.name if provider else "mock", "live" if live_model else "mock", model_calls)
    model_status = provider_status(
        provider=provider.name if provider else "mock",
        model_mode="live" if live_model else "mock",
        live_model_enabled=live_model,
        authentication_configured=_auth_configured(provider),
        sanitized_base_url=getattr(provider, "sanitized_base_url", None),
        requested_model=getattr(provider, "model", None),
        records=model_calls,
    )
    manifest = RunManifest(
        run_id=run_id,
        project_id=project.project_id,
        offline_mode=not live_network,
        live_network_enabled=live_network,
        live_model_enabled=live_model,
        model_mode="live" if live_model else "mock",
        model_provider=provider.name if provider else "mock",
        sanitized_model_base_url=getattr(provider, "sanitized_base_url", None),
        requested_model=getattr(provider, "model", None),
        returned_models=sorted({record.returned_model for record in model_calls if record.returned_model}),
        literature_mode=project.literature.mode,
        model_call_budget=model_call_budget,
        model_usage=model_usage.model_dump(mode="json"),
        provider_failures=model_usage.failed_call_count,
        structured_output_failures=model_usage.structured_output_failures,
        repair_attempts=model_usage.repair_attempts,
        run_status=run_status,
        created_at=started,
        completed_at=datetime.now(UTC),
        artifacts=REQUIRED_V1_ARTIFACTS,
    )
    write_literature_artifacts(project, corpus_result, run_dir)
    write_jsonl(run_dir / "model_calls.jsonl", model_calls)
    write_json(run_dir / "model_usage.json", model_usage)
    write_json(run_dir / "model_provider_status.json", model_status)
    write_jsonl(run_dir / "hypotheses_initial.jsonl", rounds["initial"])
    write_jsonl(run_dir / "reviews.jsonl", read_json(run_dir / "reviews_round_0.json"))
    write_jsonl(run_dir / "evidence_verification.jsonl", all_verifications)
    write_jsonl(run_dir / "rankings.jsonl", read_json(run_dir / "ranking_round_0.json"))
    write_jsonl(run_dir / "evolution_round_1.jsonl", rounds["evolution_round_1"])
    write_jsonl(run_dir / "evolution_round_2.jsonl", rounds["evolution_round_2"])
    write_json(run_dir / "hypotheses_final.json", rounds["final"])
    write_json(run_dir / "evaluation_by_round.json", evaluations)
    write_json(run_dir / "round_comparison.json", comparison)
    write_json(run_dir / "lineage.json", {hypothesis.id: hypothesis.parent_ids for hypothesis in rounds["final"]})
    write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "report.md").write_text(
        build_pilot_report(project, rounds["final"], evaluations, comparison, all_verifications),
        encoding="utf-8",
    )
    (run_dir / "human_review.md").write_text(
        build_human_review_package(project, corpus, rounds["final"], comparison, all_verifications),
        encoding="utf-8",
    )
    return run_dir


def build_literature_artifacts(
    project: ResearchProjectSpec,
    corpus_result: ScholarlyCorpusResult,
    run_dir: Path,
    run_id: str,
    live_network: bool,
    live_model: bool,
    *,
    provider_name: str = "mock",
    dry_run: bool = False,
) -> Path:
    started = datetime.now(UTC)
    artifacts = [
        "run_manifest.json",
        "project_snapshot.json",
        "resolved_configuration.json",
        "corpus.jsonl",
        "normalized_papers.jsonl",
        "literature_queries.jsonl",
        "literature_search_events.jsonl",
        "provider_status.json",
        "provider_usage.json",
        "model_calls.jsonl",
        "model_usage.json",
        "model_provider_status.json",
        "raw_openalex_records.jsonl",
        "raw_arxiv_records.jsonl",
        "crossref_enrichment.jsonl",
        "unpaywall_enrichment.jsonl",
        "metadata_conflicts.jsonl",
        "deduplication_report.json",
        "corpus_manifest.json",
    ]
    write_literature_artifacts(project, corpus_result, run_dir)
    model_usage = summarize_model_usage(provider_name, "live" if live_model else "mock", [])
    model_status = provider_status(
        provider=provider_name,
        model_mode="live" if live_model else "mock",
        live_model_enabled=live_model,
        authentication_configured=bool(os.getenv("OPENAI_API_KEY")) if provider_name == "openai" else False,
        sanitized_base_url=_sanitized_env_base_url() if provider_name == "openai" else None,
        requested_model=os.getenv("OPENAI_MODEL") if provider_name == "openai" else None,
        records=[],
        dry_run=dry_run,
    )
    write_jsonl(run_dir / "model_calls.jsonl", [])
    write_json(run_dir / "model_usage.json", model_usage)
    write_json(run_dir / "model_provider_status.json", model_status)
    manifest = RunManifest(
        run_id=run_id,
        project_id=project.project_id,
        offline_mode=not live_network,
        live_network_enabled=live_network,
        live_model_enabled=live_model,
        model_mode="live" if live_model else "mock",
        model_provider=provider_name,
        literature_mode=project.literature.mode,
        model_usage=model_usage.model_dump(mode="json"),
        run_status="dry_run" if dry_run else "complete",
        created_at=started,
        completed_at=datetime.now(UTC),
        artifacts=artifacts,
    )
    write_json(run_dir / "run_manifest.json", manifest)
    if dry_run:
        (run_dir / "report.md").write_text("Dry-run complete. No network calls were made.\n", encoding="utf-8")
    return run_dir


def write_literature_artifacts(project: ResearchProjectSpec, corpus_result: ScholarlyCorpusResult, run_dir: Path) -> None:
    write_json(run_dir / "project_snapshot.json", project)
    write_json(run_dir / "resolved_configuration.json", project.literature)
    write_jsonl(run_dir / "corpus.jsonl", corpus_result.papers)
    write_jsonl(run_dir / "normalized_papers.jsonl", corpus_result.papers)
    write_jsonl(run_dir / "literature_queries.jsonl", corpus_result.query_records)
    write_jsonl(run_dir / "literature_search_events.jsonl", corpus_result.provider_events)
    write_json(run_dir / "provider_status.json", corpus_result.provider_status)
    write_json(run_dir / "provider_usage.json", corpus_result.provider_usage)
    write_jsonl(run_dir / "raw_openalex_records.jsonl", corpus_result.raw_records.get("openalex", []))
    write_jsonl(run_dir / "raw_arxiv_records.jsonl", corpus_result.raw_records.get("arxiv", []))
    write_jsonl(run_dir / "crossref_enrichment.jsonl", corpus_result.metadata_resolutions)
    write_jsonl(run_dir / "unpaywall_enrichment.jsonl", corpus_result.full_text_locations)
    write_jsonl(run_dir / "metadata_conflicts.jsonl", corpus_result.metadata_conflicts)
    write_json(run_dir / "deduplication_report.json", corpus_result.deduplication_report)
    write_json(run_dir / "corpus_manifest.json", corpus_result.corpus_manifest)


def _resolve_literature_config(
    project: ResearchProjectSpec,
    *,
    literature_mode: str | None,
    search_providers: list[str] | None,
    enrichment_providers: list[str] | None,
    max_literature_results: int | None,
    corpus_path: str | Path | None,
) -> ProjectLiteratureConfig:
    updates = {}
    if literature_mode:
        updates["mode"] = literature_mode
    if search_providers:
        updates["search_providers"] = search_providers
        updates["queries"] = [
            {**query.model_dump(), "providers": search_providers}
            for query in project.literature.queries
        ]
    if enrichment_providers:
        updates["enrichment_providers"] = enrichment_providers
    elif literature_mode == "live" and project.literature.enrichment_providers == ["mock"]:
        updates["enrichment_providers"] = []
    if max_literature_results:
        updates["max_total_results"] = max_literature_results
        updates["max_results_per_provider"] = max_literature_results
    if corpus_path:
        updates["existing_corpus_path"] = str(corpus_path)
        if not literature_mode:
            updates["mode"] = "existing"
    return ProjectLiteratureConfig.model_validate({**project.literature.model_dump(), **updates})


def _make_provider(provider_name: str, live_model: bool) -> StructuredLLMProvider:
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "openai":
        if not live_model:
            raise ValueError("--provider openai requires --live-model")
        return OpenAICompatibleProvider()
    raise ValueError(f"unknown provider: {provider_name}")


def _model_call_budget(project: ResearchProjectSpec, override: int | None, smoke: bool) -> int:
    if override is not None:
        return override
    if smoke:
        return min(project.maximum_model_call_budget, 4)
    return project.maximum_model_call_budget


def _evolution_rounds(project: ResearchProjectSpec, override: int | None, smoke: bool) -> int:
    if override is not None:
        return override
    if smoke:
        return 0
    return project.maximum_evolution_rounds


def _model_call_records(provider: StructuredLLMProvider | None) -> list[ModelCallRecord]:
    return list(getattr(provider, "call_records", [])) if provider else []


def _auth_configured(provider: StructuredLLMProvider | None) -> bool:
    return bool(getattr(provider, "api_key", None)) if provider else False


def _sanitized_env_base_url() -> str | None:
    base = os.getenv("OPENAI_BASE_URL")
    if not base:
        return None
    parsed = urlparse(base)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid-base-url>"
    return f"{parsed.scheme}://{parsed.netloc}"


def run_pilot_project_sync(*args, **kwargs) -> Path:
    return asyncio.run(run_pilot_project(*args, **kwargs))


def _load_hypotheses(path: Path) -> list[Hypothesis]:
    return [Hypothesis.model_validate(item) for item in read_json(path)]


def _resolve_project_path(project_file: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return project_file.parent / path
