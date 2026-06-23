from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from coscientist.config import WorkflowConfig
from coscientist.literature.http import NetworkDisabledError
from coscientist.literature.scholarly import ScholarlyCorpusResult, ScholarlyLiteratureOrchestrator
from coscientist.orchestration.workflow import run_workflow
from coscientist.pilot.artifacts import REQUIRED_V1_ARTIFACTS, read_json, write_json, write_jsonl
from coscientist.pilot.evaluation import compare_rounds, evaluate_round
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.project_io import load_project_spec
from coscientist.pilot.reports import build_human_review_package, build_pilot_report
from coscientist.providers.mock import MockProvider
from coscientist.schemas.evaluation import RunManifest
from coscientist.schemas.hypothesis import Hypothesis
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
    literature_mode: str | None = None,
    search_providers: list[str] | None = None,
    enrichment_providers: list[str] | None = None,
    max_literature_results: int | None = None,
    corpus_path: str | Path | None = None,
    acquire_literature_only: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    if live_model:
        raise ValueError("V1 run-project uses the deterministic mock model provider only.")
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
        return build_literature_artifacts(project, corpus_result, run_dir, run_id, live_network, live_model, dry_run=dry_run)

    config = WorkflowConfig(
        max_llm_calls=project.maximum_model_call_budget,
        evolution_rounds=project.maximum_evolution_rounds,
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
    await run_workflow(goal, MockProvider(), config, LocalStore(runs_dir), run_id=run_id)
    return build_v1_artifacts(project, corpus_result, run_dir, run_id, live_network=live_network, live_model=live_model)


def build_v1_artifacts(
    project: ResearchProjectSpec,
    corpus_result: ScholarlyCorpusResult,
    run_dir: Path,
    run_id: str,
    *,
    live_network: bool = False,
    live_model: bool = False,
) -> Path:
    started = datetime.now(UTC)
    corpus = corpus_result.papers
    initial = _load_hypotheses(run_dir / "hypotheses_initial.json")
    round_1 = _load_hypotheses(run_dir / "hypotheses_round_1.json")
    round_2 = _load_hypotheses(run_dir / "hypotheses_round_2.json")
    final_ids = set(read_json(run_dir / "run_state.json").get("active_hypothesis_ids", []))
    final = [hypothesis for hypothesis in round_2 if hypothesis.id in final_ids] or round_2[:3]
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
    manifest = RunManifest(
        run_id=run_id,
        project_id=project.project_id,
        offline_mode=not live_network,
        live_network_enabled=live_network,
        live_model_enabled=live_model,
        created_at=started,
        completed_at=datetime.now(UTC),
        artifacts=REQUIRED_V1_ARTIFACTS,
    )
    write_literature_artifacts(project, corpus_result, run_dir)
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
        "raw_openalex_records.jsonl",
        "raw_arxiv_records.jsonl",
        "crossref_enrichment.jsonl",
        "unpaywall_enrichment.jsonl",
        "metadata_conflicts.jsonl",
        "deduplication_report.json",
        "corpus_manifest.json",
    ]
    write_literature_artifacts(project, corpus_result, run_dir)
    manifest = RunManifest(
        run_id=run_id,
        project_id=project.project_id,
        offline_mode=not live_network,
        live_network_enabled=live_network,
        live_model_enabled=live_model,
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
