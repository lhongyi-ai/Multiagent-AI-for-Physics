from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from coscientist.config import WorkflowConfig
from coscientist.orchestration.workflow import run_workflow
from coscientist.pilot.artifacts import REQUIRED_V1_ARTIFACTS, read_json, write_json, write_jsonl
from coscientist.pilot.evaluation import compare_rounds, evaluate_round
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.project_io import load_fixture_corpus, load_project_spec
from coscientist.pilot.reports import build_human_review_package, build_pilot_report
from coscientist.providers.mock import MockProvider
from coscientist.schemas.evaluation import RunManifest
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.project import ResearchProjectSpec
from coscientist.schemas.research_goal import ResearchGoal
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
    force: bool = False,
) -> Path:
    if live_network or live_model:
        raise ValueError("V1 deterministic pilot defaults to offline mode; live execution is not implemented for run-project.")
    project_file = Path(project_path)
    project = load_project_spec(project_file)
    run_id = run_id or f"{project.project_id}-pilot"
    run_dir = Path(runs_dir) / run_id
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        if manifest.get("completed_at"):
            raise CompletedRunError(f"completed run artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = _resolve_project_path(project_file, project.literature_fixture_path)
    corpus = load_fixture_corpus(corpus_path) if corpus_path else []
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
    return build_v1_artifacts(project, corpus, run_dir, run_id)


def build_v1_artifacts(project: ResearchProjectSpec, corpus, run_dir: Path, run_id: str) -> Path:
    started = datetime.now(UTC)
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
        offline_mode=True,
        live_network_enabled=False,
        live_model_enabled=False,
        created_at=started,
        completed_at=datetime.now(UTC),
        artifacts=REQUIRED_V1_ARTIFACTS,
    )
    write_json(run_dir / "project_snapshot.json", project)
    write_jsonl(run_dir / "corpus.jsonl", corpus)
    write_jsonl(run_dir / "normalized_papers.jsonl", corpus)
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
