from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.v17 import (
    BeamSelection,
    CandidateArchiveSnapshot,
    CandidateSolution,
    CandidateStatus,
    CandidateStatusEvent,
    DiscoveryProject,
    ProblemFormalization,
    SearchCheckpoint,
    SearchConfig,
    SearchStrategyMetrics,
    SearchTask,
    TaskBudget,
    TaskResult,
    TournamentComparison,
    VerifierResult,
)
from coscientist.schemas.v20 import (
    AdaptiveBudgetAllocation,
    CandidateRating,
    EloTournamentState,
    ImplementationComparison,
    ProviderRoutingPlan,
    ReproductionResult,
    ReproductionRun,
    RoleModelRoute,
    StrategyAllocation,
)
from coscientist.verifiers.registry import default_verifier_registry


def load_discovery_project(path: str | Path) -> DiscoveryProject:
    project_path = Path(path)
    data = yaml.safe_load(project_path.read_text(encoding="utf-8")) or {}
    if isinstance(data.get("created_at"), str):
        data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
    return DiscoveryProject.model_validate(data)


def adapt_hypothesis_to_candidate(hypothesis: Hypothesis, *, problem_id: str, step: int = 0) -> CandidateSolution:
    return CandidateSolution(
        candidate_id=f"cand-{hypothesis.id}",
        problem_id=problem_id,
        candidate_type="hypothesis",
        title=hypothesis.title,
        summary=hypothesis.core_claim,
        formal_representation=hypothesis.mechanism,
        assumptions=hypothesis.assumptions,
        predicted_observables=hypothesis.testable_predictions,
        falsification_conditions=hypothesis.falsification_criteria,
        parent_ids=[f"cand-{item}" for item in hypothesis.parent_ids],
        root_candidate_id=f"cand-{hypothesis.parent_ids[0]}" if hypothesis.parent_ids else f"cand-{hypothesis.id}",
        lineage_depth=max(0, hypothesis.version - 1),
        generation_strategy="mainstream_extension",
        linked_evidence_ids=[paper_id for link in hypothesis.evidence_links for paper_id in link.supporting_paper_ids],
        created_step=step,
        updated_step=step,
        provenance=[f"adapted_from_hypothesis:{hypothesis.id}"],
    )


class CandidateArchive:
    def __init__(self, problem_id: str) -> None:
        self.problem_id = problem_id
        self.candidates: dict[str, CandidateSolution] = {}
        self.status_history: list[CandidateStatusEvent] = []

    def add(self, candidate: CandidateSolution) -> None:
        if candidate.candidate_id not in self.candidates:
            self.candidates[candidate.candidate_id] = candidate

    def get(self, candidate_id: str) -> CandidateSolution:
        return self.candidates[candidate_id]

    def transition(self, candidate_id: str, status: CandidateStatus, *, step: int, reason: str) -> None:
        candidate = self.candidates[candidate_id]
        previous = candidate.scientific_status
        self.candidates[candidate_id] = candidate.model_copy(update={"scientific_status": status, "updated_step": step})
        self.status_history.append(CandidateStatusEvent(candidate_id=candidate_id, from_status=previous, to_status=status, step=step, reason=reason))

    def duplicate_groups(self) -> list[list[str]]:
        by_form: dict[str, list[str]] = defaultdict(list)
        for candidate in self.candidates.values():
            key = " ".join((candidate.formal_representation or candidate.summary).lower().split())
            by_form[key].append(candidate.candidate_id)
        return [ids for ids in by_form.values() if len(ids) > 1]

    def lineage_graph(self) -> dict[str, list[str]]:
        return {candidate.candidate_id: candidate.parent_ids for candidate in self.candidates.values()}

    def active_candidates(self) -> list[CandidateSolution]:
        return [candidate for candidate in self.candidates.values() if candidate.scientific_status in {"proposed", "awaiting_verification", "partially_verified", "promising", "strong_verification_passed", "expert_review_required"}]

    def cluster_representatives(self) -> list[CandidateSolution]:
        groups: dict[str, list[CandidateSolution]] = defaultdict(list)
        for candidate in self.active_candidates():
            key = (candidate.formal_representation or candidate.summary).lower().split()[:4]
            groups[" ".join(key)].append(candidate)
        return [max(group, key=lambda item: (item.aggregate_search_score, -item.lineage_depth, item.candidate_id)) for group in groups.values()]

    def underexplored_branches(self) -> list[CandidateSolution]:
        by_root = Counter(candidate.root_candidate_id or candidate.candidate_id for candidate in self.candidates.values())
        return sorted(self.active_candidates(), key=lambda item: (by_root[item.root_candidate_id or item.candidate_id], item.candidate_id))

    def snapshot(self) -> CandidateArchiveSnapshot:
        return CandidateArchiveSnapshot(
            problem_id=self.problem_id,
            candidates=sorted(self.candidates.values(), key=lambda item: item.candidate_id),
            status_history=self.status_history,
            duplicate_groups=self.duplicate_groups(),
        )

    @classmethod
    def from_snapshot(cls, snapshot: CandidateArchiveSnapshot) -> CandidateArchive:
        archive = cls(snapshot.problem_id)
        archive.candidates = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
        archive.status_history = list(snapshot.status_history)
        return archive


class SearchTaskQueue:
    def __init__(self, tasks: list[SearchTask] | None = None) -> None:
        self.tasks = {task.task_id: task for task in tasks or []}

    def add(self, task: SearchTask) -> None:
        if task.task_id not in self.tasks:
            self.tasks[task.task_id] = task

    def ready(self) -> list[SearchTask]:
        completed = {task.task_id for task in self.tasks.values() if task.status == "completed"}
        ready = []
        for task in self.tasks.values():
            if task.status in {"pending", "ready"} and set(task.dependencies).issubset(completed):
                ready.append(task.model_copy(update={"status": "ready"}))
        return sorted(ready, key=lambda item: (item.priority, item.created_step, item.task_id))

    def mark(self, task_id: str, status: str, *, step: int, artifacts: list[str] | None = None, failure: str | None = None) -> None:
        task = self.tasks[task_id]
        updates: dict[str, Any] = {"status": status}
        if status == "running":
            updates["started_step"] = step
        if status in {"completed", "failed", "cancelled"}:
            updates["completed_step"] = step
        if artifacts is not None:
            updates["result_artifact_ids"] = artifacts
        if failure:
            updates["failure_reason"] = failure
        self.tasks[task_id] = task.model_copy(update=updates)

    def cancel(self, task_id: str, *, step: int) -> None:
        self.mark(task_id, "cancelled", step=step)

    def completed_ids(self) -> list[str]:
        return sorted(task.task_id for task in self.tasks.values() if task.status == "completed")

    def list(self) -> list[SearchTask]:
        return sorted(self.tasks.values(), key=lambda item: (item.created_step, item.task_id))


class SearchController:
    def __init__(self, project: DiscoveryProject) -> None:
        self.project = project

    def initial_tasks(self) -> list[SearchTask]:
        problem_id = self.project.problem.problem_id
        return [
            SearchTask(task_id="task-formalize", problem_id=problem_id, task_type="formalize_problem", priority=10, created_step=0),
            SearchTask(task_id="task-cheap-filter", problem_id=problem_id, task_type="cheap_filter", priority=20, dependencies=["task-formalize"], created_step=0),
            SearchTask(task_id="task-verify", problem_id=problem_id, task_type="verify_candidate", priority=30, dependencies=["task-cheap-filter"], created_step=0, budget=TaskBudget(verifier_call_budget=self.project.search.verifier_call_budget)),
            SearchTask(task_id="task-compare", problem_id=problem_id, task_type="compare_candidates", priority=40, dependencies=["task-verify"], created_step=0),
            SearchTask(task_id="task-summary", problem_id=problem_id, task_type="summarize_search_state", priority=90, dependencies=["task-compare"], created_step=0),
        ]

    def adaptive_tasks(self, archive: CandidateArchive, verifier_results: list[VerifierResult], plateau: bool, step: int) -> list[SearchTask]:
        tasks: list[SearchTask] = []
        failures = Counter(check for result in verifier_results for check in result.checks_failed)
        duplicates = archive.duplicate_groups()
        existing_strategies = {candidate.generation_strategy for candidate in archive.candidates.values()}
        if duplicates and "repair_failed_candidate" not in existing_strategies:
            tasks.append(SearchTask(task_id=f"task-repair-duplicates-{step}", problem_id=self.project.problem.problem_id, task_type="repair_candidate", priority=45, candidate_ids=duplicates[0][:1], created_step=step))
        if failures and failures.most_common(1)[0][1] >= 2 and "counterexample_search" not in existing_strategies:
            tasks.append(SearchTask(task_id=f"task-counterexample-{step}", problem_id=self.project.problem.problem_id, task_type="search_counterexample", priority=46, created_step=step))
        if plateau and "cross_domain_transfer" not in existing_strategies:
            tasks.append(SearchTask(task_id=f"task-plateau-transfer-{step}", problem_id=self.project.problem.problem_id, task_type="cross_domain_transfer", priority=47, created_step=step))
        if len(archive.active_candidates()) <= self.project.search.beam_width:
            tasks.append(SearchTask(task_id=f"task-expert-review-{step}", problem_id=self.project.problem.problem_id, task_type="request_expert_review", priority=95, candidate_ids=[candidate.candidate_id for candidate in archive.active_candidates()], created_step=step))
        return tasks


def run_discovery_project(project_path: str | Path, *, runs_dir: str | Path = "runs", run_id: str | None = None, force: bool = False, stop_after_tasks: int | None = None) -> Path:
    project_file = Path(project_path)
    project = load_discovery_project(project_file)
    run_id = run_id or f"{project.project_id}-discovery"
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"discovery artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = CandidateArchive(project.problem.problem_id)
    for candidate in project.initial_candidates:
        archive.add(candidate)
    queue = SearchTaskQueue(SearchController(project).initial_tasks())
    state = _SearchState(project, run_id, run_dir, archive, queue)
    _run_state(state, stop_after_tasks=stop_after_tasks)
    return run_dir


def resume_discovery_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = SearchCheckpoint.model_validate_json(Path(checkpoint_path).read_text(encoding="utf-8"))
    run_dir = Path(checkpoint_path).parent
    project = DiscoveryProject.model_validate_json((run_dir / "discovery_project.json").read_text(encoding="utf-8"))
    if checkpoint.project_hash != _hash_json(_agent_visible_project(project).model_dump(mode="json")):
        raise ValueError("stale checkpoint: project hash mismatch")
    archive = CandidateArchive.from_snapshot(checkpoint.archive)
    queue = SearchTaskQueue(checkpoint.queue)
    state = _SearchState(project, checkpoint.run_id, run_dir, archive, queue, verifier_results=checkpoint.verifier_results, current_step=checkpoint.current_step, resume_count=checkpoint.resume_count + 1)
    _run_state(state)
    return run_dir


class _SearchState:
    def __init__(self, project: DiscoveryProject, run_id: str, run_dir: Path, archive: CandidateArchive, queue: SearchTaskQueue, *, verifier_results: list[VerifierResult] | None = None, current_step: int = 0, resume_count: int = 0) -> None:
        self.project = project
        self.run_id = run_id
        self.run_dir = run_dir
        self.archive = archive
        self.queue = queue
        self.verifier_results = verifier_results or []
        self.current_step = current_step
        self.resume_count = resume_count
        self.completed_task_count = len(queue.completed_ids())
        self.beam_history: list[BeamSelection] = []
        self.tournament: list[TournamentComparison] = []
        self.elo_ratings: dict[str, CandidateRating] = {}
        self.deep_comparison_pairings: list[str] = []
        self.reproduction_results: list[ReproductionResult] = []
        self.plateau_history: list[dict[str, float | int | str]] = []
        self.strategy_metrics = {strategy: SearchStrategyMetrics(strategy=strategy) for strategy in ["mainstream_extension", "counterexample_search", "assumption_relaxation", "repair_failed_candidate", "combine_partial_solutions", "cross_domain_transfer"]}
        self.budgets_spent = {"model_calls": 0, "tokens": 0, "verifier_calls": 0}


def _run_state(state: _SearchState, *, stop_after_tasks: int | None = None) -> None:
    _write_static_artifacts(state)
    controller = SearchController(state.project)
    tasks_run = 0
    while state.current_step < state.project.search.max_steps:
        ready = state.queue.ready()
        if not ready:
            break
        task = ready[0]
        state.current_step += 1
        state.queue.mark(task.task_id, "running", step=state.current_step)
        result = _execute_task(state, task)
        state.queue.mark(task.task_id, result.status, step=state.current_step, artifacts=result.artifact_ids, failure=result.notes if result.status == "failed" else None)
        tasks_run += 1
        plateau = _detect_plateau(state)
        for new_task in controller.adaptive_tasks(state.archive, state.verifier_results, plateau, state.current_step):
            if not any(existing.task_type == new_task.task_type for existing in state.queue.tasks.values()):
                state.queue.add(new_task)
        _write_checkpoint(state)
        if stop_after_tasks is not None and tasks_run >= stop_after_tasks:
            break
    _write_outputs(state)


def persist_expert_feedback(run_dir: str | Path, *, candidate_id: str, decision: str, rationale: str, reviewer: str = "local-human") -> Path:
    path = Path(run_dir)
    candidates = [CandidateSolution.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "candidate_archive.jsonl")]
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if candidate_id not in candidate_ids:
        raise ValueError(f"unknown candidate_id for expert feedback: {candidate_id}")
    record = {
        "schema_version": "v17",
        "candidate_id": candidate_id,
        "decision": decision,
        "rationale": rationale,
        "reviewer": reviewer,
        "created_at": datetime.now(UTC).isoformat(),
    }
    feedback_path = path / "expert_feedback.jsonl"
    existing = read_jsonl(feedback_path) if feedback_path.exists() else []
    write_jsonl(feedback_path, [*existing, record])
    return feedback_path


def _execute_task(state: _SearchState, task: SearchTask) -> TaskResult:
    if task.task_type == "formalize_problem":
        formalization = ProblemFormalization(
            problem_id=state.project.problem.problem_id,
            created_at=datetime.now(UTC),
            formal_statement=state.project.problem.precise_statement,
            normalized_constraints=[item.description for item in state.project.problem.known_constraints],
            search_space_summary=f"{len(state.project.initial_candidates)} initial candidates; candidate types: {', '.join(state.project.problem.candidate_types)}",
            evaluator_only_fields_present=bool(state.project.evaluator_only_ground_truth),
        )
        write_json(state.run_dir / "problem_formalization.json", formalization)
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["problem_formalization.json"])
    if task.task_type == "cheap_filter":
        for candidate in list(state.archive.candidates.values()):
            reasons = _cheap_filter_reasons(candidate, state)
            if reasons:
                failed = candidate.model_copy(update={"failure_reason_ids": reasons})
                state.archive.candidates[candidate.candidate_id] = failed
                state.archive.transition(candidate.candidate_id, "cheap_filter_failed", step=state.current_step, reason="; ".join(reasons))
            else:
                state.archive.transition(candidate.candidate_id, "awaiting_verification", step=state.current_step, reason="cheap filters passed")
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["candidate_status_history.jsonl"])
    if task.task_type == "verify_candidate":
        registry = default_verifier_registry()
        evidence_ids = set(state.project.evidence_ids)
        for candidate in state.archive.active_candidates():
            result_ids: list[str] = []
            for verifier in registry.enabled(state.project.enabled_verifiers):
                if state.budgets_spent["verifier_calls"] >= state.project.search.verifier_call_budget:
                    break
                result = verifier.verify(candidate, state.project.problem, evidence_ids=evidence_ids)
                state.verifier_results.append(result)
                result_ids.append(result.verifier_result_id)
                state.budgets_spent["verifier_calls"] += 1
            if result_ids:
                current = state.archive.candidates[candidate.candidate_id]
                merged = [*current.verification_result_ids, *result_ids]
                state.archive.candidates[candidate.candidate_id] = current.model_copy(update={"verification_result_ids": sorted(set(merged))})
            _score_candidate(state, candidate.candidate_id)
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["verifier_results.jsonl"])
    if task.task_type == "compare_candidates":
        state.beam_history.append(_select_beam(state))
        if state.project.search.tournament_ranking_mode in {"elo", "bradley_terry"}:
            state.tournament.extend(_rating_tournament(state))
        else:
            state.tournament.extend(_bounded_tournament(state))
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["beam_selection.json", "tournament_comparisons.jsonl", "elo_tournament_state.json"])
    if task.task_type in {"repair_candidate", "cross_domain_transfer", "search_counterexample"}:
        generated = _derive_candidate_from_task(state, task)
        if generated:
            state.archive.add(generated)
            return TaskResult(task_id=task.task_id, status="completed", candidate_ids=[generated.candidate_id], artifact_ids=["candidate_archive.jsonl"])
    if task.task_type == "request_expert_review":
        for candidate_id in task.candidate_ids:
            if candidate_id in state.archive.candidates:
                state.archive.transition(candidate_id, "expert_review_required", step=state.current_step, reason="bounded search reached review point")
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["expert_review.md"])
    if task.task_type == "summarize_search_state":
        return TaskResult(task_id=task.task_id, status="completed", artifact_ids=["discovery_report.md"])
    return TaskResult(task_id=task.task_id, status="completed")


def _cheap_filter_reasons(candidate: CandidateSolution, state: _SearchState) -> list[str]:
    reasons = []
    if candidate.candidate_type not in state.project.problem.candidate_types:
        reasons.append("unsupported_candidate_type")
    if not candidate.assumptions:
        reasons.append("missing_assumptions")
    if not candidate.predicted_observables:
        reasons.append("missing_predictions")
    if not candidate.falsification_conditions:
        reasons.append("missing_falsification_conditions")
    if candidate.lineage_depth > state.project.search.max_lineage_depth:
        reasons.append("lineage_depth_overflow")
    if set(candidate.linked_evidence_ids) - set(state.project.evidence_ids):
        reasons.append("invalid_evidence_reference")
    if "unrestricted shell" in candidate.summary.lower():
        reasons.append("forbidden_tool_request")
    return reasons


def _score_candidate(state: _SearchState, candidate_id: str) -> None:
    candidate = state.archive.candidates[candidate_id]
    results = [result for result in state.verifier_results if result.candidate_id == candidate_id]
    verifier_score = sum(result.score for result in results) / len(results) if results else 0.0
    contradiction_penalty = 0.2 if any(result.verdict == "fail" for result in results) else 0.0
    components = {
        "verifier_score": round(verifier_score, 3),
        "grounding_score": 1.0 if candidate.linked_evidence_ids else 0.3,
        "novelty": 0.7 if candidate.novelty_status in {"unknown", "possibly_novel"} else 0.2,
        "predictive_specificity": min(1.0, len(candidate.predicted_observables) / 3),
        "contradiction_penalty": contradiction_penalty,
        "complexity_penalty": min(0.3, len(candidate.assumptions) * 0.03),
        "diversity_bonus": 0.1 if candidate.candidate_type == "counterexample" else 0.0,
        "lineage_diversity": min(0.2, candidate.lineage_depth * 0.04),
    }
    aggregate = components["verifier_score"] * 0.45 + components["grounding_score"] * 0.2 + components["novelty"] * 0.1 + components["predictive_specificity"] * 0.15 + components["diversity_bonus"] - components["contradiction_penalty"] - components["complexity_penalty"]
    status = "promising" if aggregate >= 0.55 and not contradiction_penalty else "falsified" if contradiction_penalty else "partially_verified"
    updated = candidate.model_copy(update={"component_scores": components, "aggregate_search_score": max(0.0, min(1.0, round(aggregate, 3)))})
    state.archive.candidates[candidate_id] = updated
    state.archive.transition(candidate_id, status, step=state.current_step, reason="verification-weighted deterministic scoring")


def _select_beam(state: _SearchState) -> BeamSelection:
    candidates = sorted(state.archive.active_candidates(), key=lambda item: (item.aggregate_search_score, item.candidate_type == "counterexample", item.candidate_id), reverse=True)
    selected: list[CandidateSolution] = []
    seen_types: set[str] = set()
    for candidate in candidates:
        if state.project.search.preserve_counterexample_branch and candidate.candidate_type == "counterexample" and candidate not in selected:
            selected.append(candidate)
            seen_types.add(candidate.candidate_type)
            break
    for candidate in candidates:
        if len(selected) >= state.project.search.beam_width:
            break
        if state.project.search.preserve_diverse_clusters and candidate.candidate_type in seen_types and len(selected) < len(seen_types):
            continue
        if candidate not in selected:
            selected.append(candidate)
            seen_types.add(candidate.candidate_type)
    return BeamSelection(
        step=state.current_step,
        selected_candidate_ids=[candidate.candidate_id for candidate in selected[: state.project.search.beam_width]],
        component_formula="0.45*verifier + 0.2*grounding + 0.1*novelty + 0.15*specificity + diversity - contradiction - complexity",
        component_inputs={candidate.candidate_id: candidate.component_scores for candidate in selected},
    )


def _bounded_tournament(state: _SearchState) -> list[TournamentComparison]:
    pool = sorted(state.archive.active_candidates(), key=lambda item: (item.aggregate_search_score, item.candidate_id), reverse=True)[: state.project.search.tournament_max_candidates]
    comparisons: list[TournamentComparison] = []
    limit = state.project.search.tournament_max_comparisons
    for index, left in enumerate(pool):
        for right in pool[index + 1:]:
            if len(comparisons) >= limit:
                return comparisons
            winner = left if left.aggregate_search_score >= right.aggregate_search_score else right
            comparisons.append(TournamentComparison(
                comparison_id=f"cmp-{left.candidate_id}-{right.candidate_id}",
                candidate_a_id=left.candidate_id,
                candidate_b_id=right.candidate_id,
                winner_id=winner.candidate_id,
                rationale="Single-turn deterministic verifier-weighted comparison.",
            ))
    return comparisons


def _rating_tournament(state: _SearchState) -> list[TournamentComparison]:
    pool = sorted(state.archive.active_candidates(), key=lambda item: (item.aggregate_search_score, item.candidate_id), reverse=True)[: state.project.search.tournament_max_candidates]
    completed = {_pair_key(item.candidate_a_id, item.candidate_b_id) for item in state.tournament}
    comparisons: list[TournamentComparison] = []
    limit = state.project.search.tournament_max_comparisons
    candidate_ids = {candidate.candidate_id for candidate in pool}
    for candidate in pool:
        if candidate.candidate_id not in state.elo_ratings:
            state.elo_ratings[candidate.candidate_id] = CandidateRating(
                candidate_id=candidate.candidate_id,
                rating=round(state.project.search.tournament_initial_rating + candidate.aggregate_search_score * 100.0, 3),
                uncertainty=state.project.search.tournament_initial_uncertainty,
            )
    candidate_lookup = {candidate.candidate_id: candidate for candidate in pool}
    candidate_pairs: list[tuple[float, float, str, str]] = []
    for index, left in enumerate(pool):
        for right in pool[index + 1:]:
            key = _pair_key(left.candidate_id, right.candidate_id)
            if key in completed:
                continue
            rating_gap = abs(state.elo_ratings[left.candidate_id].rating - state.elo_ratings[right.candidate_id].rating)
            top_score = max(left.aggregate_search_score, right.aggregate_search_score)
            candidate_pairs.append((rating_gap, -top_score, left.candidate_id, right.candidate_id))
    for _, _, left_id, right_id in sorted(candidate_pairs):
        if len(comparisons) + len(state.tournament) >= limit:
            break
        if left_id not in candidate_ids or right_id not in candidate_ids:
            continue
        left = candidate_lookup[left_id]
        right = candidate_lookup[right_id]
        score_a, score_b = _comparison_scores(left, right)
        winner_id = None
        if score_a > score_b:
            winner_id = left_id
        elif score_b > score_a:
            winner_id = right_id
        deep = _allocate_deep_comparison(state, left_id, right_id)
        _update_rating_pair(state, left_id, right_id, score_a, score_b)
        comparisons.append(TournamentComparison(
            comparison_id=f"cmp-{left_id}-{right_id}",
            candidate_a_id=left_id,
            candidate_b_id=right_id,
            winner_id=winner_id,
            rationale=f"{state.project.search.tournament_ranking_mode} deterministic verifier-weighted update; {'deeper comparison allocated' if deep else 'single bounded comparison'}",
            single_turn=not deep,
        ))
    return comparisons


def _comparison_scores(left: CandidateSolution, right: CandidateSolution) -> tuple[float, float]:
    left_score = left.aggregate_search_score
    right_score = right.aggregate_search_score
    if abs(left_score - right_score) < 1e-9:
        return 0.5, 0.5
    return (1.0, 0.0) if left_score > right_score else (0.0, 1.0)


def _update_rating_pair(state: _SearchState, left_id: str, right_id: str, score_a: float, score_b: float) -> None:
    left = state.elo_ratings[left_id]
    right = state.elo_ratings[right_id]
    k_factor = state.project.search.tournament_k_factor
    if state.project.search.tournament_ranking_mode == "bradley_terry":
        k_factor *= 0.75
    expected_a = 1.0 / (1.0 + 10 ** ((right.rating - left.rating) / 400.0))
    expected_b = 1.0 - expected_a
    left_rating = left.rating + k_factor * (score_a - expected_a)
    right_rating = right.rating + k_factor * (score_b - expected_b)
    state.elo_ratings[left_id] = left.model_copy(update={
        "rating": round(left_rating, 3),
        "uncertainty": round(max(40.0, left.uncertainty * 0.92), 3),
        "comparisons": left.comparisons + 1,
        "wins": left.wins + int(score_a > score_b),
        "losses": left.losses + int(score_b > score_a),
        "draws": left.draws + int(score_a == score_b),
    })
    state.elo_ratings[right_id] = right.model_copy(update={
        "rating": round(right_rating, 3),
        "uncertainty": round(max(40.0, right.uncertainty * 0.92), 3),
        "comparisons": right.comparisons + 1,
        "wins": right.wins + int(score_b > score_a),
        "losses": right.losses + int(score_a > score_b),
        "draws": right.draws + int(score_a == score_b),
    })


def _allocate_deep_comparison(state: _SearchState, left_id: str, right_id: str) -> bool:
    if len(state.deep_comparison_pairings) >= state.project.search.tournament_max_deep_comparisons:
        return False
    key = _pair_key(left_id, right_id)
    if key in state.deep_comparison_pairings:
        return False
    left = state.elo_ratings[left_id]
    right = state.elo_ratings[right_id]
    if abs(left.rating - right.rating) <= state.project.search.tournament_close_match_gap:
        state.deep_comparison_pairings.append(key)
        return True
    ranked = sorted(state.elo_ratings.values(), key=lambda item: item.rating, reverse=True)
    top_ids = {item.candidate_id for item in ranked[: max(2, state.project.search.beam_width)]}
    if left_id in top_ids and right_id in top_ids:
        state.deep_comparison_pairings.append(key)
        return True
    return False


def _pair_key(left_id: str, right_id: str) -> str:
    return "::".join(sorted([left_id, right_id]))


def _detect_plateau(state: _SearchState) -> bool:
    scores = [candidate.aggregate_search_score for candidate in state.archive.active_candidates()]
    best = max(scores) if scores else 0.0
    window = state.project.search.plateau_window
    prior = [*state.plateau_history, {"step": state.current_step, "best_score": round(best, 3)}]
    plateau = False
    if len(prior) >= window:
        recent = prior[-window:]
        plateau = float(recent[-1]["best_score"]) - float(recent[0]["best_score"]) < state.project.search.plateau_minimum_improvement
    state.plateau_history.append({
        "step": state.current_step,
        "best_score": round(best, 3),
        "surviving_lineages": len({candidate.root_candidate_id or candidate.candidate_id for candidate in state.archive.active_candidates()}),
        "plateau": "true" if plateau else "false",
    })
    return plateau


def _derive_candidate_from_task(state: _SearchState, task: SearchTask) -> CandidateSolution | None:
    parent = state.archive.underexplored_branches()[0] if state.archive.active_candidates() else None
    if not parent or len(state.archive.candidates) >= state.project.search.max_candidates_total:
        return None
    strategy = {
        "repair_candidate": "repair_failed_candidate",
        "cross_domain_transfer": "cross_domain_transfer",
        "search_counterexample": "counterexample_search",
    }[task.task_type]
    candidate_type = "counterexample" if task.task_type == "search_counterexample" else parent.candidate_type
    suffix = hashlib.sha1(f"{task.task_id}-{parent.candidate_id}".encode()).hexdigest()[:8]
    return CandidateSolution(
        candidate_id=f"cand-{suffix}",
        problem_id=state.project.problem.problem_id,
        candidate_type=candidate_type,  # type: ignore[arg-type]
        title=f"{strategy.replace('_', ' ').title()} from {parent.candidate_id}",
        summary=f"Deterministic {strategy} child preserving partial components from {parent.candidate_id}.",
        formal_representation=parent.formal_representation,
        assumptions=parent.assumptions[:],
        construction_or_model=parent.construction_or_model,
        predicted_observables=parent.predicted_observables[:],
        falsification_conditions=parent.falsification_conditions[:],
        parent_ids=[parent.candidate_id],
        root_candidate_id=parent.root_candidate_id or parent.candidate_id,
        lineage_depth=parent.lineage_depth + 1,
        generation_strategy=strategy,  # type: ignore[arg-type]
        linked_evidence_ids=parent.linked_evidence_ids[:],
        created_step=state.current_step,
        updated_step=state.current_step,
        provenance=[f"derived_by_task:{task.task_id}"],
    )


def _write_static_artifacts(state: _SearchState) -> None:
    write_json(state.run_dir / "discovery_project.json", _agent_visible_project(state.project))
    if state.project.evaluator_only_ground_truth:
        write_json(state.run_dir / "evaluator_ground_truth.json", {
            "schema_version": "v17",
            "project_id": state.project.project_id,
            "problem_id": state.project.problem.problem_id,
            "hidden_from_agents": True,
            "ground_truth": state.project.evaluator_only_ground_truth,
        })
    write_json(state.run_dir / "scientific_problem.json", state.project.problem)


def _write_checkpoint(state: _SearchState) -> None:
    _refresh_strategy_metrics(state)
    checkpoint = SearchCheckpoint(
        project_id=state.project.project_id,
        problem_id=state.project.problem.problem_id,
        run_id=state.run_id,
        random_seed=state.project.random_seed,
        current_step=state.current_step,
        queue=state.queue.list(),
        completed_task_ids=state.queue.completed_ids(),
        archive=state.archive.snapshot(),
        strategy_metrics=list(state.strategy_metrics.values()),
        verifier_results=state.verifier_results,
        grounding_references=state.project.evidence_ids,
        budgets_spent=state.budgets_spent,
        plateau_history=state.plateau_history,
        project_hash=_hash_json(_agent_visible_project(state.project).model_dump(mode="json")),
        corpus_hash=_hash_json(state.project.evidence_ids),
        created_at=datetime.now(UTC),
        resume_count=state.resume_count,
    )
    write_json(state.run_dir / "search_checkpoint.json", checkpoint)


def _write_outputs(state: _SearchState) -> None:
    _refresh_strategy_metrics(state)
    state.reproduction_results = _independent_reproduction_results(state)
    write_jsonl(state.run_dir / "candidate_archive.jsonl", list(state.archive.candidates.values()))
    write_json(state.run_dir / "candidate_lineage_graph.json", state.archive.lineage_graph())
    write_jsonl(state.run_dir / "candidate_status_history.jsonl", state.archive.status_history)
    write_json(state.run_dir / "candidate_failure_catalog.json", _failure_catalog(state.archive))
    write_json(state.run_dir / "search_strategy_metrics.json", list(state.strategy_metrics.values()))
    allocation = _adaptive_budget_allocation(state)
    write_json(state.run_dir / "adaptive_budget_allocation.json", allocation)
    write_json(state.run_dir / "provider_routing_plan.json", _provider_routing_plan(state))
    write_jsonl(state.run_dir / "search_tasks.jsonl", state.queue.list())
    write_jsonl(state.run_dir / "verifier_results.jsonl", state.verifier_results)
    write_json(state.run_dir / "beam_selection.json", state.beam_history[-1] if state.beam_history else None)
    write_jsonl(state.run_dir / "tournament_comparisons.jsonl", state.tournament)
    write_json(state.run_dir / "elo_tournament_state.json", _elo_tournament_state(state))
    write_jsonl(state.run_dir / "reproduction_results.jsonl", state.reproduction_results)
    write_json(state.run_dir / "reproduction_discrepancies.json", _reproduction_discrepancies(state.reproduction_results))
    write_jsonl(state.run_dir / "claim_ledger.jsonl", _claim_ledger(state))
    write_jsonl(state.run_dir / "prediction_ledger.jsonl", _prediction_ledger(state))
    write_json(state.run_dir / "plateau_history.json", state.plateau_history)
    write_json(state.run_dir / "model_usage.json", {"model_mode": "mock", "provider": "mock", "call_count": state.budgets_spent["model_calls"], "total_tokens": state.budgets_spent["tokens"]})
    if not (state.run_dir / "expert_feedback.jsonl").exists():
        write_jsonl(state.run_dir / "expert_feedback.jsonl", [])
    (state.run_dir / "discovery_report.md").write_text(_discovery_report(state), encoding="utf-8")
    (state.run_dir / "expert_review.md").write_text(_expert_review(state), encoding="utf-8")
    _write_checkpoint(state)


def _refresh_strategy_metrics(state: _SearchState) -> None:
    grouped: dict[str, list[CandidateSolution]] = defaultdict(list)
    for candidate in state.archive.candidates.values():
        grouped[candidate.generation_strategy].append(candidate)
    for strategy, candidates in grouped.items():
        verified_ids = {result.candidate_id for result in state.verifier_results if result.verdict in {"pass", "partial", "inconclusive"}}
        passed_ids = {candidate.candidate_id for candidate in candidates if candidate.scientific_status != "cheap_filter_failed"}
        surviving_lineages = {candidate.root_candidate_id or candidate.candidate_id for candidate in candidates if candidate.scientific_status in {"partially_verified", "promising", "strong_verification_passed", "expert_review_required"}}
        metrics = state.strategy_metrics.get(strategy)
        if metrics:
            state.strategy_metrics[strategy] = metrics.model_copy(update={
                "candidates_generated": len(candidates),
                "cheap_filter_pass_rate": round(len(passed_ids) / max(1, len(candidates)), 3),
                "verification_pass_rate": round(len(verified_ids & {candidate.candidate_id for candidate in candidates}) / max(1, len(candidates)), 3),
                "novelty_yield": round(len([candidate for candidate in candidates if candidate.novelty_status in {"unknown", "possibly_novel"}]) / max(1, len(candidates)), 3),
                "score_improvement": round(max([candidate.aggregate_search_score for candidate in candidates] or [0.0]), 3),
                "token_cost": state.budgets_spent["tokens"],
                "model_calls": state.budgets_spent["model_calls"],
                "surviving_lineages": len(surviving_lineages),
            })


def _adaptive_budget_allocation(state: _SearchState) -> AdaptiveBudgetAllocation:
    _refresh_strategy_metrics(state)
    metrics = sorted(state.strategy_metrics.values(), key=lambda item: item.strategy)
    duplicate_strategies = {
        state.archive.candidates[candidate_id].generation_strategy
        for group in state.archive.duplicate_groups()
        for candidate_id in group
        if candidate_id in state.archive.candidates
    }
    falsified_by_strategy: Counter[str] = Counter(candidate.generation_strategy for candidate in state.archive.candidates.values() if candidate.scientific_status in {"falsified", "cheap_filter_failed"})
    raw_weights: dict[str, float] = {}
    for metric in metrics:
        historical_yield = max(0.0, min(1.0, (metric.verification_pass_rate * 0.45) + (metric.novelty_yield * 0.2) + (min(1.0, metric.score_improvement) * 0.35)))
        duplicate_penalty = 0.35 if metric.strategy in duplicate_strategies else 0.0
        falsification_penalty = min(0.5, falsified_by_strategy[metric.strategy] * 0.15)
        preserve = metric.strategy == "counterexample_search" and state.project.search.preserve_minimum_contrarian_branches > 0
        if state.project.search.adaptive_compute_enabled:
            raw_weights[metric.strategy] = max(0.02 if preserve else 0.0, historical_yield * (1.0 - duplicate_penalty) * (1.0 - falsification_penalty))
        else:
            raw_weights[metric.strategy] = 1.0
    if not any(raw_weights.values()) and metrics:
        raw_weights = {metric.strategy: 1.0 for metric in metrics}
    allocations = _bounded_allocations(state, metrics, raw_weights, duplicate_strategies, falsified_by_strategy)
    verifier_stage_yield = _verifier_stage_yield(state.verifier_results)
    lineage_yield = _lineage_yield(state)
    return AdaptiveBudgetAllocation(
        project_id=state.project.project_id,
        run_id=state.run_id,
        total_token_budget=state.project.search.token_budget,
        total_model_call_budget=state.project.search.model_call_budget,
        total_verifier_call_budget=state.project.search.verifier_call_budget,
        allocations=allocations,
        verifier_stage_yield=verifier_stage_yield,
        role_yield={"generation": _mean([metric.novelty_yield for metric in metrics]), "comparison": _mean([metric.score_improvement for metric in metrics]), "verification": _mean(list(verifier_stage_yield.values()))},
        lineage_yield=lineage_yield,
    )


def _bounded_allocations(state: _SearchState, metrics: list[SearchStrategyMetrics], raw_weights: dict[str, float], duplicate_strategies: set[str], falsified_by_strategy: Counter[str]) -> list[StrategyAllocation]:
    total_weight = sum(raw_weights.values()) or 1.0
    remaining_tokens = state.project.search.token_budget
    remaining_model_calls = state.project.search.model_call_budget
    remaining_verifier_calls = state.project.search.verifier_call_budget
    allocations: list[StrategyAllocation] = []
    for index, metric in enumerate(metrics):
        last = index == len(metrics) - 1
        share = raw_weights.get(metric.strategy, 0.0) / total_weight
        preserve = metric.strategy == "counterexample_search" and state.project.search.preserve_minimum_contrarian_branches > 0
        token_budget = remaining_tokens if last else int(state.project.search.token_budget * share)
        model_call_budget = remaining_model_calls if last else int(state.project.search.model_call_budget * share)
        verifier_call_budget = remaining_verifier_calls if last else int(state.project.search.verifier_call_budget * share)
        if preserve and state.project.search.verifier_call_budget and verifier_call_budget == 0:
            verifier_call_budget = min(1, remaining_verifier_calls)
        remaining_tokens = max(0, remaining_tokens - token_budget)
        remaining_model_calls = max(0, remaining_model_calls - model_call_budget)
        remaining_verifier_calls = max(0, remaining_verifier_calls - verifier_call_budget)
        historical_yield = max(0.0, min(1.0, (metric.verification_pass_rate * 0.45) + (metric.novelty_yield * 0.2) + (min(1.0, metric.score_improvement) * 0.35)))
        duplicate_penalty = 0.35 if metric.strategy in duplicate_strategies else 0.0
        falsification_penalty = min(0.5, falsified_by_strategy[metric.strategy] * 0.15)
        allocations.append(StrategyAllocation(
            strategy=metric.strategy,
            historical_yield=round(historical_yield, 3),
            duplicate_penalty=round(duplicate_penalty, 3),
            falsification_penalty=round(falsification_penalty, 3),
            token_budget=token_budget,
            model_call_budget=model_call_budget,
            verifier_call_budget=verifier_call_budget,
            preserve_branch=preserve,
            rationale="Preserved contrarian branch floor." if preserve else "Allocated from verifier pass, novelty, score improvement, duplicate and falsification history.",
        ))
    return allocations


def _provider_routing_plan(state: _SearchState) -> ProviderRoutingPlan:
    roles = ["generation", "review", "comparison", "deep_reasoning", "novelty_audit", "meta_review"]
    routes = []
    for role in roles:
        configured_model = state.project.search.role_model_routing.get(role, "deterministic-mock")
        routes.append(RoleModelRoute(
            role=role,  # type: ignore[arg-type]
            provider="mock",
            model=configured_model,
            model_mode="mock",
            max_context_characters=min(12000, state.project.search.token_budget * 4 if state.project.search.token_budget else 0),
            max_output_tokens=900,
            live_permission_required=False,
        ))
    return ProviderRoutingPlan(project_id=state.project.project_id, run_id=state.run_id, live_model_enabled=False, routes=routes)


def _elo_tournament_state(state: _SearchState) -> EloTournamentState:
    for candidate in state.archive.active_candidates():
        if candidate.candidate_id not in state.elo_ratings:
            state.elo_ratings[candidate.candidate_id] = CandidateRating(
                candidate_id=candidate.candidate_id,
                rating=round(state.project.search.tournament_initial_rating + candidate.aggregate_search_score * 100.0, 3),
                uncertainty=state.project.search.tournament_initial_uncertainty,
            )
    return EloTournamentState(
        ranking_mode=state.project.search.tournament_ranking_mode,
        ratings=sorted(state.elo_ratings.values(), key=lambda item: item.candidate_id),
        completed_pairings=sorted({_pair_key(item.candidate_a_id, item.candidate_b_id) for item in state.tournament}),
        deep_comparison_pairings=sorted(state.deep_comparison_pairings),
        maximum_comparisons=state.project.search.tournament_max_comparisons,
    )


def _independent_reproduction_results(state: _SearchState) -> list[ReproductionResult]:
    top = sorted(state.archive.active_candidates(), key=lambda item: (item.aggregate_search_score, item.candidate_id), reverse=True)[: max(1, min(2, state.project.search.beam_width))]
    results: list[ReproductionResult] = []
    for candidate in top:
        direct = _direct_reproduction_score(candidate, state.verifier_results)
        independent = _independent_reproduction_score(candidate, state.verifier_results)
        if direct is None and independent is None:
            outcome = "inconclusive"
            difference = None
            summary = "No verifier outputs were available for independent reproduction."
        else:
            left = direct or 0.0
            right = independent or 0.0
            difference = abs(left - right)
            if difference <= 0.1:
                outcome = "reproduced"
            elif difference <= 0.3:
                outcome = "partially_reproduced"
            else:
                outcome = "not_reproduced"
            summary = f"Independent paths differed by {difference:.3f}; discrepancy preserved."
        path_a = ReproductionRun(run_id=f"repr-{candidate.candidate_id}-direct", path_id="direct_verifier_mean", method="Mean stored verifier scores without recomputing check ratios.", output_value=direct, output_summary="direct verifier-score reproduction")
        path_b = ReproductionRun(run_id=f"repr-{candidate.candidate_id}-ratio", path_id="check_ratio_reimplementation", method="Independently recomputed pass ratios from check lists.", output_value=independent, output_summary="independent check-ratio reproduction")
        comparison = ImplementationComparison(candidate_id=candidate.candidate_id, compared_path_ids=[path_a.path_id, path_b.path_id], tolerance=0.1, absolute_difference=difference, discrepancy_summary=summary)
        results.append(ReproductionResult(
            reproduction_result_id=f"repr-{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            requested_at=datetime.now(UTC),
            runs=[path_a, path_b],
            comparison=comparison,
            outcome=outcome,  # type: ignore[arg-type]
        ))
    return results


def _direct_reproduction_score(candidate: CandidateSolution, verifier_results: list[VerifierResult]) -> float | None:
    scores = [result.score for result in verifier_results if result.candidate_id == candidate.candidate_id]
    return round(sum(scores) / len(scores), 3) if scores else None


def _independent_reproduction_score(candidate: CandidateSolution, verifier_results: list[VerifierResult]) -> float | None:
    ratios = []
    for result in verifier_results:
        if result.candidate_id != candidate.candidate_id:
            continue
        total = len(result.checks_passed) + len(result.checks_failed)
        if total:
            ratios.append(len(result.checks_passed) / total)
    return round(sum(ratios) / len(ratios), 3) if ratios else None


def _reproduction_discrepancies(results: list[ReproductionResult]) -> dict[str, Any]:
    return {
        "schema_version": "v20",
        "discrepancies": [
            {
                "candidate_id": item.candidate_id,
                "outcome": item.outcome,
                "absolute_difference": item.comparison.absolute_difference,
                "summary": item.comparison.discrepancy_summary,
            }
            for item in results
        ],
        "validation_status": "validated",
    }


def _claim_ledger(state: _SearchState) -> list[dict[str, Any]]:
    reproduction_by_candidate = {item.candidate_id: item.outcome for item in state.reproduction_results}
    records = []
    for candidate in sorted(state.archive.candidates.values(), key=lambda item: item.candidate_id):
        status = {
            "promising": "supported_by_current_data",
            "partially_verified": "internally_consistent",
            "falsified": "falsified",
            "cheap_filter_failed": "withdrawn",
            "expert_review_required": "awaiting_expert_review",
        }.get(candidate.scientific_status, "draft")
        records.append({
            "schema_version": "v20",
            "claim_id": f"claim-{candidate.candidate_id}",
            "claim_type": "mechanistic" if candidate.candidate_type in {"hypothesis", "mechanistic_model"} else "experimental_recommendation",
            "candidate_ids": [candidate.candidate_id],
            "claim_text": candidate.summary,
            "evidence_ids": candidate.linked_evidence_ids,
            "verifier_result_ids": candidate.verification_result_ids,
            "status": status,
            "novelty_status": candidate.novelty_status,
            "reproduction_status": reproduction_by_candidate.get(candidate.candidate_id, "not_requested"),
            "uncertainty": round(1.0 - candidate.aggregate_search_score, 3),
            "scope": "bounded deterministic offline discovery run",
        })
    return records


def _prediction_ledger(state: _SearchState) -> list[dict[str, Any]]:
    records = []
    for candidate in sorted(state.archive.candidates.values(), key=lambda item: item.candidate_id):
        for index, prediction in enumerate(candidate.predicted_observables):
            records.append({
                "schema_version": "v20",
                "prediction_id": f"pred-{candidate.candidate_id}-{index + 1}",
                "candidate_id": candidate.candidate_id,
                "observable": prediction,
                "status": "testable_now" if candidate.linked_evidence_ids else "proposed",
                "used_for_fitting": False,
                "preregistered": False,
                "evaluation_status": "not_evaluated",
            })
    return records


def _verifier_stage_yield(verifier_results: list[VerifierResult]) -> dict[str, float]:
    grouped: dict[str, list[VerifierResult]] = defaultdict(list)
    for result in verifier_results:
        grouped[result.stage].append(result)
    return {stage: round(len([item for item in items if item.verdict in {"pass", "partial"}]) / max(1, len(items)), 3) for stage, items in sorted(grouped.items())}


def _lineage_yield(state: _SearchState) -> dict[str, float]:
    grouped: dict[str, list[CandidateSolution]] = defaultdict(list)
    for candidate in state.archive.candidates.values():
        grouped[candidate.root_candidate_id or candidate.candidate_id].append(candidate)
    return {root: round(max(candidate.aggregate_search_score for candidate in candidates), 3) for root, candidates in sorted(grouped.items())}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def validate_discovery_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors: list[str] = []
    required = ["discovery_project.json", "scientific_problem.json", "candidate_archive.jsonl", "candidate_lineage_graph.json", "candidate_status_history.jsonl", "candidate_failure_catalog.json", "search_strategy_metrics.json", "adaptive_budget_allocation.json", "provider_routing_plan.json", "search_tasks.jsonl", "verifier_results.jsonl", "search_checkpoint.json", "discovery_report.md", "expert_review.md", "expert_feedback.jsonl", "elo_tournament_state.json", "reproduction_results.jsonl", "reproduction_discrepancies.json", "claim_ledger.jsonl", "prediction_ledger.jsonl"]
    for name in required:
        if not (path / name).exists():
            errors.append(f"missing discovery artifact: {name}")
    if errors:
        return errors
    project = DiscoveryProject.model_validate_json((path / "discovery_project.json").read_text(encoding="utf-8"))
    candidates = [CandidateSolution.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "candidate_archive.jsonl")]
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    for candidate in candidates:
        if candidate.problem_id != project.problem.problem_id:
            errors.append(f"candidate {candidate.candidate_id} has wrong problem_id")
        missing_parents = set(candidate.parent_ids) - candidate_ids
        if missing_parents:
            errors.append(f"candidate {candidate.candidate_id} references missing parents: {sorted(missing_parents)}")
        if set(candidate.linked_evidence_ids) - set(project.evidence_ids):
            errors.append(f"candidate {candidate.candidate_id} references missing evidence")
    if _has_cycle({candidate.candidate_id: candidate.parent_ids for candidate in candidates}):
        errors.append("candidate lineage graph contains a cycle")
    verifier_ids = {item["verifier_result_id"] for item in read_jsonl(path / "verifier_results.jsonl")}
    for candidate in candidates:
        if set(candidate.verification_result_ids) - verifier_ids:
            errors.append(f"candidate {candidate.candidate_id} references missing verifier result")
    tasks = [SearchTask.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "search_tasks.jsonl")]
    task_ids = {task.task_id for task in tasks}
    for task in tasks:
        if set(task.dependencies) - task_ids:
            errors.append(f"task {task.task_id} references missing dependency")
    checkpoint = SearchCheckpoint.model_validate_json((path / "search_checkpoint.json").read_text(encoding="utf-8"))
    if project.evaluator_only_ground_truth:
        errors.append("agent-visible discovery_project.json contains evaluator-only ground truth")
    evaluator_ground_truth = path / "evaluator_ground_truth.json"
    if evaluator_ground_truth.exists() and not read_json(evaluator_ground_truth).get("hidden_from_agents"):
        errors.append("evaluator ground truth artifact is not marked hidden_from_agents")
    if checkpoint.project_hash != _hash_json(_agent_visible_project(project).model_dump(mode="json")):
        errors.append("checkpoint project hash mismatch")
    beam = read_json(path / "beam_selection.json")
    if beam and len(beam.get("selected_candidate_ids", [])) > project.search.beam_width:
        errors.append("beam selection exceeds beam_width")
    if len(read_jsonl(path / "tournament_comparisons.jsonl")) > project.search.tournament_max_comparisons:
        errors.append("tournament comparison limit exceeded")
    try:
        elo_state = EloTournamentState.model_validate_json((path / "elo_tournament_state.json").read_text(encoding="utf-8"))
        if len(elo_state.deep_comparison_pairings) > project.search.tournament_max_deep_comparisons:
            errors.append("deep comparison budget exceeded")
        if {rating.candidate_id for rating in elo_state.ratings} - candidate_ids:
            errors.append("elo tournament references missing candidate")
    except Exception as exc:
        errors.append(f"invalid elo tournament state: {exc}")
    try:
        allocation = AdaptiveBudgetAllocation.model_validate_json((path / "adaptive_budget_allocation.json").read_text(encoding="utf-8"))
        if allocation.total_model_call_budget != project.search.model_call_budget:
            errors.append("adaptive allocation model-call budget mismatch")
    except Exception as exc:
        errors.append(f"invalid adaptive budget allocation: {exc}")
    try:
        routing = ProviderRoutingPlan.model_validate_json((path / "provider_routing_plan.json").read_text(encoding="utf-8"))
        if any(route.model_mode == "live" for route in routing.routes) and project.model_mode != "live":
            errors.append("provider routing enables live model without project live mode")
    except Exception as exc:
        errors.append(f"invalid provider routing plan: {exc}")
    for item in read_jsonl(path / "reproduction_results.jsonl"):
        result = ReproductionResult.model_validate_json(json.dumps(item))
        if result.candidate_id not in candidate_ids:
            errors.append(f"reproduction references missing candidate: {result.candidate_id}")
    for item in read_jsonl(path / "claim_ledger.jsonl"):
        if set(item.get("candidate_ids", [])) - candidate_ids:
            errors.append(f"claim references missing candidate: {item.get('claim_id')}")
        if set(item.get("verifier_result_ids", [])) - verifier_ids:
            errors.append(f"claim references missing verifier result: {item.get('claim_id')}")
    for item in read_jsonl(path / "prediction_ledger.jsonl"):
        if item.get("candidate_id") not in candidate_ids:
            errors.append(f"prediction references missing candidate: {item.get('prediction_id')}")
    for artifact in [*path.rglob("*.json"), *path.rglob("*.jsonl")]:
        text = artifact.read_text(encoding="utf-8").lower()
        if "openai_api_key" in text or re.search(r"\bsk-[a-z0-9_-]{16,}", text) or re.search(r"\bbearer\s+[a-z0-9_.-]{12,}", text):
            errors.append(f"secret-like content appears in {artifact.relative_to(path)}")
    return errors


def _failure_catalog(archive: CandidateArchive) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = defaultdict(list)
    for candidate in archive.candidates.values():
        for reason in candidate.failure_reason_ids:
            catalog[reason].append(candidate.candidate_id)
    return dict(catalog)


def _discovery_report(state: _SearchState) -> str:
    active = sorted(state.archive.active_candidates(), key=lambda item: item.aggregate_search_score, reverse=True)
    falsified = [candidate for candidate in state.archive.candidates.values() if candidate.scientific_status in {"falsified", "cheap_filter_failed"}]
    lines = [
        f"# Discovery Search Report: {state.project.title}",
        "",
        "> Deterministic V1.7 search-runtime benchmark. Generic verifiers check encoded constraints, not scientific truth.",
        "",
        f"- Problem: {state.project.problem.title}",
        f"- Constraints: {len(state.project.problem.known_constraints)}",
        f"- Search mode: {state.project.search.mode}",
        f"- Beam width: {state.project.search.beam_width}",
        f"- Candidates archived: {len(state.archive.candidates)}",
        f"- Active candidates: {len(active)}",
        f"- Falsified or cheap-filter-failed candidates: {len(falsified)}",
        f"- Surviving lineages: {len({candidate.root_candidate_id or candidate.candidate_id for candidate in active})}",
        f"- Verifier calls: {state.budgets_spent['verifier_calls']}",
        f"- Model calls: {state.budgets_spent['model_calls']}",
        f"- Tournament mode: {state.project.search.tournament_ranking_mode}",
        f"- Deep comparisons allocated: {len(state.deep_comparison_pairings)}",
        f"- Reproduction checks: {len(state.reproduction_results)}",
        f"- Plateau events: {sum(1 for item in state.plateau_history if item.get('plateau') == 'true')}",
        "",
        "## Top Candidates",
        "",
    ]
    for candidate in active[:5]:
        lines.append(f"- `{candidate.candidate_id}` {candidate.title}: score {candidate.aggregate_search_score:.3f}, status {candidate.scientific_status}")
    lines.extend(["", "## Common Failure Modes", ""])
    for reason, ids in _failure_catalog(state.archive).items():
        lines.append(f"- {reason}: {', '.join(ids)}")
    if state.reproduction_results:
        lines.extend(["", "## Independent Reproduction", ""])
        for result in state.reproduction_results:
            lines.append(f"- `{result.candidate_id}`: {result.outcome}; {result.comparison.discrepancy_summary}")
    lines.extend(["", "Expert review remains required before treating any candidate as validated.", ""])
    return "\n".join(lines)


def _expert_review(state: _SearchState) -> str:
    return "\n".join([
        f"# Expert Review Package: {state.project.title}",
        "",
        "- Is the problem formalization scientifically fair?",
        "- Are the constraints complete?",
        "- Is the candidate representation sufficient?",
        "- Did the verifier test the scientifically important claim?",
        "- Did duplicate candidates inflate support?",
        "- Was a promising unusual branch pruned too early?",
        "- Is the candidate genuinely novel?",
        "- Are any assumptions hidden?",
        "- What experiment or calculation would be decisive?",
        "- Is the claimed confidence justified?",
        "",
    ])


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for parent in graph.get(node, []):
            if visit(parent):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _agent_visible_project(project: DiscoveryProject) -> DiscoveryProject:
    return project.model_copy(update={"evaluator_only_ground_truth": {}})
