from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from coscientist.agents.proximity import ProximityAgent
from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.pilot.project_io import load_fixture_corpus
from coscientist.schemas.evidence import ClaimEvidenceLink, EvidenceExcerpt
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import Paper
from coscientist.schemas.ranking import HypothesisRanking
from coscientist.schemas.v16 import (
    AnswerCalibrationSummary,
    AnswerEvidenceCell,
    ClosedQuestion,
    ClosedQuestionEvaluation,
    ClosedQuestionProject,
    ContextBuildRecord,
    FinalAnswer,
    GroundTruth,
    HypothesisAnswerLink,
)


def load_closed_question_project(path: str | Path) -> ClosedQuestionProject:
    project_path = Path(path)
    with project_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data.get("created_at"), str):
        data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
    return ClosedQuestionProject.model_validate(data)


def run_closed_question_project(
    project_path: str | Path,
    *,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    force: bool = False,
    controlled_feedback: bool = False,
) -> Path:
    project_file = Path(project_path)
    project = load_closed_question_project(project_file)
    if project.model_mode != "mock" or project.literature_mode not in {"fixture", "existing"}:
        raise ValueError("closed-question runs default to mock model and offline fixture/existing corpus")
    run_id = run_id or f"{project.project_id}-closed"
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"closed-question artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_fixture_corpus(_resolve_project_path(project_file, project.corpus_path))
    ground_truth_by_id = {truth.question_id: truth for truth in project.ground_truth}
    evaluations: list[ClosedQuestionEvaluation] = []
    answers: list[FinalAnswer] = []
    contexts: list[ContextBuildRecord] = []
    report_context: dict[str, dict[str, Any]] = {}
    total_calls = 0
    total_tokens = 0

    write_json(run_dir / "closed_question_project.json", project)
    write_jsonl(run_dir / "closed_question_corpus.jsonl", corpus)
    for question in project.questions:
        prefix = _artifact_prefix(question.question_id)
        question_visible = question.model_copy()
        write_json(run_dir / f"{prefix}_closed_question.json", question_visible)
        write_json(run_dir / f"{prefix}_answer_options.json", question.answer_options)
        if question.question_id in ground_truth_by_id:
            write_json(run_dir / f"{prefix}_ground_truth_evaluator_only.json", ground_truth_by_id[question.question_id])
        context_record = build_context_record(project, run_id, question, corpus)
        contexts.append(context_record)
        hypotheses = build_evidence_hypotheses(project, run_id, question, corpus, max_hypotheses=project.config.generation.max_hypotheses)
        rankings = _rankings(hypotheses)
        proximity = ProximityAgent().analyze(
            project_id=project.project_id,
            run_id=run_id,
            round_label=question.question_id,
            round_number=0,
            hypotheses=hypotheses,
            rankings=rankings,
            verifications=[],
            config=__import__("coscientist.schemas.v15b", fromlist=["ProximityConfig"]).ProximityConfig(),
            model_mode="mock",
            literature_mode=project.literature_mode,
        )
        links = link_hypotheses_to_answers(project.project_id, run_id, question, hypotheses, corpus, proximity)
        cells = build_answer_evidence_matrix(project.project_id, run_id, question, links, proximity)
        final = synthesize_final_answer(project, run_id, question, cells, links, controlled_feedback=controlled_feedback)
        final = FinalAnswerValidator().validate(final, question, cells, hypotheses, proximity, corpus, project.config.answer_policy)
        answers.append(final)
        evaluation = evaluate_final_answer(project, run_id, question, final, ground_truth_by_id.get(question.question_id), calls=0, tokens=0)
        evaluations.append(evaluation)
        write_json(run_dir / f"{prefix}_hypotheses.json", hypotheses)
        write_json(run_dir / f"{prefix}_proximity.json", proximity)
        write_json(run_dir / f"{prefix}_hypothesis_answer_links_round_0.json", links)
        write_json(run_dir / f"{prefix}_answer_evidence_matrix_round_0.json", cells)
        write_json(run_dir / f"{prefix}_final_answer.json", final)
        write_json(run_dir / f"{prefix}_closed_question_evaluation.json", evaluation)
        report_context[question.question_id] = {
            "question": question,
            "cells": cells,
            "links": links,
            "corpus": _question_corpus(question, corpus),
        }

    calibration = calibration_summary(project.project_id, run_id, evaluations, total_tokens)
    write_json(run_dir / "final_answers.json", answers)
    write_json(run_dir / "closed_question_evaluations.json", evaluations)
    write_json(run_dir / "answer_calibration.json", calibration)
    write_json(run_dir / "context_build_records.json", contexts)
    write_json(run_dir / "model_usage.json", {
        "provider": "mock",
        "model_mode": "mock",
        "call_count": total_calls,
        "total_tokens": total_tokens,
        "cost": None,
    })
    (run_dir / "report.md").write_text(build_closed_question_report(project, answers, evaluations, calibration, report_context=report_context), encoding="utf-8")
    (run_dir / "human_review.md").write_text(build_closed_question_human_review(project, answers), encoding="utf-8")
    return run_dir


def validate_closed_question_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors: list[str] = []
    for name in ["closed_question_project.json", "final_answers.json", "closed_question_evaluations.json", "answer_calibration.json", "context_build_records.json", "report.md", "human_review.md"]:
        if not (path / name).exists():
            errors.append(f"missing closed-question artifact: {name}")
    if errors:
        return errors
    project = ClosedQuestionProject.model_validate_json((path / "closed_question_project.json").read_text(encoding="utf-8"))
    for question in project.questions:
        prefix = _artifact_prefix(question.question_id)
        for suffix in [
            "closed_question.json",
            "answer_options.json",
            "hypotheses.json",
            "hypothesis_answer_links_round_0.json",
            "answer_evidence_matrix_round_0.json",
            "final_answer.json",
            "closed_question_evaluation.json",
        ]:
            if not (path / f"{prefix}_{suffix}").exists():
                errors.append(f"{question.question_id}: missing {suffix}")
        if errors:
            continue
        valid_answers = {option.answer_id for option in question.answer_options}
        hypotheses = {item["id"] for item in read_json(path / f"{prefix}_hypotheses.json")}
        proximity = read_json(path / f"{prefix}_proximity.json")
        clusters = {cluster["cluster_id"] for cluster in proximity.get("clusters", [])}
        corpus_ids = {item["id"] for item in read_jsonl(path / "closed_question_corpus.jsonl")}
        links = read_json(path / f"{prefix}_hypothesis_answer_links_round_0.json")
        final = read_json(path / f"{prefix}_final_answer.json")
        if set(final.get("selected_answer_ids", [])) - valid_answers:
            errors.append(f"{question.question_id}: final answer uses invalid answer id")
        if len(final.get("selected_answer_ids", [])) > question.allowed_answer_count:
            errors.append(f"{question.question_id}: final answer exceeds allowed answer count")
        for link in links:
            if link["answer_id"] not in valid_answers:
                errors.append(f"{question.question_id}: link references invalid answer id")
            if link["hypothesis_id"] not in hypotheses:
                errors.append(f"{question.question_id}: link references invalid hypothesis id")
            if set(link.get("supporting_evidence_ids", [])) - corpus_ids:
                errors.append(f"{question.question_id}: link references invalid evidence id")
        if set(final.get("supporting_hypothesis_ids", [])) - hypotheses:
            errors.append(f"{question.question_id}: final answer references invalid hypothesis")
        if set(final.get("supporting_cluster_ids", [])) - clusters:
            errors.append(f"{question.question_id}: final answer references invalid cluster")
        if set(final.get("supporting_evidence_ids", [])) - corpus_ids:
            errors.append(f"{question.question_id}: final answer references invalid evidence")
        if (path / f"{prefix}_closed_question.json").read_text(encoding="utf-8").find("correct_answer_ids") >= 0:
            errors.append(f"{question.question_id}: ground truth leaked into agent-visible question artifact")
    for artifact in path.rglob("*.json"):
        text = artifact.read_text(encoding="utf-8").lower()
        if "openai_api_key" in text or "sk-" in text or "bearer " in text:
            errors.append(f"secret-like content appears in {artifact.relative_to(path)}")
    return errors


def build_evidence_hypotheses(project: ClosedQuestionProject, run_id: str, question: ClosedQuestion, corpus: list[Paper], *, max_hypotheses: int) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []
    for paper in _question_corpus(question, corpus):
        meta = paper.source_metadata
        support = list(meta.get("supports", []))
        contradicts = list(meta.get("contradicts", []))
        if not support and not contradicts and question.question_type != "numeric":
            continue
        answer_ids = support or contradicts or ["numeric"]
        link = ClaimEvidenceLink(
            claim_id=f"claim-{paper.id}-{question.question_id}",
            claim_text=paper.abstract or paper.title,
            claim_kind="source_observation",
            supporting_paper_ids=[paper.id] if not meta.get("metadata_only", False) else [],
            contradicting_paper_ids=[],
            evidence=[EvidenceExcerpt(
                paper_id=paper.id,
                excerpt=(paper.abstract or paper.title)[: project.config.grounding.max_chunk_characters],
                evidence_type="metadata" if meta.get("metadata_only", False) else "fixture_excerpt",
                source_field="abstract",
            )],
            evidence_type="metadata" if meta.get("metadata_only", False) else "fixture_excerpt",
            confidence=float(meta.get("quality", 0.7)),
        )
        hypotheses.append(Hypothesis(
            id=f"hyp-{hashlib.sha1((question.question_id + paper.id).encode()).hexdigest()[:12]}",
            title=f"Evidence-derived hypothesis from {paper.id}",
            core_claim=f"Evidence item {paper.id} is relevant to answer option(s): {', '.join(answer_ids)}.",
            mechanism=paper.abstract or paper.title,
            assumptions=["Curated evidence metadata correctly labels the relation to the closed answer space."],
            supporting_evidence=[paper.id],
            contradicting_evidence=[],
            novelty_statement="Not a novelty claim; deterministic evidence-derived hypothesis.",
            testable_predictions=["Additional independent characterization should preserve or refute the labeled relation."],
            falsification_criteria=["Independent evidence contradicts the labeled answer relation."],
            proposed_experiments=["Review the linked source or perform the recommended discriminating measurement."],
            uncertainty=round(1.0 - float(meta.get("quality", 0.7)), 3),
            generation_strategy="mechanistic",
            parent_ids=[],
            version=1,
            status="active",
            evidence_links=[link],
        ))
        if len(hypotheses) >= max_hypotheses:
            break
    return hypotheses


def link_hypotheses_to_answers(project_id: str, run_id: str, question: ClosedQuestion, hypotheses: list[Hypothesis], corpus: list[Paper], proximity) -> list[HypothesisAnswerLink]:
    valid = {option.answer_id for option in question.answer_options}
    paper_by_id = {paper.id: paper for paper in corpus}
    links: list[HypothesisAnswerLink] = []
    for hypothesis in hypotheses:
        evidence_ids = [paper_id for link in hypothesis.evidence_links for paper_id in link.supporting_paper_ids]
        support: set[str] = set()
        contradict: set[str] = set()
        metadata_only = False
        for paper_id in evidence_ids:
            meta = paper_by_id.get(paper_id, Paper(id="missing", title="missing", source_provider="missing")).source_metadata
            support.update(meta.get("supports", []))
            contradict.update(meta.get("contradicts", []))
            metadata_only = metadata_only or bool(meta.get("metadata_only", False))
        for answer_id in sorted(valid):
            relation = "neutral"
            if answer_id in support and not metadata_only:
                relation = "supports"
            elif answer_id in contradict:
                relation = "contradicts"
            elif answer_id in support and metadata_only:
                relation = "insufficient"
            links.append(HypothesisAnswerLink(
                project_id=project_id,
                run_id=run_id,
                question_id=question.question_id,
                hypothesis_id=hypothesis.id,
                answer_id=answer_id,
                relation=relation,  # type: ignore[arg-type]
                relevance=0.8 if relation in {"supports", "contradicts"} else 0.2,
                supporting_evidence_ids=evidence_ids if relation == "supports" else [],
                contradicting_evidence_ids=evidence_ids if relation == "contradicts" else [],
                rationale_summary=f"Deterministic link from curated evidence metadata for {hypothesis.id}.",
            ))
    return links


def build_answer_evidence_matrix(project_id: str, run_id: str, question: ClosedQuestion, links: list[HypothesisAnswerLink], proximity) -> list[AnswerEvidenceCell]:
    cluster_by_hypothesis = {}
    for cluster in proximity.clusters:
        for member in cluster.member_ids:
            cluster_by_hypothesis[member] = cluster.cluster_id
    cells: list[AnswerEvidenceCell] = []
    for option in question.answer_options:
        support_links = [link for link in links if link.answer_id == option.answer_id and link.relation == "supports"]
        contradiction_links = [link for link in links if link.answer_id == option.answer_id and link.relation == "contradicts"]
        support_hypotheses = _stable_unique([link.hypothesis_id for link in support_links])
        contradiction_hypotheses = _stable_unique([link.hypothesis_id for link in contradiction_links])
        support_evidence = _stable_unique([evidence_id for link in support_links for evidence_id in link.supporting_evidence_ids])
        contradiction_evidence = _stable_unique([evidence_id for link in contradiction_links for evidence_id in link.contradicting_evidence_ids])
        clusters = _stable_unique([cluster_by_hypothesis.get(hid, f"singleton-{hid}") for hid in support_hypotheses])
        duplicate_adjusted = min(len(support_evidence), len(clusters)) if clusters else 0
        cells.append(AnswerEvidenceCell(
            project_id=project_id,
            run_id=run_id,
            question_id=question.question_id,
            answer_id=option.answer_id,
            supporting_hypothesis_ids=support_hypotheses,
            contradicting_hypothesis_ids=contradiction_hypotheses,
            independent_cluster_ids=clusters,
            verified_support_count=len(support_evidence),
            verified_contradiction_count=len(contradiction_evidence),
            evidence_quality=round(duplicate_adjusted / max(1, len(support_evidence) + len(contradiction_evidence)), 3),
            duplicate_adjusted_support=float(duplicate_adjusted),
            unresolved_issues=["contradicting evidence present"] if contradiction_evidence else [],
        ))
    return cells


def synthesize_final_answer(project: ClosedQuestionProject, run_id: str, question: ClosedQuestion, cells: list[AnswerEvidenceCell], links: list[HypothesisAnswerLink], *, controlled_feedback: bool = False) -> FinalAnswer:
    now = datetime.now(UTC)
    if question.question_type == "numeric":
        values = [float(item) for item in question.metadata.get("numeric_observations", [])]
        if not values:
            return _abstain(project, run_id, question, "No numeric observations were available.")
        estimate = sum(values) / len(values)
        return FinalAnswer(
            project_id=project.project_id,
            run_id=run_id,
            question_id=question.question_id,
            numeric_estimate=round(estimate, 3),
            numeric_interval=[round(min(values), 3), round(max(values), 3)],
            confidence=0.75 if len(values) >= 2 else 0.55,
            rationale_summary="Deterministic numeric aggregation over curated observations.",
            recommended_next_action="Collect additional measurements if interval width is too large.",
            created_at=now,
        )
    best = sorted(cells, key=lambda cell: (cell.duplicate_adjusted_support - cell.verified_contradiction_count, cell.evidence_quality, cell.answer_id), reverse=True)
    sufficient = [
        cell for cell in best
        if cell.verified_support_count >= project.config.answer_policy.minimum_verified_evidence_items
        and len(cell.independent_cluster_ids) >= project.config.answer_policy.minimum_independent_clusters
        and cell.verified_contradiction_count == 0
    ]
    if not sufficient and question.allow_abstention and project.config.answer_policy.allow_abstention:
        return _abstain(project, run_id, question, "Insufficient independent verified support under the answer policy.")
    selected = sufficient[: question.allowed_answer_count] if question.question_type != "ranking" else best
    selected_ids = [cell.answer_id for cell in selected[: question.allowed_answer_count]]
    if question.question_type == "ranking":
        selected_ids = [best[0].answer_id] if best else []
    confidence = min(0.95, 0.45 + sum(cell.duplicate_adjusted_support for cell in selected[: question.allowed_answer_count]) * 0.12)
    if controlled_feedback and confidence >= 0.5:
        confidence = min(0.95, confidence + 0.03)
    support_hypotheses = _stable_unique([hid for cell in selected for hid in cell.supporting_hypothesis_ids])
    support_clusters = _stable_unique([cid for cell in selected for cid in cell.independent_cluster_ids])
    support_evidence = _stable_unique([eid for link in links if link.answer_id in selected_ids for eid in link.supporting_evidence_ids])
    contradiction_evidence = _stable_unique([eid for link in links if link.answer_id in selected_ids for eid in link.contradicting_evidence_ids])
    return FinalAnswer(
        project_id=project.project_id,
        run_id=run_id,
        question_id=question.question_id,
        selected_answer_ids=selected_ids,
        ranking=[cell.answer_id for cell in best] if question.question_type == "ranking" else [],
        confidence=round(confidence, 3),
        supporting_hypothesis_ids=support_hypotheses,
        supporting_cluster_ids=support_clusters,
        supporting_evidence_ids=support_evidence,
        contradicting_evidence_ids=contradiction_evidence,
        rationale_summary="Deterministic answer synthesis from duplicate-adjusted verified support.",
        remaining_uncertainty=[issue for cell in selected for issue in cell.unresolved_issues],
        recommended_next_action="Human review should inspect evidence independence and contradictions.",
        created_at=now,
    )


class FinalAnswerValidator:
    def validate(self, answer: FinalAnswer, question: ClosedQuestion, cells: list[AnswerEvidenceCell], hypotheses: list[Hypothesis], proximity, corpus: list[Paper], policy) -> FinalAnswer:
        valid_answers = {option.answer_id for option in question.answer_options}
        hypothesis_ids = {hypothesis.id for hypothesis in hypotheses}
        clusters = {cluster.cluster_id for cluster in proximity.clusters}
        evidence_ids = {paper.id for paper in corpus if not paper.source_metadata.get("metadata_only", False)}
        selected = [answer_id for answer_id in answer.selected_answer_ids if answer_id in valid_answers]
        if len(selected) > question.allowed_answer_count:
            selected = selected[: question.allowed_answer_count]
        support_ids = [item for item in answer.supporting_evidence_ids if item in evidence_ids]
        support_hyp = [item for item in answer.supporting_hypothesis_ids if item in hypothesis_ids]
        support_clusters = [item for item in answer.supporting_cluster_ids if item in clusters or item.startswith("singleton-")]
        answer = answer.model_copy(update={
            "selected_answer_ids": selected,
            "supporting_evidence_ids": support_ids,
            "supporting_hypothesis_ids": support_hyp,
            "supporting_cluster_ids": support_clusters,
            "ranking": [item for item in answer.ranking if item in valid_answers],
        })
        if question.question_type != "numeric" and not answer.abstained:
            selected_cells = [cell for cell in cells if cell.answer_id in selected]
            verified = sum(cell.verified_support_count for cell in selected_cells)
            clusters_count = len(_stable_unique([cid for cell in selected_cells for cid in cell.independent_cluster_ids]))
            contradictions = sum(cell.verified_contradiction_count for cell in selected_cells)
            if verified < policy.minimum_verified_evidence_items or clusters_count < policy.minimum_independent_clusters or (policy.require_contradiction_review and contradictions):
                answer = answer.model_copy(update={
                    "abstained": True,
                    "abstention_reason": "FinalAnswerValidator enforced abstention due to insufficient independent verified support or unresolved contradictions.",
                    "selected_answer_ids": [],
                    "confidence": min(answer.confidence, 0.5),
                })
        return answer.model_copy(update={"validation_status": "validated"})


def evaluate_final_answer(project: ClosedQuestionProject, run_id: str, question: ClosedQuestion, answer: FinalAnswer, truth: GroundTruth | None, *, calls: int, tokens: int) -> ClosedQuestionEvaluation:
    if truth is None:
        return ClosedQuestionEvaluation(
            project_id=project.project_id,
            run_id=run_id,
            question_id=question.question_id,
            correct=False,
            confidence=answer.confidence,
            calibration_bin=_bin(answer.confidence),
            evidence_sufficiency="unscored_no_ground_truth",
            token_cost=tokens,
            call_count=calls,
            outcome="insufficient_evidence",
            evaluated_at=datetime.now(UTC),
        )
    if answer.abstained:
        correct = truth.acceptable_abstention
        return ClosedQuestionEvaluation(
            project_id=project.project_id,
            run_id=run_id,
            question_id=question.question_id,
            correct=correct,
            exact_match=False,
            abstention_correct=correct,
            confidence=answer.confidence,
            calibration_bin=_bin(answer.confidence),
            evidence_sufficiency="abstained",
            token_cost=tokens,
            call_count=calls,
            outcome="correct_abstention" if correct else "unnecessary_abstention",
            evaluated_at=datetime.now(UTC),
        )
    selected = answer.selected_answer_ids
    correct = False
    precision = recall = f1 = None
    ranking_score = None
    numeric_error = relative_error = None
    within_tolerance = None
    if question.question_type == "single_choice":
        correct = selected == truth.correct_answer_ids
    elif question.question_type == "multi_choice":
        selected_set = set(selected)
        truth_set = set(truth.correct_answer_ids)
        true_pos = len(selected_set & truth_set)
        precision = true_pos / len(selected_set) if selected_set else 0.0
        recall = true_pos / len(truth_set) if truth_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        correct = selected_set == truth_set
    elif question.question_type == "ranking":
        ranking_score = _ranking_agreement(answer.ranking, truth.correct_ranking)
        correct = bool(answer.ranking and truth.correct_ranking and answer.ranking[0] == truth.correct_ranking[0])
    elif question.question_type == "numeric":
        target = truth.numeric_value
        if target is not None and answer.numeric_estimate is not None:
            numeric_error = abs(answer.numeric_estimate - target)
            relative_error = numeric_error / abs(target) if target else None
            within_tolerance = numeric_error <= (question.tolerance or 0.0)
            correct = bool(within_tolerance)
    outcome = "correct" if correct else "incorrect"
    if not correct and answer.confidence >= 0.8:
        outcome = "overconfident_error"
    return ClosedQuestionEvaluation(
        project_id=project.project_id,
        run_id=run_id,
        question_id=question.question_id,
        correct=correct,
        exact_match=correct if question.question_type in {"single_choice", "multi_choice"} else None,
        top_k_correct=bool(set(selected) & set(truth.correct_answer_ids)) if truth.correct_answer_ids else None,
        precision=precision,
        recall=recall,
        f1=f1,
        ranking_score=ranking_score,
        numeric_error=numeric_error,
        relative_error=relative_error,
        within_tolerance=within_tolerance,
        abstention_correct=False,
        confidence=answer.confidence,
        calibration_bin=_bin(answer.confidence),
        evidence_sufficiency="sufficient" if answer.supporting_evidence_ids else "weak",
        token_cost=tokens,
        call_count=calls,
        outcome=outcome,  # type: ignore[arg-type]
        evaluated_at=datetime.now(UTC),
    )


def calibration_summary(project_id: str, run_id: str, evaluations: list[ClosedQuestionEvaluation], total_tokens: int) -> AnswerCalibrationSummary:
    bins: dict[str, dict[str, float | int]] = {}
    answered = [item for item in evaluations if item.outcome not in {"correct_abstention", "unnecessary_abstention", "insufficient_evidence"}]
    for label in ["0.0-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
        members = [item for item in evaluations if item.calibration_bin == label]
        bins[label] = {
            "count": len(members),
            "accuracy": round(sum(1 for item in members if item.correct) / len(members), 3) if members else 0.0,
            "mean_confidence": round(sum(item.confidence for item in members) / len(members), 3) if members else 0.0,
        }
    correct = sum(1 for item in evaluations if item.correct)
    return AnswerCalibrationSummary(
        project_id=project_id,
        run_id=run_id,
        question_count=len(evaluations),
        coverage=round(len(answered) / len(evaluations), 3) if evaluations else 0.0,
        selective_accuracy=round(sum(1 for item in answered if item.correct) / len(answered), 3) if answered else None,
        bins=bins,
        correct_answers_per_million_tokens=round(correct / total_tokens * 1_000_000, 3) if total_tokens else None,
        cost_per_correct_answer=None,
    )


def build_context_record(project: ClosedQuestionProject, run_id: str, question: ClosedQuestion, corpus: list[Paper]) -> ContextBuildRecord:
    selected = _compress_evidence(question, corpus, project.config.grounding.max_total_context_characters)
    text = json.dumps({
        "question_id": question.question_id,
        "prompt": question.prompt,
        "answer_options": [option.model_dump() for option in question.answer_options],
        "evidence_ids": [paper.id for paper in selected],
    }, sort_keys=True)
    omitted = [paper.id for paper in _question_corpus(question, corpus) if paper not in selected]
    return ContextBuildRecord(
        project_id=project.project_id,
        run_id=run_id,
        question_id=question.question_id,
        context_type="answer_synthesis",
        input_character_count=sum(len((paper.abstract or "")[: project.config.grounding.max_chunk_characters]) for paper in _question_corpus(question, corpus)),
        output_character_count=len(text),
        omitted_items=omitted,
        context_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


def compare_closed_feedback(project_path: str | Path, *, runs_dir: str | Path = "runs", experiment_id: str | None = None, force: bool = False) -> Path:
    project = load_closed_question_project(project_path)
    experiment_id = experiment_id or f"{project.project_id}-closed-feedback-ab"
    out = Path(runs_dir) / experiment_id
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"closed feedback comparison artifacts are immutable; use a new experiment id or --force: {out}")
    out.mkdir(parents=True, exist_ok=True)
    control_dir = run_closed_question_project(project_path, runs_dir=out, run_id="control_advisory", force=True, controlled_feedback=False)
    treatment_dir = run_closed_question_project(project_path, runs_dir=out, run_id="treatment_controlled_feedback", force=True, controlled_feedback=True)
    control = [ClosedQuestionEvaluation.model_validate_json(json.dumps(item)) for item in read_json(control_dir / "closed_question_evaluations.json")]
    treatment = [ClosedQuestionEvaluation.model_validate_json(json.dumps(item)) for item in read_json(treatment_dir / "closed_question_evaluations.json")]
    comparison = _closed_ab_comparison(project, control, treatment, experiment_id)
    write_json(out / "closed_question_ab_comparison.json", comparison)
    (out / "report.md").write_text(_closed_ab_report(project, comparison), encoding="utf-8")
    (out / "human_review.md").write_text(_closed_ab_human_review(project, comparison), encoding="utf-8")
    return out


def validate_closed_feedback_artifacts(experiment_dir: str | Path) -> list[str]:
    path = Path(experiment_dir)
    errors = []
    for name in ["control_advisory", "treatment_controlled_feedback"]:
        branch = path / name
        if not branch.exists():
            errors.append(f"missing closed-feedback branch: {name}")
        else:
            errors.extend(validate_closed_question_artifacts(branch))
    if not (path / "closed_question_ab_comparison.json").exists():
        errors.append("missing closed_question_ab_comparison.json")
    return errors


def build_closed_question_report(
    project: ClosedQuestionProject,
    answers: list[FinalAnswer],
    evaluations: list[ClosedQuestionEvaluation],
    calibration: AnswerCalibrationSummary,
    *,
    report_context: dict[str, dict[str, Any]] | None = None,
) -> str:
    lines = [f"# Closed Question Report: {project.title}", "", "> Offline deterministic closed-question evaluation. Do not treat outputs as scientific proof.", ""]
    eval_by_q = {item.question_id: item for item in evaluations}
    for answer in answers:
        evaluation = eval_by_q.get(answer.question_id)
        context = (report_context or {}).get(answer.question_id)
        lines.extend([
            f"## {answer.question_id}",
            "",
            f"- Selected answers: {', '.join(answer.selected_answer_ids) if answer.selected_answer_ids else 'none'}",
            f"- Abstained: {answer.abstained}",
            f"- Confidence: {answer.confidence:.3f}",
            f"- Supporting evidence: {', '.join(answer.supporting_evidence_ids) or 'none'}",
            f"- Contradicting evidence: {', '.join(answer.contradicting_evidence_ids) or 'none'}",
            f"- Independent clusters: {', '.join(answer.supporting_cluster_ids) or 'none'}",
            f"- Evaluation outcome: {evaluation.outcome if evaluation else 'unscored'}",
            f"- Recommended next action: {answer.recommended_next_action}",
            "",
        ])
        if context:
            lines.extend(_evidence_interpretation_lines(context["question"], context["cells"], context["links"], context["corpus"]))
    lines.extend([
        "## Calibration",
        "",
        f"- Coverage: {calibration.coverage:.3f}",
        f"- Selective accuracy: {calibration.selective_accuracy if calibration.selective_accuracy is not None else 'unavailable'}",
        "- Token cost: unavailable for deterministic mock closed-question runs.",
        "",
    ])
    return "\n".join(lines)


def _evidence_interpretation_lines(
    question: ClosedQuestion,
    cells: list[AnswerEvidenceCell],
    links: list[HypothesisAnswerLink],
    corpus: list[Paper],
) -> list[str]:
    cells_by_answer = {cell.answer_id: cell for cell in cells}
    support_by_answer = {
        option.answer_id: _stable_unique([evidence_id for link in links if link.answer_id == option.answer_id for evidence_id in link.supporting_evidence_ids])
        for option in question.answer_options
    }
    contradict_by_answer = {
        option.answer_id: _stable_unique([evidence_id for link in links if link.answer_id == option.answer_id for evidence_id in link.contradicting_evidence_ids])
        for option in question.answer_options
    }
    paper_by_id = {paper.id: paper for paper in corpus}
    lines = ["### Evidence Interpretation", ""]
    for option in question.answer_options:
        cell = cells_by_answer.get(option.answer_id)
        support_ids = support_by_answer.get(option.answer_id, [])
        contradict_ids = contradict_by_answer.get(option.answer_id, [])
        support_text = _evidence_summary(support_ids, paper_by_id)
        contradict_text = _evidence_summary(contradict_ids, paper_by_id)
        if cell and cell.verified_support_count:
            status = (
                f"supported by {cell.verified_support_count} evidence item(s), "
                f"{len(cell.independent_cluster_ids)} independent cluster(s), "
                f"duplicate-adjusted support {cell.duplicate_adjusted_support:.1f}"
            )
        elif support_ids:
            status = "mentioned by evidence but not counted as verified support"
        else:
            status = "not directly supported by current curated evidence"
        if cell and cell.verified_contradiction_count:
            status += f"; contradicted by {cell.verified_contradiction_count} item(s)"
        lines.extend([
            f"#### {option.answer_id}: {option.statement}",
            "",
            f"- Status: {status}.",
            f"- Supporting evidence: {support_text}",
            f"- Contradicting evidence: {contradict_text}",
        ])
        if cell and cell.unresolved_issues:
            lines.append(f"- Unresolved issues: {'; '.join(cell.unresolved_issues)}")
        lines.append("")
    return lines


def _evidence_summary(evidence_ids: list[str], paper_by_id: dict[str, Paper]) -> str:
    if not evidence_ids:
        return "none"
    parts = []
    for evidence_id in evidence_ids[:4]:
        paper = paper_by_id.get(evidence_id)
        if not paper:
            parts.append(evidence_id)
            continue
        excerpt = (paper.abstract or paper.title).strip()
        if len(excerpt) > 180:
            excerpt = excerpt[:177].rstrip() + "..."
        parts.append(f"{evidence_id} ({excerpt})")
    if len(evidence_ids) > 4:
        parts.append(f"+{len(evidence_ids) - 4} more")
    return "; ".join(parts)


def build_closed_question_human_review(project: ClosedQuestionProject, answers: list[FinalAnswer]) -> str:
    return "\n".join([
        f"# Closed Question Human Review: {project.title}",
        "",
        "## Review Questions",
        "",
        "- Is the answer space well constructed?",
        "- Was the selected answer scientifically defensible?",
        "- Should the system have abstained?",
        "- Were any answer options unfairly broad or overlapping?",
        "- Were evidence sources independent?",
        "- Did duplicate hypotheses inflate support?",
        "- Was the confidence calibrated?",
        "- Was the recommended experiment truly discriminating?",
        "- Did controlled feedback improve the final answer?",
        "- Was the improvement worth the token cost?",
        "",
    ])


def _closed_ab_comparison(project: ClosedQuestionProject, control: list[ClosedQuestionEvaluation], treatment: list[ClosedQuestionEvaluation], experiment_id: str) -> dict[str, Any]:
    control_correct = sum(1 for item in control if item.correct)
    treatment_correct = sum(1 for item in treatment if item.correct)
    control_abstain = sum(1 for item in control if item.outcome in {"correct_abstention", "unnecessary_abstention"})
    treatment_abstain = sum(1 for item in treatment if item.outcome in {"correct_abstention", "unnecessary_abstention"})
    delta = treatment_correct - control_correct
    if delta > 0:
        outcome = "improved"
    elif delta < 0:
        outcome = "regressed"
    elif treatment_abstain != control_abstain:
        outcome = "mixed"
    else:
        outcome = "no_material_change"
    return {
        "schema_version": "v16",
        "experiment_id": experiment_id,
        "project_id": project.project_id,
        "control_correct": control_correct,
        "treatment_correct": treatment_correct,
        "control_abstentions": control_abstain,
        "treatment_abstentions": treatment_abstain,
        "outcome_label": outcome,
        "outcome_rationale": "Bounded comparison over objective closed-question evaluations.",
        "validation_status": "validated",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _closed_ab_report(project: ClosedQuestionProject, comparison: dict[str, Any]) -> str:
    return "\n".join([
        f"# Closed Feedback A/B Report: {project.title}",
        "",
        f"- Outcome: `{comparison['outcome_label']}`",
        f"- Control correct: {comparison['control_correct']}",
        f"- Treatment correct: {comparison['treatment_correct']}",
        f"- Control abstentions: {comparison['control_abstentions']}",
        f"- Treatment abstentions: {comparison['treatment_abstentions']}",
        "",
    ])


def _closed_ab_human_review(project: ClosedQuestionProject, comparison: dict[str, Any]) -> str:
    return "\n".join([
        f"# Closed Feedback Human Review: {project.title}",
        "",
        "- Did controlled feedback improve final-answer correctness?",
        "- Did abstention behavior improve?",
        "- Were any changes worth the additional process complexity?",
        "",
    ])


def _abstain(project: ClosedQuestionProject, run_id: str, question: ClosedQuestion, reason: str) -> FinalAnswer:
    return FinalAnswer(
        project_id=project.project_id,
        run_id=run_id,
        question_id=question.question_id,
        confidence=0.35,
        abstained=True,
        abstention_reason=reason,
        rationale_summary="The deterministic sufficiency policy required abstention.",
        remaining_uncertainty=[reason],
        recommended_next_action="Acquire more independent verified evidence before answering.",
        created_at=datetime.now(UTC),
    )


def _question_corpus(question: ClosedQuestion, corpus: list[Paper]) -> list[Paper]:
    return [paper for paper in corpus if question.question_id in paper.source_metadata.get("question_ids", [])]


def _compress_evidence(question: ClosedQuestion, corpus: list[Paper], limit: int) -> list[Paper]:
    selected = []
    total = 0
    candidates = sorted(_question_corpus(question, corpus), key=lambda paper: (
        bool(paper.source_metadata.get("metadata_only", False)),
        -int(bool(paper.source_metadata.get("contradicts", []))),
        paper.id,
    ))
    seen_sources: dict[str, int] = defaultdict(int)
    for paper in candidates:
        source = str(paper.source_metadata.get("source_group", paper.id))
        if seen_sources[source] >= 2:
            continue
        excerpt = paper.abstract or paper.title
        if total + len(excerpt) > limit and selected:
            continue
        selected.append(paper)
        total += len(excerpt)
        seen_sources[source] += 1
    return selected


def _rankings(hypotheses: list[Hypothesis]) -> list[HypothesisRanking]:
    return [
        HypothesisRanking(
            hypothesis_id=hypothesis.id,
            correctness=6.0,
            novelty=5.0,
            testability=6.0,
            explanatory_power=6.0,
            feasibility=6.0,
            discriminative_power=6.0,
            evidence_quality=7.0 if hypothesis.evidence_links else 3.0,
            impact=5.0,
            parsimony=6.0,
            weighted_total=6.0,
        )
        for hypothesis in hypotheses
    ]


def _ranking_agreement(predicted: list[str], truth: list[str]) -> float:
    if not predicted or not truth:
        return 0.0
    pairs = 0
    agree = 0
    pred_pos = {item: index for index, item in enumerate(predicted)}
    truth_pos = {item: index for index, item in enumerate(truth)}
    common = [item for item in truth if item in pred_pos]
    for i, left in enumerate(common):
        for right in common[i + 1:]:
            pairs += 1
            agree += int((pred_pos[left] < pred_pos[right]) == (truth_pos[left] < truth_pos[right]))
    return round(agree / pairs, 3) if pairs else 1.0


def _bin(confidence: float) -> str:
    if confidence < 0.5:
        return "0.0-0.5"
    if confidence < 0.7:
        return "0.5-0.7"
    if confidence < 0.9:
        return "0.7-0.9"
    return "0.9-1.0"


def _stable_unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _artifact_prefix(question_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in question_id)


def _resolve_project_path(project_file: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return project_file.parent / path
