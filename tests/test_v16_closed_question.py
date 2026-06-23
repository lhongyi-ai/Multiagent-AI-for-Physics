from __future__ import annotations

from pathlib import Path

from coscientist.closed_question import (
    compare_closed_feedback,
    evaluate_final_answer,
    load_closed_question_project,
    run_closed_question_project,
    validate_closed_feedback_artifacts,
    validate_closed_question_artifacts,
)
from coscientist.pilot.artifacts import read_json


BENCHMARK = "examples/closed_question_benchmark/project.yaml"
CAFE = "examples/cafe4al8_closed_pilot/project.yaml"


def test_closed_question_benchmark_runs_and_scores_objectively(tmp_path: Path) -> None:
    run_dir = run_closed_question_project(BENCHMARK, runs_dir=tmp_path, run_id="closed-demo")
    assert validate_closed_question_artifacts(run_dir) == []
    evaluations = read_json(run_dir / "closed_question_evaluations.json")
    by_id = {item["question_id"]: item for item in evaluations}
    assert len(evaluations) == 8
    assert by_id["q_single_sufficient"]["outcome"] == "correct"
    assert by_id["q_single_abstain"]["outcome"] == "correct_abstention"
    assert by_id["q_multi"]["precision"] == 1.0
    assert by_id["q_numeric"]["within_tolerance"] is True
    assert by_id["q_metadata_only"]["outcome"] == "correct_abstention"


def test_hypothesis_answer_links_preserve_valid_ids_and_metadata_only_is_not_support(tmp_path: Path) -> None:
    run_dir = run_closed_question_project(BENCHMARK, runs_dir=tmp_path, run_id="closed-demo")
    links = read_json(run_dir / "q_metadata_only_hypothesis_answer_links_round_0.json")
    assert links
    assert all(link["relation"] != "supports" for link in links)
    final = read_json(run_dir / "q_metadata_only_final_answer.json")
    assert final["abstained"] is True
    assert final["supporting_evidence_ids"] == []


def test_duplicate_cluster_support_is_discounted(tmp_path: Path) -> None:
    run_dir = run_closed_question_project(BENCHMARK, runs_dir=tmp_path, run_id="closed-demo")
    matrix = read_json(run_dir / "q_duplicate_answer_evidence_matrix_round_0.json")
    cell = next(item for item in matrix if item["answer_id"] == "independent_support_required")
    assert cell["verified_support_count"] == 2
    assert cell["duplicate_adjusted_support"] <= cell["verified_support_count"]


def test_closed_feedback_comparison_uses_bounded_label_and_validates(tmp_path: Path) -> None:
    experiment = compare_closed_feedback(BENCHMARK, runs_dir=tmp_path, experiment_id="closed-ab")
    assert validate_closed_feedback_artifacts(experiment) == []
    comparison = read_json(experiment / "closed_question_ab_comparison.json")
    assert comparison["outcome_label"] in {"improved", "mixed", "no_material_change", "regressed", "insufficient_evidence"}
    assert (experiment / "control_advisory" / "closed_question_evaluations.json").exists()
    assert (experiment / "treatment_controlled_feedback" / "closed_question_evaluations.json").exists()


def test_cafe4al8_closed_pilot_outputs_mechanism_and_experiment_answers(tmp_path: Path) -> None:
    run_dir = run_closed_question_project(CAFE, runs_dir=tmp_path, run_id="cafe")
    assert validate_closed_question_artifacts(run_dir) == []
    evaluations = read_json(run_dir / "closed_question_evaluations.json")
    assert {item["question_id"] for item in evaluations} == {"cafe-mechanism", "cafe-next-experiment"}
    mechanism = read_json(run_dir / "cafe-mechanism_final_answer.json")
    experiment = read_json(run_dir / "cafe-next-experiment_final_answer.json")
    assert mechanism["selected_answer_ids"] == ["F"]
    assert experiment["selected_answer_ids"] == ["icp_bulk"]


def test_missing_ground_truth_produces_insufficient_evidence() -> None:
    project = load_closed_question_project(BENCHMARK)
    question = project.questions[0]
    run_dir = Path("/tmp/not-used")
    answer = read_json(run_closed_question_project(BENCHMARK, runs_dir=run_dir, run_id="tmp", force=True) / "q_single_sufficient_final_answer.json")
    from coscientist.schemas.v16 import FinalAnswer

    evaluation = evaluate_final_answer(project, "r", question, FinalAnswer.model_validate_json(__import__("json").dumps(answer)), None, calls=0, tokens=0)
    assert evaluation.outcome == "insufficient_evidence"
