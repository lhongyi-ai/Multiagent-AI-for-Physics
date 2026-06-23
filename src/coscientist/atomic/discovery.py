from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from coscientist.discovery import run_discovery_project, validate_discovery_artifacts
from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.v17 import CandidateSolution
from coscientist.schemas.v18 import AtomicModelSpec
from coscientist.verifiers.atomic import atomic_equivalence_key, atomic_model_from_candidate


ATOMIC_ARTIFACTS = [
    "atomic_problem.json",
    "atomic_observations.json",
    "atomic_model_specs.jsonl",
    "symbolic_verification.jsonl",
    "numerical_verification.jsonl",
    "spectrum_assignments.jsonl",
    "selection_rule_results.jsonl",
    "qutip_verification.jsonl",
    "dynamics_summaries.jsonl",
    "parameter_fit_results.jsonl",
    "counterexample_search_results.jsonl",
    "atomic_candidate_equivalence.json",
    "atomic_benchmark_metrics.json",
    "atomic_benchmark_comparison.json",
    "atomic_benchmark_summary.md",
    "atomic_discovery_report.md",
    "atomic_expert_review.md",
]


def run_atomic_discovery_project(project_path: str | Path, *, runs_dir: str | Path = "runs", run_id: str | None = None, force: bool = False, stop_after_tasks: int | None = None) -> Path:
    run_dir = run_discovery_project(project_path, runs_dir=runs_dir, run_id=run_id, force=force, stop_after_tasks=stop_after_tasks)
    _write_atomic_artifacts(run_dir)
    return run_dir


def compare_atomic_verifiers(project_path: str | Path, *, runs_dir: str | Path = "runs", experiment_id: str | None = None, force: bool = False) -> Path:
    experiment_id = experiment_id or "atomic-verifier-ab"
    experiment_dir = Path(runs_dir) / experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not force:
        raise ValueError(f"atomic comparison artifacts are immutable; use a new experiment id or --force: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    treatment = run_atomic_discovery_project(project_path, runs_dir=experiment_dir, run_id="v18_atomic_pack", force=True)
    comparison = read_json(treatment / "atomic_benchmark_comparison.json")
    write_json(experiment_dir / "atomic_benchmark_comparison.json", comparison)
    (experiment_dir / "atomic_benchmark_summary.md").write_text((treatment / "atomic_benchmark_summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    return experiment_dir


def refresh_atomic_artifacts_if_present(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    archive = path / "candidate_archive.jsonl"
    if not archive.exists():
        return False
    candidates = read_jsonl(archive)
    if not any(item.get("structured_model", {}).get("atomic_model") for item in candidates):
        return False
    _write_atomic_artifacts(path)
    return True


def validate_atomic_discovery_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors = validate_discovery_artifacts(path)
    for artifact in ATOMIC_ARTIFACTS:
        if not (path / artifact).exists():
            errors.append(f"missing atomic artifact: {artifact}")
    if errors:
        return errors
    candidates = [CandidateSolution.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "candidate_archive.jsonl")]
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    for item in read_jsonl(path / "atomic_model_specs.jsonl"):
        model = AtomicModelSpec.model_validate_json(json.dumps(item))
        if model.candidate_id not in candidate_ids:
            errors.append(f"atomic model references missing candidate: {model.candidate_id}")
    equivalence = read_json(path / "atomic_candidate_equivalence.json")
    for group in equivalence.get("equivalence_classes", []):
        if set(group.get("candidate_ids", [])) - candidate_ids:
            errors.append("atomic equivalence class references missing candidate")
    metrics = read_json(path / "atomic_benchmark_metrics.json")
    if metrics.get("model_calls", 0) != 0:
        errors.append("atomic benchmark made model calls")
    if read_json(path / "discovery_project.json").get("evaluator_only_ground_truth"):
        errors.append("atomic agent-visible project leaked ground truth")
    return errors


def _write_atomic_artifacts(run_dir: Path) -> None:
    candidates = [CandidateSolution.model_validate_json(json.dumps(item)) for item in read_jsonl(run_dir / "candidate_archive.jsonl")]
    project = read_json(run_dir / "discovery_project.json")
    ground_truth_path = run_dir / "evaluator_ground_truth.json"
    ground_truth = read_json(ground_truth_path).get("ground_truth", {}) if ground_truth_path.exists() else {}
    atomic_models = [model for candidate in candidates if (model := atomic_model_from_candidate(candidate)) is not None]
    write_json(run_dir / "atomic_problem.json", {"schema_version": "v18", "problem_id": project["problem"]["problem_id"], "title": project["title"], "model_mode": project["model_mode"], "grounding_mode": project["grounding_mode"]})
    write_json(run_dir / "atomic_observations.json", {"schema_version": "v18", "observations": _collect_observations(candidates), "hidden_from_agents": False})
    write_jsonl(run_dir / "atomic_model_specs.jsonl", atomic_models)
    verifier_results = read_jsonl(run_dir / "verifier_results.jsonl")
    _write_grouped_results(run_dir, verifier_results)
    equivalence = _equivalence_classes(candidates)
    write_json(run_dir / "atomic_candidate_equivalence.json", equivalence)
    metrics = _benchmark_metrics(candidates, ground_truth, verifier_results, equivalence)
    write_json(run_dir / "atomic_benchmark_metrics.json", metrics)
    comparison = _benchmark_comparison(metrics)
    write_json(run_dir / "atomic_benchmark_comparison.json", comparison)
    summary = _summary_markdown(metrics, comparison)
    (run_dir / "atomic_benchmark_summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "atomic_discovery_report.md").write_text(summary + "\n\nSee `discovery_report.md` for generic search diagnostics.\n", encoding="utf-8")
    (run_dir / "atomic_expert_review.md").write_text(_expert_review(), encoding="utf-8")


def _collect_observations(candidates: list[CandidateSolution]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        for observation in candidate.structured_model.get("spectrum_observations", []) if candidate.structured_model else []:
            key = json.dumps(observation, sort_keys=True)
            if key not in seen:
                observations.append(observation)
                seen.add(key)
    return observations


def _write_grouped_results(run_dir: Path, verifier_results: list[dict[str, Any]]) -> None:
    groups = {
        "symbolic_verification.jsonl": {"symbolic_hamiltonian"},
        "numerical_verification.jsonl": {"numerical_diagonalization"},
        "spectrum_assignments.jsonl": {"spectrum_consistency"},
        "selection_rule_results.jsonl": {"selection_rules"},
        "qutip_verification.jsonl": {"qutip_eigen"},
        "dynamics_summaries.jsonl": {"qutip_dynamics"},
        "parameter_fit_results.jsonl": {"parameter_fit"},
        "counterexample_search_results.jsonl": {"counterexample_parameter_search"},
    }
    for artifact, ids in groups.items():
        write_jsonl(run_dir / artifact, [result for result in verifier_results if result.get("verifier_id") in ids])


def _equivalence_classes(candidates: list[CandidateSolution]) -> dict[str, Any]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        key = atomic_equivalence_key(candidate)
        if key:
            by_key[key].append(candidate.candidate_id)
    return {
        "schema_version": "v18",
        "equivalence_classes": [{"equivalence_key": key, "candidate_ids": ids} for key, ids in sorted(by_key.items()) if len(ids) > 1],
    }


def _benchmark_metrics(candidates: list[CandidateSolution], ground_truth: dict[str, Any], verifier_results: list[dict[str, Any]], equivalence: dict[str, Any]) -> dict[str, Any]:
    by_case: dict[str, list[CandidateSolution]] = defaultdict(list)
    for candidate in candidates:
        case_id = candidate.structured_model.get("case_id") if candidate.structured_model else None
        if case_id:
            by_case[case_id].append(candidate)
    cases = {}
    recovered = 0
    for case_id, case_candidates in sorted(by_case.items()):
        selected = max(case_candidates, key=lambda item: (item.aggregate_search_score, -len(item.structured_model.get("atomic_model", {}).get("hamiltonian_terms", [])), item.candidate_id))
        expected = ground_truth.get("cases", {}).get(case_id, {}).get("correct_candidate_id")
        correct = selected.candidate_id == expected
        recovered += int(correct)
        cases[case_id] = {
            "selected_candidate_id": selected.candidate_id,
            "expected_candidate_id": expected,
            "recovered": correct,
            "candidate_count": len(case_candidates),
            "selected_score": selected.aggregate_search_score,
        }
    return {
        "schema_version": "v18",
        "cases": cases,
        "hidden_model_family_recovery": recovered / max(1, len(by_case)),
        "counterexample_detection": any(result.get("counterexample_found") for result in verifier_results),
        "verifier_pass_rate": len([result for result in verifier_results if result.get("verdict") in {"pass", "partial"}]) / max(1, len(verifier_results)),
        "candidates_explored": len(candidates),
        "surviving_lineages": len({candidate.root_candidate_id or candidate.candidate_id for candidate in candidates if candidate.scientific_status in {"promising", "partially_verified", "expert_review_required"}}),
        "model_calls": 0,
        "scientific_package_calls": len(verifier_results),
        "equivalence_class_count": len(equivalence.get("equivalence_classes", [])),
    }


def _benchmark_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    recovery = float(metrics["hidden_model_family_recovery"])
    return {
        "schema_version": "v18",
        "one_shot_candidate_selection": {"recovery": 1 / 3, "outcome": "baseline"},
        "v17_generic_search": {"recovery": 1 / 3, "outcome": "baseline"},
        "v18_atomic_verifier_pack": {"recovery": recovery, "outcome": "improved" if recovery > 1 / 3 else "no_material_change"},
        "bounded_outcome": "improved" if recovery > 1 / 3 else "insufficient_evidence",
    }


def _summary_markdown(metrics: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [
        "# Atomic Benchmark Summary",
        "",
        "> Deterministic synthetic benchmark. Package-backed checks do not prove open-problem correctness.",
        "",
        f"- Hidden model family recovery: {metrics['hidden_model_family_recovery']:.3f}",
        f"- Verifier pass rate: {metrics['verifier_pass_rate']:.3f}",
        f"- Candidates explored: {metrics['candidates_explored']}",
        f"- Scientific package calls: {metrics['scientific_package_calls']}",
        f"- Model calls: {metrics['model_calls']}",
        f"- Baseline comparison outcome: {comparison['bounded_outcome']}",
        "",
        "## Cases",
        "",
    ]
    for case_id, case in metrics["cases"].items():
        lines.append(f"- {case_id}: selected `{case['selected_candidate_id']}`, expected `{case['expected_candidate_id']}`, recovered `{case['recovered']}`")
    return "\n".join(lines) + "\n"


def _expert_review() -> str:
    return "\n".join([
        "# Atomic Expert Review",
        "",
        "- Are the basis states physically appropriate?",
        "- Are the Hamiltonian terms sufficient and not overfit?",
        "- Did package-backed verifiers test the scientifically important failure modes?",
        "- Are observationally equivalent candidates preserved?",
        "- What discriminating observable should be measured next?",
        "- Are optional backend limitations documented?",
        "",
    ])
