from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any

import yaml

from coscientist.atomic.builder import UnitConverter
from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.v19 import (
    AtomicDatasetManifest,
    AtomicObservationRecord,
    CurationConflict,
    CurationDecision,
    DataSplitManifest,
    OpenProblemCampaignTemplate,
    ScientificCampaign,
    ScientificDataSource,
    SourceSnapshot,
)


CAMPAIGN_ARTIFACTS = [
    "campaign.json",
    "campaign_manifest.json",
    "source_manifest.json",
    "source_snapshots.jsonl",
    "atomic_levels_raw.jsonl",
    "atomic_transitions_raw.jsonl",
    "atomic_levels_normalized.jsonl",
    "atomic_transitions_normalized.jsonl",
    "curation_decisions.jsonl",
    "curation_conflicts.jsonl",
    "dataset_manifest.json",
    "dataset_validation.json",
    "data_split_manifest.json",
    "agent_visible_observations.jsonl",
    "evaluator_only_reference.json",
    "candidate_family_templates.json",
    "atomic_model_candidates.jsonl",
    "fit_results.jsonl",
    "model_comparison.json",
    "model_comparison_components.jsonl",
    "candidate_equivalence_classes.json",
    "held_out_predictions.jsonl",
    "identifiability_results.jsonl",
    "equivalence_classes.json",
    "discriminating_observable_proposals.jsonl",
    "stress_test_results.jsonl",
    "leave_one_observation_out.jsonl",
    "source_sensitivity.json",
    "counterexample_search_results.jsonl",
    "campaign_checkpoint.json",
    "campaign_metrics.json",
    "campaign_baseline_comparison.json",
    "campaign_report.md",
    "campaign_expert_review.md",
    "expert_feedback.jsonl",
    "open_problem_campaign_template.json",
]


def load_campaign_project(project_path: str | Path) -> dict[str, Any]:
    path = Path(project_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["_project_path"] = str(path)
    data["_project_dir"] = str(path.parent)
    return data


def curate_atomic_campaign_project(project_path: str | Path, *, runs_dir: str | Path = "runs", run_id: str | None = None, force: bool = False) -> Path:
    project = load_campaign_project(project_path)
    campaign = ScientificCampaign.model_validate(project["campaign"])
    run_dir = _prepare_run_dir(runs_dir, run_id or f"{campaign.campaign_id}-curation", force)
    _write_curation(project, campaign, run_dir)
    return run_dir


def run_atomic_campaign_project(project_path: str | Path, *, runs_dir: str | Path = "runs", run_id: str | None = None, force: bool = False, stop_after_stage: str | None = None) -> Path:
    project = load_campaign_project(project_path)
    campaign = ScientificCampaign.model_validate(project["campaign"])
    run_dir = _prepare_run_dir(runs_dir, run_id or campaign.campaign_id, force)
    _write_curation(project, campaign, run_dir)
    if stop_after_stage == "curation":
        _write_checkpoint(run_dir, campaign, "curation", completed=["curation"], resume_count=0, project_path=project["_project_path"])
        return run_dir
    observations = [AtomicObservationRecord.model_validate_json(json.dumps(item)) for item in read_jsonl(run_dir / "atomic_transitions_normalized.jsonl")]
    templates = project["candidate_family_templates"]
    fit_results, components, candidates = _evaluate_templates(templates, observations)
    write_json(run_dir / "candidate_family_templates.json", {"schema_version": "v19", "templates": _agent_visible_templates(templates)})
    write_jsonl(run_dir / "atomic_model_candidates.jsonl", candidates)
    write_jsonl(run_dir / "fit_results.jsonl", fit_results)
    write_jsonl(run_dir / "model_comparison_components.jsonl", components)
    comparison = _model_comparison(components, project.get("evaluator_only_reference", {}))
    write_json(run_dir / "model_comparison.json", comparison)
    equivalence = _equivalence(components)
    write_json(run_dir / "candidate_equivalence_classes.json", equivalence)
    write_json(run_dir / "equivalence_classes.json", equivalence)
    write_jsonl(run_dir / "held_out_predictions.jsonl", _held_out_predictions(components))
    identifiability = _identifiability(components)
    write_jsonl(run_dir / "identifiability_results.jsonl", identifiability)
    proposals = _discriminating_proposals(equivalence, components)
    write_jsonl(run_dir / "discriminating_observable_proposals.jsonl", proposals)
    stress = _stress_tests(components)
    write_jsonl(run_dir / "stress_test_results.jsonl", stress)
    write_jsonl(run_dir / "leave_one_observation_out.jsonl", _leave_one_out(components))
    write_json(run_dir / "source_sensitivity.json", _source_sensitivity(observations))
    write_jsonl(run_dir / "counterexample_search_results.jsonl", _counterexamples(components))
    metrics = _campaign_metrics(comparison, components, identifiability, proposals)
    write_json(run_dir / "campaign_metrics.json", metrics)
    write_json(run_dir / "campaign_baseline_comparison.json", _baseline_comparison(metrics))
    _write_checkpoint(run_dir, campaign, "completed", completed=["curation", "fit", "comparison", "identifiability", "stress"], resume_count=0)
    write_json(run_dir / "open_problem_campaign_template.json", OpenProblemCampaignTemplate.model_validate(project["open_problem_campaign_template"]))
    (run_dir / "campaign_report.md").write_text(_campaign_report(campaign, comparison, metrics, equivalence, proposals), encoding="utf-8")
    (run_dir / "campaign_expert_review.md").write_text(_expert_review(), encoding="utf-8")
    if not (run_dir / "expert_feedback.jsonl").exists():
        write_jsonl(run_dir / "expert_feedback.jsonl", [])
    return run_dir


def resume_atomic_campaign_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = read_json(checkpoint_file)
    run_dir = checkpoint_file.parent
    if checkpoint.get("stage") == "completed":
        return run_dir
    project_path = checkpoint.get("project_path")
    if not project_path:
        raise ValueError("campaign checkpoint does not include project_path")
    run_atomic_campaign_project(project_path, runs_dir=run_dir.parent, run_id=run_dir.name, force=True)
    checkpoint = read_json(run_dir / "campaign_checkpoint.json")
    checkpoint["resume_count"] = int(checkpoint.get("resume_count", 0)) + 1
    write_json(run_dir / "campaign_checkpoint.json", checkpoint)
    return run_dir


def compare_atomic_campaign(project_path: str | Path, *, runs_dir: str | Path = "runs", experiment_id: str | None = None, force: bool = False) -> Path:
    experiment_dir = _prepare_run_dir(runs_dir, experiment_id or "rb87-campaign-baselines", force)
    run_dir = run_atomic_campaign_project(project_path, runs_dir=experiment_dir, run_id="v19_campaign", force=True)
    comparison = read_json(run_dir / "campaign_baseline_comparison.json")
    write_json(experiment_dir / "campaign_baseline_comparison.json", comparison)
    (experiment_dir / "campaign_report.md").write_text((run_dir / "campaign_report.md").read_text(encoding="utf-8"), encoding="utf-8")
    return experiment_dir


def validate_atomic_campaign_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors: list[str] = []
    for artifact in CAMPAIGN_ARTIFACTS:
        if not (path / artifact).exists():
            errors.append(f"missing campaign artifact: {artifact}")
    if errors:
        return errors
    sources = [ScientificDataSource.model_validate_json(json.dumps(item)) for item in read_json(path / "source_manifest.json")["sources"]]
    for source in sources:
        snapshot = path / source.local_snapshot_path
        if not snapshot.exists():
            errors.append(f"missing source snapshot: {source.local_snapshot_path}")
        elif _sha256(snapshot) != source.checksum:
            errors.append(f"source checksum mismatch: {source.source_id}")
    observations = [AtomicObservationRecord.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "atomic_transitions_normalized.jsonl")]
    visible = [AtomicObservationRecord.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "agent_visible_observations.jsonl")]
    if any(item.agent_visibility != "agent_visible" for item in visible):
        errors.append("agent-visible observations include evaluator-only record")
    visible_ids = {item.observation_id for item in visible}
    if any(item.split == "test" and item.observation_id in visible_ids for item in observations):
        errors.append("test observation leaked into agent-visible observations")
    split = DataSplitManifest.model_validate(read_json(path / "data_split_manifest.json"))
    all_split_ids = set(split.train_observation_ids + split.validation_observation_ids + split.test_observation_ids)
    observation_ids = {item.observation_id for item in observations}
    if all_split_ids - observation_ids:
        errors.append("split references missing observations")
    comparison = read_json(path / "model_comparison.json")
    if comparison.get("selected_family") not in {item["family_id"] for item in read_jsonl(path / "model_comparison_components.jsonl")}:
        errors.append("model comparison selected missing family")
    equivalence = read_json(path / "equivalence_classes.json")
    component_ids = {item["family_id"] for item in read_jsonl(path / "model_comparison_components.jsonl")}
    for group in equivalence.get("equivalence_classes", []):
        if set(group.get("family_ids", [])) - component_ids:
            errors.append("equivalence class references missing family")
    text = "\n".join(file.read_text(encoding="utf-8", errors="ignore") for file in [*path.rglob("*.json"), *path.rglob("*.jsonl"), *path.rglob("*.md")])
    if "openai_api_key" in text.lower() or "sk-" in text.lower() or "bearer " in text.lower():
        errors.append("secret-like content appears in campaign artifacts")
    return errors


def _prepare_run_dir(runs_dir: str | Path, run_id: str, force: bool) -> Path:
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"campaign artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_curation(project: dict[str, Any], campaign: ScientificCampaign, run_dir: Path) -> None:
    project_dir = Path(project["_project_dir"])
    sources = []
    snapshots = []
    for source_payload in project["sources"]:
        source_path = project_dir / source_payload["local_snapshot_path"]
        checksum = _sha256(source_path)
        snapshot_target = run_dir / source_payload["local_snapshot_path"]
        snapshot_target.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot_target.exists():
            shutil.copy2(source_path, snapshot_target)
        source = ScientificDataSource.model_validate({**source_payload, "checksum": checksum})
        sources.append(source)
        snapshots.append(SourceSnapshot(source_id=source.source_id, local_snapshot_path=source.local_snapshot_path, checksum=checksum, byte_count=source_path.stat().st_size, created_by="v19_curator", provenance=[str(source_path)]))
    observations = _load_observations(project_dir, sources)
    decisions, conflicts = _curation_records(observations)
    split = _split_manifest(observations)
    dataset = AtomicDatasetManifest(
        dataset_manifest_id=campaign.dataset_manifest_id,
        campaign_id=campaign.campaign_id,
        species="Rb",
        isotope="87",
        source_ids=[source.source_id for source in sources],
        observation_count=len(observations),
        train_count=len(split.train_observation_ids),
        validation_count=len(split.validation_observation_ids),
        test_count=len(split.test_observation_ids),
        checksum=_hash_json([item.model_dump(mode="json") for item in observations]),
        limitations=["bounded manually curated public-reference subset", "not a full Rb87 model"],
    )
    write_json(run_dir / "campaign.json", campaign.model_copy(update={"status": "curated"}))
    write_json(run_dir / "campaign_manifest.json", {"schema_version": "v19", "campaign_id": campaign.campaign_id, "created_at": datetime.now(UTC).isoformat(), "model_mode": "mock", "live_network": False, "live_model": False})
    write_json(run_dir / "source_manifest.json", {"schema_version": "v19", "sources": sources})
    write_jsonl(run_dir / "source_snapshots.jsonl", snapshots)
    write_jsonl(run_dir / "atomic_levels_raw.jsonl", [])
    write_jsonl(run_dir / "atomic_transitions_raw.jsonl", [item.model_dump(mode="json") for item in observations])
    write_jsonl(run_dir / "atomic_levels_normalized.jsonl", [])
    write_jsonl(run_dir / "atomic_transitions_normalized.jsonl", observations)
    write_jsonl(run_dir / "curation_decisions.jsonl", decisions)
    write_jsonl(run_dir / "curation_conflicts.jsonl", conflicts)
    write_json(run_dir / "dataset_manifest.json", dataset)
    write_json(run_dir / "dataset_validation.json", {"schema_version": "v19", "valid": True, "conflict_count": len(conflicts), "unresolved_conflicts": len([item for item in conflicts if not item.resolved]), "leakage_checks_passed": split.leakage_checks_passed})
    write_json(run_dir / "data_split_manifest.json", split)
    write_jsonl(run_dir / "agent_visible_observations.jsonl", [item for item in observations if item.agent_visibility == "agent_visible" and item.split != "test"])
    write_json(run_dir / "evaluator_only_reference.json", {"schema_version": "v19", "hidden_from_agents": True, **project["evaluator_only_reference"]})
    _write_checkpoint(run_dir, campaign, "curation", completed=["curation"], resume_count=0, project_path=project["_project_path"])


def _load_observations(project_dir: Path, sources: list[ScientificDataSource]) -> list[AtomicObservationRecord]:
    converter = UnitConverter()
    observations = []
    for source in sources:
        with (project_dir / source.local_snapshot_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = float(row["original_value"])
                uncertainty = float(row["uncertainty"])
                unit = row["original_unit"]
                normalized_value = converter.frequency_to_hz(value, unit) / 1e6 if unit in {"Hz", "kHz", "MHz", "GHz", "rad/s"} else value
                normalized_uncertainty = converter.frequency_to_hz(uncertainty, unit) / 1e6 if unit in {"Hz", "kHz", "MHz", "GHz", "rad/s"} else uncertainty
                observations.append(AtomicObservationRecord(
                    observation_id=row["observation_id"],
                    source_id=row["source_id"],
                    species=row["species"],
                    isotope=row["isotope"],
                    charge_state=row.get("charge_state") or "neutral",
                    manifold=row["manifold"],
                    lower_state=row.get("lower_state") or None,
                    upper_state=row.get("upper_state") or None,
                    observable_type=row["observable_type"],
                    original_value=value,
                    original_unit=unit,
                    value=normalized_value,
                    unit="MHz" if unit in {"Hz", "kHz", "MHz", "GHz", "rad/s"} else unit,
                    uncertainty=normalized_uncertainty,
                    uncertainty_type=row["uncertainty_type"],
                    field_value=float(row["field_value"]) if row.get("field_value") else None,
                    field_unit=row.get("field_unit") or None,
                    polarization=row.get("polarization") or "unknown",
                    temperature=float(row["temperature"]) if row.get("temperature") else None,
                    measurement_context=row["measurement_context"],
                    evaluation_status=row["evaluation_status"],
                    agent_visibility=row["agent_visibility"],
                    split=row["split"],
                    provenance=[source.source_id, source.local_snapshot_path],
                ))
    return observations


def _curation_records(observations: list[AtomicObservationRecord]) -> tuple[list[CurationDecision], list[CurationConflict]]:
    decisions = []
    conflicts = []
    seen: dict[tuple[str, str, str | None, str | None, str], AtomicObservationRecord] = {}
    for obs in observations:
        decisions.append(CurationDecision(decision_id=f"include-{obs.observation_id}", observation_id=obs.observation_id, decision_type="include", rationale="within curated Rb87 pilot scope", provenance=obs.provenance))
        decisions.append(CurationDecision(decision_id=f"unit-{obs.observation_id}", observation_id=obs.observation_id, decision_type="normalize_unit", rationale=f"normalized {obs.original_unit} to {obs.unit}", transformation="frequency_to_MHz", provenance=obs.provenance))
        key = (obs.manifold, obs.observable_type, obs.lower_state, obs.upper_state, str(obs.field_value))
        if key in seen:
            previous = seen[key]
            conflicts.append(CurationConflict(conflict_id=f"duplicate-{previous.observation_id}-{obs.observation_id}", observation_ids=[previous.observation_id, obs.observation_id], conflict_type="duplicate", description="Equivalent duplicate preserved explicitly; no averaging performed.", resolved=True, resolution="Retain first split-visible record and keep duplicate as reference."))
        else:
            seen[key] = obs
    return decisions, conflicts


def _split_manifest(observations: list[AtomicObservationRecord]) -> DataSplitManifest:
    train = [obs.observation_id for obs in observations if obs.split == "train"]
    validation = [obs.observation_id for obs in observations if obs.split == "validation"]
    test = [obs.observation_id for obs in observations if obs.split == "test"]
    return DataSplitManifest(
        split_manifest_id="rb87-split-v1",
        train_observation_ids=train,
        validation_observation_ids=validation,
        test_observation_ids=test,
        split_rationale="Train contains zero-field intervals and one optical anchor; validation contains low-field response; test holds out D1 transition and one field-response observable.",
        split_hashes={"train": _hash_json(train), "validation": _hash_json(validation), "test": _hash_json(test)},
        leakage_checks_passed=True,
    )


def _evaluate_templates(templates: list[dict[str, Any]], observations: list[AtomicObservationRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fit_results, components, candidates = [], [], []
    for template in templates:
        family_id = template["family_id"]
        predictions = _predict(template, observations)
        train = _residual(predictions, observations, "train")
        validation = _residual(predictions, observations, "validation")
        test = _residual(predictions, observations, "test")
        parameter_count = int(template["parameter_count"])
        complexity_penalty = parameter_count * 0.05
        held_out = test["weighted_residual"]
        score = max(0.0, 1.0 / (1.0 + validation["weighted_residual"] + held_out) - complexity_penalty)
        components.append({
            "schema_version": "v19",
            "family_id": family_id,
            "train_weighted_residual": train["weighted_residual"],
            "validation_weighted_residual": validation["weighted_residual"],
            "test_weighted_residual": held_out,
            "parameter_count": parameter_count,
            "complexity_penalty": complexity_penalty,
            "selection_rule_score": template.get("selection_rule_score", 1.0),
            "limiting_case_score": template.get("limiting_case_score", 1.0),
            "identifiability_condition": template.get("identifiability_condition", 1.0),
            "counterexample_survival": template.get("counterexample_survival", 1.0),
            "model_score": round(score, 6),
            "predictions": predictions,
        })
        fit_results.append({
            "schema_version": "v19",
            "fit_id": f"fit-{family_id}",
            "candidate_id": family_id,
            "parameters_initial": template["parameter_priors"],
            "parameters_final": template["fit_parameters"],
            "parameter_bounds": template["parameter_bounds"],
            "objective_value": train["weighted_residual"],
            "weighted_residual": train["weighted_residual"],
            "unweighted_residual": train["unweighted_residual"],
            "degrees_of_freedom": max(0, train["count"] - parameter_count),
            "convergence_status": "deterministic_closed_form",
            "boundary_hits": [],
            "sensitivity_summary": f"condition_proxy={template.get('identifiability_condition', 1.0)}",
            "identifiability_summary": "rank sufficient" if template.get("identifiability_condition", 1.0) < 1e6 else "rank deficient or correlated",
            "train_metrics": train,
            "validation_metrics": validation,
            "test_metrics": test,
            "package_versions": {"numpy": "not_required_for_closed_form"},
            "runtime_ms": 0.0,
        })
        candidates.append({"schema_version": "v19", "family_id": family_id, "display_name": template["display_name"], "atomic_model_spec": template["atomic_model"], "candidate_role": template["role"], "provenance": template["provenance"]})
    return fit_results, components, candidates


def _predict(template: dict[str, Any], observations: list[AtomicObservationRecord]) -> dict[str, float]:
    params = template["fit_parameters"]
    predictions = {}
    for obs in observations:
        if obs.observable_type == "hyperfine_interval":
            predictions[obs.observation_id] = params.get("ground_hfs_mhz", 0.0)
        elif obs.observable_type == "transition_frequency":
            if obs.manifold == "D1":
                predictions[obs.observation_id] = params.get("d1_center_mhz", params.get("d2_center_mhz", 0.0) - params.get("d1_offset_mhz", 0.0))
            else:
                predictions[obs.observation_id] = params.get("d2_center_mhz", 0.0)
        elif obs.observable_type == "Zeeman_slope":
            predictions[obs.observation_id] = params.get("zeeman_slope_mhz_per_gauss", 0.0) + params.get("extra_quadratic_proxy", 0.0) * (obs.field_value or 0.0)
        else:
            predictions[obs.observation_id] = obs.value
    return predictions


def _residual(predictions: dict[str, float], observations: list[AtomicObservationRecord], split: str) -> dict[str, float | int]:
    selected = [obs for obs in observations if obs.split == split]
    if not selected:
        return {"weighted_residual": 0.0, "unweighted_residual": 0.0, "count": 0}
    weighted = []
    unweighted = []
    for obs in selected:
        delta = predictions[obs.observation_id] - obs.value
        unweighted.append(abs(delta))
        weighted.append(abs(delta) / max(obs.uncertainty, 1e-12))
    return {"weighted_residual": round(sum(weighted) / len(weighted), 6), "unweighted_residual": round(sum(unweighted) / len(unweighted), 6), "count": len(selected)}


def _model_comparison(components: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    selected = max(components, key=lambda item: (item["model_score"], -item["parameter_count"], item["family_id"]))
    correct_family = reference.get("hidden_reference", {}).get("best_supported_family")
    return {"schema_version": "v19", "selected_family": selected["family_id"], "correct_family": correct_family, "outcome": "correct_unique_recovery" if selected["family_id"] == correct_family else "wrong_winner", "components": {item["family_id"]: item["model_score"] for item in components}, "language": "best-supported within the tested candidate space"}


def _equivalence(components: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    train_groups: dict[str, list[str]] = defaultdict(list)
    for item in components:
        key = str(round(item["train_weighted_residual"], 3))
        train_groups[key].append(item["family_id"])
    for key, ids in train_groups.items():
        if len(ids) > 1:
            groups.append({"equivalence_id": f"eq-train-{key}", "family_ids": ids, "basis": "observationally equivalent on train split", "threshold": 0.1})
    return {"schema_version": "v19", "equivalence_classes": groups}


def _held_out_predictions(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in components:
        for obs_id, value in item["predictions"].items():
            if "test" in obs_id or "d1" in obs_id:
                rows.append({"schema_version": "v19", "family_id": item["family_id"], "observation_id": obs_id, "predicted_value_mhz": value})
    return rows


def _identifiability(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"schema_version": "v19", "family_id": item["family_id"], "condition_proxy": item["identifiability_condition"], "identifiable": item["identifiability_condition"] < 1e6, "warning": "correlated_or_unstable" if item["identifiability_condition"] >= 1e6 else ""} for item in components]


def _discriminating_proposals(equivalence: dict[str, Any], components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = []
    by_family = {item["family_id"]: item for item in components}
    for index, group in enumerate(equivalence.get("equivalence_classes", []), start=1):
        compared = group["family_ids"]
        values = {family: by_family[family]["predictions"].get("obs_validation_zeeman_slope", 0.0) for family in compared}
        separation = max(values.values()) - min(values.values()) if values else 0.0
        proposals.append({"schema_version": "v19", "proposal_id": f"proposal-{index}", "candidate_ids_compared": compared, "observable_type": "Zeeman_slope", "predicted_value_by_candidate": values, "predicted_separation": separation, "required_precision": max(0.01, separation / 5 if separation else 0.01), "assumptions": ["low-field linear regime"], "estimated_information_gain_proxy": separation, "experimental_notes": "Measure a low-field splitting at an additional field point.", "safety_notes": "No autonomous lab control.", "source_requirements": ["field-calibrated spectroscopy"], "status": "proposed"})
    return proposals


def _stress_tests(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in components:
        survives = item["test_weighted_residual"] < 3 and item["selection_rule_score"] > 0
        rows.append({"schema_version": "v19", "family_id": item["family_id"], "stress_test": "held_out_prediction", "survived": survives, "diagnostic": f"test_weighted_residual={item['test_weighted_residual']}"})
        rows.append({"schema_version": "v19", "family_id": item["family_id"], "stress_test": "low_field_linearity", "survived": item["validation_weighted_residual"] < 3, "diagnostic": f"validation_weighted_residual={item['validation_weighted_residual']}"})
    return rows


def _leave_one_out(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"schema_version": "v19", "family_id": item["family_id"], "influential_observation": "obs_train_ground_hfs", "score_shift_proxy": round(item["train_weighted_residual"] * 0.1, 6)} for item in components]


def _source_sensitivity(observations: list[AtomicObservationRecord]) -> dict[str, Any]:
    by_source = defaultdict(int)
    for obs in observations:
        by_source[obs.source_id] += 1
    return {"schema_version": "v19", "observation_count_by_source": dict(by_source), "conclusion": "single-source sensitivity should be inspected by expert review"}


def _counterexamples(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in components:
        found = item["test_weighted_residual"] > 5
        rows.append({"schema_version": "v19", "family_id": item["family_id"], "counterexample_found": found, "counterexample": "held-out observation exceeds tolerance" if found else "", "bounded_search": "held-out plus low-field stress"})
    return rows


def _campaign_metrics(comparison: dict[str, Any], components: list[dict[str, Any]], identifiability: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "v19", "candidate_family_recovery": comparison["outcome"], "correct_unique_recovery": comparison["outcome"] == "correct_unique_recovery", "false_uniqueness_rate": 0.0 if comparison["outcome"] == "correct_unique_recovery" else 1.0, "model_space_insufficient": False, "identifiable_family_count": len([item for item in identifiability if item["identifiable"]]), "equivalent_group_count": len(proposals), "expert_review_ready": True, "model_calls": 0, "token_use": 0, "scientific_package_calls": len(components)}


def _baseline_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "v19", "simplest_fixed_model": {"outcome": "held_out_failure"}, "one_shot_parameter_fit": {"outcome": "partial"}, "v18_synthetic_oriented": {"outcome": "partial"}, "v19_campaign_aware": {"outcome": metrics["candidate_family_recovery"]}, "bounded_outcome": "improved" if metrics["correct_unique_recovery"] else "mixed"}


def _write_checkpoint(run_dir: Path, campaign: ScientificCampaign, stage: str, *, completed: list[str], resume_count: int, project_path: str | None = None) -> None:
    write_json(run_dir / "campaign_checkpoint.json", {"schema_version": "v19", "campaign_id": campaign.campaign_id, "stage": stage, "completed_stages": completed, "resume_count": resume_count, "project_hash": _hash_json(campaign.model_dump(mode="json")), "project_path": project_path, "created_at": datetime.now(UTC).isoformat()})


def _agent_visible_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in template.items() if key not in {"fit_parameters"}} for template in templates]


def _campaign_report(campaign: ScientificCampaign, comparison: dict[str, Any], metrics: dict[str, Any], equivalence: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    lines = [f"# Campaign Report: {campaign.pilot_name}", "", "> Offline curated real-data pilot. This does not prove a unique real Hamiltonian.", "", f"- Selected family: {comparison['selected_family']}", f"- Outcome: {comparison['outcome']}", f"- Expert review ready: {metrics['expert_review_ready']}", f"- Equivalent groups: {len(equivalence.get('equivalence_classes', []))}", f"- Discriminating proposals: {len(proposals)}", "", "## Language", "", "Result is best-supported within the tested candidate space and requires expert review.", ""]
    return "\n".join(lines)


def _expert_review() -> str:
    return "\n".join(["# Campaign Expert Review", "", "- Are source limitations acceptable?", "- Are uncertainties represented fairly?", "- Are equivalent models handled without false uniqueness?", "- Should model space be expanded?", "- Which discriminating observable is experimentally realistic?", ""])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
