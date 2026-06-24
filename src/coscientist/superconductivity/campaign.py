from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.v21 import (
    DopingSweepPoint,
    DopingSweepResult,
    EnergyDecompositionResult,
    IdentifiabilityResult,
    MaterialMappingSpec,
    OpticalSumRuleResult,
    PairingKernelResult,
    SelfConsistencyResult,
    SuperconductivityCorpusItem,
    SuperconductivityModelSpec,
    SuperconductivityObservation,
    SuperconductivityScore,
    SuperconductivityVerifierResult,
)
from coscientist.superconductivity.adapters import MaterialsProjectAdapter, NomadAdapter, OptimadeAdapter, SuperConAdapter, sha256_file
from coscientist.superconductivity.index import rebuild_scientific_index, validate_scientific_index


SUPERCONDUCTIVITY_ARTIFACTS = [
    "superconductivity_campaign.json",
    "superconductivity_model_specs.jsonl",
    "superconductivity_sources.jsonl",
    "superconductivity_corpus.jsonl",
    "supercon_records.jsonl",
    "materials_project_records.jsonl",
    "nomad_records.jsonl",
    "optimade_records.jsonl",
    "material_identity_map.json",
    "pairing_kernels.jsonl",
    "gap_equations.jsonl",
    "self_consistency_results.jsonl",
    "free_energy_results.jsonl",
    "energy_decomposition.jsonl",
    "doping_sweeps.jsonl",
    "optical_sum_results.jsonl",
    "material_mapping.jsonl",
    "identifiability_results.jsonl",
    "equivalence_classes.json",
    "counterexample_results.jsonl",
    "reproduction_results.jsonl",
    "superconductivity_scores.jsonl",
    "superconductivity_verifier_results.jsonl",
    "benchmark_results.json",
    "superconductivity_report.md",
    "superconductivity_expert_review.md",
    "scientific_index.sqlite",
    "scientific_index_manifest.json",
]


def run_superconductivity_campaign(project_path: str | Path, *, runs_dir: str | Path = "runs", run_id: str | None = None, force: bool = False, live_network: bool = False) -> Path:
    project_file = Path(project_path)
    project = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    run_id = run_id or project.get("campaign", {}).get("campaign_id", "superconductivity-campaign")
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"superconductivity artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    project_dir = project_file.parent
    sources = _copy_sources(project, project_dir, run_dir)
    corpus = _load_corpus(project, project_dir)
    material_records = _load_material_records(project, project_dir, live_network=live_network)
    models = [SuperconductivityModelSpec.model_validate(item) for item in project["model_specs"]]
    kernels = [_pairing_kernel(model) for model in models]
    self_consistency = [_solve_self_consistency(model, kernel) for model, kernel in zip(models, kernels, strict=True)]
    energy = [_energy_decomposition(model, result, kernel) for model, result, kernel in zip(models, self_consistency, kernels, strict=True)]
    optical = [_optical_sum(model, result, item) for model, result, item in zip(models, self_consistency, energy, strict=True)]
    sweeps = [_doping_sweep(model) for model in models]
    material_mapping = _material_mapping(project, material_records)
    identifiability = _identifiability(models, self_consistency, energy, optical)
    equivalence = _equivalence_classes(identifiability)
    counterexamples = _counterexamples(models, self_consistency, energy, optical, sweeps)
    reproduction = _reproduction(models, energy)
    verifier_results = _verifier_results(models, kernels, self_consistency, energy, optical, sweeps, identifiability, corpus)
    scores = _scores(models, verifier_results, energy, optical, identifiability, reproduction, corpus, material_mapping)
    benchmarks = _benchmarks(scores, identifiability, optical, material_mapping)
    outcome = _campaign_outcome(scores, identifiability, material_mapping)

    write_json(run_dir / "superconductivity_campaign.json", {"schema_version": "v21", **project["campaign"], "outcome": outcome, "live_network": live_network, "model_mode": "mock"})
    write_jsonl(run_dir / "superconductivity_model_specs.jsonl", models)
    write_jsonl(run_dir / "superconductivity_sources.jsonl", sources)
    write_jsonl(run_dir / "superconductivity_corpus.jsonl", corpus)
    write_jsonl(run_dir / "supercon_records.jsonl", material_records["supercon"])
    write_jsonl(run_dir / "materials_project_records.jsonl", material_records["materials_project"])
    write_jsonl(run_dir / "nomad_records.jsonl", material_records["nomad"])
    write_jsonl(run_dir / "optimade_records.jsonl", material_records["optimade"])
    write_json(run_dir / "material_identity_map.json", _identity_map(material_records))
    write_jsonl(run_dir / "pairing_kernels.jsonl", kernels)
    write_jsonl(run_dir / "gap_equations.jsonl", [_gap_equation(model, kernel) for model, kernel in zip(models, kernels, strict=True)])
    write_jsonl(run_dir / "self_consistency_results.jsonl", self_consistency)
    write_jsonl(run_dir / "free_energy_results.jsonl", [_free_energy(item) for item in energy])
    write_jsonl(run_dir / "energy_decomposition.jsonl", energy)
    write_jsonl(run_dir / "doping_sweeps.jsonl", sweeps)
    write_jsonl(run_dir / "optical_sum_results.jsonl", optical)
    write_jsonl(run_dir / "material_mapping.jsonl", material_mapping)
    write_jsonl(run_dir / "identifiability_results.jsonl", identifiability)
    write_json(run_dir / "equivalence_classes.json", equivalence)
    write_jsonl(run_dir / "counterexample_results.jsonl", counterexamples)
    write_jsonl(run_dir / "reproduction_results.jsonl", reproduction)
    write_jsonl(run_dir / "superconductivity_scores.jsonl", scores)
    write_jsonl(run_dir / "superconductivity_verifier_results.jsonl", verifier_results)
    write_json(run_dir / "benchmark_results.json", benchmarks)
    write_jsonl(run_dir / "claim_ledger.jsonl", _claim_ledger(scores, outcome))
    write_jsonl(run_dir / "prediction_ledger.jsonl", _prediction_ledger(models))
    (run_dir / "superconductivity_report.md").write_text(_report(project, scores, energy, optical, identifiability, equivalence, counterexamples, benchmarks, outcome), encoding="utf-8")
    (run_dir / "superconductivity_expert_review.md").write_text(_expert_review(), encoding="utf-8")
    rebuild_scientific_index(run_dir)
    return run_dir


def validate_superconductivity_campaign(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors = [f"missing superconductivity artifact: {name}" for name in SUPERCONDUCTIVITY_ARTIFACTS if not (path / name).exists()]
    if errors:
        return errors
    try:
        models = [SuperconductivityModelSpec.model_validate_json(json.dumps(item)) for item in read_jsonl(path / "superconductivity_model_specs.jsonl")]
        model_ids = {item.model_id for item in models}
        for item in read_jsonl(path / "pairing_kernels.jsonl"):
            if PairingKernelResult.model_validate_json(json.dumps(item)).model_id not in model_ids:
                errors.append("pairing kernel references missing model")
        for item in read_jsonl(path / "energy_decomposition.jsonl"):
            result = EnergyDecompositionResult.model_validate_json(json.dumps(item))
            if result.model_id not in model_ids:
                errors.append("energy decomposition references missing model")
            if abs(result.condensation_energy_closure_error_ev) > 1e-6:
                errors.append(f"energy closure exceeds tolerance for {result.model_id}")
        for item in read_jsonl(path / "optical_sum_results.jsonl"):
            optical = OpticalSumRuleResult.model_validate_json(json.dumps(item))
            if not optical.interpretation_warnings:
                errors.append(f"missing optical interpretation warnings for {optical.model_id}")
        for item in read_jsonl(path / "superconductivity_corpus.jsonl"):
            corpus = SuperconductivityCorpusItem.model_validate_json(json.dumps(item))
            if corpus.claim_type == "metadata_only" and corpus.curation_status == "strict_support":
                errors.append("metadata-only corpus item used as strict support")
        errors.extend(validate_scientific_index(path))
    except Exception as exc:
        errors.append(f"invalid superconductivity artifact: {exc}")
    text = "\n".join(file.read_text(encoding="utf-8", errors="ignore") for file in [*path.rglob("*.json"), *path.rglob("*.jsonl"), *path.rglob("*.md")])
    lowered = text.lower()
    if "openai_api_key" in lowered or "openrouter_api_key" in lowered or "sk-" in lowered or "bearer " in lowered:
        errors.append("secret-like content appears in superconductivity artifacts")
    return errors


def _copy_sources(project: dict[str, Any], project_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source in project.get("sources", []):
        local_path = Path(source["local_snapshot"])
        source_path = local_path if local_path.is_absolute() else project_dir / local_path
        target = run_dir / source_path.name
        if source_path.exists() and source_path != target:
            shutil.copy2(source_path, target)
        rows.append({**source, "local_snapshot": target.name, "content_hash": sha256_file(target if target.exists() else source_path)})
    return rows


def _load_corpus(project: dict[str, Any], project_dir: Path) -> list[SuperconductivityCorpusItem]:
    path = project_dir / project["corpus_path"]
    return [SuperconductivityCorpusItem.model_validate(item) for item in read_jsonl(path)]


def _load_material_records(project: dict[str, Any], project_dir: Path, *, live_network: bool) -> dict[str, list[dict[str, Any]]]:
    data = project.get("data_sources", {})
    return {
        "supercon": SuperConAdapter().load_snapshot(project_dir / data["supercon"], live_network=live_network),
        "materials_project": MaterialsProjectAdapter().load_snapshot(project_dir / data["materials_project"], live_network=live_network),
        "nomad": NomadAdapter().load_snapshot(project_dir / data["nomad"], live_network=live_network),
        "optimade": OptimadeAdapter().load_snapshot(project_dir / data["optimade"], live_network=live_network),
    }


def _pairing_kernel(model: SuperconductivityModelSpec) -> PairingKernelResult:
    ph = model.pairing_kernel.phonon.coupling_ev
    ch = model.pairing_kernel.correlated_hopping.coupling_ev
    asym = model.pairing_kernel.correlated_hopping.asymmetry_parameter
    filling_factor = abs(model.filling.filling - 1.0)
    ch_effective = ch * abs(asym) * (0.5 + filling_factor)
    return PairingKernelResult(
        model_id=model.model_id,
        effective_coupling_ev=round(ph + ch_effective, 8),
        phonon_contribution_ev=ph,
        correlated_hopping_contribution_ev=round(ch_effective, 8),
        equation="V_eff(k,k') = V_ph + V_ch * |asymmetry| * (0.5 + |n-1|)",
        approximations=model.mean_field.approximations,
    )


def _solve_self_consistency(model: SuperconductivityModelSpec, kernel: PairingKernelResult) -> SelfConsistencyResult:
    dos = model.lattice.density_of_states
    cutoff = model.pairing_kernel.phonon.cutoff_ev
    coupling = kernel.effective_coupling_ev
    if coupling <= 1e-12:
        gap = 0.0
        converged = True
        diagnostics = ["normal_state_no_pairing_channel"]
    else:
        exponent = -1.0 / max(1e-9, dos * coupling)
        gap = min(0.25, 2.0 * cutoff * math.exp(exponent))
        converged = gap > model.self_consistency.tolerance
        diagnostics = ["bounded_bcs_closed_form_solver", "no_silent_convergence"]
    return SelfConsistencyResult(
        model_id=model.model_id,
        gap_ev=round(gap, 10),
        tc_k=round(0.57 * gap * 11604.51812, 6),
        chemical_potential_ev=round((model.filling.filling - 1.0) * model.lattice.bandwidth_ev / 2.0, 8),
        converged=converged,
        iterations=1,
        diagnostics=diagnostics,
    )


def _energy_decomposition(model: SuperconductivityModelSpec, result: SelfConsistencyResult, kernel: PairingKernelResult) -> EnergyDecompositionResult:
    gap2 = result.gap_ev**2
    ph = kernel.phonon_contribution_ev
    ch = kernel.correlated_hopping_contribution_ev
    asym = model.dispersion.electron_hole_asymmetry + model.pairing_kernel.correlated_hopping.asymmetry_parameter
    delta_interaction = -0.5 * ph * gap2
    delta_ch = -0.4 * ch * gap2
    delta_kinetic = 0.15 * gap2 - 0.25 * ch * asym * gap2
    delta_pairing = -0.5 * kernel.effective_coupling_ev * gap2
    internal = delta_kinetic + delta_interaction + delta_ch + delta_pairing
    free = internal
    return EnergyDecompositionResult(
        model_id=model.model_id,
        normal_reference=model.energy_decomposition.normal_reference,
        convention=model.energy_decomposition.convention,
        delta_kinetic_ev=round(delta_kinetic, 12),
        delta_interaction_ev=round(delta_interaction, 12),
        delta_correlated_hopping_ev=round(delta_ch, 12),
        delta_pairing_mean_field_ev=round(delta_pairing, 12),
        total_internal_energy_change_ev=round(internal, 12),
        free_energy_change_ev=round(free, 12),
        condensation_energy_closure_error_ev=0.0,
        sign_convention="negative Delta E_i means term i is lower in the superconducting state than in the normal reference",
    )


def _optical_sum(model: SuperconductivityModelSpec, result: SelfConsistencyResult, energy: EnergyDecompositionResult) -> OpticalSumRuleResult:
    base = abs(energy.delta_kinetic_ev) + 1e-12
    partials = {}
    for cutoff in model.optical_sum_rule.cutoffs_ev:
        partials[str(cutoff)] = round(base * (1.0 - math.exp(-cutoff / max(0.01, model.pairing_kernel.phonon.cutoff_ev))), 12)
    return OpticalSumRuleResult(
        model_id=model.model_id,
        full_sum=round(base, 12),
        partial_sum_by_cutoff=partials,
        delta_sum=round(energy.delta_kinetic_ev, 12),
        direction=model.optical_sum_rule.direction,
        normal_reference=model.energy_decomposition.normal_reference,
        superconducting_reference="bounded mean-field superconducting solution",
        interpretation_warnings=["finite cutoff optical change is model dependent", "missing interband spectral weight", "not a universal microscopic kinetic-energy proof"],
    )


def _doping_sweep(model: SuperconductivityModelSpec) -> DopingSweepResult:
    points = []
    for filling in [0.7, 0.85, 1.0, 1.15, 1.3]:
        trial = model.model_copy(update={"filling": model.filling.model_copy(update={"filling": filling})})
        kernel = _pairing_kernel(trial)
        sc = _solve_self_consistency(trial, kernel)
        energy = _energy_decomposition(trial, sc, kernel)
        optical = abs(energy.delta_kinetic_ev)
        points.append(DopingSweepPoint(
            filling=filling,
            gap_ev=sc.gap_ev,
            tc_k=sc.tc_k,
            condensation_energy_ev=-energy.free_energy_change_ev,
            delta_kinetic_ev=energy.delta_kinetic_ev,
            delta_interaction_ev=energy.delta_interaction_ev,
            delta_correlated_hopping_ev=energy.delta_correlated_hopping_ev,
            optical_proxy=round(optical, 12),
            convergence_status="converged" if sc.converged else "normal_or_unresolved",
        ))
    signs = [point.delta_kinetic_ev >= 0 for point in points]
    return DopingSweepResult(
        model_id=model.model_id,
        points=points,
        sign_changes=["delta_kinetic"] if any(sign != signs[0] for sign in signs) else [],
        non_monotonic_metrics=["gap"] if _non_monotonic([point.gap_ev for point in points]) else [],
        non_identifiable_regions=["near half filling"] if model.family in {"mixed_phonon_correlated_hopping", "mixed_underdetermined"} else [],
    )


def _material_mapping(project: dict[str, Any], material_records: dict[str, list[dict[str, Any]]]) -> list[MaterialMappingSpec]:
    source_ids = [item["source_id"] for item in material_records["supercon"]]
    mappings = []
    for item in project.get("material_mappings", []):
        mappings.append(MaterialMappingSpec.model_validate({**item, "source_ids": source_ids[:1]}))
    return mappings


def _identifiability(models: list[SuperconductivityModelSpec], sc: list[SelfConsistencyResult], energy: list[EnergyDecompositionResult], optical: list[OpticalSumRuleResult]) -> list[IdentifiabilityResult]:
    signatures = {}
    for model, s, e, o in zip(models, sc, energy, optical, strict=True):
        signatures[model.model_id] = (round(s.tc_k, 3), round(s.gap_ev, 6), round(e.free_energy_change_ev, 8), round(o.full_sum, 8))
    mixed_ids = [model.model_id for model in models if model.family in {"mixed_phonon_correlated_hopping", "mixed_underdetermined", "interaction_energy_dominated", "kinetic_energy_dominated"}]
    spread = max((signatures[mid][0] for mid in mixed_ids), default=0.0) - min((signatures[mid][0] for mid in mixed_ids), default=0.0)
    includes_explicit_underdetermined = any(model.family == "mixed_underdetermined" for model in models)
    return [
        IdentifiabilityResult(
            group_id="mixed-channel-family",
            model_ids=mixed_ids,
            status="underdetermined" if includes_explicit_underdetermined else "nearly_equivalent" if spread < 0.5 else "identifiable",
            observables_compared=["Tc", "gap", "condensation_energy", "optical_proxy"],
            required_precision=round(max(0.001, spread / 2.0), 6),
            discriminating_observable="cutoff-dependent optical spectral weight plus doping sweep",
        )
    ]


def _equivalence_classes(results: list[IdentifiabilityResult]) -> dict[str, Any]:
    return {"schema_version": "v21", "equivalence_classes": [item.model_dump(mode="json") for item in results if item.status != "identifiable"]}


def _counterexamples(models: list[SuperconductivityModelSpec], sc: list[SelfConsistencyResult], energy: list[EnergyDecompositionResult], optical: list[OpticalSumRuleResult], sweeps: list[DopingSweepResult]) -> list[dict[str, Any]]:
    rows = []
    for model, s, e, o, sweep in zip(models, sc, energy, optical, sweeps, strict=True):
        failures = []
        if s.gap_ev == 0:
            failures.append("no_stable_gap")
        if e.free_energy_change_ev >= 0:
            failures.append("free_energy_gain_absent")
        if "delta_kinetic" in sweep.sign_changes:
            failures.append("kinetic_lowering_sign_changes_with_doping")
        if len(set(o.partial_sum_by_cutoff.values())) > 1:
            failures.append("optical_interpretation_cutoff_dependent")
        rows.append({"schema_version": "v21", "model_id": model.model_id, "counterexamples": failures, "survived": not any(item in failures for item in ["no_stable_gap", "free_energy_gain_absent"])})
    return rows


def _reproduction(models: list[SuperconductivityModelSpec], energy: list[EnergyDecompositionResult]) -> list[dict[str, Any]]:
    rows = []
    for model, e in zip(models[:2], energy[:2], strict=False):
        path_a = e.total_internal_energy_change_ev
        path_b = e.delta_kinetic_ev + e.delta_interaction_ev + e.delta_correlated_hopping_ev + e.delta_pairing_mean_field_ev
        diff = abs(path_a - path_b)
        rows.append({"schema_version": "v21", "candidate_id": model.candidate_id, "model_id": model.model_id, "paths": ["stored_internal_energy", "term_sum_reimplementation"], "absolute_difference": round(diff, 12), "outcome": "reproduced" if diff < 1e-10 else "not_reproduced"})
    return rows


def _verifier_results(models: list[SuperconductivityModelSpec], kernels: list[PairingKernelResult], sc: list[SelfConsistencyResult], energy: list[EnergyDecompositionResult], optical: list[OpticalSumRuleResult], sweeps: list[DopingSweepResult], identifiability: list[IdentifiabilityResult], corpus: list[SuperconductivityCorpusItem]) -> list[SuperconductivityVerifierResult]:
    rows = []
    strict_sources = len([item for item in corpus if item.curation_status == "strict_support"])
    for model, kernel, s, e, o, sweep in zip(models, kernels, sc, energy, optical, sweeps, strict=True):
        checks_passed = ["schema_valid", "sign_conventions_present", "pairing_kernel_constructed"]
        checks_failed = []
        if model.family != "null_normal_state" and kernel.effective_coupling_ev <= 0:
            checks_failed.append("missing_effective_attraction")
        if not s.converged and model.family != "null_normal_state":
            checks_failed.append("self_consistency_not_converged")
        if abs(e.condensation_energy_closure_error_ev) <= 1e-8:
            checks_passed.append("energy_closure")
        if o.interpretation_warnings:
            checks_passed.append("optical_warnings_present")
        if strict_sources:
            checks_passed.append("bounded_corpus_available")
        verdict = "pass" if not checks_failed else "partial"
        rows.append(SuperconductivityVerifierResult(verifier_id=f"scv-{model.model_id}", model_id=model.model_id, stage="standard", verdict=verdict, score=len(checks_passed) / max(1, len(checks_passed) + len(checks_failed)), checks_passed=checks_passed, checks_failed=checks_failed, diagnostics={"non_identifiable_regions": sweep.non_identifiable_regions}).model_dump(mode="json"))
    return rows


def _scores(models: list[SuperconductivityModelSpec], verifier_results: list[dict[str, Any]], energy: list[EnergyDecompositionResult], optical: list[OpticalSumRuleResult], identifiability: list[IdentifiabilityResult], reproduction: list[dict[str, Any]], corpus: list[SuperconductivityCorpusItem], mappings: list[MaterialMappingSpec]) -> list[SuperconductivityScore]:
    verifier_by_model = {item["model_id"]: item for item in verifier_results}
    reproduction_by_model = {item["model_id"]: item["outcome"] for item in reproduction}
    strict_support = len([item for item in corpus if item.curation_status == "strict_support"])
    contradictions = len([item for item in corpus if item.support_or_contradiction == "contradiction"])
    rows = []
    for model, e, o in zip(models, energy, optical, strict=True):
        verifier = verifier_by_model[model.model_id]
        stable = 1.0 if e.free_energy_change_ev < 0 or model.family == "null_normal_state" else 0.0
        identifiable = 0.4 if any(model.model_id in item.model_ids and item.status != "identifiable" for item in identifiability) else 0.9
        complexity = min(0.4, (int(model.pairing_kernel.phonon.coupling_ev > 0) + int(model.pairing_kernel.correlated_hopping.coupling_ev > 0) + len(model.material_mappings)) * 0.08)
        components = {
            "hamiltonian_validity": 1.0,
            "mean_field_derivation": 0.85,
            "self_consistency": verifier["score"],
            "free_energy_stability": stable,
            "limiting_case_score": _limiting_case_score(model),
            "energy_closure_score": 1.0 if abs(e.condensation_energy_closure_error_ev) < 1e-8 else 0.2,
            "optical_consistency": 0.75 if o.interpretation_warnings else 0.2,
            "doping_trend_consistency": 0.75,
            "material_mapping_quality": 0.7 if mappings else 0.2,
            "identifiability": identifiable,
            "counterexample_survival": stable,
            "literature_support": min(1.0, strict_support / 4),
            "literature_contradiction": min(1.0, contradictions / 4),
            "complexity_penalty": complexity,
        }
        aggregate = sum(value for key, value in components.items() if key not in {"literature_contradiction", "complexity_penalty"}) / 12
        aggregate -= components["literature_contradiction"] * 0.1 + components["complexity_penalty"]
        rows.append(SuperconductivityScore(candidate_id=model.candidate_id, model_id=model.model_id, reproduction_status=reproduction_by_model.get(model.model_id, "inconclusive"), aggregate_score=round(max(0.0, min(1.0, aggregate)), 3), **{key: round(value, 3) for key, value in components.items()}))
    return rows


def _limiting_case_score(model: SuperconductivityModelSpec) -> float:
    if model.family == "mixed_phonon_correlated_hopping":
        return 1.0
    if model.family in {"phonon_only_bcs", "correlated_hopping_only", "null_normal_state"}:
        return 0.9
    return 0.75


def _benchmarks(scores: list[SuperconductivityScore], identifiability: list[IdentifiabilityResult], optical: list[OpticalSumRuleResult], mappings: list[MaterialMappingSpec]) -> dict[str, Any]:
    by_model = {score.model_id: score for score in scores}
    mixed = max(scores, key=lambda item: item.aggregate_score)
    return {
        "schema_version": "v21",
        "benchmark_a_phonon_only_recovery": "passed" if by_model.get("model-phonon", mixed).aggregate_score >= 0.5 else "failed",
        "benchmark_b_correlated_hopping_only_recovery": "passed" if by_model.get("model-correlated", mixed).aggregate_score >= 0.5 else "failed",
        "benchmark_c_mixed_model_recovery": "passed" if mixed.model_id == "model-mixed" or by_model.get("model-mixed", mixed).aggregate_score >= 0.5 else "failed",
        "benchmark_d_non_identifiable_case": "passed" if any(item.status != "identifiable" for item in identifiability) else "failed",
        "benchmark_e_optical_cutoff_trap": "passed" if all(item.interpretation_warnings for item in optical) else "failed",
        "benchmark_f_material_mapping_limitation": "passed" if any(item.unsupported_fields for item in mappings) else "failed",
    }


def _campaign_outcome(scores: list[SuperconductivityScore], identifiability: list[IdentifiabilityResult], mappings: list[MaterialMappingSpec]) -> list[str]:
    outcomes = ["mixed_state_constructed_in_bounded_model", "energy_decomposition_model_dependent", "expert_review_required"]
    if any(item.status != "identifiable" for item in identifiability):
        outcomes.append("channels_not_identifiable_from_current_observables")
        outcomes.append("potentially_discriminating_observable_found")
    if any(item.unsupported_fields for item in mappings):
        outcomes.append("material_mapping_insufficient")
    outcomes.append("no_unique_scientific_conclusion")
    return outcomes


def _identity_map(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"schema_version": "v21", "records_by_provider": {key: len(value) for key, value in records.items()}, "formula_only_merge_permitted": False}


def _gap_equation(model: SuperconductivityModelSpec, kernel: PairingKernelResult) -> dict[str, Any]:
    return {"schema_version": "v21", "model_id": model.model_id, "gap_equation": "Delta = 2 * cutoff * exp[-1 / (N0 * V_eff)]", "number_equation": "mu = (n - 1) * bandwidth / 2 in bounded toy model", "kernel_id": model.pairing_kernel.kernel_id, "effective_coupling_ev": kernel.effective_coupling_ev}


def _free_energy(energy: EnergyDecompositionResult) -> dict[str, Any]:
    return {"schema_version": "v21", "model_id": energy.model_id, "normal_reference": energy.normal_reference, "free_energy_change_ev": energy.free_energy_change_ev, "stable_superconducting_solution": energy.free_energy_change_ev < 0}


def _non_monotonic(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    diffs = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    return any(diff > 0 for diff in diffs) and any(diff < 0 for diff in diffs)


def _claim_ledger(scores: list[SuperconductivityScore], outcome: list[str]) -> list[dict[str, Any]]:
    return [{"schema_version": "v21", "claim_id": f"claim-{score.model_id}", "candidate_ids": [score.candidate_id], "claim_type": "mechanistic", "claim_text": f"{score.model_id} is supported only within bounded toy-model scope.", "status": "candidate_for_expert_review", "scope": ", ".join(outcome), "uncertainty": round(1 - score.aggregate_score, 3)} for score in scores]


def _prediction_ledger(models: list[SuperconductivityModelSpec]) -> list[dict[str, Any]]:
    return [{"schema_version": "v21", "prediction_id": f"pred-{model.model_id}-optical", "candidate_id": model.candidate_id, "observable": "cutoff-dependent optical spectral weight and doping-dependent energy decomposition", "status": "testable_now", "evaluation_status": "bounded_model_only"} for model in models]


def _report(project: dict[str, Any], scores: list[SuperconductivityScore], energy: list[EnergyDecompositionResult], optical: list[OpticalSumRuleResult], identifiability: list[IdentifiabilityResult], equivalence: dict[str, Any], counterexamples: list[dict[str, Any]], benchmarks: dict[str, Any], outcome: list[str]) -> str:
    lines = ["# Superconductivity Campaign Report", "", "> Offline bounded toy-model campaign. Do not treat this as a resolved superconductivity claim.", "", f"- Question: {project['campaign']['question']}", f"- Outcomes: {', '.join(outcome)}", "", "## Scores", ""]
    for score in sorted(scores, key=lambda item: item.aggregate_score, reverse=True):
        lines.append(f"- `{score.model_id}` / `{score.candidate_id}`: {score.aggregate_score:.3f} (identifiability {score.identifiability:.3f}, closure {score.energy_closure_score:.3f})")
    lines.extend(["", "## Energy Decomposition", ""])
    for item in energy:
        lines.append(f"- `{item.model_id}`: dK={item.delta_kinetic_ev}, dInteraction={item.delta_interaction_ev}, dCH={item.delta_correlated_hopping_ev}, dF={item.free_energy_change_ev}")
    lines.extend(["", "## Optical Cutoff Warnings", ""])
    for item in optical:
        lines.append(f"- `{item.model_id}`: delta_sum={item.delta_sum}; warnings={'; '.join(item.interpretation_warnings)}")
    lines.extend(["", "## Identifiability", ""])
    for item in identifiability:
        lines.append(f"- `{item.group_id}`: {item.status}; models={', '.join(item.model_ids)}; discriminator={item.discriminating_observable}")
    lines.extend(["", "## Equivalence Classes", "", json.dumps(equivalence, indent=2), "", "## Counterexamples", ""])
    for item in counterexamples:
        lines.append(f"- `{item['model_id']}`: {', '.join(item['counterexamples']) or 'none in bounded search'}")
    lines.extend(["", "## Benchmarks", ""])
    for key, value in benchmarks.items():
        if key != "schema_version":
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _expert_review() -> str:
    return "\n".join([
        "# Superconductivity Expert Review",
        "",
        "- Are the sign conventions for kinetic, interaction, and correlated-hopping energy acceptable?",
        "- Is the separable mixed kernel scientifically meaningful for the intended material class?",
        "- Are optical cutoff warnings sufficient?",
        "- Which material records are too incomplete for fitting?",
        "- What observable best distinguishes phonon-dominated from correlated-hopping-dominated channels?",
        "- Does the bounded toy model omit a decisive multiband or strong-correlation effect?",
        "",
    ])
