from __future__ import annotations

from pathlib import Path

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.superconductivity.minimal_model import run_minimal_mixed_bcs_project, validate_minimal_mixed_bcs_run
from coscientist.superconductivity.phase2_data import phase2_coverage, run_phase2_data_coverage_tool, validate_phase2_data_coverage_tool


def test_minimal_mixed_bcs_phase1_run_outputs_hamiltonian_energy_and_phase_diagram(tmp_path: Path) -> None:
    run_dir = run_minimal_mixed_bcs_project(runs_dir=tmp_path, run_id="minimal")
    assert validate_minimal_mixed_bcs_run(run_dir) == []

    hamiltonian = read_json(run_dir / "hamiltonian_spec.json")
    assert set(hamiltonian["terms"]) == {"H_t", "H_U", "H_ph", "H_e_ph", "H_corr"}

    base = read_json(run_dir / "base_solution.json")
    assert base["gap_ev"] > 0
    assert base["stable_superconducting_solution"] is True
    assert base["free_energy_change_ev"] < 0
    assert "delta_kinetic_ev" in base
    assert "delta_phonon_interaction_ev" in base
    assert "delta_correlated_hopping_ev" in base

    comparison = read_jsonl(run_dir / "four_model_comparison.jsonl")
    assert {item["model_id"] for item in comparison} == {"null", "phonon_only", "correlated_only", "mixed"}
    mixed = next(item for item in comparison if item["model_id"] == "mixed")
    assert mixed["stable_superconducting_solution"] is True

    phase = read_json(run_dir / "phase_diagram_summary.json")
    assert phase["point_count"] > 0
    assert phase["stable_point_count"] > 0
    assert phase["best_point"] is not None

    verifier_results = read_jsonl(run_dir / "minimal_bcs_verifier_results.jsonl")
    verifier_ids = {item["verifier_id"] for item in verifier_results}
    assert "hamiltonian_term_verifier" in verifier_ids
    assert "energy_ledger_closure_verifier" in verifier_ids
    assert "four_model_ablation_verifier" in verifier_ids
    assert "phase2_existing_data_verifier" in verifier_ids
    assert next(item for item in verifier_results if item["verifier_id"] == "hamiltonian_term_verifier")["verdict"] == "pass"

    phase2 = read_json(run_dir / "phase2_held_out_evaluation.json")
    assert phase2["status"] == "blocked_insufficient_existing_data"
    assert phase2["can_claim_material_separation"] is False


def test_minimal_bcs_cli_style_validation_rejects_missing_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    errors = validate_minimal_mixed_bcs_run(run_dir)
    assert any("missing minimal BCS artifact" in item for item in errors)


def test_frontend_can_run_minimal_bcs_solver_and_show_equations(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = app.run_minimal_bcs(run_id="frontend-minimal")

    assert app.validate_minimal_bcs(run_dir) == []
    hamiltonian = app.minimal_bcs_hamiltonian(run_dir)
    assert "H_corr" in hamiltonian["terms"]
    method = app.minimal_bcs_solution_method(run_dir)
    assert "gap_equation" in method
    assert app.minimal_bcs_verifier_task_rows(run_dir)
    assert app.minimal_bcs_verifier_result_rows(run_dir)
    assert app.phase2_evaluation(run_dir)["status"] == "blocked_insufficient_existing_data"
    assert "Required local table columns" in app.phase2_connection_plan(run_dir)
    report = app.report_text(run_dir)
    assert "H_corr =" in report
    assert "gap equation" in report


def test_phase2_coverage_uses_tiered_readiness_not_only_complete_cases() -> None:
    observations = []
    for doping in ["x=0.060", "x=0.090", "x=0.120"]:
        observations.extend(
            [
                {"material_family": "cuprate", "material_id": f"LSCO-film-{doping}", "doping": doping, "observable": "tc_k", "value": 20, "split": "train", "usable_for_fit": True},
                {"material_family": "cuprate", "material_id": f"LSCO-film-{doping}", "doping": doping, "observable": "penetration_depth_nm", "value": 300, "split": "train", "usable_for_fit": True},
                {"material_family": "cuprate", "material_id": f"LSCO-film-{doping}", "doping": doping, "observable": "optical_spectral_weight_proxy", "value": 0.1, "split": "test", "usable_for_fit": True},
            ]
        )
    coverage = phase2_coverage(observations)
    assert coverage["complete_doping_point_count"] == 0
    assert coverage["tier_a"]["status"] == "pass"
    assert coverage["tier_b"]["status"] == "pass"
    assert coverage["tier_c"]["status"] == "blocked"
    assert coverage["overall_readiness_status"] == "partial_tier_b_local_multi_observable"


def test_phase2_data_tool_reports_partial_readiness_for_tier_a_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "phase2_partial.csv"
    csv_path.write_text(
        "observation_id,material_family,material_id,doping,observable,value,unit,uncertainty,split,source_id,provenance,usable_for_fit\n"
        "obs-1,cuprate,LSCO,x=0.080,tc_k,20,K,,train,s1,table,true\n"
        "obs-2,cuprate,LSCO,x=0.080,isotope_alpha,0.2,dimensionless,,train,s2,table,true\n",
        encoding="utf-8",
    )
    run_dir = run_phase2_data_coverage_tool(tmp_path / "coverage", source_path=csv_path)
    assert validate_phase2_data_coverage_tool(run_dir) == []
    evaluation = read_json(run_dir / "phase2_data_tool_evaluation.json")
    assert evaluation["status"] == "partial_tier_a_minimal_material_trend"
    assert evaluation["can_run_partial_material_trend"] is True
    assert evaluation["can_claim_full_quantitative_separation"] is False
