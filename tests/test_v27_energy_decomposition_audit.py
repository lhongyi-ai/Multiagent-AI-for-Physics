from __future__ import annotations

from pathlib import Path

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.superconductivity.energy_decomposition import run_energy_decomposition_audit, validate_energy_decomposition_audit


QUESTION = (
    "Determine whether the decomposition of superconducting condensation energy into bare kinetic-energy, "
    "phonon-interaction-energy, and correlated-hopping-energy contributions is uniquely defined, gauge invariant, "
    "and physically observable."
)


def test_energy_decomposition_audit_outputs_counterexample_and_guardrails(tmp_path: Path) -> None:
    run_dir = run_energy_decomposition_audit(tmp_path / "audit")
    assert validate_energy_decomposition_audit(run_dir) == []

    hamiltonian = read_json(run_dir / "gauge_coupled_hamiltonian.json")
    assert "H_corr[A]" in hamiltonian["terms"]
    assert "electromagnetic" in hamiltonian["critical_correction"]

    counterexample = read_json(run_dir / "representation_counterexample.json")
    assert counterexample["spectra_equal"] is True
    assert counterexample["total_hamiltonians_equal"] is True
    assert abs(counterexample["component_difference"]["total"]) < 1e-12
    assert abs(counterexample["component_difference"]["kinetic"]) > 1e-6

    outcome = read_json(run_dir / "final_theory_outcome.json")
    assert outcome["status"] == "counterexample_found"
    assert outcome["finite_size_numerics_are_general_proof"] is False
    assert "bare kinetic contribution" in outcome["model_dependent_quantities"]

    verifiers = read_jsonl(run_dir / "energy_decomposition_verifier_results.jsonl")
    assert next(item for item in verifiers if item["verifier_id"] == "unique_component_percentage_claim")["verdict"] == "fail"


def test_live_agent_meeting_uses_energy_audit_and_stops_no_progress(tmp_path: Path) -> None:
    service = create_app(runs_dir=tmp_path)
    run_dir = service.run_live_agent_meeting(
        tmp_path / "meeting",
        QUESTION,
        live_model=False,
        max_rounds=4,
        force=True,
        phase2_data_path="data/phase2_lsco.csv",
    )
    context = service.live_agent_tool_context(run_dir)
    audit = context["energy_decomposition_audit"]
    assert audit["enabled"] is True
    assert audit["outcome"]["status"] == "counterexample_found"

    status = read_json(Path(run_dir) / "meeting_session.json")
    assert status["status"] == "stopped_no_progress"
    assert "STOPPED_NO_PROGRESS" in status["stopping_reason"]

    task_rows = service.verifier_task_rows(run_dir)
    assert any(row["verifier_type"] == "representation_counterexample" and row["status"] == "complete" for row in task_rows)
    transcript = service.live_agent_transcript(run_dir)
    assert "Energy decomposition audit" in transcript
    assert "LSCO data are a parallel validation line" in transcript
