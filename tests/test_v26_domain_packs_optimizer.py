from __future__ import annotations

from pathlib import Path

import pytest

from coscientist.core import GenericDataAcquisitionAgent, HypothesisOptimizerV2, get_default_domain_registry
from coscientist.core.hypotheses_v2 import migrate_hypothesis_to_v2, validate_finalist_requirements
from coscientist.core.optimizer_v2 import mutation_operators
from coscientist.core.tasks import ScientificTaskType, TaskPolicyRegistry
from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl


def _hypotheses() -> list[dict[str, object]]:
    return [
        {
            "id": "good",
            "title": "Good branch",
            "core_claim": "A bounded model can be falsified by a failed verifier.",
            "assumptions": ["bounded"],
            "testable_predictions": ["verifier should pass"],
            "falsification_criteria": ["verifier fails"],
            "supporting_evidence": ["artifact.json"],
            "required_tools": ["verifier"],
        },
        {
            "id": "bad",
            "title": "Bad branch",
            "core_claim": "This is always true and can prove everything.",
            "assumptions": [],
            "testable_predictions": [],
            "falsification_criteria": [],
        },
        {
            "id": "contrarian",
            "title": "Contrarian branch",
            "core_claim": "A simpler explanation may beat the complex model.",
            "assumptions": ["held-out comparison"],
            "testable_predictions": ["simple model wins"],
            "falsification_criteria": ["complex model wins held-out comparison"],
        },
    ]


def test_domain_pack_registry_discovers_required_packs() -> None:
    registry = get_default_domain_registry()
    domain_ids = {item.domain_id for item in registry.list()}
    assert {"superconductivity_lsco", "magnetic_transport_crse", "mathematical_physics", "xrd_phase_identification"}.issubset(domain_ids)
    with pytest.raises(KeyError):
        registry.get("missing_domain")
    assert "Hall" in " ".join(registry.get("magnetic_transport_crse").guardrails())


def test_task_policy_registry_has_scientific_task_types() -> None:
    registry = TaskPolicyRegistry.default()
    policy = registry.get(ScientificTaskType.DATA_EXTRACTION)
    assert "review" in policy.required_stages
    assert registry.get("theory_derivation").required_artifacts


def test_hypothesis_v2_migration_and_finalist_requirements() -> None:
    hypothesis = migrate_hypothesis_to_v2(_hypotheses()[0], domain_id="superconductivity_lsco")
    assert hypothesis.hypothesis_id == "good"
    assert hypothesis.domain_id == "superconductivity_lsco"
    assert validate_finalist_requirements(hypothesis) == []
    bad = migrate_hypothesis_to_v2({"id": "bad", "title": "Bad", "core_claim": "No tests"})
    assert "missing falsification criteria" in validate_finalist_requirements(bad)


def test_optimizer_v2_writes_gates_kills_portfolio_and_memory(tmp_path: Path) -> None:
    result = HypothesisOptimizerV2(domain_id="superconductivity_lsco", run_id="opt").optimize(_hypotheses(), run_dir=tmp_path)
    assert result.hypothesis_count == 3
    assert result.killed_count == 1
    assert result.information_gain_actions
    assert result.failure_memory
    assert (tmp_path / "hard_gate_results.jsonl").exists()
    assert read_json(tmp_path / "optimizer_v2_summary.json")["killed_count"] == 1
    portfolio = read_jsonl(tmp_path / "hypothesis_portfolio_v2.jsonl")
    assert any(row["role"] == "contrarian_guardrail" for row in portfolio)
    assert mutation_operators()


def test_generic_acquisition_non_lsco_fixture(tmp_path: Path) -> None:
    result = GenericDataAcquisitionAgent().run(
        domain_id="magnetic_transport_crse",
        question="CrSe AHE altermagnetism claim",
        task_type="data_extraction",
        runs_dir=tmp_path,
        run_id="crse",
        force=True,
    )
    assert result.status in {"fixture_supported", "needs_review"}
    assert Path(result.run_dir, "domain_pack_manifest.json").exists()
    validations = read_jsonl(Path(result.run_dir) / "record_validations.jsonl")
    assert validations


def test_generic_acquisition_lsco_delegates_to_existing_phase2(tmp_path: Path) -> None:
    result = GenericDataAcquisitionAgent().run(
        domain_id="superconductivity_lsco",
        question="LSCO Phase 2 fixture",
        task_type="data_extraction",
        runs_dir=tmp_path,
        run_id="lsco",
        force=True,
    )
    assert result.delegated_to == "phase2_lsco_acquisition"
    assert Path(result.run_dir, "generic_acquisition_result.json").exists()


def test_frontend_exposes_domain_pack_and_optimizer(tmp_path: Path) -> None:
    service = create_app(runs_dir=tmp_path)
    rows = service.domain_pack_rows()
    assert any(row["domain_id"] == "xrd_phase_identification" for row in rows)
    run_dir = service.run_hypothesis_optimizer_v2(domain_id="superconductivity_lsco", run_id="front-opt")
    assert service.hypothesis_optimizer_portfolio_rows(run_dir)
    assert service.failure_memory_rows(run_dir)
    assert service.information_gain_rows(run_dir)


def test_live_agent_meeting_receives_optimizer_context(tmp_path: Path) -> None:
    service = create_app(runs_dir=tmp_path)
    run_dir = service.run_live_agent_meeting(
        tmp_path / "meeting",
        "Can a generalized BCS superconducting state be generated by mixed phonon attraction and correlated hopping?",
        live_model=False,
        max_rounds=1,
        force=True,
    )
    context = service.live_agent_tool_context(run_dir)
    assert context["optimizer_v2"]["killed_count"] == 1
    assert context["optimizer_v2"]["top_information_gain_actions"]
    transcript = service.live_agent_transcript(run_dir)
    assert "Hypothesis Optimizer V2" in transcript
