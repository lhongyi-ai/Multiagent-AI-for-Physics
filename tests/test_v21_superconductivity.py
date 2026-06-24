from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.schemas.v21 import CorrelatedHoppingSpec, PhononAttractionSpec, SuperconductivityModelSpec
from coscientist.superconductivity import (
    query_scientific_index,
    rebuild_scientific_index,
    run_superconductivity_campaign,
    validate_scientific_index,
    validate_superconductivity_campaign,
)
from coscientist.superconductivity.adapters import MaterialsProjectAdapter, NomadAdapter, OptimadeAdapter, SuperConAdapter


FIXTURE = "examples/superconductivity_bcs_campaign/project.yaml"
SOURCE_DIR = Path("examples/superconductivity_bcs_campaign/sources")


def test_superconductivity_schema_rejects_inconsistent_family() -> None:
    with pytest.raises(ValidationError):
        SuperconductivityModelSpec(
            model_id="bad",
            candidate_id="cand",
            family="phonon_only_bcs",
            pairing_kernel={
                "kernel_id": "k",
                "phonon": PhononAttractionSpec(coupling_ev=0.2),
                "correlated_hopping": CorrelatedHoppingSpec(coupling_ev=0.1),
            },
        )


def test_database_adapters_parse_local_fixtures_and_preserve_identity() -> None:
    supercon = SuperConAdapter().load_snapshot(SOURCE_DIR / "supercon_fixture.csv")
    mp = MaterialsProjectAdapter().load_snapshot(SOURCE_DIR / "materials_project_fixture.jsonl")
    nomad = NomadAdapter().load_snapshot(SOURCE_DIR / "nomad_fixture.jsonl")
    optimade = OptimadeAdapter().load_snapshot(SOURCE_DIR / "optimade_fixture.jsonl")
    assert supercon[0]["mechanism_inferred"] is False
    assert supercon[2]["doping_label"] == "x~0.15 ambiguous"
    assert mp[0]["computed_or_experimental"] == "computed"
    assert "archive_id" in nomad[0]
    assert optimade[0]["formula_only_merge_permitted"] is False


def test_superconductivity_campaign_runs_validates_and_scores_differentiate(tmp_path: Path) -> None:
    run_dir = run_superconductivity_campaign(FIXTURE, runs_dir=tmp_path, run_id="sc")
    assert validate_superconductivity_campaign(run_dir) == []
    benchmarks = read_json(run_dir / "benchmark_results.json")
    assert all(value == "passed" for key, value in benchmarks.items() if key != "schema_version")
    scores = read_jsonl(run_dir / "superconductivity_scores.jsonl")
    assert len({item["aggregate_score"] for item in scores}) > 1
    energy = read_jsonl(run_dir / "energy_decomposition.jsonl")
    assert all(abs(item["condensation_energy_closure_error_ev"]) <= 1e-8 for item in energy)
    optical = read_jsonl(run_dir / "optical_sum_results.jsonl")
    assert all(item["interpretation_warnings"] for item in optical)
    equivalence = read_json(run_dir / "equivalence_classes.json")
    assert equivalence["equivalence_classes"]


def test_scientific_index_rebuild_query_and_stale_detection(tmp_path: Path) -> None:
    run_dir = run_superconductivity_campaign(FIXTURE, runs_dir=tmp_path, run_id="sc-index")
    db = rebuild_scientific_index(run_dir)
    assert db.exists()
    assert validate_scientific_index(run_dir) == []
    rows = query_scientific_index(run_dir, "materials", limit=2)
    assert rows
    (run_dir / "superconductivity_report.md").write_text("changed\n", encoding="utf-8")
    assert "scientific index is stale" in validate_scientific_index(run_dir)


def test_frontend_runs_superconductivity_campaign(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_superconductivity_fixture(FIXTURE, run_id="sc-ui"))
    assert app.validate_superconductivity(run_dir) == []
    assert app.superconductivity_score_rows(run_dir)
    assert app.superconductivity_energy_rows(run_dir)
    assert app.superconductivity_optical_rows(run_dir)
    assert app.superconductivity_material_rows(run_dir)
