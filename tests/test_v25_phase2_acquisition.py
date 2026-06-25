from __future__ import annotations

import csv
from pathlib import Path

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.superconductivity.phase2_acquisition import (
    CandidateReviewDecision,
    PaperRecord,
    build_default_lsco_task,
    build_lsco_queries,
    evaluate_phase2_readiness_from_rows,
    import_digitized_points,
    parse_extraction_source_text,
    parse_supplementary_csv,
    promote_reviewed_candidates,
    review_candidate_rows,
    run_phase2_acquisition,
    validate_phase2_acquisition_run,
)


def test_phase2_query_generation_targets_observables() -> None:
    task = build_default_lsco_task(mode="fixture", max_queries=6)
    queries = build_lsco_queries(task)
    assert len(queries) == 6
    assert any("LSCO" in item["query"] for item in queries)
    assert {item["observable"] for item in queries}


def test_phase2_fixture_pipeline_stages_rows_and_digitization(tmp_path: Path) -> None:
    canonical = tmp_path / "phase2_lsco.csv"
    run_dir = run_phase2_acquisition(runs_dir=tmp_path, run_id="fixture", canonical_dataset=canonical)

    assert not validate_phase2_acquisition_run(run_dir)
    summary = read_json(run_dir / "acquisition_summary.json")
    assert summary["status"] == "completed"
    assert summary["candidate_rows_staged"] == 4
    assert summary["rows_promoted"] == 0
    assert summary["figure_digitization_tasks_created"] == 1

    rows = read_jsonl(run_dir / "candidate_rows.jsonl")
    assert {row["observable"] for row in rows} == {"tc_k", "gap_ev", "isotope_alpha", "penetration_depth_nm"}
    assert not canonical.exists()


def test_phase2_live_mode_requires_network_permission(tmp_path: Path) -> None:
    run_dir = run_phase2_acquisition(mode="live", live_network=False, runs_dir=tmp_path, run_id="blocked", canonical_dataset=tmp_path / "phase2.csv")
    summary = read_json(run_dir / "acquisition_summary.json")
    assert summary["status"] == "blocked_live_network_permission_required"
    assert summary["queries_executed"] == 0
    assert not validate_phase2_acquisition_run(run_dir)


def test_phase2_auto_promotion_updates_canonical_copy(tmp_path: Path) -> None:
    canonical = tmp_path / "phase2_promoted.csv"
    run_dir = run_phase2_acquisition(
        mode="fixture",
        runs_dir=tmp_path,
        run_id="promoted",
        canonical_dataset=canonical,
        auto_promote=True,
    )
    diff = read_json(run_dir / "canonical_dataset_diff.json")
    assert diff["changed"] is True
    assert diff["promoted_count"] == 4

    promoted = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines()))
    assert len(promoted) == 4
    assert {row["observable"] for row in promoted} == {"tc_k", "gap_ev", "isotope_alpha", "penetration_depth_nm"}


def test_phase2_digitized_import_preserves_review_gate(tmp_path: Path) -> None:
    run_dir = run_phase2_acquisition(runs_dir=tmp_path, run_id="digitization", canonical_dataset=tmp_path / "phase2.csv")
    csv_path = tmp_path / "digitized.csv"
    csv_path.write_text(
        "doping_x,observable_value,x_uncertainty,y_uncertainty,series_label,digitization_method,reviewer,source_figure\n"
        "0.24,0.31,0.005,0.03,LSCO overdoped,manual-webplotdigitizer,tester,Figure 2e\n",
        encoding="utf-8",
    )
    output = import_digitized_points(run_dir, task_id="digitize-fixture-optical-fig2e", csv_path=csv_path, reviewer="tester")
    rows = read_jsonl(output)
    assert rows[0]["review_status"] == "needs_review"
    assert rows[0]["observable_value"] == 0.31


def test_phase2_frontend_adapter_exposes_acquisition_views(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = app.run_phase2_acquisition(run_id="frontend", canonical_dataset=tmp_path / "phase2.csv")
    assert app.validate_phase2_acquisition(run_dir) == []
    assert app.phase2_acquisition_status(run_dir)["candidate_rows_staged"] == 4
    assert app.phase2_acquisition_candidate_sources(run_dir)
    assert app.phase2_acquisition_imported_rows(run_dir)
    assert app.phase2_acquisition_digitization_queue(run_dir)
    assert app.phase2_acquisition_readiness(run_dir)["phase2_status"] == "blocked_missing_optical_data"


def test_phase2_text_parser_extracts_values_and_queues_figures() -> None:
    paper = PaperRecord(paper_id="paper-parser", title="LSCO table and optical figure")
    text = """
    La2-xSrxCuO4 x=0.150 Tc=38 K gap=12.5 meV lambda=240 nm alpha=0.08.
    Figure 2e shows S_delta/Sn for LSCO but does not tabulate values.
    """
    records, digitization, fallback = parse_extraction_source_text(paper, text, source_kind="tex")
    assert {item.observable for item in records} == {"tc_k", "gap_ev", "penetration_depth_nm", "isotope_alpha"}
    assert next(item for item in records if item.observable == "gap_ev").normalized_value == 0.0125
    assert digitization
    assert fallback[0].status == "parsed"


def test_phase2_supplementary_csv_parser_preserves_provenance(tmp_path: Path) -> None:
    paper = PaperRecord(paper_id="paper-supp", title="LSCO supplement")
    csv_path = tmp_path / "supp.csv"
    csv_path.write_text(
        "observable,value,unit,doping,sample_id,uncertainty,measurement_method,definition\n"
        "gap_ev,12.5,meV,x=0.150,s1,2 meV,ARPES,superconducting gap\n",
        encoding="utf-8",
    )
    records, fallback = parse_supplementary_csv(paper, csv_path)
    assert records[0].normalized_value == 0.0125
    assert records[0].table_id == "supp.csv"
    assert "Supplementary CSV row" in records[0].evidence_text
    assert fallback[0].status == "parsed"


def test_phase2_review_then_promote_reviewed_rows(tmp_path: Path) -> None:
    canonical = tmp_path / "reviewed.csv"
    run_dir = run_phase2_acquisition(runs_dir=tmp_path, run_id="review", canonical_dataset=canonical)
    first = read_jsonl(run_dir / "candidate_rows.jsonl")[0]
    review_candidate_rows(run_dir, [CandidateReviewDecision(candidate_row_id=first["candidate_row_id"], decision="approve", rationale="fixture table reviewed")])
    promote_reviewed_candidates(run_dir, canonical)
    rows = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0]["observable"] == first["observable"]


def test_phase2_readiness_adversarial_gates() -> None:
    base = [
        {"observation_id": f"obs-{obs}", "material_id": "same", "doping": "x=0.150", "observable": obs, "value": 1, "unit": "x", "split": "train", "provenance": "table", "usable_for_fit": True, "curation_note": "spectral proxy"}
        for obs in ["tc_k", "gap_ev", "penetration_depth_nm", "isotope_alpha", "optical_spectral_weight_proxy"]
    ]
    assert evaluate_phase2_readiness_from_rows(base, require_held_out=True).status == "ready_for_exploratory_comparison"
    bad_provenance = [dict(row) for row in base]
    bad_provenance[0]["provenance"] = ""
    assert evaluate_phase2_readiness_from_rows(bad_provenance).status == "blocked_insufficient_existing_data"
    no_overlap = [dict(row, doping=f"x=0.{150 + index:03d}") for index, row in enumerate(base)]
    assert evaluate_phase2_readiness_from_rows(no_overlap).status == "blocked_missing_overlap"
    no_optical = [row for row in base if row["observable"] != "optical_spectral_weight_proxy"]
    assert evaluate_phase2_readiness_from_rows(no_optical).status == "blocked_missing_optical_data"
