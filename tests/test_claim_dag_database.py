from __future__ import annotations

from pathlib import Path

from coscientist.claim_dag import (
    create_claim_dag_artifacts_from_run,
    query_claim_dag_database,
    rebuild_claim_dag_database,
    validate_claim_dag_database,
)
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.superconductivity import run_v22_campaign


FIXTURE = "examples/v22_superconductivity_real_data/project.yaml"


def test_claim_dag_artifacts_and_database_from_v22_run(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    create_claim_dag_artifacts_from_run(run_dir)
    db = rebuild_claim_dag_database(run_dir)
    assert db.exists()
    assert validate_claim_dag_database(run_dir) == []

    dag = read_json(run_dir / "claim_dag.json")
    claims = read_jsonl(run_dir / "atomic_claims.jsonl")
    assert dag["main_claim_id"].startswith("claim-main-")
    assert any(item["load_bearing"] for item in claims)

    nodes = query_claim_dag_database(run_dir, "claim_nodes", limit=50)
    edges = query_claim_dag_database(run_dir, "claim_edges", limit=50)
    checks = query_claim_dag_database(run_dir, "claim_checks", limit=50)
    gate = query_claim_dag_database(run_dir, "total_gate_results", limit=1)[0]
    assert len(nodes) == len(claims) + 1
    assert edges
    assert checks
    assert gate["terminal_status"] in {"internally_validated", "needs_experiment", "refuted", "insufficient_evidence", "verifier_insufficient"}


def test_claim_dag_builder_is_idempotent_without_force(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    create_claim_dag_artifacts_from_run(run_dir, candidate_id="first", force=True)
    first = read_json(run_dir / "formal_claim.json")
    create_claim_dag_artifacts_from_run(run_dir, candidate_id="second", force=False)
    second = read_json(run_dir / "formal_claim.json")
    assert first == second


def test_claim_dag_query_rejects_unknown_table(tmp_path: Path) -> None:
    run_dir = run_v22_campaign(FIXTURE, runs_dir=tmp_path, run_id="v22")
    create_claim_dag_artifacts_from_run(run_dir)
    rebuild_claim_dag_database(run_dir)
    try:
        query_claim_dag_database(run_dir, "not_a_table")
    except ValueError as exc:
        assert "unsupported claim DAG table" in str(exc)
    else:
        raise AssertionError("unknown claim DAG table should be rejected")
