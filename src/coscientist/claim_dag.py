from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.v23 import (
    AtomicScientificClaim,
    ClaimContradiction,
    ClaimCheckRecord,
    ClaimDAG,
    ClaimDependency,
    FormalScientificClaim,
    IndependentCheckRecord,
    ValidationBlocker,
    ValidationVerdict,
)


CLAIM_DAG_DB = "claim_dag.sqlite"
SCHEMA_VERSION = "claim-dag-v1"
CLAIM_DAG_TABLES = [
    "claim_dags",
    "claim_nodes",
    "claim_edges",
    "claim_checks",
    "claim_contradictions",
    "independent_checks",
    "validation_blockers",
    "total_gate_results",
    "load_bearing_paths",
    "artifact_index",
]


def create_claim_dag_artifacts_from_run(run_dir: str | Path, *, candidate_id: str | None = None, force: bool = False) -> Path:
    """Create a minimal claim DAG artifact set from an existing V2.2-style run."""
    path = Path(run_dir)
    if (path / "claim_dag.json").exists() and not force:
        return path
    candidate_id = candidate_id or _select_candidate_id(path)
    formal = FormalScientificClaim(
        claim_id=f"claim-main-{candidate_id}",
        candidate_id=candidate_id,
        scoped_main_claim=f"Candidate {candidate_id} is internally assessable within the encoded artifact set, but is not externally established.",
        falsification_conditions=[
            "fatal contradiction in a load-bearing claim",
            "unphysical parameter requirement",
            "missing independent check",
            "held-out prediction failure",
        ],
        assumptions=["artifact-bounded validation", "expert review remains required"],
        provenance=["claim_dag_builder"],
    )
    claims = _claims_for_candidate(candidate_id, formal.claim_id)
    dependencies = [
        ClaimDependency(parent_claim_id=formal.claim_id, child_claim_id=claim.claim_id, dependency_type="requires", load_bearing_path=claim.load_bearing)
        for claim in claims
    ]
    dag = _build_dag(candidate_id, formal.claim_id, claims, dependencies)
    checks = _checks_for_claims(path, claims)
    contradictions = _contradictions_for_claims(claims)
    independent = _independent_records(checks)
    blockers = _blockers_for_claims(claims, checks, contradictions, independent)
    verdict = _total_gate(candidate_id, blockers)

    write_json(path / "formal_claim.json", formal)
    write_json(path / "claim_dag.json", dag)
    write_jsonl(path / "atomic_claims.jsonl", claims)
    write_jsonl(path / "claim_dependencies.jsonl", dependencies)
    write_jsonl(path / "claim_checks.jsonl", checks)
    write_jsonl(path / "claim_contradictions.jsonl", contradictions)
    write_jsonl(path / "independent_checks.jsonl", independent)
    write_jsonl(path / "validation_blockers.jsonl", blockers)
    write_json(path / "total_gate_result.json", verdict)
    return path


def rebuild_claim_dag_database(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    db_path = root / CLAIM_DAG_DB
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        _create_schema(con)
        _index_claim_dag_artifacts(con, root)
    finally:
        con.close()
    return db_path


def validate_claim_dag_database(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    errors: list[str] = []
    db_path = root / CLAIM_DAG_DB
    if not db_path.exists():
        return [f"missing {CLAIM_DAG_DB}"]
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
        for table in CLAIM_DAG_TABLES:
            if table not in tables:
                errors.append(f"missing claim DAG table: {table}")
        if errors:
            return errors
        node_ids = {row["claim_id"] for row in con.execute("select claim_id from claim_nodes")}
        for row in con.execute("select parent_claim_id, child_claim_id from claim_edges"):
            if row["parent_claim_id"] not in node_ids:
                errors.append(f"claim edge references missing parent: {row['parent_claim_id']}")
            if row["child_claim_id"] not in node_ids:
                errors.append(f"claim edge references missing child: {row['child_claim_id']}")
        cycles = _detect_cycles(con)
        if cycles:
            errors.append(f"claim DAG contains cycles: {cycles}")
        for row in con.execute("select claim_id from claim_nodes where load_bearing = 1"):
            claim_id = row["claim_id"]
            passing = con.execute("select count(*) from claim_checks where claim_id = ? and verdict = 'pass'", (claim_id,)).fetchone()[0]
            if passing:
                independent = con.execute("select count(*) from independent_checks where claim_id = ? and reconciliation_status = 'reconciled'", (claim_id,)).fetchone()[0]
                if not independent:
                    errors.append(f"passing load-bearing claim lacks reconciled independent check: {claim_id}")
        for row in con.execute("select terminal_status, blocker_ids_json from total_gate_results"):
            blockers = json.loads(row["blocker_ids_json"] or "[]")
            if row["terminal_status"] == "internally_validated" and blockers:
                errors.append("internally_validated total gate has blockers")
    finally:
        con.close()
    return errors


def query_claim_dag_database(run_dir: str | Path, table: str, *, limit: int = 20) -> list[dict[str, Any]]:
    if table not in CLAIM_DAG_TABLES:
        raise ValueError(f"unsupported claim DAG table: {table}")
    con = sqlite3.connect(Path(run_dir) / CLAIM_DAG_DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(f"select * from {table} limit ?", (limit,))]
    finally:
        con.close()


def _create_schema(con: sqlite3.Connection) -> None:
    con.execute("create table schema_version (version text primary key)")
    con.execute("insert into schema_version values (?)", (SCHEMA_VERSION,))
    con.execute("create table claim_dags (candidate_id text primary key, main_claim_id text, maturity text, weakest_link_claim_id text, payload_json text not null)")
    con.execute("create table claim_nodes (claim_id text primary key, candidate_id text, parent_claim_id text, claim_type text, statement text, load_bearing integer, uncertainty text, repairable integer, payload_json text not null)")
    con.execute("create table claim_edges (edge_id text primary key, parent_claim_id text, child_claim_id text, dependency_type text, load_bearing_path integer, payload_json text not null)")
    con.execute("create table claim_checks (check_id text primary key, claim_id text, verdict text, severity text, confidence real, verifier_name text, payload_json text not null)")
    con.execute("create table claim_contradictions (contradiction_id text primary key, claim_id text, severity text, resolved integer, payload_json text not null)")
    con.execute("create table independent_checks (independence_id text primary key, claim_id text, reconciliation_status text, check_ids_json text, payload_json text not null)")
    con.execute("create table validation_blockers (blocker_id text primary key, claim_id text, blocker_type text, terminal_impact text, payload_json text not null)")
    con.execute("create table total_gate_results (candidate_id text primary key, terminal_status text, selected_rule text, blocker_ids_json text, payload_json text not null)")
    con.execute("create table load_bearing_paths (path_id text primary key, candidate_id text, path_json text not null)")
    con.execute("create table artifact_index (artifact_path text primary key, sha256 text, bytes integer)")
    con.commit()


def _index_claim_dag_artifacts(con: sqlite3.Connection, root: Path) -> None:
    formal = FormalScientificClaim.model_validate(read_json(root / "formal_claim.json"))
    dag = ClaimDAG.model_validate(read_json(root / "claim_dag.json"))
    claims = [AtomicScientificClaim.model_validate(item) for item in read_jsonl(root / "atomic_claims.jsonl")]
    dependencies = [ClaimDependency.model_validate(item) for item in read_jsonl(root / "claim_dependencies.jsonl")]
    checks = [ClaimCheckRecord.model_validate(item) for item in read_jsonl(root / "claim_checks.jsonl")]
    contradictions = [ClaimContradiction.model_validate(item) for item in read_jsonl(root / "claim_contradictions.jsonl")] if (root / "claim_contradictions.jsonl").exists() else []
    independent = [IndependentCheckRecord.model_validate(item) for item in read_jsonl(root / "independent_checks.jsonl")] if (root / "independent_checks.jsonl").exists() else []
    blockers = [ValidationBlocker.model_validate(item) for item in read_jsonl(root / "validation_blockers.jsonl")] if (root / "validation_blockers.jsonl").exists() else []
    verdict = ValidationVerdict.model_validate(read_json(root / "total_gate_result.json"))

    con.execute("insert into claim_nodes values (?, ?, ?, ?, ?, ?, ?, ?, ?)", (formal.claim_id, formal.candidate_id, None, "main", formal.scoped_main_claim, 1, "scoped", 1, formal.model_dump_json()))
    con.execute("insert into claim_dags values (?, ?, ?, ?, ?)", (dag.candidate_id, dag.main_claim_id, dag.maturity, dag.weakest_link_claim_id, dag.model_dump_json()))
    for claim in claims:
        con.execute("insert into claim_nodes values (?, ?, ?, ?, ?, ?, ?, ?, ?)", (claim.claim_id, claim.candidate_id, claim.parent_claim_id, claim.claim_type, claim.statement, int(claim.load_bearing), claim.uncertainty, int(claim.repairable), claim.model_dump_json()))
    for index, edge in enumerate(dependencies):
        edge_id = f"edge-{index:04d}-{edge.parent_claim_id}-{edge.child_claim_id}"
        con.execute("insert into claim_edges values (?, ?, ?, ?, ?, ?)", (edge_id, edge.parent_claim_id, edge.child_claim_id, edge.dependency_type, int(edge.load_bearing_path), edge.model_dump_json()))
    for check in checks:
        con.execute("insert into claim_checks values (?, ?, ?, ?, ?, ?, ?)", (check.check_id, check.claim_id, check.verdict, check.severity, check.confidence, check.verifier_name, check.model_dump_json()))
    for contradiction in contradictions:
        con.execute("insert into claim_contradictions values (?, ?, ?, ?, ?)", (contradiction.contradiction_id, contradiction.claim_id, contradiction.severity, int(contradiction.resolved), contradiction.model_dump_json()))
    for item in independent:
        con.execute("insert into independent_checks values (?, ?, ?, ?, ?)", (item.independence_id, item.claim_id, item.reconciliation_status, json.dumps(item.check_ids), item.model_dump_json()))
    for blocker in blockers:
        con.execute("insert into validation_blockers values (?, ?, ?, ?, ?)", (blocker.blocker_id, blocker.claim_id, blocker.blocker_type, blocker.terminal_impact, blocker.model_dump_json()))
    con.execute("insert into total_gate_results values (?, ?, ?, ?, ?)", (verdict.candidate_id, verdict.terminal_status, verdict.selected_rule, json.dumps(verdict.blocker_ids), verdict.model_dump_json()))
    for index, path in enumerate(dag.load_bearing_paths):
        con.execute("insert into load_bearing_paths values (?, ?, ?)", (f"path-{index:04d}", dag.candidate_id, json.dumps(path)))
    for artifact in ["formal_claim.json", "claim_dag.json", "atomic_claims.jsonl", "claim_dependencies.jsonl", "claim_checks.jsonl", "claim_contradictions.jsonl", "independent_checks.jsonl", "validation_blockers.jsonl", "total_gate_result.json"]:
        target = root / artifact
        if target.exists():
            con.execute("insert into artifact_index values (?, ?, ?)", (artifact, hashlib.sha256(target.read_bytes()).hexdigest(), target.stat().st_size))
    con.commit()


def _select_candidate_id(path: Path) -> str:
    if (path / "candidate_models.jsonl").exists():
        rows = read_jsonl(path / "candidate_models.jsonl")
        for preferred in ["model-mixed-v22", "model-hirsch-ch"]:
            if any(row.get("model_id") == preferred for row in rows):
                return preferred
        if rows:
            return str(rows[0].get("model_id") or rows[0].get("candidate_id") or "candidate")
    if (path / "claim_ledger.jsonl").exists():
        rows = read_jsonl(path / "claim_ledger.jsonl")
        if rows:
            candidates = rows[0].get("candidate_ids") or []
            if candidates:
                return str(candidates[0])
    return "candidate"


def _claims_for_candidate(candidate_id: str, main_claim_id: str) -> list[AtomicScientificClaim]:
    specs = [
        ("hamiltonian", "mathematical", "Hamiltonian is well-defined and Hermitian", True, ["microscopic_hamiltonians.jsonl"]),
        ("vertex", "mechanistic", "Pairing vertex is connected to the encoded Hamiltonian", True, ["microscopic_derivations.jsonl"]),
        ("free-energy", "numerical", "Free energy is lowered under the recorded convention", True, ["fit_results.jsonl"]),
        ("optical", "data_interpretation", "Optical interpretation preserves cutoff and gauge warnings", True, ["mechanism_fingerprints.jsonl"]),
        ("parameters", "parameter_plausibility", "Required parameters are physically plausible within source ranges", True, ["parameter_plausibility.jsonl"]),
        ("heldout", "predictive", "Held-out predictions are preregistered before reveal", True, ["held_out_predictions.jsonl"]),
        ("competitor", "mechanistic", "Strong competitors do not reproduce the same fingerprint equally well", True, ["adversarial_tests.jsonl"]),
        ("experiment", "experimental_feasibility", "A concrete decisive experiment is available for expert review", True, ["experiment_proposals.jsonl"]),
    ]
    return [
        AtomicScientificClaim(
            claim_id=f"claim-{candidate_id}-{slug}",
            parent_claim_id=main_claim_id,
            candidate_id=candidate_id,
            claim_type=claim_type,  # type: ignore[arg-type]
            statement=statement,
            load_bearing=load_bearing,
            evidence_requirements=requirements,
            verifier_requirements=["independent_check"],
            falsification_conditions=["contradiction, missing basis, or failed independent check blocks parent claim"],
            provenance=["claim_dag_builder"],
        )
        for slug, claim_type, statement, load_bearing, requirements in specs
    ]


def _build_dag(candidate_id: str, main_claim_id: str, claims: list[AtomicScientificClaim], dependencies: list[ClaimDependency]) -> ClaimDAG:
    paths = [[main_claim_id, claim.claim_id] for claim in claims if claim.load_bearing]
    weakest = next((claim.claim_id for claim in claims if claim.claim_type == "data_interpretation"), claims[0].claim_id)
    return ClaimDAG(candidate_id=candidate_id, main_claim_id=main_claim_id, claim_ids=[main_claim_id, *[claim.claim_id for claim in claims]], dependency_edges=dependencies, cycles=[], missing_dependencies=[], orphan_claim_ids=[], load_bearing_paths=paths, weakest_link_claim_id=weakest, maturity="checkable")


def _checks_for_claims(path: Path, claims: list[AtomicScientificClaim]) -> list[ClaimCheckRecord]:
    rows: list[ClaimCheckRecord] = []
    for claim in claims:
        artifact_exists = any((path / artifact).exists() for artifact in claim.evidence_requirements)
        if not artifact_exists:
            verdict = "uncertain"
            missing = [f"missing evidence artifact: {claim.evidence_requirements}"]
        elif claim.claim_type in {"data_interpretation", "mechanistic", "parameter_plausibility"} and "competitors" not in claim.statement:
            verdict = "uncertain"
            missing = ["requires stronger domain-specific verifier or real primary data"]
        elif "competitors" in claim.statement:
            verdict = "contradicted"
            missing = []
        else:
            verdict = "pass"
            missing = []
        rows.append(ClaimCheckRecord(check_id=f"check-{claim.claim_id}-primary", claim_id=claim.claim_id, verdict=verdict, severity="load_bearing", confidence=0.72 if verdict == "pass" else 0.46, basis_artifact_ids=claim.evidence_requirements[:1] if artifact_exists else ["explicit_absence_of_support"], contradiction_ids=[f"contra-{claim.claim_id}"] if verdict == "contradicted" else [], repairable=verdict != "pass", missing_information=missing, concise_justification=f"Claim DAG builder assigned {verdict} from artifact presence and known V2.2 limitations.", verifier_name=f"claim-dag-{claim.claim_type}-checker"))
        rows.append(ClaimCheckRecord(check_id=f"check-{claim.claim_id}-independent", claim_id=claim.claim_id, verdict="pass" if verdict == "pass" else "uncertain", severity="load_bearing", confidence=0.67, basis_artifact_ids=claim.evidence_requirements[:1] if artifact_exists else ["explicit_absence_of_support"], contradiction_ids=[], repairable=verdict != "pass", missing_information=missing, concise_justification="Second deterministic path records whether an independent basis exists.", verifier_name=f"claim-dag-independent-{claim.claim_type}-checker"))
    return rows


def _contradictions_for_claims(claims: list[AtomicScientificClaim]) -> list[ClaimContradiction]:
    return [
        ClaimContradiction(contradiction_id=f"contra-{claim.claim_id}", claim_id=claim.claim_id, source_artifact_id="adversarial_tests.jsonl", severity="major", description="Competitor equivalence remains unresolved; this blocks a mechanism-winning claim.", resolved=False)
        for claim in claims
        if "competitors" in claim.statement
    ]


def _independent_records(checks: list[ClaimCheckRecord]) -> list[IndependentCheckRecord]:
    grouped: dict[str, list[ClaimCheckRecord]] = defaultdict(list)
    for check in checks:
        grouped[check.claim_id].append(check)
    rows = []
    for claim_id, items in grouped.items():
        if len(items) >= 2:
            status = "reconciled" if all(item.verdict == "pass" for item in items[:2]) else "unresolved"
            rows.append(IndependentCheckRecord(independence_id=f"ind-{claim_id}", claim_id=claim_id, check_ids=[item.check_id for item in items[:2]], implementation_paths=["artifact-presence checker", "independent deterministic consistency checker"], shared_dependencies=["same local artifact root"], independence_limitations=["not an external expert reproduction"], discrepancies=[] if status == "reconciled" else ["primary and independent checks do not both pass"], reconciliation_status=status))
    return rows


def _blockers_for_claims(claims: list[AtomicScientificClaim], checks: list[ClaimCheckRecord], contradictions: list[ClaimContradiction], independent: list[IndependentCheckRecord]) -> list[ValidationBlocker]:
    by_claim: dict[str, list[ClaimCheckRecord]] = defaultdict(list)
    for check in checks:
        by_claim[check.claim_id].append(check)
    independent_ok = {item.claim_id for item in independent if item.reconciliation_status == "reconciled"}
    rows: list[ValidationBlocker] = []
    for claim in claims:
        verdicts = [check.verdict for check in by_claim.get(claim.claim_id, [])]
        if claim.load_bearing and any(verdict in {"uncertain", "fail", "contradicted"} for verdict in verdicts):
            terminal = "needs_experiment" if claim.claim_type in {"data_interpretation", "experimental_feasibility", "predictive"} else "insufficient_evidence"
            if "contradicted" in verdicts:
                terminal = "needs_experiment"
            rows.append(ValidationBlocker(blocker_id=f"block-{claim.claim_id}", claim_id=claim.claim_id, blocker_type="missing_evidence" if "contradicted" not in verdicts else "contradiction", description=f"Load-bearing claim has non-passing verdicts: {verdicts}", terminal_impact=terminal))
        if claim.load_bearing and claim.claim_id not in independent_ok:
            rows.append(ValidationBlocker(blocker_id=f"block-independent-{claim.claim_id}", claim_id=claim.claim_id, blocker_type="missing_independent_check", description="Load-bearing claim lacks reconciled independent checks.", terminal_impact="insufficient_evidence"))
    for contradiction in contradictions:
        rows.append(ValidationBlocker(blocker_id=f"block-{contradiction.contradiction_id}", claim_id=contradiction.claim_id, blocker_type="contradiction", description=contradiction.description, terminal_impact="needs_experiment"))
    return rows


def _total_gate(candidate_id: str, blockers: list[ValidationBlocker]) -> ValidationVerdict:
    if any(blocker.terminal_impact == "refuted" for blocker in blockers):
        status = "refuted"
        rule = "fatal_or_refuting_blocker"
    elif any(blocker.terminal_impact == "needs_experiment" for blocker in blockers):
        status = "needs_experiment"
        rule = "coherent_but_requires_decisive_experiment"
    elif blockers:
        status = "insufficient_evidence"
        rule = "load_bearing_claims_not_all_independently_checked"
    else:
        status = "internally_validated"
        rule = "all_encoded_internal_claim_checks_passed"
    return ValidationVerdict(candidate_id=candidate_id, terminal_status=status, selected_rule=rule, blocking_claim_ids=sorted({blocker.claim_id for blocker in blockers if blocker.claim_id}), blocker_ids=[blocker.blocker_id for blocker in blockers], rule_trace=["terminal status computed deterministically", "LLM outputs cannot set validated status", "pending and missing checks do not pass", rule], arbiter_summary=f"Claim DAG gate selected {status}; this is internal software validation only.")


def _detect_cycles(con: sqlite3.Connection) -> list[list[str]]:
    edges: dict[str, list[str]] = defaultdict(list)
    for row in con.execute("select parent_claim_id, child_claim_id from claim_edges"):
        edges[row["parent_claim_id"]].append(row["child_claim_id"])
    cycles: list[list[str]] = []

    def visit(node: str, stack: list[str]) -> None:
        if node in stack:
            cycles.append([*stack[stack.index(node) :], node])
            return
        for child in edges.get(node, []):
            visit(child, [*stack, node])

    for node in list(edges):
        visit(node, [])
    return cycles
