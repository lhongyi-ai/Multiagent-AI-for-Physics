from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json
from coscientist.schemas.v21 import ScientificIndexManifest


SCHEMA_VERSION = "v21-sqlite-1"


TABLES = [
    "runs",
    "campaigns",
    "providers",
    "provider_connection_status",
    "retrieval_jobs",
    "source_snapshots",
    "sources",
    "papers",
    "datasets",
    "materials",
    "samples",
    "material_aliases",
    "doping_records",
    "observations",
    "optical_records",
    "thermodynamic_records",
    "isotope_records",
    "structures",
    "computational_records",
    "model_candidates",
    "candidate_models",
    "fingerprint_predictions",
    "hamiltonian_terms",
    "fit_results",
    "verifier_results",
    "claims",
    "predictions",
    "objections",
    "expert_reviews",
    "dialogue_turns",
    "model_calls",
    "usage_costs",
    "artifact_index",
]


def rebuild_scientific_index(run_or_project_path: str | Path) -> Path:
    root = Path(run_or_project_path)
    if root.is_file():
        root = root.parent
    db_path = root / "scientific_index.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        _create_schema(con)
        _index_artifacts(con, root)
        _index_known_records(con, root)
        manifest = _manifest(con, root, db_path)
        write_json(root / "scientific_index_manifest.json", manifest)
    finally:
        con.close()
    return db_path


def validate_scientific_index(run_or_project_path: str | Path) -> list[str]:
    root = Path(run_or_project_path)
    if root.is_file():
        root = root.parent
    errors = []
    db_path = root / "scientific_index.sqlite"
    manifest_path = root / "scientific_index_manifest.json"
    if not db_path.exists():
        return ["missing scientific_index.sqlite"]
    if not manifest_path.exists():
        return ["missing scientific_index_manifest.json"]
    manifest = ScientificIndexManifest.model_validate(read_json(manifest_path))
    current_hash = _artifact_hash(root)
    if manifest.artifact_hash != current_hash:
        errors.append("scientific index is stale")
    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
        for table in TABLES:
            if table not in tables:
                errors.append(f"missing index table: {table}")
    finally:
        con.close()
    return errors


def query_scientific_index(run_or_project_path: str | Path, table: str, *, limit: int = 20) -> list[dict[str, Any]]:
    if table not in TABLES:
        raise ValueError(f"unsupported scientific index table: {table}")
    root = Path(run_or_project_path)
    if root.is_file():
        root = root.parent
    con = sqlite3.connect(root / "scientific_index.sqlite")
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(f"select * from {table} limit ?", (limit,))]
    finally:
        con.close()


def _create_schema(con: sqlite3.Connection) -> None:
    con.execute("create table schema_version (version text primary key)")
    con.execute("insert into schema_version values (?)", (SCHEMA_VERSION,))
    con.execute("create table runs (run_id text primary key, artifact_root text, schema_version text)")
    for table in TABLES[1:]:
        con.execute(f"create table {table} (id text primary key, artifact_path text, payload_json text)")
    con.commit()


def _index_artifacts(con: sqlite3.Connection, root: Path) -> None:
    con.execute("insert into runs values (?, ?, ?)", (root.name, str(root), SCHEMA_VERSION))
    for path in sorted([*root.glob("*.json"), *root.glob("*.jsonl"), *root.glob("*.md")]):
        payload = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        con.execute("insert into artifact_index values (?, ?, ?)", (path.name, path.name, json.dumps(payload, sort_keys=True)))
    con.commit()


def _index_known_records(con: sqlite3.Connection, root: Path) -> None:
    mapping = {
        "superconductivity_campaign.json": ("campaigns", "campaign_id"),
        "superconductivity_sources.jsonl": ("sources", "source_id"),
        "superconductivity_corpus.jsonl": ("papers", "source_id"),
        "supercon_records.jsonl": ("materials", "material_id"),
        "material_mapping.jsonl": ("doping_records", "material_id"),
        "self_consistency_results.jsonl": ("fit_results", "model_id"),
        "superconductivity_verifier_results.jsonl": ("verifier_results", "verifier_id"),
        "provider_connection_results.json": ("provider_connection_status", "provider"),
        "live_agent_dialogues.jsonl": ("dialogue_turns", "turn_id"),
        "model_call_records.jsonl": ("model_calls", "request_sequence_number"),
        "model_usage_summary.json": ("usage_costs", "model_mode"),
        "candidate_models.jsonl": ("candidate_models", "model_id"),
        "mechanism_fingerprints.jsonl": ("model_candidates", "fingerprint_id"),
        "held_out_predictions.jsonl": ("fingerprint_predictions", "prediction_id"),
        "microscopic_hamiltonians.jsonl": ("hamiltonian_terms", "model_id"),
        "real_observations.jsonl": ("observations", "record_id"),
        "expert_curated_dossier.json": ("datasets", "record_id"),
        "objection_board.jsonl": ("objections", "objection_id"),
        "claim_ledger.jsonl": ("claims", "claim_id"),
        "prediction_ledger.jsonl": ("predictions", "prediction_id"),
    }
    for artifact, (table, key) in mapping.items():
        path = root / artifact
        if not path.exists():
            continue
        raw = read_json(path) if path.suffix == ".json" else read_jsonl(path)
        if artifact == "provider_connection_results.json":
            records = raw.get("providers", [])
        elif artifact == "expert_curated_dossier.json":
            records = raw.get("records", [])
        else:
            records = [raw] if path.suffix == ".json" else raw
        for index, record in enumerate(records):
            identifier = str(record.get(key) or record.get("id") or f"{artifact}:{index}")
            con.execute(f"insert or replace into {table} values (?, ?, ?)", (identifier, artifact, json.dumps(record, sort_keys=True)))
    con.commit()


def _manifest(con: sqlite3.Connection, root: Path, db_path: Path) -> ScientificIndexManifest:
    counts = {table: con.execute(f"select count(*) from {table}").fetchone()[0] for table in TABLES}
    return ScientificIndexManifest(database_path=str(db_path), artifact_root=str(root), artifact_hash=_artifact_hash(root), table_counts=counts)


def _artifact_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*root.glob("*.json"), *root.glob("*.jsonl"), *root.glob("*.md")]):
        if path.name == "scientific_index_manifest.json":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
