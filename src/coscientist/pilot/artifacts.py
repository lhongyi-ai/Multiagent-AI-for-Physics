from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coscientist.pilot.project_io import dump_jsonable


REQUIRED_V1_ARTIFACTS = [
    "run_manifest.json",
    "project_snapshot.json",
    "resolved_configuration.json",
    "corpus.jsonl",
    "normalized_papers.jsonl",
    "literature_queries.jsonl",
    "literature_search_events.jsonl",
    "provider_status.json",
    "provider_usage.json",
    "model_calls.jsonl",
    "model_usage.json",
    "model_provider_status.json",
    "raw_openalex_records.jsonl",
    "raw_arxiv_records.jsonl",
    "crossref_enrichment.jsonl",
    "unpaywall_enrichment.jsonl",
    "metadata_conflicts.jsonl",
    "deduplication_report.json",
    "corpus_manifest.json",
    "hypotheses_initial.jsonl",
    "reviews.jsonl",
    "evidence_verification.jsonl",
    "rankings.jsonl",
    "evolution_round_1.jsonl",
    "evolution_round_2.jsonl",
    "hypotheses_final.json",
    "evaluation_by_round.json",
    "round_comparison.json",
    "lineage.json",
    "proximity_round_final.json",
    "hypothesis_graph_round_final.json",
    "clusters_round_final.json",
    "search_space_coverage_round_final.json",
    "meta_review_round_final.json",
    "meta_review_decisions_round_final.json",
    "grounding_diagnostics_round_final.json",
    "grounding_packets_round_final.json",
    "v15b_summary.json",
    "report.md",
    "human_review.md",
]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(dump_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, payloads: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(dump_jsonable(payload), sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_v1_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors = []
    for name in REQUIRED_V1_ARTIFACTS:
        if not (path / name).exists():
            errors.append(f"missing artifact: {name}")
    manifest_path = path / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid manifest JSON: {exc}")
        else:
            if manifest.get("artifact_schema_version") != "v1":
                errors.append("incompatible artifact_schema_version")
            if not manifest.get("live_model_enabled", False):
                calls_path = path / "model_calls.jsonl"
                if calls_path.exists() and calls_path.read_text(encoding="utf-8").strip():
                    errors.append("manifest disables live model but model_calls.jsonl is non-empty")
            if manifest.get("live_model_enabled", False) and manifest.get("model_mode") != "live":
                errors.append("manifest enables live model but model_mode is not live")
            for artifact in manifest.get("artifacts", []):
                if not (path / artifact).exists():
                    errors.append(f"manifest references missing artifact: {artifact}")
    errors.extend(_validate_v15b_artifacts(path))
    for name in REQUIRED_V1_ARTIFACTS:
        artifact = path / name
        if not artifact.exists() or artifact.suffix not in {".json", ".jsonl"}:
            continue
        try:
            if artifact.suffix == ".jsonl":
                read_jsonl(artifact)
            else:
                read_json(artifact)
        except json.JSONDecodeError as exc:
            errors.append(f"corrupt JSON artifact {name}: {exc}")
    return errors


def _validate_v15b_artifacts(path: Path) -> list[str]:
    errors: list[str] = []
    hypotheses_path = path / "hypotheses_final.json"
    if not hypotheses_path.exists():
        return errors
    try:
        hypothesis_ids = {item["id"] for item in read_json(hypotheses_path)}
    except Exception as exc:
        return [f"cannot validate V1.5B artifacts without hypotheses_final.json: {exc}"]
    evidence_ids = set()
    for item in read_jsonl(path / "evidence_verification.jsonl") if (path / "evidence_verification.jsonl").exists() else []:
        evidence_ids.add(item.get("claim_id"))
    proximity_path = path / "proximity_round_final.json"
    if proximity_path.exists():
        proximity = read_json(proximity_path)
        cluster_ids = {cluster.get("cluster_id") for cluster in proximity.get("clusters", [])}
        for item in proximity.get("pairwise_similarities", []):
            if item.get("hypothesis_a_id") not in hypothesis_ids or item.get("hypothesis_b_id") not in hypothesis_ids:
                errors.append("proximity pair references missing hypothesis")
        for cluster in proximity.get("clusters", []):
            members = set(cluster.get("member_ids", []))
            if not members.issubset(hypothesis_ids):
                errors.append(f"cluster {cluster.get('cluster_id')} references missing hypothesis")
            if cluster.get("representative_hypothesis_id") not in members:
                errors.append(f"cluster {cluster.get('cluster_id')} representative is not a member")
        for edge in proximity.get("graph_edges", []):
            if edge.get("source_id") not in hypothesis_ids or edge.get("target_id") not in hypothesis_ids:
                errors.append("hypothesis graph edge references missing node")
        meta_path = path / "meta_review_round_final.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            for hid in (
                meta.get("strongest_hypotheses", [])
                + meta.get("recommended_hypothesis_branches", [])
                + meta.get("recommended_hypothesis_repairs", [])
                + meta.get("recommended_hypotheses_to_hold", [])
            ):
                if hid not in hypothesis_ids:
                    errors.append(f"meta-review references missing hypothesis: {hid}")
            for group in meta.get("recommended_hypothesis_merges", []):
                if not set(group).issubset(hypothesis_ids):
                    errors.append("meta-review merge recommendation references missing hypothesis")
            for cluster_id in meta.get("referenced_cluster_ids", []):
                if cluster_id not in cluster_ids:
                    errors.append(f"meta-review references missing cluster: {cluster_id}")
            for evidence_id in meta.get("referenced_verification_ids", []):
                if evidence_id not in evidence_ids:
                    errors.append(f"meta-review references missing verification: {evidence_id}")
    packet_path = path / "grounding_packets_round_final.json"
    if packet_path.exists():
        packet = read_json(packet_path)
        corpus_ids = {item.get("id") for item in read_jsonl(path / "corpus.jsonl")} if (path / "corpus.jsonl").exists() else set()
        for item in packet.get("evidence_items", []):
            if item.get("paper_id") not in corpus_ids:
                errors.append(f"grounding packet references missing corpus paper: {item.get('paper_id')}")
    for name in [
        "proximity_round_final.json",
        "meta_review_round_final.json",
        "grounding_diagnostics_round_final.json",
        "grounding_packets_round_final.json",
        "v15b_summary.json",
    ]:
        target = path / name
        if target.exists() and "OPENAI_API_KEY" in target.read_text(encoding="utf-8"):
            errors.append(f"secret-like key name leaked into {name}")
    return errors
