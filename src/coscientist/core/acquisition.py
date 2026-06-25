from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from coscientist.core.domain_packs import DomainPack, ResearchObjective, SourceRecord, get_default_domain_registry
from coscientist.core.tasks import ScientificTaskType
from coscientist.pilot.artifacts import read_json, write_json, write_jsonl


class GenericAcquisitionResult(BaseModel):
    schema_version: str = "v26-generic-acquisition-result"
    domain_id: str
    run_dir: str
    mode: str
    status: str
    task_type: str
    query_count: int
    normalized_record_count: int
    validation_error_count: int
    delegated_to: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


class GenericDataAcquisitionAgent:
    def __init__(self, registry=None):
        self.registry = registry or get_default_domain_registry()

    def run(
        self,
        *,
        domain_id: str,
        question: str,
        task_type: str | ScientificTaskType = ScientificTaskType.DATA_EXTRACTION,
        mode: str = "fixture",
        runs_dir: str | Path = "runs",
        run_id: str | None = None,
        live_network: bool = False,
        force: bool = False,
    ) -> GenericAcquisitionResult:
        pack = self.registry.get(domain_id)
        run_name = run_id or f"generic-acquisition-{domain_id}"
        root = Path(runs_dir) / run_name
        if root.exists() and any(root.iterdir()) and not force:
            raise ValueError(f"generic acquisition artifacts are immutable; use --force or a new run id: {root}")
        root.mkdir(parents=True, exist_ok=True)
        parsed_task_type = ScientificTaskType(task_type)
        if parsed_task_type not in pack.supported_task_types():
            raise ValueError(f"domain {domain_id} does not support task type {parsed_task_type.value}")
        if domain_id == "superconductivity_lsco" and mode in {"fixture", "existing", "live"}:
            from coscientist.superconductivity import run_phase2_acquisition

            delegated = run_phase2_acquisition(
                mode=mode,  # type: ignore[arg-type]
                live_network=live_network,
                runs_dir=runs_dir,
                run_id=run_name,
            )
            summary = read_json(Path(delegated) / "acquisition_summary.json")
            result = GenericAcquisitionResult(
                domain_id=domain_id,
                run_dir=str(delegated),
                mode=mode,
                status=str(summary.get("status")),
                task_type=parsed_task_type.value,
                query_count=int(summary.get("query_count", 0)),
                normalized_record_count=int(summary.get("candidate_rows_staged", 0)),
                validation_error_count=0,
                delegated_to="phase2_lsco_acquisition",
                artifact_ids=["acquisition_summary.json", "paper_records.jsonl", "candidate_rows.jsonl", "readiness_gates.json"],
            )
            write_json(Path(delegated) / "generic_acquisition_result.json", result)
            return result
        objective = ResearchObjective(
            objective_id=f"objective-{domain_id}",
            question=question or f"fixture acquisition for {domain_id}",
            task_type=parsed_task_type,
        )
        queries = pack.search_queries(objective)
        fixture_sources = [
            SourceRecord(
                source_id=f"{domain_id}-fixture-source",
                title=f"{domain_id} fixture source",
                source_type="fixture",
                text=question,
                metadata={"mode": mode},
            )
        ]
        classifications = [pack.classify_source(source) for source in fixture_sources]
        records = [record for source in fixture_sources for record in pack.normalize_records(source)]
        validations = [pack.validate_record(record) for record in records]
        status = "fixture_supported" if all(item.status == "valid" for item in validations) else "needs_review"
        if mode == "live" and not live_network:
            status = "blocked_live_network_permission_required"
        write_json(root / "domain_pack_manifest.json", _pack_manifest(pack))
        write_jsonl(root / "search_queries.jsonl", queries)
        write_jsonl(root / "source_classifications.jsonl", classifications)
        write_jsonl(root / "normalized_records.jsonl", records)
        write_jsonl(root / "record_validations.jsonl", validations)
        write_json(root / "readiness_gates.json", {"schema_version": "v26", "domain_id": domain_id, "gates": [gate.model_dump() for gate in pack.readiness_gates()]})
        result = GenericAcquisitionResult(
            domain_id=domain_id,
            run_dir=str(root),
            mode=mode,
            status=status,
            task_type=parsed_task_type.value,
            query_count=len(queries),
            normalized_record_count=len(records),
            validation_error_count=sum(1 for item in validations if item.status != "valid"),
            artifact_ids=[
                "domain_pack_manifest.json",
                "search_queries.jsonl",
                "source_classifications.jsonl",
                "normalized_records.jsonl",
                "record_validations.jsonl",
                "readiness_gates.json",
                "generic_acquisition_result.json",
            ],
        )
        write_json(root / "generic_acquisition_result.json", result)
        write_json(root / "generic_acquisition_summary.json", result.model_dump(mode="json") | {"created_at": datetime.now(UTC).isoformat()})
        return result


def validate_generic_acquisition_run(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    required = ["domain_pack_manifest.json", "generic_acquisition_result.json"]
    return [f"missing generic acquisition artifact: {name}" for name in required if not (root / name).exists()]


def _pack_manifest(pack: DomainPack) -> dict[str, object]:
    return {
        "schema_version": "v26-domain-pack-manifest",
        "domain_id": pack.domain_id,
        "version": pack.version,
        "supported_task_types": sorted(item.value for item in pack.supported_task_types()),
        "tools": [item.model_dump() for item in pack.tools()],
        "benchmark_cases": [item.model_dump() for item in pack.benchmark_cases()],
        "guardrails": pack.guardrails(),
    }
