from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from coscientist.config import LiteratureConfig
from coscientist.literature.pipeline import build_literature_pipeline
from coscientist.literature.providers.base import ProviderConfigurationError
from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.schemas.literature import MetadataResolveRequest, Paper, SearchQuery
from coscientist.superconductivity.phase2_data import run_phase2_data_coverage_tool


ACQUISITION_SCHEMA_VERSION = "v25-phase2-acquisition"
DEFAULT_CANONICAL_DATASET = Path("data/phase2_lsco.csv")
DEFAULT_STAGING_DIR = Path("data/staging")

ObservableName = Literal[
    "tc_k",
    "gap_ev",
    "isotope_alpha",
    "penetration_depth_nm",
    "superfluid_density_proxy",
    "optical_spectral_weight_proxy",
    "optical_s_delta_over_sn",
    "optical_su_over_sn",
]

AcquisitionMode = Literal["fixture", "existing", "live"]
CandidateStatus = Literal["DISCOVERED", "EXTRACTED", "NORMALIZED", "NEEDS_REVIEW", "VALIDATED", "CONFLICT", "REJECTED", "PROMOTED", "SUPERSEDED"]


class ObservableDefinition(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    observable: str
    canonical_unit: str
    synonyms: list[str]
    measurement_methods: list[str] = Field(default_factory=list)
    definition: str = ""
    material_claim_requires_definition: bool = False


class SampleIdentity(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    sample_id: str
    material: str = "La2-xSrxCuO4"
    nominal_doping_x: float | None = None
    measured_doping_x: float | None = None
    doping_definition: Literal["nominal", "measured", "inferred", "unknown"] = "unknown"
    sample_form: str = "unknown"
    growth_method: str = "unknown"
    paper_id: str
    evidence_text: str = ""


class AcquisitionTask(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    task_id: str = "phase2-lsco-acquisition"
    campaign_id: str = "lsco_phase2"
    material_family: str = "La2-xSrxCuO4"
    aliases: list[str] = Field(default_factory=lambda: ["LSCO", "La2-xSrxCuO4", "La_{2-x}Sr_xCuO_4", "La1.85Sr0.15CuO4"])
    observable_targets: list[ObservableName] = Field(default_factory=lambda: [
        "tc_k",
        "gap_ev",
        "isotope_alpha",
        "penetration_depth_nm",
        "superfluid_density_proxy",
        "optical_spectral_weight_proxy",
        "optical_s_delta_over_sn",
        "optical_su_over_sn",
    ])
    doping_range: dict[str, float] = Field(default_factory=lambda: {"x_min": 0.0, "x_max": 0.35})
    allowed_sources: list[str] = Field(default_factory=lambda: ["arxiv", "openalex", "crossref", "unpaywall"])
    network_permission_required: bool = True
    canonical_dataset: str = str(DEFAULT_CANONICAL_DATASET)
    mode: AcquisitionMode = "fixture"
    live_network: bool = False
    max_results_per_query: int = 5
    max_queries: int = 8
    auto_promote: bool = False
    runs_dir: str = "runs"
    run_id: str = "phase2-lsco-acquisition"


class AcquisitionTaskHandle(BaseModel):
    task_id: str
    run_dir: str
    status: str
    executor: str


class AcquisitionTaskStatus(BaseModel):
    task_id: str
    status: str
    current_stage: str
    message: str = ""


class PaperRecord(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    journal: str | None = None
    abstract: str | None = None
    landing_page_url: str | None = None
    open_access_url: str | None = None
    pdf_url: str | None = None
    source_connectors: list[str] = Field(default_factory=list)
    is_open_access: bool = False
    retrieval_status: str = "metadata_only"
    checksum: str | None = None


class PaperClassification(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    paper_id: str
    relevance: Literal["relevant", "maybe_relevant", "not_relevant"]
    observable_labels: list[str] = Field(default_factory=list)
    data_location_labels: list[str] = Field(default_factory=list)
    sample_type: str = "unknown"
    doping_coverage: list[str] = Field(default_factory=list)
    extraction_difficulty: Literal["low", "medium", "high"] = "medium"
    source_reliability: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


class ExtractionRecord(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    extraction_id: str
    paper_id: str
    observable: str
    raw_value: str
    raw_unit: str
    normalized_value: float | None = None
    normalized_unit: str = ""
    doping_raw: str = ""
    doping_x: float | None = None
    sample_id: str | None = None
    sample_type: str = "unknown"
    measurement_temperature_k: float | None = None
    uncertainty_raw: str = ""
    uncertainty_normalized: float | None = None
    page: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    caption: str | None = None
    evidence_text: str = ""
    extraction_method: str = "deterministic_fixture"
    extractor_version: str = ACQUISITION_SCHEMA_VERSION
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    observable_definition: str = ""
    doping_definition: Literal["nominal", "measured", "inferred", "unknown"] = "unknown"
    measurement_method: str = "unknown"


class CandidateDataRow(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    candidate_row_id: str
    material_family: str = "cuprate"
    material_id: str = "LSCO"
    doping: str
    doping_x: float | None = None
    observable: str
    value: float | str
    unit: str
    uncertainty: str | None = None
    split: str = "train"
    source_id: str
    provenance: str
    usable_for_fit: bool = True
    source_url: str | None = None
    curation_note: str = ""
    paper_id: str
    extraction_id: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    status: CandidateStatus = "NORMALIZED"
    review_status: Literal["not_reviewed", "approved", "rejected"] = "not_reviewed"
    validation_errors: list[str] = Field(default_factory=list)
    sample_id: str | None = None
    measurement_method: str = "unknown"
    observable_definition: str = ""
    doping_definition: Literal["nominal", "measured", "inferred", "unknown"] = "unknown"


class DigitizationTask(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    digitization_task_id: str
    paper_id: str
    figure_id: str
    panel_id: str = ""
    page: str | None = None
    observable: str
    x_axis_name: str = "doping"
    x_axis_unit: str = "x"
    y_axis_name: str
    y_axis_unit: str
    series_label: str = ""
    required_doping_points: list[float] = Field(default_factory=list)
    figure_image_artifact_id: str = ""
    status: Literal["QUEUED", "READY", "IN_PROGRESS", "DIGITIZED", "NEEDS_REVIEW", "APPROVED", "REJECTED"] = "QUEUED"
    assigned_to: str = ""
    digitization_method: str = "manual_required"
    estimated_uncertainty: float | None = None
    review_status: str = "not_reviewed"


class PromotionDecision(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    candidate_row_id: str
    decision: Literal["promoted", "rejected", "needs_review", "duplicate", "conflict"]
    reason: str
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CandidateReviewDecision(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    candidate_row_id: str
    decision: Literal["approve", "reject", "edit"]
    reviewer: str = "local-human"
    rationale: str = ""
    edited_value: float | str | None = None
    edited_unit: str | None = None
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ParserFallbackRecord(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    paper_id: str
    source_kind: str
    parser: str
    status: Literal["parsed", "figure_only", "unsupported", "failed", "no_values_found"]
    extracted_count: int = 0
    figure_tasks_created: int = 0
    message: str = ""


class DataClaimRecord(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    claim_id: str
    claim_text: str
    claim_type: str
    status: Literal["resolved", "blocked", "needs_review"]
    depends_on: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    candidate_row_id: str | None = None


class ReadinessGateResult(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    status: Literal[
        "blocked_insufficient_existing_data",
        "blocked_missing_optical_data",
        "blocked_missing_overlap",
        "ready_for_exploratory_comparison",
        "ready_for_held_out_comparison",
        "comparison_complete",
        "comparison_inconclusive",
    ]
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    available_any_source: dict[str, int] = Field(default_factory=dict)
    available_same_doping: list[str] = Field(default_factory=list)
    available_same_sample: list[str] = Field(default_factory=list)
    same_paper_overlap: list[str] = Field(default_factory=list)
    explanation: str = ""


class ComparisonRobustnessReport(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    status: Literal["not_run_blocked", "ready_not_run", "comparison_inconclusive"]
    holdout_strategy: str = "paper_or_sample_grouped"
    candidate_models: list[str] = Field(default_factory=lambda: ["pure_phonon", "pure_correlated_hopping", "mixed"])
    metrics: list[str] = Field(default_factory=lambda: ["per_observable_mae", "weighted_residual", "macro_score", "model_complexity_penalty", "bootstrap_uncertainty", "parameter_identifiability"])
    leakage_checks: list[str] = Field(default_factory=list)
    reason: str = ""


class AcquisitionResult(BaseModel):
    schema_version: str = ACQUISITION_SCHEMA_VERSION
    task_id: str
    run_dir: str
    status: str
    summary: dict[str, Any]
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class AcquisitionExecutor(Protocol):
    async def submit(self, task: AcquisitionTask) -> AcquisitionTaskHandle:
        ...

    async def get_status(self, task_id: str) -> AcquisitionTaskStatus:
        ...

    async def get_result(self, task_id: str) -> AcquisitionResult:
        ...


class MockAcquisitionExecutor:
    def __init__(self) -> None:
        self._results: dict[str, AcquisitionResult] = {}
        self._statuses: dict[str, AcquisitionTaskStatus] = {}

    async def submit(self, task: AcquisitionTask) -> AcquisitionTaskHandle:
        task = task.model_copy(update={"mode": "fixture", "live_network": False})
        result = await _run_acquisition(task, executor_name="mock")
        self._results[task.task_id] = result
        self._statuses[task.task_id] = AcquisitionTaskStatus(task_id=task.task_id, status=result.status, current_stage="complete")
        return AcquisitionTaskHandle(task_id=task.task_id, run_dir=result.run_dir, status=result.status, executor="mock")

    async def get_status(self, task_id: str) -> AcquisitionTaskStatus:
        return self._statuses[task_id]

    async def get_result(self, task_id: str) -> AcquisitionResult:
        return self._results[task_id]


class LocalAcquisitionExecutor:
    def __init__(self) -> None:
        self._results: dict[str, AcquisitionResult] = {}
        self._statuses: dict[str, AcquisitionTaskStatus] = {}

    async def submit(self, task: AcquisitionTask) -> AcquisitionTaskHandle:
        self._statuses[task.task_id] = AcquisitionTaskStatus(task_id=task.task_id, status="running", current_stage="submitted")
        result = await _run_acquisition(task, executor_name="local")
        self._results[task.task_id] = result
        self._statuses[task.task_id] = AcquisitionTaskStatus(task_id=task.task_id, status=result.status, current_stage="complete")
        return AcquisitionTaskHandle(task_id=task.task_id, run_dir=result.run_dir, status=result.status, executor="local")

    async def get_status(self, task_id: str) -> AcquisitionTaskStatus:
        return self._statuses[task_id]

    async def get_result(self, task_id: str) -> AcquisitionResult:
        return self._results[task_id]


class RemoteAcquisitionExecutor:
    """Extension point for a future cloud/container worker."""

    async def submit(self, task: AcquisitionTask) -> AcquisitionTaskHandle:
        raise NotImplementedError("RemoteAcquisitionExecutor is an extension point; configure a worker endpoint before use.")

    async def get_status(self, task_id: str) -> AcquisitionTaskStatus:
        raise NotImplementedError("RemoteAcquisitionExecutor is an extension point; configure a worker endpoint before use.")

    async def get_result(self, task_id: str) -> AcquisitionResult:
        raise NotImplementedError("RemoteAcquisitionExecutor is an extension point; configure a worker endpoint before use.")


class Phase2DataAcquisitionAgent:
    agent_name = "Phase2DataAcquisitionAgent"

    def __init__(self, executor: AcquisitionExecutor | None = None) -> None:
        self.executor = executor or LocalAcquisitionExecutor()

    async def run(self, task: AcquisitionTask) -> AcquisitionResult:
        handle = await self.executor.submit(task)
        return await self.executor.get_result(handle.task_id)


def build_default_lsco_task(
    *,
    mode: AcquisitionMode = "fixture",
    live_network: bool = False,
    runs_dir: str | Path = "runs",
    run_id: str = "phase2-lsco-acquisition",
    canonical_dataset: str | Path = DEFAULT_CANONICAL_DATASET,
    auto_promote: bool = False,
    max_queries: int = 8,
    max_results_per_query: int = 5,
) -> AcquisitionTask:
    return AcquisitionTask(
        task_id=run_id,
        mode=mode,
        live_network=live_network,
        runs_dir=str(runs_dir),
        run_id=run_id,
        canonical_dataset=str(canonical_dataset),
        auto_promote=auto_promote,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
    )


def run_phase2_acquisition(
    *,
    mode: AcquisitionMode = "fixture",
    live_network: bool = False,
    runs_dir: str | Path = "runs",
    run_id: str = "phase2-lsco-acquisition",
    canonical_dataset: str | Path = DEFAULT_CANONICAL_DATASET,
    auto_promote: bool = False,
    max_queries: int = 8,
    max_results_per_query: int = 5,
) -> Path:
    task = build_default_lsco_task(
        mode=mode,
        live_network=live_network,
        runs_dir=runs_dir,
        run_id=run_id,
        canonical_dataset=canonical_dataset,
        auto_promote=auto_promote,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
    )
    result = asyncio.run(Phase2DataAcquisitionAgent().run(task))
    return Path(result.run_dir)


def validate_phase2_acquisition_run(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    required = [
        "search_queries.json",
        "search_results.json",
        "paper_registry.json",
        "deduplication_report.json",
        "full_text_retrieval_report.json",
        "paper_classification.json",
        "extraction_records.jsonl",
        "candidate_rows.jsonl",
        "normalization_report.json",
        "conflict_report.json",
        "promotion_report.json",
        "digitization_queue.json",
        "coverage_before.json",
        "coverage_after.json",
    "canonical_dataset_diff.json",
    "comparison_trigger.json",
        "readiness_gates.json",
        "comparison_robustness.json",
        "data_claims.jsonl",
        "parser_fallback_report.json",
        "observable_ontology.json",
        "run_summary.md",
        "acquisition_summary.json",
    ]
    errors = [f"missing phase2 acquisition artifact: {name}" for name in required if not (root / name).exists()]
    for name in required:
        path = root / name
        if not path.exists() or path.suffix not in {".json", ".jsonl"}:
            continue
        try:
            if path.suffix == ".json":
                read_json(path)
            else:
                read_jsonl(path)
        except Exception as exc:
            errors.append(f"invalid artifact {name}: {exc}")
    return errors


def import_digitized_points(run_dir: str | Path, *, task_id: str, csv_path: str | Path, reviewer: str = "local-human") -> Path:
    root = Path(run_dir)
    rows = list(csv.DictReader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
    imported = []
    for index, row in enumerate(rows, start=1):
        imported.append({
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "digitization_task_id": task_id,
            "imported_point_id": f"{task_id}-point-{index}",
            "doping_x": _maybe_float(row.get("doping_x")),
            "observable_value": _maybe_float(row.get("observable_value")),
            "x_uncertainty": _maybe_float(row.get("x_uncertainty")),
            "y_uncertainty": _maybe_float(row.get("y_uncertainty")),
            "series_label": row.get("series_label") or "",
            "digitization_method": row.get("digitization_method") or "manual",
            "reviewer": row.get("reviewer") or reviewer,
            "source_figure": row.get("source_figure") or "",
            "review_status": "needs_review",
        })
    output = root / "digitized_points_imported.jsonl"
    write_jsonl(output, imported)
    return output


def phase2_acquisition_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "acquisition_summary.json"
    return read_json(path) if path.exists() else {}


def phase2_candidate_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "candidate_rows.jsonl"
    return read_jsonl(path) if path.exists() else []


def phase2_candidate_sources(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "paper_registry.json"
    return read_json(path).get("papers", []) if path.exists() else []


def phase2_digitization_queue(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "digitization_queue.json"
    return read_json(path).get("tasks", []) if path.exists() else []


def phase2_acquisition_gaps(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "coverage_after_tool" / "phase2_missing_observables.jsonl"
    return read_jsonl(path) if path.exists() else []


def phase2_readiness(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "comparison_trigger.json"
    return read_json(path) if path.exists() else {}


def observable_ontology() -> dict[str, ObservableDefinition]:
    definitions = [
        ObservableDefinition(observable="tc_k", canonical_unit="K", synonyms=["Tc", "transition temperature", "critical temperature"], measurement_methods=["resistivity", "susceptibility", "mutual_inductance"], definition="Superconducting transition temperature."),
        ObservableDefinition(observable="gap_ev", canonical_unit="eV", synonyms=["gap", "Delta", "superconducting gap", "order parameter"], measurement_methods=["ARPES", "STS", "tunneling"], definition="Superconducting gap; distinguish from pseudogap before material-level use.", material_claim_requires_definition=True),
        ObservableDefinition(observable="isotope_alpha", canonical_unit="dimensionless", synonyms=["alpha", "isotope exponent", "oxygen isotope"], measurement_methods=["isotope_substitution"], definition="Isotope coefficient, usually alpha = -d ln Tc / d ln M."),
        ObservableDefinition(observable="penetration_depth_nm", canonical_unit="nm", synonyms=["lambda", "penetration depth", "lambda_ab"], measurement_methods=["mutual_inductance", "muSR"], definition="Magnetic penetration depth, usually lambda_ab(0).", material_claim_requires_definition=True),
        ObservableDefinition(observable="superfluid_density_proxy", canonical_unit="um^-2", synonyms=["lambda^-2", "superfluid density"], measurement_methods=["mutual_inductance", "THz"], definition="Proxy proportional to superfluid density."),
        ObservableDefinition(observable="optical_spectral_weight_proxy", canonical_unit="relative", synonyms=["spectral weight", "missing area", "optical sum rule"], measurement_methods=["optical_conductivity", "TDTS"], definition="Optical spectral-weight quantity; normalization must be explicit.", material_claim_requires_definition=True),
        ObservableDefinition(observable="optical_s_delta_over_sn", canonical_unit="dimensionless", synonyms=["S_delta/Sn", "S_delta / S_n"], measurement_methods=["TDTS", "mutual_inductance"], definition="Condensed superfluid spectral weight normalized by normal-state spectral weight.", material_claim_requires_definition=True),
        ObservableDefinition(observable="optical_su_over_sn", canonical_unit="dimensionless", synonyms=["Su/Sn", "S_u / S_n"], measurement_methods=["TDTS"], definition="Uncondensed low-frequency spectral weight normalized by normal-state spectral weight.", material_claim_requires_definition=True),
    ]
    return {item.observable: item for item in definitions}


def parse_extraction_source_text(paper: PaperRecord, text: str, *, source_kind: str = "text") -> tuple[list[ExtractionRecord], list[DigitizationTask], list[ParserFallbackRecord]]:
    cleaned = _strip_tex_markup(text)
    extractions: list[ExtractionRecord] = []
    for line_number, line in enumerate(cleaned.splitlines(), start=1):
        if not _is_lsco_text(line.lower()) and "x=" not in line.lower():
            continue
        extractions.extend(_extract_values_from_line(paper.paper_id, line, line_number=line_number, source_kind=source_kind))
    digitization: list[DigitizationTask] = []
    lower = cleaned.lower()
    figure_only = any(term in lower for term in ["s_delta", "s_u", "su/sn", "sδ", "spectral weight"]) and not any(item.observable.startswith("optical") for item in extractions)
    if figure_only:
        digitization.append(DigitizationTask(
            digitization_task_id=f"digitize-{_digest(paper.paper_id + source_kind)}-optical",
            paper_id=paper.paper_id,
            figure_id=_first_figure_id(cleaned),
            observable="optical_s_delta_over_sn",
            y_axis_name="S_delta / S_n",
            y_axis_unit="dimensionless",
            status="QUEUED",
            series_label="LSCO optical spectral weight",
        ))
    status: Literal["parsed", "figure_only", "unsupported", "failed", "no_values_found"] = "parsed" if extractions else ("figure_only" if digitization else "no_values_found")
    fallback = [ParserFallbackRecord(
        paper_id=paper.paper_id,
        source_kind=source_kind,
        parser=f"{source_kind}_deterministic_value_parser",
        status=status,
        extracted_count=len(extractions),
        figure_tasks_created=len(digitization),
        message="Parsed explicit numeric values only; figure-only values are queued for review." if status != "no_values_found" else "No explicit supported LSCO values found.",
    )]
    return extractions, digitization, fallback


def parse_supplementary_csv(paper: PaperRecord, csv_path: str | Path) -> tuple[list[ExtractionRecord], list[ParserFallbackRecord]]:
    rows = list(csv.DictReader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
    records: list[ExtractionRecord] = []
    for index, row in enumerate(rows, start=1):
        observable = str(row.get("observable") or "").strip()
        raw_value = str(row.get("value") or "").strip()
        raw_unit = str(row.get("unit") or "").strip()
        normalized_value, normalized_unit = normalize_value_unit(observable, raw_value, raw_unit)
        doping = str(row.get("doping") or row.get("doping_x") or "").strip()
        records.append(ExtractionRecord(
            extraction_id=f"extract-{_digest(str(csv_path) + str(index))}",
            paper_id=paper.paper_id,
            observable=observable,
            raw_value=raw_value,
            raw_unit=raw_unit,
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
            doping_raw=doping if doping.startswith("x=") else f"x={doping}",
            doping_x=_parse_doping(doping),
            sample_id=row.get("sample_id") or None,
            sample_type=row.get("sample_type") or "unknown",
            uncertainty_raw=row.get("uncertainty") or "",
            table_id=Path(csv_path).name,
            evidence_text=f"Supplementary CSV row {index}: {row}",
            extraction_method="supplementary_csv_table",
            confidence="HIGH",
            observable_definition=row.get("definition") or "",
            doping_definition="measured" if row.get("measured_doping") else "nominal",
            measurement_method=row.get("measurement_method") or "unknown",
        ))
    return records, [ParserFallbackRecord(paper_id=paper.paper_id, source_kind="supplementary_csv", parser="csv.DictReader", status="parsed" if records else "no_values_found", extracted_count=len(records))]


def parse_supplementary_zip(paper: PaperRecord, zip_path: str | Path, output_dir: str | Path) -> tuple[list[ExtractionRecord], list[ParserFallbackRecord]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[ExtractionRecord] = []
    fallbacks: list[ParserFallbackRecord] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                fallbacks.append(ParserFallbackRecord(paper_id=paper.paper_id, source_kind="supplementary_zip", parser="zipfile", status="unsupported", message=f"Unsupported supplementary member: {member}"))
                continue
            target = root / Path(member).name
            target.write_bytes(archive.read(member))
            parsed, parsed_fallback = parse_supplementary_csv(paper, target)
            records.extend(parsed)
            fallbacks.extend(parsed_fallback)
    return records, fallbacks


def normalize_value_unit(observable: str, raw_value: str, raw_unit: str) -> tuple[float | None, str]:
    value = _maybe_float(str(raw_value).replace("~", "").replace("±", " ").split()[0])
    unit = raw_unit.strip()
    if value is None:
        return None, unit
    if observable == "gap_ev" and unit.lower() in {"mev", "milliev"}:
        return value / 1000.0, "eV"
    if observable == "tc_k" and unit.lower() in {"k", "kelvin"}:
        return value, "K"
    if observable == "penetration_depth_nm" and unit.lower() in {"nm", "nanometer", "nanometers"}:
        return value, "nm"
    if observable == "penetration_depth_nm" and unit.lower() in {"um", "micron", "microns"}:
        return value * 1000.0, "nm"
    if observable == "isotope_alpha":
        return value, "dimensionless"
    if observable.startswith("optical_") or observable in {"superfluid_density_proxy", "optical_spectral_weight_proxy"}:
        return value, unit or "dimensionless"
    return value, unit


def review_candidate_rows(run_dir: str | Path, decisions: list[CandidateReviewDecision]) -> Path:
    root = Path(run_dir)
    rows = [CandidateDataRow.model_validate(item) for item in read_jsonl(root / "candidate_rows.jsonl")]
    by_id = {decision.candidate_row_id: decision for decision in decisions}
    for row in rows:
        decision = by_id.get(row.candidate_row_id)
        if not decision:
            continue
        if decision.decision == "approve":
            row.review_status = "approved"
            row.status = "VALIDATED" if not row.validation_errors else "NEEDS_REVIEW"
        elif decision.decision == "reject":
            row.review_status = "rejected"
            row.status = "REJECTED"
        elif decision.decision == "edit":
            if decision.edited_value is not None:
                row.value = decision.edited_value
            if decision.edited_unit:
                row.unit = decision.edited_unit
            row.review_status = "approved"
            row.curation_note = f"{row.curation_note}; human edited: {decision.rationale}".strip("; ")
    write_jsonl(root / "candidate_rows.jsonl", [row.model_dump(mode="json") for row in rows])
    write_jsonl(root / "review_decisions.jsonl", [decision.model_dump(mode="json") for decision in decisions])
    return root / "review_decisions.jsonl"


def promote_reviewed_candidates(run_dir: str | Path, canonical_dataset: str | Path) -> Path:
    root = Path(run_dir)
    rows = [CandidateDataRow.model_validate(item) for item in read_jsonl(root / "candidate_rows.jsonl")]
    reviewed = [row for row in rows if row.review_status == "approved" and row.status == "VALIDATED"]
    decisions, diff = _promote_rows(reviewed, Path(canonical_dataset), auto_promote=True)
    write_json(root / "reviewed_promotion_report.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "decisions": [item.model_dump(mode="json") for item in decisions], "diff": diff})
    coverage_dir = root / "coverage_after_reviewed_promotion"
    run_phase2_data_coverage_tool(coverage_dir, source_path=canonical_dataset)
    coverage = read_json(coverage_dir / "phase2_data_coverage.json")
    canonical = Path(canonical_dataset)
    canonical_rows = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines())) if canonical.exists() else []
    readiness = evaluate_phase2_readiness_from_rows(canonical_rows)
    trigger = _comparison_trigger(diff, coverage, readiness)
    write_json(root / "readiness_gates.json", readiness.model_dump(mode="json"))
    write_json(root / "comparison_trigger.json", trigger)
    write_json(root / "comparison_robustness.json", _comparison_robustness_report(trigger, readiness).model_dump(mode="json"))
    return root / "reviewed_promotion_report.json"


def evaluate_phase2_readiness_from_rows(rows: list[dict[str, Any]], *, require_held_out: bool = True) -> ReadinessGateResult:
    usable = [row for row in rows if _truthy(row.get("usable_for_fit", True))]
    by_observable: dict[str, list[dict[str, Any]]] = {}
    by_doping: dict[str, set[str]] = {}
    by_sample: dict[str, set[str]] = {}
    by_paper: dict[str, set[str]] = {}
    bad_provenance = []
    ambiguous_definitions = []
    for row in usable:
        observable = str(row.get("observable") or "")
        by_observable.setdefault(observable, []).append(row)
        doping = str(row.get("doping") or "")
        sample = str(row.get("material_id") or row.get("sample_id") or "")
        paper = str(row.get("source_id") or row.get("paper_id") or "")
        by_doping.setdefault(doping, set()).add(observable)
        by_sample.setdefault(sample, set()).add(observable)
        by_paper.setdefault(paper, set()).add(observable)
        if not str(row.get("provenance") or "").strip():
            bad_provenance.append(row.get("observation_id") or row.get("candidate_row_id"))
        if observable in {"gap_ev", "optical_spectral_weight_proxy", "optical_s_delta_over_sn", "optical_su_over_sn"}:
            note = f"{row.get('curation_note', '')} {row.get('observable_definition', '')}".lower()
            if observable.startswith("optical") and not any(term in note for term in ["s_delta", "s_n", "lambda", "sum", "spectral", "proxy"]):
                ambiguous_definitions.append(row.get("observation_id") or row.get("candidate_row_id"))
            if observable == "gap_ev" and "pseudogap" in note and "superconducting" not in note:
                ambiguous_definitions.append(row.get("observation_id") or row.get("candidate_row_id"))
    required = {"tc_k", "gap_ev", "penetration_depth_nm", "isotope_alpha", "optical_spectral_weight_proxy"}
    optical = {"optical_spectral_weight_proxy", "optical_s_delta_over_sn", "optical_su_over_sn", "superfluid_density_proxy"}
    available = {key: len(value) for key, value in by_observable.items()}
    available_required = set(by_observable)
    has_optical = bool(available_required & optical)
    same_doping = sorted(doping for doping, observed in by_doping.items() if required.issubset(observed) or ({"tc_k", "gap_ev", "penetration_depth_nm", "isotope_alpha"}.issubset(observed) and observed & optical))
    same_sample = sorted(sample for sample, observed in by_sample.items() if len(observed & required) >= 4 and observed & optical)
    same_paper = sorted(paper for paper, observed in by_paper.items() if len(observed & required) >= 3)
    passed = []
    failed = []
    if not bad_provenance:
        passed.append("provenance_complete")
    else:
        failed.append("bad_provenance")
    if not ambiguous_definitions:
        passed.append("definitions_usable")
    else:
        failed.append("ambiguous_observable_definitions")
    if required - available_required and not (required - available_required == {"optical_spectral_weight_proxy"} and has_optical):
        failed.append("missing_required_observables")
    else:
        passed.append("required_observables_available")
    if has_optical:
        passed.append("optical_available")
    else:
        failed.append("missing_optical_data")
    if same_doping:
        passed.append("same_doping_overlap")
    else:
        failed.append("missing_same_doping_overlap")
    split_counts = {split: sum(1 for row in usable if str(row.get("split") or "train") == split) for split in ["train", "validation", "test"]}
    held_out = split_counts["validation"] + split_counts["test"] > 0
    if held_out:
        passed.append("held_out_available")
    elif require_held_out:
        failed.append("held_out_split_impossible")
    if bad_provenance or ambiguous_definitions:
        status = "blocked_insufficient_existing_data"
        explanation = "Rows exist, but provenance or observable definitions are not reliable enough for comparison."
    elif not has_optical:
        status = "blocked_missing_optical_data"
        explanation = "No usable optical or superfluid spectral-weight observable is available."
    elif not same_doping:
        status = "blocked_missing_overlap"
        explanation = "Observables exist but do not overlap at the same doping point."
    elif not held_out:
        status = "ready_for_exploratory_comparison"
        explanation = "Same-doping overlap exists, but held-out split is unavailable."
    elif len(same_doping) >= 3:
        status = "ready_for_held_out_comparison"
        explanation = "At least three overlapping doping points and held-out data are available."
    else:
        status = "ready_for_exploratory_comparison"
        explanation = "Overlap exists but data volume is too small for held-out material-level claims."
    return ReadinessGateResult(
        status=status,  # type: ignore[arg-type]
        passed_gates=passed,
        failed_gates=failed,
        available_any_source=available,
        available_same_doping=same_doping,
        available_same_sample=same_sample,
        same_paper_overlap=same_paper,
        explanation=explanation,
    )


def build_lsco_queries(task: AcquisitionTask) -> list[dict[str, Any]]:
    observable_terms: dict[str, list[str]] = {
        "tc_k": ["superconducting transition temperature", "Tc", "critical temperature"],
        "gap_ev": ["ARPES superconducting gap", "gap doping", "d-wave order parameter"],
        "isotope_alpha": ["oxygen isotope exponent", "isotope effect"],
        "penetration_depth_nm": ["penetration depth", "lambda inverse squared", "superfluid density"],
        "superfluid_density_proxy": ["superfluid density", "lambda^-2", "mutual inductance"],
        "optical_spectral_weight_proxy": ["optical spectral weight", "missing area optical conductivity", "optical sum rule"],
        "optical_s_delta_over_sn": ["S_delta Sn", "superfluid spectral weight"],
        "optical_su_over_sn": ["Su Sn", "uncondensed spectral weight"],
    }
    queries: list[dict[str, Any]] = []
    aliases = task.aliases[:3]
    for observable in task.observable_targets:
        for alias in aliases[:2]:
            terms = observable_terms.get(observable, [observable.replace("_", " ")])
            queries.append({
                "schema_version": ACQUISITION_SCHEMA_VERSION,
                "query_id": f"query-{len(queries) + 1:03d}",
                "query": f"{alias} {terms[0]} doping",
                "observable": observable,
                "aliases": [alias],
                "connectors": task.allowed_sources,
            })
            if len(queries) >= task.max_queries:
                return queries
    return queries


async def _run_acquisition(task: AcquisitionTask, *, executor_name: str) -> AcquisitionResult:
    run_dir = Path(task.runs_dir) / task.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    coverage_before_dir = run_dir / "coverage_before_tool"
    coverage_after_dir = run_dir / "coverage_after_tool"
    canonical = Path(task.canonical_dataset)
    run_phase2_data_coverage_tool(coverage_before_dir, source_path=canonical)
    coverage_before = read_json(coverage_before_dir / "phase2_data_coverage.json")
    queries = build_lsco_queries(task)

    blocked = task.mode == "live" and not task.live_network
    if blocked:
        papers: list[PaperRecord] = []
        raw_search_results: list[dict[str, Any]] = []
        classifications: list[PaperClassification] = []
        extractions: list[ExtractionRecord] = []
        digitization_tasks: list[DigitizationTask] = []
    elif task.mode == "fixture":
        papers, raw_search_results = _fixture_search_results(task, queries)
        classifications = [_classify_paper(paper) for paper in papers]
        extractions, digitization_tasks = _fixture_extractions(papers)
        parser_fallbacks = [ParserFallbackRecord(paper_id=paper.paper_id, source_kind="fixture", parser="fixture_extractor", status="parsed" if paper.paper_id != "paper-fixture-lsco-optical-figure" else "figure_only", extracted_count=4 if paper.paper_id != "paper-fixture-lsco-optical-figure" else 0, figure_tasks_created=1 if paper.paper_id == "paper-fixture-lsco-optical-figure" else 0) for paper in papers]
    elif task.mode == "existing":
        papers, raw_search_results = _existing_source_records(canonical)
        classifications = [_classify_paper(paper) for paper in papers]
        extractions, digitization_tasks = [], []
        parser_fallbacks = []
    else:
        papers, raw_search_results = await _live_search(task, queries)
        classifications = [_classify_paper(paper) for paper in papers]
        extractions, digitization_tasks = _extract_from_metadata(papers, classifications)
        parser_fallbacks = [ParserFallbackRecord(paper_id=paper.paper_id, source_kind="live_metadata", parser="metadata_classifier", status="figure_only" if any(task.paper_id == paper.paper_id for task in digitization_tasks) else "no_values_found", figure_tasks_created=sum(1 for task in digitization_tasks if task.paper_id == paper.paper_id)) for paper in papers]
    if blocked:
        parser_fallbacks = []

    papers, dedupe_report = _deduplicate_papers(papers)
    candidate_rows = _candidate_rows_from_extractions(extractions, papers)
    candidate_rows, normalization_report = _validate_candidate_rows(candidate_rows)
    conflicts = _detect_conflicts(candidate_rows, canonical)
    for row in candidate_rows:
        if any(item["candidate_row_id"] == row.candidate_row_id for item in conflicts):
            row.status = "CONFLICT"
            row.validation_errors.append("conflict_or_duplicate_detected")
    promotion_decisions, canonical_diff = _promote_rows(candidate_rows, canonical, auto_promote=task.auto_promote)
    run_phase2_data_coverage_tool(coverage_after_dir, source_path=canonical)
    coverage_after = read_json(coverage_after_dir / "phase2_data_coverage.json")
    canonical_rows = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines())) if canonical.exists() else []
    readiness_gate = evaluate_phase2_readiness_from_rows(canonical_rows)
    comparison_trigger = _comparison_trigger(canonical_diff, coverage_after, readiness_gate)
    robustness = _comparison_robustness_report(comparison_trigger, readiness_gate)
    data_claims = _data_claims(papers, extractions, candidate_rows, readiness_gate, comparison_trigger)

    summary = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "task_id": task.task_id,
        "mode": task.mode,
        "executor": executor_name,
        "queries_executed": 0 if blocked else len(queries),
        "papers_found": sum(int(item.get("result_count", 0)) for item in raw_search_results),
        "papers_deduplicated": len(papers),
        "papers_relevant": sum(1 for item in classifications if item.relevance != "not_relevant"),
        "full_texts_retrieved": sum(int(item.get("full_text_location_count", 0)) for item in raw_search_results),
        "text_values_extracted": sum(1 for item in extractions if item.extraction_method in {"deterministic_fixture_text", "metadata_text"}),
        "table_values_extracted": sum(1 for item in extractions if "table" in item.extraction_method),
        "figure_digitization_tasks_created": len(digitization_tasks),
        "candidate_rows_staged": len(candidate_rows),
        "rows_promoted": sum(1 for item in promotion_decisions if item.decision == "promoted"),
        "rows_rejected": sum(1 for item in promotion_decisions if item.decision in {"rejected", "needs_review", "conflict", "duplicate"}),
        "conflicts_detected": len(conflicts),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "comparison_rerun": comparison_trigger["comparison_rerun"],
        "final_phase2_status": comparison_trigger["phase2_status"],
        "status": "blocked_live_network_permission_required" if blocked else "completed",
    }
    artifacts = _write_acquisition_artifacts(
        run_dir,
        task=task,
        queries=queries,
        search_results=raw_search_results,
        papers=papers,
        dedupe_report=dedupe_report,
        classifications=classifications,
        extractions=extractions,
        candidate_rows=candidate_rows,
        normalization_report=normalization_report,
        conflicts=conflicts,
        promotion_decisions=promotion_decisions,
        digitization_tasks=digitization_tasks,
        parser_fallbacks=parser_fallbacks,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        canonical_diff=canonical_diff,
        comparison_trigger=comparison_trigger,
        readiness_gate=readiness_gate,
        robustness=robustness,
        data_claims=data_claims,
        summary=summary,
        blocked=blocked,
    )
    status = "blocked_live_network_permission_required" if blocked else "completed"
    return AcquisitionResult(task_id=task.task_id, run_dir=str(run_dir), status=status, summary=summary, artifact_paths=artifacts)


def _fixture_search_results(task: AcquisitionTask, queries: list[dict[str, Any]]) -> tuple[list[PaperRecord], list[dict[str, Any]]]:
    papers = [
        PaperRecord(
            paper_id="paper-fixture-lsco-tc-gap",
            title="Fixture LSCO transition temperature and superconducting gap table",
            authors=["Curated Fixture"],
            year=2026,
            doi="10.0000/fixture.lsco.tc-gap",
            journal="Fixture Journal",
            abstract="La2-xSrxCuO4 table reports Tc and superconducting gap values at x=0.15.",
            landing_page_url="fixture://lsco/tc-gap",
            source_connectors=["fixture"],
            is_open_access=True,
            retrieval_status="fixture_full_text_available",
        ),
        PaperRecord(
            paper_id="paper-fixture-lsco-isotope-lambda",
            title="Fixture LSCO isotope exponent and penetration depth table",
            authors=["Curated Fixture"],
            year=2026,
            doi="10.0000/fixture.lsco.isotope-lambda",
            abstract="LSCO oxygen isotope exponent and penetration depth are tabulated for x=0.15.",
            landing_page_url="fixture://lsco/isotope-lambda",
            source_connectors=["fixture"],
            is_open_access=True,
            retrieval_status="fixture_full_text_available",
        ),
        PaperRecord(
            paper_id="paper-fixture-lsco-optical-figure",
            title="Fixture LSCO optical spectral weight figure",
            authors=["Curated Fixture"],
            year=2026,
            doi="10.0000/fixture.lsco.optical",
            abstract="Optical spectral weight appears only in Figure 2e as S_delta/Sn and Su/Sn.",
            landing_page_url="fixture://lsco/optical",
            source_connectors=["fixture"],
            is_open_access=True,
            retrieval_status="figure_only",
        ),
    ]
    results = [
        {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "query_id": query["query_id"],
            "query": query["query"],
            "provider": "fixture",
            "result_count": len(papers),
            "paper_ids": [paper.paper_id for paper in papers],
        }
        for query in queries
    ]
    return papers, results


def _existing_source_records(canonical: Path) -> tuple[list[PaperRecord], list[dict[str, Any]]]:
    if not canonical.exists():
        return [], []
    rows = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines()))
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row.get("source_id") or "unknown", []).append(row)
    papers = []
    for source_id, source_rows in sorted(by_source.items()):
        url = next((row.get("source_url") for row in source_rows if row.get("source_url")), None)
        title = source_id.replace("-", " ")
        papers.append(PaperRecord(
            paper_id=f"paper-existing-{_digest(source_id)}",
            title=title,
            landing_page_url=url,
            source_connectors=["existing"],
            retrieval_status="registered_source",
            checksum=_digest(json.dumps(source_rows, sort_keys=True)),
        ))
    return papers, [{"schema_version": ACQUISITION_SCHEMA_VERSION, "provider": "existing", "result_count": len(papers), "paper_ids": [p.paper_id for p in papers]}]


async def _live_search(task: AcquisitionTask, queries: list[dict[str, Any]]) -> tuple[list[PaperRecord], list[dict[str, Any]]]:
    literature = LiteratureConfig(
        enabled=True,
        search_providers=[name for name in task.allowed_sources if name in {"arxiv", "openalex"}],
        metadata_resolvers=[name for name in task.allowed_sources if name == "crossref"],
        full_text_locators=[name for name in task.allowed_sources if name in {"unpaywall", "arxiv"}],
        allow_live_network=task.live_network,
        max_results_per_provider=task.max_results_per_query,
    )
    try:
        pipeline = build_literature_pipeline(literature)
    except ProviderConfigurationError:
        literature = literature.model_copy(update={"full_text_locators": ["arxiv"]})
        pipeline = build_literature_pipeline(literature)
    papers: list[PaperRecord] = []
    search_results: list[dict[str, Any]] = []
    for query in queries:
        try:
            result = await pipeline.acquire(query["query"])
        except Exception as exc:
            search_results.append({"schema_version": ACQUISITION_SCHEMA_VERSION, "query_id": query["query_id"], "provider": "literature_pipeline", "status": "failed", "error": str(exc), "result_count": 0})
            continue
        search_results.append({
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "query_id": query["query_id"],
            "provider": "literature_pipeline",
            "status": "ok",
            "result_count": len(result.normalized_papers),
            "raw_result_count": len(result.raw_papers),
            "metadata_resolution_count": len(result.metadata_resolutions),
            "full_text_location_count": len(result.full_text_locations),
            "provider_request_count": len(result.provider_requests),
            "paper_ids": [paper.id for paper in result.normalized_papers],
        })
        locations_by_paper: dict[str, list[Any]] = {}
        for location in result.full_text_locations:
            locations_by_paper.setdefault(location.paper_id, []).append(location)
        for paper in result.normalized_papers:
            record = _paper_record_from_paper(paper)
            locations = locations_by_paper.get(paper.id, [])
            best = next((location for location in locations if getattr(location, "is_best", False)), locations[0] if locations else None)
            if best:
                record.open_access_url = best.landing_page_url
                record.pdf_url = best.document_url
                record.is_open_access = best.access_status == "open_location_found"
                record.retrieval_status = best.access_status
            papers.append(record)
    return papers, search_results


def _paper_record_from_paper(paper: Paper) -> PaperRecord:
    arxiv_id = None
    openalex_id = None
    for identifier in paper.identifiers:
        if identifier.scheme == "arxiv":
            arxiv_id = identifier.canonical_value
        if identifier.scheme == "openalex":
            openalex_id = identifier.canonical_value
    oa = paper.open_access or {}
    pdf = None
    landing = None
    if isinstance(oa, dict):
        pdf = oa.get("oa_url")
        landing = oa.get("oa_url")
    return PaperRecord(
        paper_id=paper.id,
        title=paper.title,
        authors=[author.name for author in paper.authors],
        year=paper.publication_year,
        doi=paper.doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        journal=paper.venue,
        abstract=paper.abstract,
        landing_page_url=landing,
        open_access_url=landing,
        pdf_url=pdf,
        source_connectors=[paper.source_provider],
        is_open_access=bool(oa.get("is_oa")) if isinstance(oa, dict) else False,
        retrieval_status="metadata_only",
        checksum=_digest(paper.model_dump_json()),
    )


def _deduplicate_papers(papers: list[PaperRecord]) -> tuple[list[PaperRecord], dict[str, Any]]:
    seen: dict[str, PaperRecord] = {}
    duplicates = []
    for paper in papers:
        key = paper.doi or paper.arxiv_id or paper.openalex_id or f"{_normalize_title(paper.title)}:{paper.year or ''}"
        if key in seen:
            existing = seen[key]
            existing.source_connectors = sorted(set(existing.source_connectors + paper.source_connectors))
            duplicates.append({"duplicate_paper_id": paper.paper_id, "kept_paper_id": existing.paper_id, "dedupe_key": key})
        else:
            seen[key] = paper
    return list(seen.values()), {"schema_version": ACQUISITION_SCHEMA_VERSION, "input_count": len(papers), "deduplicated_count": len(seen), "duplicates": duplicates}


def _classify_paper(paper: PaperRecord) -> PaperClassification:
    text = f"{paper.title} {paper.abstract or ''}".lower()
    labels = []
    if any(term in text for term in ["tc", "transition temperature", "critical temperature"]):
        labels.append("TC")
    if any(term in text for term in ["gap", "arpes", "order parameter"]):
        labels.append("GAP")
    if "isotope" in text:
        labels.append("ISOTOPE")
    if any(term in text for term in ["penetration", "lambda", "superfluid"]):
        labels.append("PENETRATION_DEPTH")
    if any(term in text for term in ["optical", "spectral weight", "sum rule", "s_delta", "su/sn"]):
        labels.append("OPTICAL_SPECTRAL_WEIGHT")
    if not labels:
        labels.append("SAMPLE_METADATA")
    relevant = "relevant" if _is_lsco_text(text) and labels != ["SAMPLE_METADATA"] else ("maybe_relevant" if _is_lsco_text(text) else "not_relevant")
    locations = ["ABSTRACT"]
    difficulty: Literal["low", "medium", "high"] = "medium"
    if "figure" in text and "table" not in text:
        locations = ["FIGURE", "FIGURE_CAPTION"]
        difficulty = "high"
    if "table" in text:
        locations.append("TABLE")
        difficulty = "low"
    return PaperClassification(
        paper_id=paper.paper_id,
        relevance=relevant,
        observable_labels=labels,
        data_location_labels=locations,
        sample_type="film" if "film" in text else "unknown",
        doping_coverage=sorted(set(re.findall(r"x\s*=\s*0\.\d+", text))),
        extraction_difficulty=difficulty,
        source_reliability="high" if paper.doi or paper.arxiv_id else "medium",
        rationale=f"Rule-based labels from title/abstract: {', '.join(labels)}",
    )


def _fixture_extractions(papers: list[PaperRecord]) -> tuple[list[ExtractionRecord], list[DigitizationTask]]:
    by_id = {paper.paper_id: paper for paper in papers}
    extractions = [
        ExtractionRecord(
            extraction_id="extract-fixture-tc-x015",
            paper_id="paper-fixture-lsco-tc-gap",
            observable="tc_k",
            raw_value="38.0",
            raw_unit="K",
            normalized_value=38.0,
            normalized_unit="K",
            doping_raw="x=0.150",
            doping_x=0.15,
            sample_id="fixture-lsco-x015",
            sample_type="fixture",
            uncertainty_raw="1 K",
            uncertainty_normalized=1.0,
            table_id="Table 1",
            evidence_text="Table 1: LSCO x=0.150 Tc = 38.0 K.",
            extraction_method="deterministic_fixture_table",
            confidence="HIGH",
        ),
        ExtractionRecord(
            extraction_id="extract-fixture-gap-x015",
            paper_id="paper-fixture-lsco-tc-gap",
            observable="gap_ev",
            raw_value="12.5 meV",
            raw_unit="meV",
            normalized_value=0.0125,
            normalized_unit="eV",
            doping_raw="x=0.150",
            doping_x=0.15,
            sample_id="fixture-lsco-x015",
            sample_type="fixture",
            uncertainty_raw="2 meV",
            uncertainty_normalized=0.002,
            table_id="Table 1",
            evidence_text="Table 1: LSCO x=0.150 superconducting gap = 12.5 meV.",
            extraction_method="deterministic_fixture_table",
            confidence="HIGH",
        ),
        ExtractionRecord(
            extraction_id="extract-fixture-alpha-x015",
            paper_id="paper-fixture-lsco-isotope-lambda",
            observable="isotope_alpha",
            raw_value="0.08",
            raw_unit="dimensionless",
            normalized_value=0.08,
            normalized_unit="dimensionless",
            doping_raw="x=0.150",
            doping_x=0.15,
            sample_id="fixture-lsco-x015",
            sample_type="fixture",
            uncertainty_raw="0.02",
            uncertainty_normalized=0.02,
            table_id="Table 2",
            evidence_text="Table 2: alpha_O = 0.08 at x=0.150.",
            extraction_method="deterministic_fixture_table",
            confidence="HIGH",
        ),
        ExtractionRecord(
            extraction_id="extract-fixture-lambda-x015",
            paper_id="paper-fixture-lsco-isotope-lambda",
            observable="penetration_depth_nm",
            raw_value="240 nm",
            raw_unit="nm",
            normalized_value=240.0,
            normalized_unit="nm",
            doping_raw="x=0.150",
            doping_x=0.15,
            sample_id="fixture-lsco-x015",
            sample_type="fixture",
            uncertainty_raw="10 nm",
            uncertainty_normalized=10.0,
            table_id="Table 2",
            evidence_text="Table 2: lambda_ab(0) = 240 nm at x=0.150.",
            extraction_method="deterministic_fixture_table",
            confidence="HIGH",
        ),
    ]
    digitization = [
        DigitizationTask(
            digitization_task_id="digitize-fixture-optical-fig2e",
            paper_id="paper-fixture-lsco-optical-figure" if "paper-fixture-lsco-optical-figure" in by_id else "unknown",
            figure_id="Figure 2",
            panel_id="e",
            observable="optical_s_delta_over_sn",
            y_axis_name="S_delta / S_n",
            y_axis_unit="dimensionless",
            required_doping_points=[0.15, 0.21, 0.24],
            status="QUEUED",
            series_label="LSCO overdoped films",
        )
    ]
    return extractions, digitization


def _extract_from_metadata(papers: list[PaperRecord], classifications: list[PaperClassification]) -> tuple[list[ExtractionRecord], list[DigitizationTask]]:
    tasks = []
    classified = {item.paper_id: item for item in classifications}
    for paper in papers:
        labels = classified.get(paper.paper_id)
        text = f"{paper.title} {paper.abstract or ''}".lower()
        if labels and "OPTICAL_SPECTRAL_WEIGHT" in labels.observable_labels:
            tasks.append(DigitizationTask(
                digitization_task_id=f"digitize-{_digest(paper.paper_id)}-optical",
                paper_id=paper.paper_id,
                figure_id="unknown",
                observable="optical_spectral_weight_proxy",
                y_axis_name="optical spectral weight",
                y_axis_unit="unknown",
                required_doping_points=[],
                status="QUEUED",
                series_label="metadata-only candidate",
            ))
        if "figure" in text and labels and any(label.startswith("OPTICAL") for label in labels.observable_labels):
            continue
    return [], tasks


def _candidate_rows_from_extractions(extractions: list[ExtractionRecord], papers: list[PaperRecord]) -> list[CandidateDataRow]:
    paper_map = {paper.paper_id: paper for paper in papers}
    rows = []
    for record in extractions:
        paper = paper_map.get(record.paper_id)
        value: float | str = record.normalized_value if record.normalized_value is not None else record.raw_value
        rows.append(CandidateDataRow(
            candidate_row_id=f"cand-{_digest(record.extraction_id)}",
            material_id=f"LSCO-{record.sample_id or record.doping_raw}",
            doping=record.doping_raw,
            doping_x=record.doping_x,
            observable=record.observable,
            value=value,
            unit=record.normalized_unit or record.raw_unit,
            uncertainty=record.uncertainty_raw or None,
            source_id=record.paper_id,
            provenance=record.evidence_text,
            source_url=(paper.landing_page_url if paper else None),
            curation_note=f"staged by Phase2DataAcquisitionAgent from {record.extraction_method}",
            paper_id=record.paper_id,
            extraction_id=record.extraction_id,
            confidence=record.confidence,
            status="NORMALIZED",
            sample_id=record.sample_id,
            measurement_method=record.measurement_method,
            observable_definition=record.observable_definition,
            doping_definition=record.doping_definition,
        ))
    return rows


def _validate_candidate_rows(rows: list[CandidateDataRow]) -> tuple[list[CandidateDataRow], dict[str, Any]]:
    allowed_units = {
        "tc_k": {"K"},
        "gap_ev": {"eV"},
        "isotope_alpha": {"dimensionless"},
        "penetration_depth_nm": {"nm"},
        "optical_spectral_weight_proxy": {"relative", "dimensionless", "um^-2"},
        "optical_s_delta_over_sn": {"dimensionless"},
        "optical_su_over_sn": {"dimensionless"},
        "superfluid_density_proxy": {"um^-2", "relative", "dimensionless"},
    }
    normalized = []
    rejected = []
    for row in rows:
        errors = []
        if row.observable not in allowed_units:
            errors.append("unknown_observable")
        elif row.unit not in allowed_units[row.observable]:
            errors.append("unrecognized_unit")
        if row.doping_x is None or not (0.0 <= row.doping_x <= 0.35):
            errors.append("invalid_or_missing_doping")
        if row.confidence == "LOW":
            errors.append("low_confidence_requires_review")
        if not row.provenance.strip():
            errors.append("missing_provenance")
        if isinstance(row.value, (int, float)) and row.observable in {"tc_k", "gap_ev", "penetration_depth_nm"} and row.value <= 0:
            errors.append("non_positive_value")
        row.validation_errors = errors
        row.status = "VALIDATED" if not errors else "NEEDS_REVIEW"
        (normalized if not errors else rejected).append(row.candidate_row_id)
    return rows, {"schema_version": ACQUISITION_SCHEMA_VERSION, "validated_candidate_row_ids": normalized, "rows_requiring_review": rejected}


def _detect_conflicts(rows: list[CandidateDataRow], canonical: Path) -> list[dict[str, Any]]:
    conflicts = []
    existing = []
    if canonical.exists():
        existing = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        for current in existing:
            same = (
                current.get("doping") == row.doping
                and current.get("observable") == row.observable
                and current.get("source_id") == row.source_id
            )
            if same:
                conflicts.append({
                    "schema_version": ACQUISITION_SCHEMA_VERSION,
                    "candidate_row_id": row.candidate_row_id,
                    "decision": "exact_duplicate" if str(current.get("value")) == str(row.value) else "genuine_conflict",
                    "existing_observation_id": current.get("observation_id"),
                    "observable": row.observable,
                    "doping": row.doping,
                })
    return conflicts


def _promote_rows(rows: list[CandidateDataRow], canonical: Path, *, auto_promote: bool) -> tuple[list[PromotionDecision], dict[str, Any]]:
    before_checksum = _file_checksum(canonical)
    decisions = []
    promotable = [row for row in rows if row.status == "VALIDATED" and row.confidence == "HIGH"]
    if not auto_promote:
        decisions = [
            PromotionDecision(candidate_row_id=row.candidate_row_id, decision="needs_review", reason="auto_promote disabled; row staged only")
            for row in rows
        ]
        return decisions, {"schema_version": ACQUISITION_SCHEMA_VERSION, "changed": False, "before_checksum": before_checksum, "after_checksum": before_checksum, "promoted_count": 0}
    if not promotable:
        decisions = [PromotionDecision(candidate_row_id=row.candidate_row_id, decision="rejected", reason="row failed validation or confidence gate") for row in rows]
        return decisions, {"schema_version": ACQUISITION_SCHEMA_VERSION, "changed": False, "before_checksum": before_checksum, "after_checksum": before_checksum, "promoted_count": 0}
    canonical.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = list(csv.DictReader(canonical.read_text(encoding="utf-8").splitlines())) if canonical.exists() else []
    fieldnames = list(existing_rows[0].keys()) if existing_rows else ["observation_id", "material_family", "material_id", "doping", "observable", "value", "unit", "uncertainty", "split", "source_id", "provenance", "usable_for_fit", "source_url", "curation_note"]
    new_rows = []
    for row in promotable:
        new_rows.append({
            "observation_id": row.candidate_row_id,
            "material_family": row.material_family,
            "material_id": row.material_id,
            "doping": row.doping,
            "observable": row.observable,
            "value": row.value,
            "unit": row.unit,
            "uncertainty": row.uncertainty or "",
            "split": row.split,
            "source_id": row.source_id,
            "provenance": row.provenance,
            "usable_for_fit": "true" if row.usable_for_fit else "false",
            "source_url": row.source_url or "",
            "curation_note": row.curation_note,
        })
        row.status = "PROMOTED"
        decisions.append(PromotionDecision(candidate_row_id=row.candidate_row_id, decision="promoted", reason="passed high-confidence deterministic gates"))
    tmp = canonical.with_suffix(canonical.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + new_rows)
    shutil.move(str(tmp), str(canonical))
    after_checksum = _file_checksum(canonical)
    for row in rows:
        if row.candidate_row_id not in {decision.candidate_row_id for decision in decisions}:
            decisions.append(PromotionDecision(candidate_row_id=row.candidate_row_id, decision="needs_review", reason="not eligible for automatic promotion"))
    return decisions, {"schema_version": ACQUISITION_SCHEMA_VERSION, "changed": before_checksum != after_checksum, "before_checksum": before_checksum, "after_checksum": after_checksum, "promoted_count": len(new_rows)}


def _comparison_trigger(canonical_diff: dict[str, Any], coverage: dict[str, Any], readiness: ReadinessGateResult | None = None) -> dict[str, Any]:
    status = coverage.get("status")
    readiness_status = readiness.status if readiness else ("ready_for_held_out_comparison" if status == "sufficient" else "blocked_insufficient_existing_data")
    ready = readiness_status == "ready_for_held_out_comparison"
    return {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "dataset_checksum_changed": bool(canonical_diff.get("changed")),
        "readiness": readiness_status,
        "phase2_status": readiness_status,
        "comparison_rerun": bool(canonical_diff.get("changed")) and ready,
        "blocked_reason": "" if ready else (readiness.explanation if readiness else "coverage gates failed; do not run material-level comparison"),
        "coverage_status": status,
        "passed_gates": readiness.passed_gates if readiness else [],
        "failed_gates": readiness.failed_gates if readiness else [],
    }


def _write_acquisition_artifacts(
    run_dir: Path,
    *,
    task: AcquisitionTask,
    queries: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    papers: list[PaperRecord],
    dedupe_report: dict[str, Any],
    classifications: list[PaperClassification],
    extractions: list[ExtractionRecord],
    candidate_rows: list[CandidateDataRow],
    normalization_report: dict[str, Any],
    conflicts: list[dict[str, Any]],
    promotion_decisions: list[PromotionDecision],
    digitization_tasks: list[DigitizationTask],
    parser_fallbacks: list[ParserFallbackRecord],
    coverage_before: dict[str, Any],
    coverage_after: dict[str, Any],
    canonical_diff: dict[str, Any],
    comparison_trigger: dict[str, Any],
    readiness_gate: ReadinessGateResult,
    robustness: ComparisonRobustnessReport,
    data_claims: list[DataClaimRecord],
    summary: dict[str, Any],
    blocked: bool,
) -> dict[str, str]:
    write_json(run_dir / "acquisition_task.json", task.model_dump(mode="json"))
    write_json(run_dir / "search_queries.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "queries": queries})
    write_json(run_dir / "search_results.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "results": search_results, "status": "blocked_live_network_permission_required" if blocked else "complete"})
    write_json(run_dir / "paper_registry.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "papers": [paper.model_dump(mode="json") for paper in papers]})
    write_json(run_dir / "deduplication_report.json", dedupe_report)
    write_json(run_dir / "full_text_retrieval_report.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "retrieved_count": 0, "records": [{"paper_id": paper.paper_id, "status": paper.retrieval_status} for paper in papers]})
    write_json(run_dir / "paper_classification.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "classifications": [item.model_dump(mode="json") for item in classifications]})
    write_jsonl(run_dir / "extraction_records.jsonl", [item.model_dump(mode="json") for item in extractions])
    write_jsonl(run_dir / "candidate_rows.jsonl", [item.model_dump(mode="json") for item in candidate_rows])
    write_json(run_dir / "normalization_report.json", normalization_report)
    write_json(run_dir / "conflict_report.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "conflicts": conflicts})
    write_json(run_dir / "promotion_report.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "decisions": [item.model_dump(mode="json") for item in promotion_decisions]})
    write_json(run_dir / "digitization_queue.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "tasks": [item.model_dump(mode="json") for item in digitization_tasks]})
    write_json(run_dir / "coverage_before.json", coverage_before)
    write_json(run_dir / "coverage_after.json", coverage_after)
    write_json(run_dir / "canonical_dataset_diff.json", canonical_diff)
    write_json(run_dir / "comparison_trigger.json", comparison_trigger)
    write_json(run_dir / "readiness_gates.json", readiness_gate.model_dump(mode="json"))
    write_json(run_dir / "comparison_robustness.json", robustness.model_dump(mode="json"))
    write_jsonl(run_dir / "data_claims.jsonl", [item.model_dump(mode="json") for item in data_claims])
    write_json(run_dir / "parser_fallback_report.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "records": [item.model_dump(mode="json") for item in parser_fallbacks]})
    write_json(run_dir / "observable_ontology.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "observables": [item.model_dump(mode="json") for item in observable_ontology().values()]})
    write_json(run_dir / "comparison_result.json", {"schema_version": ACQUISITION_SCHEMA_VERSION, "status": "not_run", "reason": comparison_trigger.get("blocked_reason") or "comparison runner not required"})
    write_json(run_dir / "acquisition_summary.json", summary)
    _write_staging_files(candidate_rows, conflicts, promotion_decisions)
    (run_dir / "run_summary.md").write_text(_run_summary_markdown(summary, comparison_trigger), encoding="utf-8")
    return {path.name: str(path) for path in run_dir.iterdir() if path.is_file()}


def _write_staging_files(rows: list[CandidateDataRow], conflicts: list[dict[str, Any]], decisions: list[PromotionDecision]) -> None:
    DEFAULT_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(DEFAULT_STAGING_DIR / "phase2_lsco_candidate_rows.jsonl", [row.model_dump(mode="json") for row in rows])
    write_jsonl(DEFAULT_STAGING_DIR / "phase2_lsco_conflicts.jsonl", conflicts)
    rejected = [decision.model_dump(mode="json") for decision in decisions if decision.decision != "promoted"]
    write_jsonl(DEFAULT_STAGING_DIR / "phase2_lsco_rejected_rows.jsonl", rejected)


def _comparison_robustness_report(comparison_trigger: dict[str, Any], readiness: ReadinessGateResult) -> ComparisonRobustnessReport:
    if comparison_trigger.get("comparison_rerun"):
        return ComparisonRobustnessReport(
            status="comparison_inconclusive",
            leakage_checks=["group_by_paper", "group_by_sample", "exclude_held_out_doping_from_fit"],
            reason="Comparison runner is permitted, but mechanism claims still require expert review.",
        )
    return ComparisonRobustnessReport(
        status="not_run_blocked" if readiness.status.startswith("blocked") else "ready_not_run",
        leakage_checks=["paper_grouped_holdout_required", "sample_grouped_holdout_required", "multi_observable_fit_required"],
        reason=comparison_trigger.get("blocked_reason") or "Ready gate did not request an automatic comparison run.",
    )


def _data_claims(
    papers: list[PaperRecord],
    extractions: list[ExtractionRecord],
    candidates: list[CandidateDataRow],
    readiness: ReadinessGateResult,
    comparison_trigger: dict[str, Any],
) -> list[DataClaimRecord]:
    claims: list[DataClaimRecord] = []
    for paper in papers:
        claims.append(DataClaimRecord(
            claim_id=f"claim-paper-{_digest(paper.paper_id)}",
            claim_text=f"Paper record `{paper.paper_id}` was discovered or registered for LSCO Phase 2 acquisition.",
            claim_type="paper_discovered",
            status="resolved",
            artifact_ids=["paper_registry.json"],
        ))
    for extraction in extractions:
        claims.append(DataClaimRecord(
            claim_id=f"claim-extraction-{_digest(extraction.extraction_id)}",
            claim_text=f"Extraction `{extraction.extraction_id}` reports {extraction.observable} for {extraction.doping_raw}.",
            claim_type="value_extracted",
            status="resolved" if extraction.confidence == "HIGH" else "needs_review",
            depends_on=[f"claim-paper-{_digest(extraction.paper_id)}"],
            artifact_ids=["extraction_records.jsonl"],
        ))
    for row in candidates:
        claims.append(DataClaimRecord(
            claim_id=f"claim-row-{_digest(row.candidate_row_id)}",
            claim_text=f"Candidate row `{row.candidate_row_id}` is staged with status {row.status}.",
            claim_type="candidate_row_staged",
            status="resolved" if row.status in {"VALIDATED", "PROMOTED"} else "needs_review",
            depends_on=[f"claim-extraction-{_digest(row.extraction_id)}"],
            artifact_ids=["candidate_rows.jsonl", "normalization_report.json"],
            candidate_row_id=row.candidate_row_id,
        ))
    claims.append(DataClaimRecord(
        claim_id="claim-phase2-readiness",
        claim_text=f"Phase 2 readiness is `{readiness.status}`.",
        claim_type="coverage_gate",
        status="resolved" if readiness.status.startswith("ready") else "blocked",
        depends_on=[f"claim-row-{_digest(row.candidate_row_id)}" for row in candidates],
        artifact_ids=["readiness_gates.json", "coverage_after.json"],
    ))
    claims.append(DataClaimRecord(
        claim_id="claim-material-level-comparison",
        claim_text="Material-level mechanism comparison may run only after readiness and held-out gates pass.",
        claim_type="material_claim_gate",
        status="resolved" if comparison_trigger.get("comparison_rerun") else "blocked",
        depends_on=["claim-phase2-readiness"],
        artifact_ids=["comparison_trigger.json", "comparison_robustness.json"],
    ))
    return claims


def _run_summary_markdown(summary: dict[str, Any], comparison_trigger: dict[str, Any]) -> str:
    return "\n".join([
        "# Phase 2 LSCO Data Acquisition Run",
        "",
        f"- Status: {summary['status']}",
        f"- Mode: {summary['mode']}",
        f"- Queries executed: {summary['queries_executed']}",
        f"- Papers found: {summary['papers_found']}",
        f"- Relevant papers: {summary['papers_relevant']}",
        f"- Candidate rows staged: {summary['candidate_rows_staged']}",
        f"- Rows promoted: {summary['rows_promoted']}",
        f"- Figure digitization tasks: {summary['figure_digitization_tasks_created']}",
        f"- Phase 2 status: {summary['final_phase2_status']}",
        f"- Comparison rerun: {comparison_trigger['comparison_rerun']}",
        "",
        "Material-level claims remain blocked unless readiness gates pass.",
        "",
    ])


def _strip_tex_markup(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = text.replace("\\pm", "±").replace("\\Delta", "Delta").replace("\\lambda", "lambda")
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _extract_values_from_line(paper_id: str, line: str, *, line_number: int, source_kind: str) -> list[ExtractionRecord]:
    doping = _parse_doping(line)
    if doping is None:
        return []
    records: list[ExtractionRecord] = []
    patterns = [
        ("tc_k", r"(?:Tc|T_c|transition temperature)\s*[=:]\s*([0-9.]+)\s*(K|kelvin)?", "K"),
        ("gap_ev", r"(?:gap|Delta|superconducting gap)\s*[=:]\s*([0-9.]+)\s*(meV|eV)", "eV"),
        ("isotope_alpha", r"(?:alpha|isotope exponent|alpha_O)\s*[=:]\s*([0-9.]+)\s*(dimensionless)?", "dimensionless"),
        ("penetration_depth_nm", r"(?:lambda_ab|lambda|penetration depth)\s*[=:]\s*([0-9.]+)\s*(nm|um)", "nm"),
        ("optical_s_delta_over_sn", r"(?:S_delta/Sn|S_delta\s*/\s*S_n)\s*[=:]\s*([0-9.]+)\s*(dimensionless)?", "dimensionless"),
        ("optical_su_over_sn", r"(?:Su/Sn|S_u\s*/\s*S_n)\s*[=:]\s*([0-9.]+)\s*(dimensionless)?", "dimensionless"),
    ]
    for observable, pattern, default_unit in patterns:
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            raw_value = match.group(1)
            raw_unit = match.group(2) or default_unit
            value, unit = normalize_value_unit(observable, raw_value, raw_unit)
            records.append(ExtractionRecord(
                extraction_id=f"extract-{_digest(paper_id + observable + str(line_number) + raw_value)}",
                paper_id=paper_id,
                observable=observable,
                raw_value=raw_value,
                raw_unit=raw_unit,
                normalized_value=value,
                normalized_unit=unit,
                doping_raw=f"x={doping:.3f}",
                doping_x=doping,
                sample_id=f"{paper_id}-x{doping:.3f}",
                sample_type="unknown",
                page=f"line {line_number}",
                evidence_text=line.strip(),
                extraction_method=f"{source_kind}_deterministic_text",
                confidence="MEDIUM" if source_kind in {"pdf_text", "text", "tex"} else "HIGH",
                observable_definition=observable_ontology().get(observable, ObservableDefinition(observable=observable, canonical_unit=unit, synonyms=[])).definition,
                doping_definition="nominal",
                measurement_method=_measurement_method_for_observable(observable),
            ))
    return records


def _measurement_method_for_observable(observable: str) -> str:
    return {
        "tc_k": "transport_or_susceptibility",
        "gap_ev": "spectroscopy",
        "isotope_alpha": "isotope_substitution",
        "penetration_depth_nm": "penetration_depth",
        "optical_s_delta_over_sn": "tdts_optical",
        "optical_su_over_sn": "tdts_optical",
    }.get(observable, "unknown")


def _first_figure_id(text: str) -> str:
    match = re.search(r"(?:Fig\.|Figure)\s*([0-9]+[a-z]?)", text, flags=re.IGNORECASE)
    return f"Figure {match.group(1)}" if match else "unknown"


def _parse_doping(value: Any) -> float | None:
    text = str(value)
    match = re.search(r"x\s*[=~]?\s*(0\.\d+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(0\.\d+)\b", text)
    return _maybe_float(match.group(1)) if match else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "n", ""}
    return bool(value)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _is_lsco_text(text: str) -> bool:
    return any(term in text for term in ["lsco", "la2-xsrxcuo4", "la2-xsrxcuo4", "la_{2-x}", "la1.85sr0.15cuo4"])


def _digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _file_checksum(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
