from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from coscientist.core.tasks import ScientificTaskType
from coscientist.schemas.literature import SearchQuery


class ResearchObjective(BaseModel):
    schema_version: str = "v26-research-objective"
    objective_id: str
    question: str
    task_type: ScientificTaskType
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    schema_version: str = "v26-source-record"
    source_id: str
    title: str
    source_type: str
    text: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceClassification(BaseModel):
    schema_version: str = "v26-source-classification"
    source_id: str
    relevance: str
    labels: list[str] = Field(default_factory=list)
    rationale: str = ""


class NormalizedRecord(BaseModel):
    schema_version: str = "v26-normalized-record"
    record_id: str
    source_id: str
    observable: str
    value: float | str
    unit: str = ""
    provenance: str = ""
    needs_review: bool = True


class ValidationResult(BaseModel):
    schema_version: str = "v26-validation-result"
    record_id: str
    status: str
    errors: list[str] = Field(default_factory=list)


class ReadinessGate(BaseModel):
    schema_version: str = "v26-readiness-gate"
    gate_id: str
    description: str
    required: bool = True


class ToolDescriptor(BaseModel):
    schema_version: str = "v26-tool-descriptor"
    tool_id: str
    name: str
    purpose: str
    implemented: bool = True


class BenchmarkCase(BaseModel):
    schema_version: str = "v26-benchmark-case"
    case_id: str
    task_type: ScientificTaskType
    prompt: str
    hidden_answer: str | None = None
    expected_status: str = "fixture_supported"


@runtime_checkable
class DomainPack(Protocol):
    domain_id: str
    version: str

    def supported_task_types(self) -> set[ScientificTaskType]:
        ...

    def search_queries(self, objective: ResearchObjective) -> list[SearchQuery]:
        ...

    def classify_source(self, source: SourceRecord) -> SourceClassification:
        ...

    def normalize_records(self, source: SourceRecord) -> list[NormalizedRecord]:
        ...

    def validate_record(self, record: NormalizedRecord) -> ValidationResult:
        ...

    def readiness_gates(self) -> list[ReadinessGate]:
        ...

    def tools(self) -> list[ToolDescriptor]:
        ...

    def benchmark_cases(self) -> list[BenchmarkCase]:
        ...

    def guardrails(self) -> list[str]:
        ...


@dataclass(frozen=True)
class DomainPackSummary:
    domain_id: str
    version: str
    task_types: list[str]
    tool_count: int
    benchmark_count: int
    guardrails: list[str]


class DomainPackRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, DomainPack] = {}

    def register(self, pack: DomainPack) -> None:
        if not pack.domain_id or not pack.version:
            raise ValueError("domain pack must define domain_id and version")
        if pack.domain_id in self._packs:
            raise ValueError(f"duplicate domain pack: {pack.domain_id}")
        self._packs[pack.domain_id] = pack

    def get(self, domain_id: str) -> DomainPack:
        try:
            return self._packs[domain_id]
        except KeyError as exc:
            raise KeyError(f"unknown domain pack: {domain_id}") from exc

    def list(self) -> list[DomainPackSummary]:
        summaries = []
        for pack in self._packs.values():
            summaries.append(DomainPackSummary(
                domain_id=pack.domain_id,
                version=pack.version,
                task_types=sorted(task.value for task in pack.supported_task_types()),
                tool_count=len(pack.tools()),
                benchmark_count=len(pack.benchmark_cases()),
                guardrails=pack.guardrails(),
            ))
        return sorted(summaries, key=lambda item: item.domain_id)


def get_default_domain_registry() -> DomainPackRegistry:
    from coscientist.domain_packs.magnetic_transport_crse import MagneticTransportCrSePack
    from coscientist.domain_packs.mathematical_physics import MathematicalPhysicsPack
    from coscientist.domain_packs.superconductivity_lsco import SuperconductivityLscoPack
    from coscientist.domain_packs.xrd_phase_identification import XrdPhaseIdentificationPack

    registry = DomainPackRegistry()
    registry.register(SuperconductivityLscoPack())
    registry.register(MagneticTransportCrSePack())
    registry.register(MathematicalPhysicsPack())
    registry.register(XrdPhaseIdentificationPack())
    return registry
