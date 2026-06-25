from __future__ import annotations

from coscientist.core.domain_packs import BenchmarkCase, NormalizedRecord, ReadinessGate, ResearchObjective, SourceClassification, SourceRecord, ToolDescriptor, ValidationResult
from coscientist.core.tasks import ScientificTaskType
from coscientist.schemas.literature import SearchQuery


class MagneticTransportCrSePack:
    domain_id = "magnetic_transport_crse"
    version = "v26.0"

    def supported_task_types(self) -> set[ScientificTaskType]:
        return {ScientificTaskType.DATA_EXTRACTION, ScientificTaskType.MATERIAL_COMPARISON, ScientificTaskType.EXPERIMENT_SELECTION}

    def search_queries(self, objective: ResearchObjective) -> list[SearchQuery]:
        return [
            SearchQuery(query="CrSe magnetic transport anomalous Hall effect"),
            SearchQuery(query="CrSe altermagnetism magnetoresistance Hall measurements"),
        ]

    def classify_source(self, source: SourceRecord) -> SourceClassification:
        text = f"{source.title} {source.text}".lower()
        labels = [label for label in ["hall", "magnetoresistance", "neel", "domain"] if label in text]
        return SourceClassification(source_id=source.source_id, relevance="relevant" if "crse" in text else "maybe_relevant", labels=labels)

    def normalize_records(self, source: SourceRecord) -> list[NormalizedRecord]:
        return [NormalizedRecord(record_id=f"{source.source_id}-ahe-note", source_id=source.source_id, observable="transport_claim", value=source.text[:120] or "fixture transport note", provenance="domain-pack fixture parser")]

    def validate_record(self, record: NormalizedRecord) -> ValidationResult:
        errors = ["AHE alone cannot prove altermagnetism"] if "altermagnet" in str(record.value).lower() and "hall" in str(record.value).lower() else []
        return ValidationResult(record_id=record.record_id, status="needs_review" if errors else "valid", errors=errors)

    def readiness_gates(self) -> list[ReadinessGate]:
        return [
            ReadinessGate(gate_id="magnetic_order_independent_probe", description="Magnetic order must be constrained by a probe beyond transport."),
            ReadinessGate(gate_id="ordinary_vs_anomalous_hall_separated", description="Transport decomposition must separate ordinary and anomalous components."),
        ]

    def tools(self) -> list[ToolDescriptor]:
        return [ToolDescriptor(tool_id="transport_tensor_checker", name="Transport tensor checker", purpose="Validate symmetry-compatible transport interpretations.", implemented=False)]

    def benchmark_cases(self) -> list[BenchmarkCase]:
        return [BenchmarkCase(case_id="crse-ahe-guardrail", task_type=ScientificTaskType.EXPERIMENT_SELECTION, prompt="AHE-only claim should remain blocked without independent magnetic-order evidence.")]

    def guardrails(self) -> list[str]:
        return ["Do not infer altermagnetism from anomalous Hall effect alone."]
