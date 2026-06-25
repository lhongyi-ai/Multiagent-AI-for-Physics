from __future__ import annotations

from coscientist.core.domain_packs import BenchmarkCase, NormalizedRecord, ReadinessGate, ResearchObjective, SourceClassification, SourceRecord, ToolDescriptor, ValidationResult
from coscientist.core.tasks import ScientificTaskType
from coscientist.schemas.literature import SearchQuery


class XrdPhaseIdentificationPack:
    domain_id = "xrd_phase_identification"
    version = "v26.0"

    def supported_task_types(self) -> set[ScientificTaskType]:
        return {ScientificTaskType.PHASE_IDENTIFICATION, ScientificTaskType.DATA_EXTRACTION}

    def search_queries(self, objective: ResearchObjective) -> list[SearchQuery]:
        return [SearchQuery(query=f"powder XRD phase identification {objective.question[:80]}")]

    def classify_source(self, source: SourceRecord) -> SourceClassification:
        text = f"{source.title} {source.text}".lower()
        labels = [label for label in ["xrd", "rietveld", "peak", "phase"] if label in text]
        return SourceClassification(source_id=source.source_id, relevance="relevant" if labels else "maybe_relevant", labels=labels)

    def normalize_records(self, source: SourceRecord) -> list[NormalizedRecord]:
        return [NormalizedRecord(record_id=f"{source.source_id}-pattern-note", source_id=source.source_id, observable="xrd_pattern_note", value=source.text[:120] or source.title, provenance="domain-pack fixture parser")]

    def validate_record(self, record: NormalizedRecord) -> ValidationResult:
        return ValidationResult(record_id=record.record_id, status="valid" if record.value else "invalid", errors=[] if record.value else ["empty pattern record"])

    def readiness_gates(self) -> list[ReadinessGate]:
        return [
            ReadinessGate(gate_id="peak_list_available", description="Peak positions or raw pattern must be available."),
            ReadinessGate(gate_id="ambiguous_phases_preserved", description="Overlapping candidate phases are preserved instead of collapsed."),
        ]

    def tools(self) -> list[ToolDescriptor]:
        return [ToolDescriptor(tool_id="xrd_peak_matcher", name="XRD peak matcher", purpose="Compare observed peaks with candidate phases.", implemented=False)]

    def benchmark_cases(self) -> list[BenchmarkCase]:
        return [BenchmarkCase(case_id="xrd-ambiguous-two-phase", task_type=ScientificTaskType.PHASE_IDENTIFICATION, prompt="Fixture pattern where two phases share the strongest peak.")]

    def guardrails(self) -> list[str]:
        return ["Do not collapse ambiguous phase mixtures into a single confident phase without residual evidence."]
