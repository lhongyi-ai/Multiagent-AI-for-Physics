from __future__ import annotations

from coscientist.core.domain_packs import BenchmarkCase, NormalizedRecord, ReadinessGate, ResearchObjective, SourceClassification, SourceRecord, ToolDescriptor, ValidationResult
from coscientist.core.tasks import ScientificTaskType
from coscientist.schemas.literature import SearchQuery
from coscientist.superconductivity.phase2_acquisition import observable_ontology


class SuperconductivityLscoPack:
    domain_id = "superconductivity_lsco"
    version = "v26.0"

    def supported_task_types(self) -> set[ScientificTaskType]:
        return {ScientificTaskType.DATA_EXTRACTION, ScientificTaskType.MATERIAL_COMPARISON, ScientificTaskType.NUMERICAL_MODELING}

    def search_queries(self, objective: ResearchObjective) -> list[SearchQuery]:
        return [
            SearchQuery(query="La2-xSrxCuO4 LSCO penetration depth doping"),
            SearchQuery(query="La2-xSrxCuO4 isotope exponent superconducting doping"),
            SearchQuery(query="LSCO optical spectral weight superconducting state doping"),
        ]

    def classify_source(self, source: SourceRecord) -> SourceClassification:
        text = f"{source.title} {source.text}".lower()
        labels = [name for name in ["tc", "gap", "penetration", "isotope", "optical"] if name in text]
        return SourceClassification(source_id=source.source_id, relevance="relevant" if "lsco" in text or "la2-xsrxcuo4" in text else "maybe_relevant", labels=labels)

    def normalize_records(self, source: SourceRecord) -> list[NormalizedRecord]:
        if not source.text:
            return []
        return [NormalizedRecord(record_id=f"{source.source_id}-fixture", source_id=source.source_id, observable="domain_fixture_note", value=source.text[:120], provenance="domain-pack fixture parser")]

    def validate_record(self, record: NormalizedRecord) -> ValidationResult:
        return ValidationResult(record_id=record.record_id, status="valid" if record.provenance else "invalid", errors=[] if record.provenance else ["missing provenance"])

    def readiness_gates(self) -> list[ReadinessGate]:
        return [
            ReadinessGate(gate_id="overlapping_doping_points", description="All target observables overlap on enough doping points."),
            ReadinessGate(gate_id="provenance_reviewed", description="Rows have exact source provenance and review state."),
            ReadinessGate(gate_id="held_out_split_possible", description="Grouped held-out split can be formed without leakage."),
        ]

    def tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(tool_id="phase1_minimal_mixed_bcs_solver", name="Minimal mixed BCS solver", purpose="Theory scan and energy ledger."),
            ToolDescriptor(tool_id="phase2_lsco_acquisition", name="LSCO Phase 2 acquisition", purpose="Search, extraction staging, review, readiness."),
            ToolDescriptor(tool_id="phase2_data_coverage_tool", name="LSCO coverage tool", purpose="Coverage and readiness analysis."),
            ToolDescriptor(tool_id="energy_decomposition_audit_tool", name="Condensation-energy decomposition audit", purpose="Gauge, representation, counterexample, and observable-classification guardrails."),
            ToolDescriptor(tool_id="representation_counterexample", name="Representation counterexample", purpose="Show whether component energy partitions are invariant under equivalent decompositions."),
            ToolDescriptor(tool_id="hellmann_feynman_diagnostic", name="Hellmann-Feynman diagnostics", purpose="Operational coupling derivatives under a fixed microscopic convention."),
        ]

    def benchmark_cases(self) -> list[BenchmarkCase]:
        return [BenchmarkCase(case_id="lsco-fixture-coverage", task_type=ScientificTaskType.DATA_EXTRACTION, prompt="Fixture LSCO data extraction keeps readiness blocked when optical data are missing.")]

    def guardrails(self) -> list[str]:
        return [
            "Do not claim material-level quantitative separation until provenance, overlap, and held-out comparison gates pass.",
            "Figure-only optical data must enter reviewed digitization, not automatic promotion.",
        ]

    def observable_ontology(self) -> list[dict[str, object]]:
        return [item.model_dump(mode="json") for item in observable_ontology()]
