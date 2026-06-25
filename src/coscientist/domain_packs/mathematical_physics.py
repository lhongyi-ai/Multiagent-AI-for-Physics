from __future__ import annotations

from coscientist.core.domain_packs import BenchmarkCase, NormalizedRecord, ReadinessGate, ResearchObjective, SourceClassification, SourceRecord, ToolDescriptor, ValidationResult
from coscientist.core.tasks import ScientificTaskType
from coscientist.schemas.literature import SearchQuery


class MathematicalPhysicsPack:
    domain_id = "mathematical_physics"
    version = "v26.0"

    def supported_task_types(self) -> set[ScientificTaskType]:
        return {ScientificTaskType.THEORY_DERIVATION, ScientificTaskType.HIDDEN_ANSWER_BENCHMARK, ScientificTaskType.NUMERICAL_MODELING}

    def search_queries(self, objective: ResearchObjective) -> list[SearchQuery]:
        return [SearchQuery(query=f"mathematical physics {objective.question[:80]}")]

    def classify_source(self, source: SourceRecord) -> SourceClassification:
        return SourceClassification(source_id=source.source_id, relevance="relevant", labels=["proof", "derivation"])

    def normalize_records(self, source: SourceRecord) -> list[NormalizedRecord]:
        return [NormalizedRecord(record_id=f"{source.source_id}-claim", source_id=source.source_id, observable="formal_claim", value=source.text[:120] or source.title, provenance="domain-pack fixture parser")]

    def validate_record(self, record: NormalizedRecord) -> ValidationResult:
        errors = ["finite numerical checks are not a proof"] if "finite numerical" in str(record.value).lower() else []
        return ValidationResult(record_id=record.record_id, status="needs_review" if errors else "valid", errors=errors)

    def readiness_gates(self) -> list[ReadinessGate]:
        return [
            ReadinessGate(gate_id="definitions_fixed", description="Definitions and quantifiers are fixed before proof attempt."),
            ReadinessGate(gate_id="proof_or_counterexample", description="Finite checks are labeled as evidence, not proof."),
        ]

    def tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(tool_id="symbolic_checker", name="Symbolic checker", purpose="Independent symbolic path.", implemented=False),
            ToolDescriptor(tool_id="numerical_counterexample_search", name="Numerical counterexample search", purpose="Bounded falsification search."),
        ]

    def benchmark_cases(self) -> list[BenchmarkCase]:
        return [BenchmarkCase(case_id="math-hidden-false", task_type=ScientificTaskType.HIDDEN_ANSWER_BENCHMARK, prompt="A plausible universal claim with a hidden counterexample.", hidden_answer="false")]

    def guardrails(self) -> list[str]:
        return ["Do not call finite numerical checks a proof."]
