from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HypothesisStatusV2(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    NEEDS_REPAIR = "needs_repair"
    KILLED = "killed"
    FINALIST = "finalist"
    HELD_CONTRARIAN = "held_contrarian"


class ScoreProvenance(BaseModel):
    schema_version: str = "v26-score-provenance"
    method: str
    artifact_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ScoreEntry(BaseModel):
    schema_version: str = "v26-score-entry"
    dimension: str
    value: float = Field(ge=0.0, le=1.0)
    direction: str = "maximize"
    provenance: ScoreProvenance


class HardGateResult(BaseModel):
    schema_version: str = "v26-hard-gate-result"
    gate_id: str
    status: str
    hypothesis_id: str
    rationale: str
    evidence_artifact_ids: list[str] = Field(default_factory=list)


class HypothesisLineageV2(BaseModel):
    schema_version: str = "v26-hypothesis-lineage"
    parent_ids: list[str] = Field(default_factory=list)
    root_id: str
    mutation_operator: str = "initial"
    generation: int = 0


class HypothesisV2(BaseModel):
    schema_version: str = "v26-hypothesis-v2"
    hypothesis_id: str
    title: str
    scoped_claim: str
    domain_id: str = "general"
    task_type: str = "theory_derivation"
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    falsification_criteria: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    contradiction_artifact_ids: list[str] = Field(default_factory=list)
    status: HypothesisStatusV2 = HypothesisStatusV2.PROPOSED
    score_vector: list[ScoreEntry] = Field(default_factory=list)
    hard_gate_results: list[HardGateResult] = Field(default_factory=list)
    lineage: HypothesisLineageV2
    novelty_notes: list[str] = Field(default_factory=list)
    failure_memory_refs: list[str] = Field(default_factory=list)


def migrate_hypothesis_to_v2(source: Any, *, domain_id: str = "general", task_type: str = "theory_derivation") -> HypothesisV2:
    data = source.model_dump() if hasattr(source, "model_dump") else dict(source)
    hypothesis_id = str(data.get("id") or data.get("hypothesis_id") or data.get("candidate_id") or _digest(repr(data)))
    title = str(data.get("title") or data.get("summary") or hypothesis_id)
    claim = str(data.get("core_claim") or data.get("scoped_claim") or data.get("claim") or data.get("summary") or title)
    assumptions = _as_strings(data.get("assumptions"))
    predictions = _as_strings(data.get("testable_predictions") or data.get("predictions") or data.get("predicted_observables"))
    falsification = _as_strings(data.get("falsification_criteria") or data.get("falsification_conditions"))
    evidence = _as_strings(data.get("supporting_evidence") or data.get("linked_evidence_ids"))
    contradictions = _as_strings(data.get("contradicting_evidence"))
    parents = _as_strings(data.get("parent_ids"))
    status = HypothesisStatusV2.ACTIVE
    if str(data.get("status", "")).lower() in {"rejected", "killed"}:
        status = HypothesisStatusV2.KILLED
    return HypothesisV2(
        hypothesis_id=hypothesis_id,
        title=title,
        scoped_claim=claim,
        domain_id=domain_id,
        task_type=task_type,
        assumptions=assumptions,
        predictions=predictions,
        falsification_criteria=falsification,
        required_tools=_as_strings(data.get("required_tools")),
        evidence_artifact_ids=evidence,
        contradiction_artifact_ids=contradictions,
        status=status,
        lineage=HypothesisLineageV2(parent_ids=parents, root_id=parents[0] if parents else hypothesis_id),
        novelty_notes=_as_strings(data.get("novelty_notes") or data.get("novelty_statement")),
    )


def validate_finalist_requirements(hypothesis: HypothesisV2) -> list[str]:
    errors: list[str] = []
    if not hypothesis.scoped_claim.strip():
        errors.append("missing scoped claim")
    if not hypothesis.falsification_criteria:
        errors.append("missing falsification criteria")
    if not hypothesis.predictions:
        errors.append("missing predictions")
    if any(result.status == "fail" for result in hypothesis.hard_gate_results):
        errors.append("hard gate failed")
    return errors


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
