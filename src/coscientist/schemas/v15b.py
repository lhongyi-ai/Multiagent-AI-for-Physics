from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


GroundingMode = Literal["strict", "permissive", "off"]
StoppingAssessment = Literal[
    "continue_exploration",
    "continue_targeted_revision",
    "request_more_evidence",
    "request_human_review",
    "ready_for_external_validation",
    "stop_due_to_budget",
    "stop_due_to_lack_of_progress",
]
MetaReviewFeedbackMode = Literal["advisory", "controlled_feedback"]


class ProximityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    method: Literal["lexical_structured"] = "lexical_structured"
    similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    include_lineage_edges: bool = True
    include_evidence_overlap: bool = True


class MetaReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    provider_mode: Literal["deterministic", "inherit"] = "deterministic"
    feedback_mode: MetaReviewFeedbackMode = "advisory"
    feed_into_next_round: bool = False
    max_recommendations: int = Field(default=10, ge=1, le=50)
    require_artifact_references: bool = True

    @model_validator(mode="after")
    def validate_feedback_gate(self) -> MetaReviewConfig:
        if self.feedback_mode == "controlled_feedback" and not self.feed_into_next_round:
            raise ValueError("controlled_feedback mode requires feed_into_next_round=true")
        return self


class GroundingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: GroundingMode = "strict"
    allow_unverified_background_knowledge: bool = False
    require_evidence_ids_for_supported_claims: bool = True
    require_exact_locations: bool = True
    max_evidence_items_per_hypothesis: int = Field(default=8, ge=1, le=50)
    max_contradiction_items_per_hypothesis: int = Field(default=5, ge=0, le=50)
    max_context_characters: int = Field(default=30000, ge=1000, le=200000)
    prioritize_verified_full_text: bool = True
    include_metadata_only_records: bool = False


class V15BConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    proximity: ProximityConfig = Field(default_factory=ProximityConfig)
    meta_review: MetaReviewConfig = Field(default_factory=MetaReviewConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)


class HypothesisSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_a_id: str
    hypothesis_b_id: str
    claim_similarity: float = Field(ge=0.0, le=1.0)
    mechanism_similarity: float = Field(ge=0.0, le=1.0)
    assumption_similarity: float = Field(ge=0.0, le=1.0)
    prediction_similarity: float = Field(ge=0.0, le=1.0)
    experiment_similarity: float = Field(ge=0.0, le=1.0)
    evidence_overlap: float = Field(ge=0.0, le=1.0)
    lineage_related: bool = False
    overall_similarity: float = Field(ge=0.0, le=1.0)
    method: str = "lexical_structured"
    notes: list[str] = Field(default_factory=list)


class HypothesisCluster(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cluster_id: str
    member_ids: list[str] = Field(min_length=1)
    representative_hypothesis_id: str
    shared_claims: list[str] = Field(default_factory=list)
    shared_mechanisms: list[str] = Field(default_factory=list)
    shared_assumptions: list[str] = Field(default_factory=list)
    distinguishing_features: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    cluster_confidence: float = Field(ge=0.0, le=1.0)


class HypothesisGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: str
    title: str
    round_label: str
    generation_strategy: str
    rank: int | None = None
    score: float | None = None
    cluster_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    status: str
    evidence_count: int = Field(ge=0)
    verified_evidence_count: int = Field(ge=0)


class HypothesisGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    target_id: str
    edge_type: Literal["similarity", "parent_child", "combination", "contradiction", "shared_evidence", "shared_assumption"]
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SearchSpaceCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    represented_mechanism_families: list[str] = Field(default_factory=list)
    represented_assumption_families: list[str] = Field(default_factory=list)
    underexplored_regions: list[str] = Field(default_factory=list)
    overrepresented_regions: list[str] = Field(default_factory=list)
    isolated_hypotheses: list[str] = Field(default_factory=list)
    duplicate_groups: list[list[str]] = Field(default_factory=list)
    diversity_score: float = Field(ge=0.0, le=1.0)
    collapse_risk: Literal["low", "medium", "high"]
    unique_cluster_count: int = Field(ge=0)
    largest_cluster_fraction: float = Field(ge=0.0, le=1.0)
    mean_pairwise_similarity: float = Field(ge=0.0, le=1.0)
    median_pairwise_similarity: float = Field(ge=0.0, le=1.0)
    isolated_hypothesis_count: int = Field(ge=0)
    effective_hypothesis_count: float = Field(ge=0.0)
    generation_strategy_coverage: list[str] = Field(default_factory=list)
    mechanism_family_coverage: int = Field(ge=0)
    evidence_source_concentration: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class ProximityAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    round_label: str
    round_number: int
    created_at: datetime
    agent_version: str = "proximity-v1"
    model_mode: str
    literature_mode: str
    pairwise_similarities: list[HypothesisSimilarity] = Field(default_factory=list)
    clusters: list[HypothesisCluster] = Field(default_factory=list)
    graph_nodes: list[HypothesisGraphNode] = Field(default_factory=list)
    graph_edges: list[HypothesisGraphEdge] = Field(default_factory=list)
    search_space_coverage: SearchSpaceCoverage
    method_metadata: dict[str, str | float | int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    validation_status: str = "unvalidated"


class GroundingEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: str
    paper_id: str
    title: str
    evidence_type: str
    excerpt: str
    verification_status: str | None = None
    source_field: str | None = None


class GroundingPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    round_label: str
    created_at: datetime
    mode: GroundingMode
    evidence_items: list[GroundingEvidenceItem] = Field(default_factory=list)
    source_limitations: list[str] = Field(default_factory=list)
    context_character_count: int = Field(ge=0)
    truncated: bool = False
    validation_status: str = "unvalidated"


class GroundingDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    round_label: str
    created_at: datetime
    grounding_mode: GroundingMode
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    claims_with_verified_evidence_ids: int = Field(ge=0)
    claims_citing_missing_evidence_ids: int = Field(ge=0)
    citation_hallucination_count: int = Field(ge=0)
    metadata_only_misuse_count: int = Field(ge=0)
    inference_as_source_fact_count: int = Field(ge=0)
    evidence_reuse_concentration: float = Field(ge=0.0, le=1.0)
    final_hypotheses_with_contradicting_evidence_fraction: float = Field(ge=0.0, le=1.0)
    grounding_coverage_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    validation_status: str = "unvalidated"


class MetaReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    round_label: str
    round_number: int
    created_at: datetime
    agent_version: str = "meta-review-v1"
    model_mode: str
    literature_mode: str
    feedback_mode: MetaReviewFeedbackMode = "advisory"
    executive_summary: str
    strongest_hypotheses: list[str] = Field(default_factory=list)
    recurring_strengths: list[str] = Field(default_factory=list)
    recurring_weaknesses: list[str] = Field(default_factory=list)
    unsupported_claim_patterns: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    novelty_risks: list[str] = Field(default_factory=list)
    feasibility_risks: list[str] = Field(default_factory=list)
    contradiction_patterns: list[str] = Field(default_factory=list)
    duplicate_or_collapse_patterns: list[str] = Field(default_factory=list)
    underexplored_directions: list[str] = Field(default_factory=list)
    ranking_concerns: list[str] = Field(default_factory=list)
    reviewer_concerns: list[str] = Field(default_factory=list)
    literature_coverage_concerns: list[str] = Field(default_factory=list)
    next_round_strategy: str
    recommended_generation_strategies: list[str] = Field(default_factory=list)
    recommended_search_queries: list[str] = Field(default_factory=list)
    recommended_hypothesis_merges: list[list[str]] = Field(default_factory=list)
    recommended_hypothesis_branches: list[str] = Field(default_factory=list)
    recommended_hypothesis_repairs: list[str] = Field(default_factory=list)
    recommended_hypotheses_to_hold: list[str] = Field(default_factory=list)
    recommended_falsification_tests: list[str] = Field(default_factory=list)
    stopping_assessment: StoppingAssessment
    stopping_reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    referenced_cluster_ids: list[str] = Field(default_factory=list)
    referenced_evidence_ids: list[str] = Field(default_factory=list)
    referenced_verification_ids: list[str] = Field(default_factory=list)
    validation_status: str = "unvalidated"


class MetaReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    round_label: str
    feedback_mode: MetaReviewFeedbackMode
    feed_into_next_round: bool
    accepted_generation_strategy_adjustments: list[str] = Field(default_factory=list)
    accepted_search_queries: list[str] = Field(default_factory=list)
    selected_repairs: list[str] = Field(default_factory=list)
    selected_branches: list[str] = Field(default_factory=list)
    selected_combinations: list[list[str]] = Field(default_factory=list)
    held_hypotheses: list[str] = Field(default_factory=list)
    rejected_recommendations: list[str] = Field(default_factory=list)
    decision_rationale: str


class V15BSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v15b"
    project_id: str
    run_id: str
    created_at: datetime
    proximity_enabled: bool
    meta_review_enabled: bool
    grounding_mode: GroundingMode
    diversity_score: float = Field(ge=0.0, le=1.0)
    collapse_risk: str
    grounding_coverage_score: float = Field(ge=0.0, le=1.0)
    stopping_assessment: StoppingAssessment
    recommended_next_task: str
