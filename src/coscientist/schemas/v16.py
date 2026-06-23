from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


QuestionType = Literal["single_choice", "multi_choice", "ranking", "numeric"]
EvaluationMethod = Literal["exact_match", "set_match", "ranking_agreement", "numeric_tolerance"]
AnswerRelation = Literal["supports", "contradicts", "neutral", "insufficient"]
FinalAnswerOutcome = Literal[
    "correct",
    "incorrect",
    "correct_abstention",
    "unnecessary_abstention",
    "overconfident_error",
    "insufficient_evidence",
]


class TokenBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_total_model_calls: int = Field(default=20, ge=0, le=200)
    max_total_tokens: int = Field(default=45000, ge=0)
    stop_on_budget_exhaustion: bool = True


class ClosedQuestionGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_hypotheses: int = Field(default=8, ge=1, le=50)


class ClosedQuestionGroundingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_evidence_items_per_hypothesis: int = Field(default=5, ge=1, le=50)
    max_contradiction_items_per_hypothesis: int = Field(default=2, ge=0, le=20)
    max_chunks_per_source: int = Field(default=2, ge=1, le=20)
    max_chunk_characters: int = Field(default=1000, ge=100, le=10000)
    max_total_context_characters: int = Field(default=12000, ge=1000, le=100000)


class ClosedQuestionReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reuse_if_unchanged: bool = True
    max_output_tokens: int = Field(default=900, ge=100, le=8000)


class ClosedQuestionRankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_pairwise_candidates: int = Field(default=6, ge=1, le=50)
    max_pairwise_calls: int = Field(default=8, ge=0, le=200)
    use_cluster_representatives: bool = True


class ClosedQuestionMetaReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    frequency_rounds: int = Field(default=2, ge=1, le=10)
    max_output_tokens: int = Field(default=1200, ge=100, le=8000)


class AnswerSynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_input_hypotheses: int = Field(default=6, ge=1, le=50)
    max_evidence_items: int = Field(default=10, ge=1, le=100)
    max_output_tokens: int = Field(default=900, ge=100, le=8000)


class AnswerPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    allow_abstention: bool = True
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    minimum_verified_evidence_items: int = Field(default=2, ge=0, le=50)
    minimum_independent_clusters: int = Field(default=2, ge=0, le=50)
    max_unresolved_fatal_flaws: int = Field(default=0, ge=0, le=50)
    require_contradiction_review: bool = True
    require_valid_artifact_references: bool = True


class ClosedQuestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    generation: ClosedQuestionGenerationConfig = Field(default_factory=ClosedQuestionGenerationConfig)
    grounding: ClosedQuestionGroundingConfig = Field(default_factory=ClosedQuestionGroundingConfig)
    review: ClosedQuestionReviewConfig = Field(default_factory=ClosedQuestionReviewConfig)
    ranking: ClosedQuestionRankingConfig = Field(default_factory=ClosedQuestionRankingConfig)
    meta_review: ClosedQuestionMetaReviewConfig = Field(default_factory=ClosedQuestionMetaReviewConfig)
    answer_synthesis: AnswerSynthesisConfig = Field(default_factory=AnswerSynthesisConfig)
    answer_policy: AnswerPolicyConfig = Field(default_factory=AnswerPolicyConfig)


class AnswerOption(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    category: str | None = None
    constraints: list[str] = Field(default_factory=list)


class ClosedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    question_type: QuestionType
    answer_options: list[AnswerOption] = Field(default_factory=list)
    allowed_answer_count: int = Field(default=1, ge=1)
    allow_abstention: bool = True
    evaluation_method: EvaluationMethod
    tolerance: float | None = Field(default=None, ge=0.0)
    units: str | None = None
    corpus_cutoff_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_question(self) -> ClosedQuestion:
        if self.question_type != "numeric" and not self.answer_options:
            raise ValueError("non-numeric closed questions require answer_options")
        if self.question_type == "single_choice" and self.allowed_answer_count != 1:
            raise ValueError("single_choice questions require allowed_answer_count=1")
        if self.question_type == "numeric" and self.tolerance is None:
            raise ValueError("numeric questions require tolerance")
        if self.question_type == "multi_choice" and self.allowed_answer_count > len(self.answer_options):
            raise ValueError("allowed_answer_count cannot exceed answer option count")
        return self


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    question_id: str
    correct_answer_ids: list[str] = Field(default_factory=list)
    correct_ranking: list[str] = Field(default_factory=list)
    numeric_value: float | None = None
    numeric_interval: list[float] | None = None
    acceptable_abstention: bool = False
    rationale: str
    provenance: list[str] = Field(default_factory=list)
    hidden_from_agents: bool = True

    @model_validator(mode="after")
    def ground_truth_hidden(self) -> GroundTruth:
        if not self.hidden_from_agents:
            raise ValueError("ground truth must be hidden from agents")
        if self.numeric_interval is not None and len(self.numeric_interval) != 2:
            raise ValueError("numeric_interval must contain lower and upper bounds")
        return self


class HypothesisAnswerLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_id: str
    hypothesis_id: str
    answer_id: str
    relation: AnswerRelation
    relevance: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    rationale_summary: str
    method: Literal["deterministic_metadata", "deterministic_text", "model_interpretation"] = "deterministic_metadata"
    validation_status: str = "validated"


class AnswerEvidenceCell(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_id: str
    answer_id: str
    supporting_hypothesis_ids: list[str] = Field(default_factory=list)
    contradicting_hypothesis_ids: list[str] = Field(default_factory=list)
    independent_cluster_ids: list[str] = Field(default_factory=list)
    verified_support_count: int = Field(ge=0)
    verified_contradiction_count: int = Field(ge=0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    duplicate_adjusted_support: float = Field(ge=0.0)
    unresolved_issues: list[str] = Field(default_factory=list)
    validation_status: str = "validated"


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_id: str
    selected_answer_ids: list[str] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    numeric_estimate: float | None = None
    numeric_interval: list[float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    abstention_reason: str | None = None
    supporting_hypothesis_ids: list[str] = Field(default_factory=list)
    supporting_cluster_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    rationale_summary: str
    remaining_uncertainty: list[str] = Field(default_factory=list)
    recommended_next_action: str
    validation_status: str = "unvalidated"
    created_at: datetime


class ClosedQuestionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_id: str
    correct: bool
    exact_match: bool | None = None
    top_k_correct: bool | None = None
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    ranking_score: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric_error: float | None = Field(default=None, ge=0.0)
    relative_error: float | None = Field(default=None, ge=0.0)
    within_tolerance: bool | None = None
    abstention_correct: bool | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    calibration_bin: str
    evidence_sufficiency: str
    token_cost: int | None = None
    call_count: int
    outcome: FinalAnswerOutcome
    evaluated_at: datetime


class AnswerCalibrationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    selective_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    bins: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    correct_answers_per_million_tokens: float | None = None
    cost_per_correct_answer: float | None = None
    validation_status: str = "validated"


class ClosedQuestionProject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    schema_version: str = "v16"
    project_id: str
    title: str
    description: str
    model_mode: Literal["mock", "live"] = "mock"
    literature_mode: Literal["fixture", "existing"] = "existing"
    grounding_mode: Literal["strict"] = "strict"
    corpus_path: str
    questions: list[ClosedQuestion] = Field(min_length=1)
    ground_truth: list[GroundTruth] = Field(default_factory=list)
    config: ClosedQuestionConfig = Field(default_factory=ClosedQuestionConfig)
    created_at: datetime


class ContextBuildRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "v16"
    project_id: str
    run_id: str
    question_id: str
    context_type: Literal["generator", "reviewer", "ranker", "answer_synthesis"]
    input_character_count: int = Field(ge=0)
    output_character_count: int = Field(ge=0)
    omitted_items: list[str] = Field(default_factory=list)
    context_hash: str
    validation_status: str = "validated"
