from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScientificTaskType(str, Enum):
    THEORY_DERIVATION = "theory_derivation"
    NUMERICAL_MODELING = "numerical_modeling"
    DATA_EXTRACTION = "data_extraction"
    EXPERIMENT_SELECTION = "experiment_selection"
    PHASE_IDENTIFICATION = "phase_identification"
    HIDDEN_ANSWER_BENCHMARK = "hidden_answer_benchmark"
    MATERIAL_COMPARISON = "material_comparison"


class TaskTypePolicy(BaseModel):
    schema_version: str = "v26-task-policy"
    task_type: ScientificTaskType
    required_stages: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    stopping_conditions: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    live_allowed: bool = False


class TaskPolicyRegistry:
    def __init__(self, policies: dict[ScientificTaskType, TaskTypePolicy] | None = None):
        self._policies = policies or {}

    @classmethod
    def default(cls) -> "TaskPolicyRegistry":
        return cls({
            ScientificTaskType.THEORY_DERIVATION: TaskTypePolicy(
                task_type=ScientificTaskType.THEORY_DERIVATION,
                required_stages=["formalize", "derive", "verify", "record_limits"],
                success_criteria=["explicit equations", "falsification criteria", "artifact-backed derivation"],
                stopping_conditions=["missing assumptions", "failed hard gate", "derivation contradiction"],
                required_artifacts=["derivation_summary.json", "verifier_results.jsonl"],
            ),
            ScientificTaskType.DATA_EXTRACTION: TaskTypePolicy(
                task_type=ScientificTaskType.DATA_EXTRACTION,
                required_stages=["discover", "extract", "normalize", "review", "promote"],
                success_criteria=["source provenance", "unit normalization", "reviewed promotion"],
                stopping_conditions=["bad provenance", "definition conflict", "coverage gate failed"],
                required_artifacts=["candidate_rows.jsonl", "readiness_gates.json"],
            ),
            ScientificTaskType.MATERIAL_COMPARISON: TaskTypePolicy(
                task_type=ScientificTaskType.MATERIAL_COMPARISON,
                required_stages=["curate", "split", "fit", "held_out_evaluate", "identifiability_check"],
                success_criteria=["no leakage", "per-observable metrics", "comparison uncertainty"],
                stopping_conditions=["no overlapping observables", "held-out split impossible", "unidentifiable parameters"],
                required_artifacts=["comparison_robustness.json"],
            ),
            ScientificTaskType.PHASE_IDENTIFICATION: TaskTypePolicy(
                task_type=ScientificTaskType.PHASE_IDENTIFICATION,
                required_stages=["ingest_pattern", "candidate_phases", "fit", "residual_audit"],
                success_criteria=["indexed peaks", "residual checks", "ambiguous phases preserved"],
                stopping_conditions=["insufficient peaks", "unresolved overlapping phases"],
                required_artifacts=["phase_candidates.jsonl"],
            ),
            ScientificTaskType.HIDDEN_ANSWER_BENCHMARK: TaskTypePolicy(
                task_type=ScientificTaskType.HIDDEN_ANSWER_BENCHMARK,
                required_stages=["hide_answer", "solve", "evaluate"],
                success_criteria=["ground truth not exposed", "objective score"],
                stopping_conditions=["answer leakage", "unsupported proof claim"],
                required_artifacts=["benchmark_result.json"],
            ),
            ScientificTaskType.NUMERICAL_MODELING: TaskTypePolicy(
                task_type=ScientificTaskType.NUMERICAL_MODELING,
                required_stages=["specify_model", "solve", "independent_reproduction", "ledger"],
                success_criteria=["bounded scan", "reproducible result", "energy or residual ledger"],
                stopping_conditions=["unstable solver", "failed reproduction"],
                required_artifacts=["model_solution.json", "independent_reproduction.jsonl"],
            ),
            ScientificTaskType.EXPERIMENT_SELECTION: TaskTypePolicy(
                task_type=ScientificTaskType.EXPERIMENT_SELECTION,
                required_stages=["define_options", "estimate_information_gain", "rank", "review"],
                success_criteria=["discriminates leading mechanisms", "cost and feasibility recorded"],
                stopping_conditions=["no discriminating observable", "unavailable measurement"],
                required_artifacts=["information_gain_queue.jsonl"],
            ),
        })

    def get(self, task_type: ScientificTaskType | str) -> TaskTypePolicy:
        parsed = ScientificTaskType(task_type)
        if parsed not in self._policies:
            raise KeyError(f"no task policy registered for {parsed.value}")
        return self._policies[parsed]

    def list(self) -> list[TaskTypePolicy]:
        return [self._policies[key] for key in sorted(self._policies, key=lambda item: item.value)]
