from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from coscientist.core.hypotheses_v2 import (
    HardGateResult,
    HypothesisLineageV2,
    HypothesisStatusV2,
    HypothesisV2,
    ScoreEntry,
    ScoreProvenance,
    migrate_hypothesis_to_v2,
)
from coscientist.pilot.artifacts import write_json, write_jsonl


MAXIMIZE_DIMENSIONS = {
    "correctness",
    "evidence_support",
    "novelty",
    "falsifiability",
    "testability",
    "explanatory_power",
    "predictive_specificity",
    "expected_information_gain",
    "computational_feasibility",
    "experimental_feasibility",
    "robustness",
    "diversity",
}


class CheapKillResult(BaseModel):
    schema_version: str = "v26-cheap-kill-result"
    hypothesis_id: str
    killed: bool
    reasons: list[str] = Field(default_factory=list)
    cost_units: int = 1


class MutationOperator(BaseModel):
    schema_version: str = "v26-mutation-operator"
    operator_id: str
    description: str
    target_failure_modes: list[str] = Field(default_factory=list)


class PortfolioEntry(BaseModel):
    schema_version: str = "v26-portfolio-entry"
    hypothesis_id: str
    role: str
    rationale: str
    pareto_frontier: bool = False


class ExpectedInformationGainAction(BaseModel):
    schema_version: str = "v26-eig-action"
    action_id: str
    hypothesis_id: str
    action_type: str
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    success_probability: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    cost: float = Field(gt=0.0)
    priority: float
    rationale: str


class FailureMemoryRecord(BaseModel):
    schema_version: str = "v26-failure-memory"
    failure_id: str
    hypothesis_id: str
    domain_id: str
    failure_mode: str
    lesson: str
    created_at: str


class CounterexampleTask(BaseModel):
    schema_version: str = "v26-counterexample-task"
    task_id: str
    hypothesis_id: str
    target_claim: str
    status: str = "queued"
    rationale: str


class OptimizerV2Result(BaseModel):
    schema_version: str = "v26-optimizer-result"
    run_id: str
    domain_id: str
    hypothesis_count: int
    killed_count: int
    pareto_frontier_ids: list[str]
    portfolio: list[PortfolioEntry]
    information_gain_actions: list[ExpectedInformationGainAction]
    failure_memory: list[FailureMemoryRecord]
    counterexample_tasks: list[CounterexampleTask]
    artifact_ids: list[str] = Field(default_factory=list)


@dataclass
class HypothesisOptimizerV2:
    domain_id: str = "general"
    run_id: str = "optimizer-v2"

    def optimize(self, hypotheses: list[Any], *, run_dir: str | Path | None = None) -> OptimizerV2Result:
        migrated = [migrate_hypothesis_to_v2(item, domain_id=self.domain_id) for item in hypotheses]
        gates = [gate for hypothesis in migrated for gate in self._hard_gates(hypothesis)]
        kills = [self._cheap_kill(hypothesis) for hypothesis in migrated]
        failure_memory = [self._failure_memory(hypothesis, kill) for hypothesis, kill in zip(migrated, kills, strict=True) if kill.killed]
        scored = [self._with_scores(hypothesis, gates, kills) for hypothesis in migrated]
        active = [hypothesis for hypothesis, kill in zip(scored, kills, strict=True) if not kill.killed]
        pareto_ids = pareto_frontier(active)
        portfolio = build_portfolio(active, pareto_ids)
        actions = build_information_gain_actions(active, kills)
        counterexamples = [
            CounterexampleTask(
                task_id=f"counterexample-{_digest(hypothesis.hypothesis_id)}",
                hypothesis_id=hypothesis.hypothesis_id,
                target_claim=hypothesis.scoped_claim,
                rationale="Try to falsify the leading scoped claim before promotion.",
            )
            for hypothesis in active[:3]
        ]
        artifacts: list[str] = []
        if run_dir is not None:
            root = Path(run_dir)
            root.mkdir(parents=True, exist_ok=True)
            write_jsonl(root / "hypotheses_v2.jsonl", scored)
            write_jsonl(root / "hard_gate_results.jsonl", gates)
            write_jsonl(root / "cheap_kill_results.jsonl", kills)
            write_jsonl(root / "hypothesis_portfolio_v2.jsonl", portfolio)
            write_jsonl(root / "information_gain_queue.jsonl", actions)
            write_jsonl(root / "failure_memory.jsonl", failure_memory)
            write_jsonl(root / "counterexample_tasks.jsonl", counterexamples)
            artifacts = [
                "hypotheses_v2.jsonl",
                "hard_gate_results.jsonl",
                "cheap_kill_results.jsonl",
                "hypothesis_portfolio_v2.jsonl",
                "information_gain_queue.jsonl",
                "failure_memory.jsonl",
                "counterexample_tasks.jsonl",
                "optimizer_v2_summary.json",
            ]
        result = OptimizerV2Result(
            run_id=self.run_id,
            domain_id=self.domain_id,
            hypothesis_count=len(scored),
            killed_count=sum(1 for item in kills if item.killed),
            pareto_frontier_ids=pareto_ids,
            portfolio=portfolio,
            information_gain_actions=actions,
            failure_memory=failure_memory,
            counterexample_tasks=counterexamples,
            artifact_ids=artifacts,
        )
        if run_dir is not None:
            write_json(Path(run_dir) / "optimizer_v2_summary.json", result)
        return result

    def _hard_gates(self, hypothesis: HypothesisV2) -> list[HardGateResult]:
        checks = [
            ("scope_defined", bool(hypothesis.scoped_claim.strip()), "Scoped claim must be explicit."),
            ("falsifiable_prediction_present", bool(hypothesis.falsification_criteria and hypothesis.predictions), "Predictions and falsification criteria are required."),
            ("required_tool_exists_or_is_implementable", True, "This checkpoint treats declared tools as implementable unless a pack marks otherwise."),
            ("no_unresolved_fatal_conflict", not hypothesis.contradiction_artifact_ids, "Unresolved contradictions block finalist promotion."),
        ]
        return [
            HardGateResult(
                gate_id=gate_id,
                status="pass" if passed else "fail",
                hypothesis_id=hypothesis.hypothesis_id,
                rationale=rationale,
                evidence_artifact_ids=hypothesis.evidence_artifact_ids,
            )
            for gate_id, passed, rationale in checks
        ]

    def _cheap_kill(self, hypothesis: HypothesisV2) -> CheapKillResult:
        reasons: list[str] = []
        lowered = hypothesis.scoped_claim.lower()
        if not hypothesis.falsification_criteria:
            reasons.append("unfalsifiable")
        if not hypothesis.predictions:
            reasons.append("no_predictions")
        if "prove everything" in lowered or "always true" in lowered:
            reasons.append("unbounded_claim")
        if "meter + second" in lowered or "meters + seconds" in lowered:
            reasons.append("dimensionally_invalid")
        return CheapKillResult(hypothesis_id=hypothesis.hypothesis_id, killed=bool(reasons), reasons=reasons)

    def _with_scores(self, hypothesis: HypothesisV2, gates: list[HardGateResult], kills: list[CheapKillResult]) -> HypothesisV2:
        failed_gate_count = sum(1 for item in gates if item.hypothesis_id == hypothesis.hypothesis_id and item.status == "fail")
        killed = any(item.hypothesis_id == hypothesis.hypothesis_id and item.killed for item in kills)
        evidence_score = min(1.0, len(hypothesis.evidence_artifact_ids) / 3.0)
        falsifiability = 1.0 if hypothesis.falsification_criteria else 0.0
        predictions = min(1.0, len(hypothesis.predictions) / 2.0)
        feasibility = 0.4 + 0.2 * bool(hypothesis.required_tools)
        if failed_gate_count:
            feasibility *= 0.5
        score_specs = {
            "correctness": 0.5 if not failed_gate_count else 0.2,
            "evidence_support": evidence_score,
            "novelty": 0.6 if hypothesis.novelty_notes else 0.4,
            "falsifiability": falsifiability,
            "testability": predictions,
            "explanatory_power": 0.6,
            "predictive_specificity": predictions,
            "expected_information_gain": 0.8 if not killed else 0.1,
            "computational_feasibility": feasibility,
            "experimental_feasibility": 0.5,
            "robustness": 0.7 if not hypothesis.contradiction_artifact_ids else 0.2,
            "diversity": 0.5,
        }
        entries = [
            ScoreEntry(
                dimension=dimension,
                value=value,
                provenance=ScoreProvenance(method="deterministic_optimizer_v2_checkpoint", rationale="Rule-based checkpoint score."),
            )
            for dimension, value in score_specs.items()
        ]
        status = HypothesisStatusV2.KILLED if killed else HypothesisStatusV2.ACTIVE
        return hypothesis.model_copy(update={"score_vector": entries, "hard_gate_results": [item for item in gates if item.hypothesis_id == hypothesis.hypothesis_id], "status": status})

    def _failure_memory(self, hypothesis: HypothesisV2, kill: CheapKillResult) -> FailureMemoryRecord:
        mode = ",".join(kill.reasons) or "unknown"
        return FailureMemoryRecord(
            failure_id=f"failure-{_digest(hypothesis.hypothesis_id + mode)}",
            hypothesis_id=hypothesis.hypothesis_id,
            domain_id=hypothesis.domain_id,
            failure_mode=mode,
            lesson=f"Avoid regenerating branches with failure mode: {mode}.",
            created_at=datetime.now(UTC).isoformat(),
        )


def pareto_frontier(hypotheses: list[HypothesisV2]) -> list[str]:
    frontier: list[str] = []
    for candidate in hypotheses:
        if not any(_dominates(other, candidate) for other in hypotheses if other.hypothesis_id != candidate.hypothesis_id):
            frontier.append(candidate.hypothesis_id)
    return frontier


def build_portfolio(hypotheses: list[HypothesisV2], pareto_ids: list[str]) -> list[PortfolioEntry]:
    entries: list[PortfolioEntry] = []
    if not hypotheses:
        return entries
    for hypothesis in hypotheses:
        role = "pareto_candidate" if hypothesis.hypothesis_id in pareto_ids else "reserve"
        if "counter" in hypothesis.scoped_claim.lower() or "contrarian" in hypothesis.title.lower():
            role = "contrarian_guardrail"
        entries.append(PortfolioEntry(
            hypothesis_id=hypothesis.hypothesis_id,
            role=role,
            rationale="Selected by deterministic Pareto and diversity portfolio checkpoint.",
            pareto_frontier=hypothesis.hypothesis_id in pareto_ids,
        ))
    if not any(item.role == "contrarian_guardrail" for item in entries):
        weakest = entries[-1]
        entries[-1] = weakest.model_copy(update={"role": "contrarian_guardrail", "rationale": "Preserved one branch for counterexample pressure."})
    return entries


def build_information_gain_actions(hypotheses: list[HypothesisV2], kills: list[CheapKillResult]) -> list[ExpectedInformationGainAction]:
    actions: list[ExpectedInformationGainAction] = []
    killed_ids = {item.hypothesis_id for item in kills if item.killed}
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in killed_ids:
            continue
        eig = _score(hypothesis, "expected_information_gain")
        feasibility = max(_score(hypothesis, "computational_feasibility"), 0.1)
        probability = max(_score(hypothesis, "falsifiability"), 0.1)
        cost = 1.0 + len(hypothesis.required_tools) * 0.25
        priority = eig * probability * feasibility / cost
        actions.append(ExpectedInformationGainAction(
            action_id=f"eig-{_digest(hypothesis.hypothesis_id)}",
            hypothesis_id=hypothesis.hypothesis_id,
            action_type="run_targeted_verifier_or_data_query",
            expected_information_gain=eig,
            success_probability=probability,
            feasibility=feasibility,
            cost=cost,
            priority=round(priority, 6),
            rationale="Prioritize actions that can falsify or materially separate active hypotheses.",
        ))
    return sorted(actions, key=lambda item: item.priority, reverse=True)


def run_optimizer_v2(hypotheses: list[Any], *, run_dir: str | Path, domain_id: str = "general", run_id: str = "optimizer-v2") -> OptimizerV2Result:
    return HypothesisOptimizerV2(domain_id=domain_id, run_id=run_id).optimize(hypotheses, run_dir=run_dir)


def mutation_operators() -> list[MutationOperator]:
    return [
        MutationOperator(operator_id="repair_scope", description="Narrow an overbroad claim.", target_failure_modes=["unbounded_claim"]),
        MutationOperator(operator_id="add_falsifier", description="Add a falsification criterion and held-out prediction.", target_failure_modes=["unfalsifiable", "no_predictions"]),
        MutationOperator(operator_id="branch_counterexample", description="Create a contrarian branch that targets the weakest assumption.", target_failure_modes=["overconfident_branch"]),
        MutationOperator(operator_id="combine_independent_support", description="Combine non-duplicate evidence paths while preserving independence metadata.", target_failure_modes=["duplicate_support"]),
    ]


def _dominates(left: HypothesisV2, right: HypothesisV2) -> bool:
    left_scores = {item.dimension: item.value for item in left.score_vector}
    right_scores = {item.dimension: item.value for item in right.score_vector}
    dims = sorted(MAXIMIZE_DIMENSIONS.intersection(left_scores).intersection(right_scores))
    if not dims:
        return False
    return all(left_scores[dim] >= right_scores[dim] for dim in dims) and any(left_scores[dim] > right_scores[dim] for dim in dims)


def _score(hypothesis: HypothesisV2, dimension: str) -> float:
    for item in hypothesis.score_vector:
        if item.dimension == dimension:
            return item.value
    return 0.0


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
