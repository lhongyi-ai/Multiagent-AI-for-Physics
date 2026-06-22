from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from coscientist.agents.evolution import EvolutionAgent
from coscientist.agents.generator import GeneratorAgent
from coscientist.agents.ranker import RankerAgent
from coscientist.agents.reviewer import ReviewerAgent
from coscientist.agents.supervisor import BudgetExhausted, Supervisor
from coscientist.config import DEFAULT_STRATEGIES, WorkflowConfig
from coscientist.literature.pipeline import LiteratureAcquisitionResult, build_literature_pipeline, no_literature_result
from coscientist.providers.base import StructuredLLMProvider
from coscientist.reporting.markdown_report import build_markdown_report
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.ranking import HypothesisRanking
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.review import Review
from coscientist.schemas.run_state import RunState
from coscientist.storage.local_store import LocalStore


class WorkflowResult:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        finalists: list[Hypothesis],
        state: RunState,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.finalists = finalists
        self.state = state


class CoScientistWorkflow:
    def __init__(
        self,
        provider: StructuredLLMProvider,
        config: WorkflowConfig,
        store: LocalStore | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.store = store or LocalStore()
        self.generator = GeneratorAgent(provider)
        self.reviewer = ReviewerAgent(provider)
        self.ranker = RankerAgent(provider)
        self.evolution = EvolutionAgent(provider)
        self.literature = build_literature_pipeline(config.literature) if config.literature.enabled else None

    async def run(self, goal: ResearchGoal, run_id: str | None = None) -> WorkflowResult:
        run_id = run_id or self._new_run_id(goal)
        state = RunState(
            run_id=run_id,
            phase="created",
            round_number=0,
            maximum_llm_calls=self.config.max_llm_calls,
            timestamps={"created": self._now()},
        )
        supervisor = Supervisor(state)
        self.store.ensure_run(run_id)
        self.store.write_json(run_id, "run_config.json", self.config)
        self.store.write_json(run_id, "research_goal.json", goal)
        self._save_state(state)

        all_hypotheses: dict[str, Hypothesis] = {}
        latest_hypotheses = await self._generate_initial(goal, supervisor)
        all_hypotheses.update({hypothesis.id: hypothesis for hypothesis in latest_hypotheses})
        self.store.write_json(run_id, "hypotheses_initial.json", latest_hypotheses)
        self._phase(state, "initial_generation", 0, [h.id for h in latest_hypotheses])

        literature_result = await self._acquire_literature(goal.question, state, round_number=0)

        reviews, rankings = await self._review_and_rank(latest_hypotheses, supervisor, round_number=0)
        self.store.write_json(run_id, "reviews_round_0.json", reviews)
        self.store.write_json(run_id, "ranking_round_0.json", rankings)
        selected = self._select(latest_hypotheses, rankings, self.config.top_k_after_review)
        self._phase(state, "ranking_round_0", 0, [h.id for h in selected])

        for round_number in range(1, self.config.evolution_rounds + 1):
            latest_hypotheses = await self._evolve(selected, reviews, supervisor, round_number)
            all_hypotheses.update({hypothesis.id: hypothesis for hypothesis in latest_hypotheses})
            self.store.write_json(run_id, f"hypotheses_round_{round_number}.json", latest_hypotheses)
            self._phase(state, f"evolution_round_{round_number}", round_number, [h.id for h in latest_hypotheses])

            reviews, rankings = await self._review_and_rank(latest_hypotheses, supervisor, round_number)
            self.store.write_json(run_id, f"reviews_round_{round_number}.json", reviews)
            self.store.write_json(run_id, f"ranking_round_{round_number}.json", rankings)
            selected = self._select(latest_hypotheses, rankings, self.config.top_k_after_review)
            self._phase(state, f"ranking_round_{round_number}", round_number, [h.id for h in selected])

        final_rankings = rankings
        finalists = self._select(selected, final_rankings, self.config.final_top_k)
        report = build_markdown_report(goal, finalists, final_rankings, state, literature_result)
        self.store.write_text(run_id, "final_report.md", report)
        self.store.write_json(run_id, "all_hypotheses.json", list(all_hypotheses.values()))
        self._phase(state, "complete", self.config.evolution_rounds, [h.id for h in finalists])
        return WorkflowResult(run_id, self.store.run_dir(run_id), finalists, state)

    async def _acquire_literature(
        self,
        query: str,
        state: RunState,
        round_number: int,
    ) -> LiteratureAcquisitionResult:
        if self.literature is None:
            return no_literature_result()
        result = await self.literature.acquire(query)
        run_id = state.run_id
        self.store.write_jsonl(run_id, f"provider_requests_round_{round_number}.jsonl", result.provider_requests)
        self.store.write_json(run_id, f"papers_raw_round_{round_number}.json", result.raw_papers)
        self.store.write_json(run_id, f"papers_normalized_round_{round_number}.json", result.normalized_papers)
        self.store.write_json(run_id, f"metadata_resolutions_round_{round_number}.json", result.metadata_resolutions)
        self.store.write_json(run_id, f"metadata_conflicts_round_{round_number}.json", result.metadata_conflicts)
        self.store.write_json(run_id, f"full_text_locations_round_{round_number}.json", result.full_text_locations)
        self.store.write_json(run_id, f"document_retrieval_round_{round_number}.json", [])
        self.store.write_json(run_id, f"citations_round_{round_number}.json", [])
        self.store.write_json(run_id, f"citation_verifications_round_{round_number}.json", result.citation_verifications)
        self.store.write_json(run_id, f"evidence_claims_round_{round_number}.json", result.evidence_claims)
        self._phase(state, f"literature_round_{round_number}", round_number, state.active_hypothesis_ids)
        return result

    async def _generate_initial(self, goal: ResearchGoal, supervisor: Supervisor) -> list[Hypothesis]:
        strategies = DEFAULT_STRATEGIES[: self.config.generators]
        supervisor.reserve(len(strategies), "initial_generation")
        semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

        async def run_one(strategy: str) -> list[Hypothesis]:
            async with semaphore:
                return await self.generator.generate(goal, strategy, self.config.hypotheses_per_generator)

        generated = await asyncio.gather(*(run_one(strategy) for strategy in strategies))
        return [hypothesis for group in generated for hypothesis in group]

    async def _review_and_rank(
        self,
        hypotheses: list[Hypothesis],
        supervisor: Supervisor,
        round_number: int,
    ) -> tuple[list[Review], list[HypothesisRanking]]:
        supervisor.reserve(1, f"review_round_{round_number}")
        reviews = await self.reviewer.review(hypotheses)
        supervisor.reserve(2, f"rank_round_{round_number}")
        rankings = await self.ranker.rank(hypotheses, reviews, self.config, round_number=round_number, seed=11)
        return reviews, rankings

    async def _evolve(
        self,
        selected: list[Hypothesis],
        reviews: list[Review],
        supervisor: Supervisor,
        round_number: int,
    ) -> list[Hypothesis]:
        calls = len(selected) * min(self.config.children_per_selected_hypothesis, 2)
        if selected and self.config.children_per_selected_hypothesis > 1:
            calls += 1
        supervisor.reserve(calls, f"evolution_round_{round_number}")
        return await self.evolution.evolve(
            selected,
            reviews,
            round_number=round_number,
            children_per_selected=self.config.children_per_selected_hypothesis,
        )

    def _select(
        self,
        hypotheses: list[Hypothesis],
        rankings: list[HypothesisRanking],
        count: int,
    ) -> list[Hypothesis]:
        by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        ordered = sorted(
            rankings,
            key=lambda item: (item.weighted_total, item.pairwise_wins, -item.pairwise_losses, item.hypothesis_id),
            reverse=True,
        )
        return [by_id[ranking.hypothesis_id] for ranking in ordered[:count] if ranking.hypothesis_id in by_id]

    def _phase(self, state: RunState, phase: str, round_number: int, active_ids: list[str]) -> None:
        state.phase = phase
        state.round_number = round_number
        state.active_hypothesis_ids = active_ids
        state.timestamps[phase] = self._now()
        self._save_state(state)

    def _save_state(self, state: RunState) -> None:
        self.store.write_json(state.run_id, "run_state.json", state)
        self.store.append_log(
            state.run_id,
            state.phase,
            {
                "round_number": state.round_number,
                "llm_call_count": state.llm_call_count,
                "active_hypothesis_ids": state.active_hypothesis_ids,
                "errors": state.errors,
            },
        )

    @staticmethod
    def _new_run_id(goal: ResearchGoal) -> str:
        return f"{goal.id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()


async def run_workflow(
    goal: ResearchGoal,
    provider: StructuredLLMProvider,
    config: WorkflowConfig,
    store: LocalStore | None = None,
    run_id: str | None = None,
) -> WorkflowResult:
    try:
        return await CoScientistWorkflow(provider, config, store).run(goal, run_id)
    except BudgetExhausted:
        raise
