from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.prompts.templates import combination_prompt, evolution_prompt
from coscientist.schemas.hypothesis import Hypothesis, HypothesisBatch
from coscientist.schemas.review import Review


class EvolutionAgent(Agent):
    async def repair(self, hypothesis: Hypothesis, reviews: list[Review], round_number: int) -> list[Hypothesis]:
        prompt = evolution_prompt(hypothesis, reviews, "repair", 1)
        batch = await self.provider.generate_structured(
            prompt,
            HypothesisBatch,
            {
                "mode": "repair",
                "strategy": "repair",
                "parent": hypothesis,
                "count": 1,
                "round_number": round_number,
                "agent_role": "evolution",
                "workflow_stage": f"repair_round_{round_number}",
            },
        )
        return batch.hypotheses

    async def branch(self, hypothesis: Hypothesis, reviews: list[Review], round_number: int) -> list[Hypothesis]:
        prompt = evolution_prompt(hypothesis, reviews, "branch", 1)
        batch = await self.provider.generate_structured(
            prompt,
            HypothesisBatch,
            {
                "mode": "branch",
                "strategy": "branch",
                "parent": hypothesis,
                "count": 1,
                "round_number": round_number,
                "agent_role": "evolution",
                "workflow_stage": f"branch_round_{round_number}",
            },
        )
        return batch.hypotheses

    async def combine(self, parents: list[Hypothesis], round_number: int, count: int = 1) -> list[Hypothesis]:
        if len(parents) < 2:
            return []
        prompt = combination_prompt(parents, count)
        batch = await self.provider.generate_structured(
            prompt,
            HypothesisBatch,
            {
                "mode": "combine",
                "strategy": "combine",
                "parents": parents,
                "count": count,
                "round_number": round_number,
                "agent_role": "evolution",
                "workflow_stage": f"combine_round_{round_number}",
            },
        )
        return batch.hypotheses

    async def evolve(
        self,
        selected: list[Hypothesis],
        reviews: list[Review],
        round_number: int,
        children_per_selected: int,
    ) -> list[Hypothesis]:
        review_map = {review.hypothesis_id: review for review in reviews}
        children: list[Hypothesis] = []
        for hypothesis in selected:
            local_reviews = [review_map[hypothesis.id]] if hypothesis.id in review_map else []
            children.extend(await self.repair(hypothesis, local_reviews, round_number))
            if children_per_selected > 1:
                children.extend(await self.branch(hypothesis, local_reviews, round_number))
        if selected and children_per_selected > 1:
            children.extend(await self.combine(selected[:2], round_number, count=1))
        return children
