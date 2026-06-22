from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.prompts.templates import generator_prompt
from coscientist.schemas.hypothesis import Hypothesis, HypothesisBatch
from coscientist.schemas.research_goal import ResearchGoal


class GeneratorAgent(Agent):
    async def generate(self, goal: ResearchGoal, strategy: str, count: int) -> list[Hypothesis]:
        prompt = generator_prompt(goal, strategy, count)
        batch = await self.provider.generate_structured(
            prompt,
            HypothesisBatch,
            {
                "mode": "generate",
                "goal_id": goal.id,
                "strategy": strategy,
                "count": count,
            },
        )
        return batch.hypotheses
