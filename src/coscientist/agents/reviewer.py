from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.prompts.templates import review_prompt
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.review import Review, ReviewBatch


class ReviewerAgent(Agent):
    async def review(self, hypotheses: list[Hypothesis]) -> list[Review]:
        if not hypotheses:
            return []
        prompt = "\n".join(review_prompt(hypothesis) for hypothesis in hypotheses)
        batch = await self.provider.generate_structured(prompt, ReviewBatch, {"hypotheses": hypotheses})
        return batch.reviews
