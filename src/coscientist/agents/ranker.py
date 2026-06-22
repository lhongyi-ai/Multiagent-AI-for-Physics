from __future__ import annotations

import itertools
import random

from coscientist.agents.base import Agent
from coscientist.config import WorkflowConfig
from coscientist.prompts.templates import ranking_prompt
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.ranking import HypothesisRanking, PairwiseBatch, RankingBatch
from coscientist.schemas.review import Review


class RankerAgent(Agent):
    async def rank(
        self,
        hypotheses: list[Hypothesis],
        reviews: list[Review],
        config: WorkflowConfig,
        round_number: int,
        seed: int = 0,
    ) -> list[HypothesisRanking]:
        if not hypotheses:
            return []
        prompt = ranking_prompt(round_number)
        batch = await self.provider.generate_structured(
            prompt,
            RankingBatch,
            {"hypotheses": hypotheses, "reviews": reviews, "weights": config.ranking_weights},
        )
        ranking_by_id = {ranking.hypothesis_id: ranking for ranking in batch.rankings}
        pairs = list(itertools.combinations(hypotheses, 2))
        rng = random.Random(seed + round_number)
        randomized_pairs: list[tuple[Hypothesis, Hypothesis]] = []
        for left, right in pairs:
            randomized_pairs.append((right, left) if rng.random() < 0.5 else (left, right))

        comparisons = await self.provider.generate_structured(
            prompt,
            PairwiseBatch,
            {"pairs": randomized_pairs},
        )
        wins = {hypothesis.id: 0 for hypothesis in hypotheses}
        losses = {hypothesis.id: 0 for hypothesis in hypotheses}
        notes = {hypothesis.id: [] for hypothesis in hypotheses}
        for comparison in comparisons.comparisons:
            a_id = comparison.hypothesis_a_id
            b_id = comparison.hypothesis_b_id
            if comparison.winner == "a":
                wins[a_id] += 1
                losses[b_id] += 1
            elif comparison.winner == "b":
                wins[b_id] += 1
                losses[a_id] += 1
            notes[a_id].append(comparison.judge_notes)
            notes[b_id].append(comparison.judge_notes)

        merged = []
        for ranking in ranking_by_id.values():
            merged.append(ranking.model_copy(update={
                "pairwise_wins": wins.get(ranking.hypothesis_id, 0),
                "pairwise_losses": losses.get(ranking.hypothesis_id, 0),
                "judge_notes": [*ranking.judge_notes, *notes.get(ranking.hypothesis_id, [])[:3]],
            }))
        return sorted(
            merged,
            key=lambda item: (item.weighted_total, item.pairwise_wins, -item.pairwise_losses, item.hypothesis_id),
            reverse=True,
        )
