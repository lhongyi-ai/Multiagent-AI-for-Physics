from __future__ import annotations

import pytest
import asyncio

from coscientist.agents.ranker import RankerAgent
from coscientist.config import WorkflowConfig
from coscientist.providers.mock import MockProvider
from coscientist.schemas.hypothesis import HypothesisBatch
from coscientist.schemas.ranking import PairwiseBatch, PairwiseComparison, RankingBatch


def test_pairwise_randomized_order_keeps_mock_result_stable() -> None:
    provider = MockProvider()
    hypotheses = (asyncio.run(provider.generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "mechanistic", "count": 5},
    ))).hypotheses
    ranker = RankerAgent(provider)
    first = asyncio.run(ranker.rank(hypotheses, [], WorkflowConfig(), round_number=0, seed=1))
    second = asyncio.run(ranker.rank(hypotheses, [], WorkflowConfig(), round_number=0, seed=999))
    assert [ranking.hypothesis_id for ranking in first] == [ranking.hypothesis_id for ranking in second]
    assert [ranking.weighted_total for ranking in first] == [ranking.weighted_total for ranking in second]


def test_single_hypothesis_skips_pairwise_call() -> None:
    provider = CountingPairwiseProvider()
    hypotheses = (asyncio.run(MockProvider().generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "mechanistic", "count": 1},
    ))).hypotheses
    rankings = asyncio.run(RankerAgent(provider).rank(hypotheses, [], WorkflowConfig(), round_number=0))
    assert len(rankings) == 1
    assert provider.pairwise_calls == 0


def test_invalid_pairwise_ids_are_ignored() -> None:
    provider = CountingPairwiseProvider(return_invalid_pair=True)
    hypotheses = (asyncio.run(MockProvider().generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "mechanistic", "count": 2},
    ))).hypotheses
    rankings = asyncio.run(RankerAgent(provider).rank(hypotheses, [], WorkflowConfig(), round_number=0))
    assert len(rankings) == 2
    assert all(ranking.pairwise_wins == 0 for ranking in rankings)


class CountingPairwiseProvider(MockProvider):
    def __init__(self, return_invalid_pair: bool = False) -> None:
        super().__init__()
        self.pairwise_calls = 0
        self.return_invalid_pair = return_invalid_pair

    async def generate_structured(self, prompt, output_schema, context=None):
        if output_schema is PairwiseBatch:
            self.pairwise_calls += 1
            if self.return_invalid_pair:
                return PairwiseBatch(comparisons=[
                    PairwiseComparison(
                        hypothesis_a_id="0",
                        hypothesis_b_id="1",
                        winner="a",
                        judge_notes="Invalid model IDs should be ignored.",
                    )
                ])
        if output_schema is RankingBatch:
            hypotheses = (context or {})["hypotheses"]
            return RankingBatch(rankings=[
                self._ranking_batch({"hypotheses": [hypothesis]}).rankings[0]
                for hypothesis in hypotheses
            ])
        return await super().generate_structured(prompt, output_schema, context)
