from __future__ import annotations

import pytest
import asyncio

from coscientist.agents.ranker import RankerAgent
from coscientist.config import WorkflowConfig
from coscientist.providers.mock import MockProvider
from coscientist.schemas.hypothesis import HypothesisBatch


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
