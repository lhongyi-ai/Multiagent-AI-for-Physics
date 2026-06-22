from __future__ import annotations

import pytest
import asyncio
from pydantic import ValidationError

from coscientist.agents.evolution import EvolutionAgent
from coscientist.providers.mock import MockProvider
from coscientist.schemas.hypothesis import HypothesisBatch


def test_evolution_preserves_parent_lineage_and_records_change() -> None:
    provider = MockProvider()
    parent = (asyncio.run(provider.generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "contrarian", "count": 1},
    ))).hypotheses[0]
    child = (asyncio.run(EvolutionAgent(provider).repair(parent, [], round_number=1)))[0]
    assert child.id != parent.id
    assert child.parent_ids == [parent.id]
    assert child.version == parent.version + 1
    assert child.change_summary


def test_parent_immutability_after_evolution() -> None:
    provider = MockProvider()
    parent = (asyncio.run(provider.generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "analogy", "count": 1},
    ))).hypotheses[0]
    asyncio.run(EvolutionAgent(provider).branch(parent, [], round_number=1))
    with pytest.raises(ValidationError):
        parent.status = "branched"  # type: ignore[misc]
