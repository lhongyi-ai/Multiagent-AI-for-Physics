from __future__ import annotations

import pytest
import asyncio

from coscientist.providers.base import ProviderError, StructuredLLMProvider
from coscientist.providers.mock import MockProvider
from coscientist.schemas.hypothesis import HypothesisBatch


def test_mock_provider_is_deterministic() -> None:
    provider = MockProvider()
    context = {"goal_id": "g", "strategy": "mechanistic", "count": 3}
    first = asyncio.run(provider.generate_structured("prompt", HypothesisBatch, context))
    second = asyncio.run(provider.generate_structured("prompt", HypothesisBatch, context))
    assert first == second
    assert len(first.hypotheses) == 3
    assert all("[MOCK]" in hypothesis.core_claim for hypothesis in first.hypotheses)


class MalformedProvider(StructuredLLMProvider):
    name = "malformed"

    async def generate_structured(self, prompt, output_schema, context=None):
        try:
            return output_schema.model_validate({"not": "valid"})
        except Exception as exc:
            raise ProviderError("Malformed structured output") from exc


def test_malformed_provider_response_raises_clear_error() -> None:
    provider = MalformedProvider()
    with pytest.raises(ProviderError, match="Malformed structured output"):
        asyncio.run(provider.generate_structured("prompt", HypothesisBatch, {}))
