from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from coscientist.providers.base import ProviderError
from coscientist.providers.openai_compatible import OpenAICompatibleProvider, extract_json_object
from coscientist.schemas.hypothesis import HypothesisBatch


def _provider(handler, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key=kwargs.pop("api_key", "test-secret-key"),
        model=kwargs.pop("model", "openrouter/test-model"),
        base_url=kwargs.pop("base_url", "https://openrouter.ai/api/v1"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=kwargs.pop("max_retries", 0),
        max_repair_attempts=kwargs.pop("max_repair_attempts", 0),
        **kwargs,
    )


def _chat_response(content: str, *, finish_reason: str = "stop", model: str = "returned-model") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }, headers={"x-request-id": "req-test"})


def _hypothesis_json() -> str:
    return json.dumps({
        "hypotheses": [{
            "id": "hyp-live-1",
            "title": "Fixture-grounded live hypothesis",
            "core_claim": "A repository-aware retrieval mechanism may improve context relevance.",
            "mechanism": "Hybrid lexical and semantic retrieval can recover exact symbols and related examples.",
            "assumptions": ["The supplied corpus is incomplete."],
            "supporting_evidence": ["paper-code-lexical-hybrid"],
            "contradicting_evidence": ["paper-code-reranking"],
            "novelty_statement": "This is a controlled test output, not a novelty claim.",
            "testable_predictions": ["Relevance should improve under fixed context budget."],
            "falsification_criteria": ["No improvement versus lexical-only retrieval."],
            "proposed_experiments": ["Compare retrieval variants on a held-out fixture task set."],
            "uncertainty": 0.45,
            "generation_strategy": "mechanistic",
            "parent_ids": [],
            "version": 1,
            "status": "active",
            "change_summary": None,
            "evidence_links": [],
        }],
    })


def test_openai_provider_requires_explicit_safe_configuration() -> None:
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider(api_key="", model="m", base_url="https://openrouter.ai/api/v1")
    with pytest.raises(ProviderError, match="OPENAI_MODEL"):
        OpenAICompatibleProvider(api_key="k", model="", base_url="https://openrouter.ai/api/v1")
    with pytest.raises(ProviderError, match="OPENAI_BASE_URL"):
        OpenAICompatibleProvider(api_key="k", model="m", base_url="")


def test_openrouter_headers_and_structured_success_are_recorded() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return _chat_response(_hypothesis_json())

    provider = _provider(handler, app_name="App", site_url="https://example.test")
    result = asyncio.run(provider.generate_structured("prompt", HypothesisBatch, {"agent_role": "generator"}))
    assert result.hypotheses[0].id == "hyp-live-1"
    assert captured["headers"]["authorization"] == "Bearer test-secret-key"
    assert captured["headers"]["x-title"] == "App"
    assert captured["headers"]["http-referer"] == "https://example.test"
    assert captured["json"]["model"] == "openrouter/test-model"
    assert provider.call_records[0].usage.total_tokens == 18
    assert provider.call_records[0].sanitized_base_url == "https://openrouter.ai"
    assert "test-secret-key" not in provider.call_records[0].model_dump_json()


def test_fenced_json_extraction() -> None:
    assert extract_json_object("```json\n{\"ok\": true}\n```") == "{\"ok\": true}"


def test_schema_repair_success_records_failure_and_success() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response('{"hypotheses": []}')
        return _chat_response(_hypothesis_json())

    provider = _provider(handler, max_repair_attempts=1)
    result = asyncio.run(provider.generate_structured("prompt", HypothesisBatch))
    assert result.hypotheses
    assert [record.structured_output_status for record in provider.call_records] == ["schema_validation_error", "success"]
    assert provider.call_records[-1].repair_attempt_count == 1


def test_auth_failure_is_not_retried_and_is_sanitized() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "bad key test-secret-key"})

    provider = _provider(handler, max_retries=2)
    with pytest.raises(ProviderError):
        asyncio.run(provider.generate_structured("prompt", HypothesisBatch))
    assert calls["count"] == 1
    assert provider.call_records[0].structured_output_status == "provider_error"
    assert "test-secret-key" not in provider.call_records[0].model_dump_json()


def test_timeout_records_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    provider = _provider(handler)
    with pytest.raises(ProviderError, match="timed out"):
        asyncio.run(provider.generate_structured("prompt", HypothesisBatch))
    assert provider.call_records[0].structured_output_status == "timeout"
