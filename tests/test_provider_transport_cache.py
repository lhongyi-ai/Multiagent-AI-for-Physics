from __future__ import annotations

import asyncio

import httpx
import pytest

from coscientist.literature.cache import ProviderResponseCache
from coscientist.literature.http import AsyncHttpTransport, NetworkDisabledError, ProviderTransportError


def test_network_disabled_blocks_live_calls() -> None:
    transport = AsyncHttpTransport(allow_live_network=False)
    with pytest.raises(NetworkDisabledError):
        asyncio.run(transport.get_json("x", "op", "https://example.test"))


def test_retry_after_is_respected_and_retries_within_bounds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, max_retries=1, client=client)
    result = asyncio.run(transport.get_json("x", "op", "https://example.test"))
    assert result.payload == {"ok": True}
    assert calls["count"] == 2


def test_permanent_auth_error_does_not_retry() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "auth"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(allow_live_network=True, max_retries=3, client=client)
    with pytest.raises(ProviderTransportError):
        asyncio.run(transport.get_json("x", "op", "https://example.test"))
    assert calls["count"] == 1


def test_cache_hit_prevents_repeated_http_request(tmp_path) -> None:
    cache = ProviderResponseCache(tmp_path)
    key = cache.key("p", "op", {"url": "u", "params": {"email": "secret@example.com"}})
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}
    assert "secret" not in key


def test_corrupted_cache_recovers(tmp_path) -> None:
    cache = ProviderResponseCache(tmp_path)
    key = cache.key("p", "op", {"url": "u"})
    (tmp_path / f"{key}.json").write_text("{bad json", encoding="utf-8")
    assert cache.get(key) is None
