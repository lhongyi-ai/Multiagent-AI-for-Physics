from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from coscientist.literature.cache import ProviderResponseCache
from coscientist.literature.http import AsyncHttpTransport, NetworkDisabledError, ProviderTransportError
from coscientist.schemas.literature import ProviderRequestLog


class ProviderConfigurationError(ValueError):
    pass


class CachedProvider:
    provider_name: str

    def __init__(
        self,
        transport: AsyncHttpTransport,
        cache: ProviderResponseCache | None = None,
        *,
        force_refresh: bool = False,
    ) -> None:
        self.transport = transport
        self.cache = cache
        self.force_refresh = force_refresh
        self.request_logs: list[ProviderRequestLog] = []

    async def get_json_cached(
        self,
        operation: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        normalized_query: dict[str, Any] | None = None,
        api_version: str = "v1",
    ) -> Any:
        request = {"url": url, "params": params or {}}
        cache_key = self.cache.key(self.provider_name, operation, request, api_version) if self.cache else ""
        if self.cache and not self.force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.request_logs.append(ProviderRequestLog(
                    provider=self.provider_name,
                    operation=operation,
                    normalized_query=normalized_query or {},
                    timestamp=datetime.now(UTC),
                    cache_hit=True,
                    status="cache_hit",
                    result_count=self._count(cached),
                    latency_ms=0.0,
                ))
                return cached
        try:
            result = await self.transport.get_json(
                self.provider_name,
                operation,
                url,
                params=params,
                normalized_query=normalized_query,
            )
        except NetworkDisabledError:
            self.request_logs.append(ProviderRequestLog(
                provider=self.provider_name,
                operation=operation,
                normalized_query=normalized_query or {},
                timestamp=datetime.now(UTC),
                cache_hit=False,
                status="network_disabled",
                result_count=0,
            ))
            raise
        except ProviderTransportError:
            self.request_logs.append(ProviderRequestLog(
                provider=self.provider_name,
                operation=operation,
                normalized_query=normalized_query or {},
                timestamp=datetime.now(UTC),
                cache_hit=False,
                status="provider_error",
                result_count=0,
            ))
            raise
        payload = result.payload
        log = result.request_log.model_copy(update={"result_count": self._count(payload)})
        self.request_logs.append(log)
        if self.cache:
            self.cache.set(cache_key, payload)
        return payload

    @staticmethod
    def _count(payload: Any) -> int:
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return len(payload["results"])
            message = payload.get("message")
            if isinstance(message, dict) and isinstance(message.get("items"), list):
                return len(message["items"])
            if isinstance(payload.get("oa_locations"), list):
                return len(payload["oa_locations"])
        return 1 if payload else 0
