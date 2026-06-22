from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from coscientist.schemas.literature import ProviderErrorRecord, ProviderRequestLog


class NetworkDisabledError(RuntimeError):
    pass


class ProviderTransportError(RuntimeError):
    def __init__(self, error: ProviderErrorRecord) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class TransportResult:
    payload: Any
    request_log: ProviderRequestLog


class AsyncHttpTransport:
    def __init__(
        self,
        *,
        allow_live_network: bool = False,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        user_agent: str = "coscientist-mvp/0.1",
        concurrency_limit: int = 4,
        client: httpx.AsyncClient | None = None,
        pace_seconds: float = 0.0,
    ) -> None:
        self.allow_live_network = allow_live_network
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.client = client
        self.pace_seconds = pace_seconds
        self._last_request_at = 0.0

    async def get_json(
        self,
        provider: str,
        operation: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        normalized_query: dict[str, Any] | None = None,
    ) -> TransportResult:
        response = await self.get(provider, operation, url, params=params, headers=headers, normalized_query=normalized_query)
        return TransportResult(response.payload.json(), response.request_log)

    async def get_text(
        self,
        provider: str,
        operation: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        normalized_query: dict[str, Any] | None = None,
    ) -> TransportResult:
        response = await self.get(provider, operation, url, params=params, headers=headers, normalized_query=normalized_query)
        return TransportResult(response.payload.text, response.request_log)

    async def get(
        self,
        provider: str,
        operation: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        normalized_query: dict[str, Any] | None = None,
    ) -> TransportResult:
        if not self.allow_live_network:
            raise NetworkDisabledError(f"Live network is disabled for {provider}:{operation}. Pass an explicit live-network flag.")
        safe_query = normalized_query or self._safe_params(params or {})
        async with self.semaphore:
            if self.pace_seconds:
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < self.pace_seconds:
                    await asyncio.sleep(self.pace_seconds - elapsed)
            started = time.monotonic()
            merged_headers = {"User-Agent": self.user_agent, **(headers or {})}
            last_error: ProviderErrorRecord | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client().get(
                        url,
                        params=params,
                        headers=merged_headers,
                        timeout=self.timeout_seconds,
                    )
                except httpx.TimeoutException as exc:
                    last_error = self._error(provider, operation, "timeout", True, None, str(exc), None)
                except httpx.HTTPError as exc:
                    last_error = self._error(provider, operation, "http_error", True, None, str(exc), None)
                else:
                    self._last_request_at = time.monotonic()
                    if 200 <= response.status_code < 300:
                        return TransportResult(
                            response,
                            ProviderRequestLog(
                                provider=provider,
                                operation=operation,
                                normalized_query=safe_query,
                                timestamp=datetime.now(UTC),
                                cache_hit=False,
                                status="success",
                                result_count=0,
                                latency_ms=round((time.monotonic() - started) * 1000, 3),
                            ),
                        )
                    retry_after = self._retry_after(response)
                    retryable = response.status_code == 429 or response.status_code >= 500
                    last_error = self._error(
                        provider,
                        operation,
                        "http_status",
                        retryable,
                        response.status_code,
                        f"{provider} returned HTTP {response.status_code}",
                        retry_after,
                    )
                    if not retryable:
                        raise ProviderTransportError(last_error)
                if attempt < self.max_retries and last_error and last_error.retryable:
                    await asyncio.sleep(last_error.retry_after_seconds or self._backoff(attempt))
                    continue
                break
            raise ProviderTransportError(last_error or self._error(provider, operation, "unknown", False, None, "Unknown provider error", None))

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient()
        return self.client

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(8.0, (2**attempt) + random.Random(attempt).random() / 5)

    @staticmethod
    def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            key: ("<redacted>" if key.lower() in {"api_key", "email", "mailto"} else value)
            for key, value in params.items()
        }

    @staticmethod
    def _error(
        provider: str,
        operation: str,
        error_type: str,
        retryable: bool,
        status_code: int | None,
        message: str,
        retry_after_seconds: float | None,
    ) -> ProviderErrorRecord:
        return ProviderErrorRecord(
            provider=provider,
            operation=operation,
            error_type=error_type,
            retryable=retryable,
            status_code=status_code,
            message=message,
            retry_after_seconds=retry_after_seconds,
            occurred_at=datetime.now(UTC),
        )
