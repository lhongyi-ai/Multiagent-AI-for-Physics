from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from coscientist.providers.base import ProviderError, StructuredLLMProvider
from coscientist.schemas.model_provider import ModelCallRecord, ModelUsage

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleProvider(StructuredLLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        max_repair_attempts: int = 1,
        timeout_seconds: float = 90.0,
        temperature: float = 0.2,
        max_output_tokens: int = 1600,
        app_name: str | None = None,
        site_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model if model is not None else os.getenv("OPENAI_MODEL")
        self.base_url = (base_url if base_url is not None else os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.max_retries = max_retries
        self.max_repair_attempts = max_repair_attempts
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.app_name = app_name if app_name is not None else os.getenv("OPENROUTER_APP_NAME")
        self.site_url = site_url if site_url is not None else os.getenv("OPENROUTER_SITE_URL")
        self.client = client
        self.call_records: list[ModelCallRecord] = []
        self._sequence = 0
        self._validate_configuration()

    @property
    def sanitized_base_url(self) -> str:
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            return "<invalid-base-url>"
        return f"{parsed.scheme}://{parsed.netloc}"

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        context = context or {}
        schema = output_schema.model_json_schema()
        errors: list[str] = []
        last_error_status = "schema_validation_error"
        total_attempts = self.max_repair_attempts + 1
        for repair_attempt in range(total_attempts):
            repair = ""
            if errors:
                repair = (
                    "\n\nPrevious response failed validation. Return corrected JSON only. "
                    f"Sanitized error: {errors[-1]}"
                )
            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON matching the requested schema. Do not use Markdown. "
                            "Do not invent citations or paper IDs; cite only supplied corpus IDs. "
                            "Mark unsupported assumptions explicitly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}{repair}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}",
                    },
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
                "response_format": {"type": "json_object"},
            }
            payload, record_base = await self._post_chat_completions(body, output_schema, context, repair_attempt)
            try:
                content, finish_reason = self._extract_content(payload)
                if finish_reason == "length":
                    last_error_status = "truncated"
                    raise ProviderError("model output was truncated before completing structured JSON")
                json_text = extract_json_object(content)
                result = output_schema.model_validate_json(json_text)
            except ProviderError as exc:
                errors.append(_safe_error(str(exc)))
                self._append_record(record_base, payload, output_schema, context, repair_attempt, False, last_error_status, str(exc))
                if repair_attempt < self.max_repair_attempts:
                    continue
                raise ProviderError(_failure_message(output_schema, errors[-1], last_error_status)) from exc
            except json.JSONDecodeError as exc:
                last_error_status = "invalid_json"
                errors.append(_safe_error(str(exc)))
                self._append_record(record_base, payload, output_schema, context, repair_attempt, False, "invalid_json", str(exc))
                if repair_attempt < self.max_repair_attempts:
                    continue
                raise ProviderError(_failure_message(output_schema, errors[-1], "invalid_json")) from exc
            except ValidationError as exc:
                last_error_status = "schema_validation_error"
                errors.append(_safe_error(str(exc)))
                self._append_record(record_base, payload, output_schema, context, repair_attempt, False, "schema_validation_error", str(exc))
                if repair_attempt < self.max_repair_attempts:
                    continue
                raise ProviderError(_failure_message(output_schema, errors[-1], "schema_validation_error")) from exc
            self._append_record(record_base, payload, output_schema, context, repair_attempt, True, "success", None)
            return result
        raise ProviderError(_failure_message(output_schema, errors[-1] if errors else "unknown error", last_error_status))

    async def _post_chat_completions(
        self,
        body: dict[str, Any],
        output_schema: type[BaseModel],
        context: dict[str, Any],
        repair_attempt: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._sequence += 1
        sequence = self._sequence
        started = time.monotonic()
        last_exc: Exception | None = None
        for retry in range(self.max_retries + 1):
            try:
                response = await self._client().post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                    timeout=self.timeout_seconds,
                )
                request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id")
                if response.status_code >= 400:
                    error_text = _safe_error(response.text[:500], [self.api_key])
                    if response.status_code in {400, 401, 403, 404}:
                        record = self._failure_record_base(sequence, retry, started)
                        record["provider_request_id"] = request_id
                        self._append_synthetic_failure(record, output_schema, context, repair_attempt, "provider_error", f"HTTP {response.status_code}: {error_text}")
                        raise ProviderError(f"OpenAI-compatible provider HTTP {response.status_code}: {error_text}")
                    if retry < self.max_retries and (response.status_code == 429 or response.status_code >= 500):
                        await asyncio.sleep(_backoff(retry))
                        continue
                    record = self._failure_record_base(sequence, retry, started)
                    record["provider_request_id"] = request_id
                    self._append_synthetic_failure(record, output_schema, context, repair_attempt, "provider_error", f"HTTP {response.status_code}: {error_text}")
                    raise ProviderError(f"OpenAI-compatible provider HTTP {response.status_code}: {error_text}")
                payload = response.json()
                return payload, {
                    "sequence": sequence,
                    "retry_count": retry,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "provider_request_id": request_id,
                }
            except httpx.TimeoutException as exc:
                last_exc = exc
                if retry < self.max_retries:
                    await asyncio.sleep(_backoff(retry))
                    continue
                record = self._failure_record_base(sequence, retry, started)
                self._append_synthetic_failure(record, output_schema, context, repair_attempt, "timeout", "provider request timed out")
                raise ProviderError("OpenAI-compatible provider request timed out") from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if retry < self.max_retries:
                    await asyncio.sleep(_backoff(retry))
                    continue
                record = self._failure_record_base(sequence, retry, started)
                self._append_synthetic_failure(record, output_schema, context, repair_attempt, "provider_error", str(exc))
                raise ProviderError(f"OpenAI-compatible provider request failed: {_safe_error(str(exc), [self.api_key])}") from exc
        raise ProviderError(f"OpenAI-compatible provider request failed: {_safe_error(str(last_exc), [self.api_key])}")

    def _extract_content(self, payload: dict[str, Any]) -> tuple[str, str | None]:
        try:
            choice = payload["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("provider response did not contain choices[0].message") from exc
        refusal = message.get("refusal")
        if refusal:
            raise ProviderError("provider returned a refusal")
        content = message.get("content")
        if content is None or str(content).strip() == "":
            raise ProviderError("provider returned an empty response")
        return str(content), choice.get("finish_reason")

    def _append_record(
        self,
        record_base: dict[str, Any],
        payload: dict[str, Any],
        output_schema: type[BaseModel],
        context: dict[str, Any],
        repair_attempt: int,
        success: bool,
        structured_status: str,
        error_message: str | None,
    ) -> None:
        choice = ((payload.get("choices") or [{}])[0] or {}) if isinstance(payload, dict) else {}
        usage = _usage(payload.get("usage") if isinstance(payload, dict) else None)
        self.call_records.append(ModelCallRecord(
            provider_type=self.name,
            model_mode="live",
            sanitized_base_url=self.sanitized_base_url,
            requested_model=self.model,
            returned_model=payload.get("model") if isinstance(payload, dict) else None,
            request_sequence_number=record_base["sequence"],
            agent_role=context.get("agent_role"),
            workflow_stage=context.get("workflow_stage"),
            schema_name=output_schema.__name__,
            temperature=self.temperature,
            maximum_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            retry_count=record_base["retry_count"],
            usage=usage,
            provider_request_id=record_base.get("provider_request_id"),
            latency_ms=record_base["latency_ms"],
            finish_reason=choice.get("finish_reason"),
            structured_output_status=structured_status,  # type: ignore[arg-type]
            repair_attempt_count=repair_attempt,
            timestamp=datetime.now(UTC),
            success=success,
            error_type=None if success else structured_status,
            error_message=None if success else _safe_error(error_message or structured_status, [self.api_key]),
            authentication_configured=bool(self.api_key),
            response_debug={} if success else {"content_excerpt": _content_excerpt(payload)},
        ))

    def _append_synthetic_failure(
        self,
        record_base: dict[str, Any],
        output_schema: type[BaseModel],
        context: dict[str, Any],
        repair_attempt: int,
        structured_status: str,
        message: str,
    ) -> None:
        self.call_records.append(ModelCallRecord(
            provider_type=self.name,
            model_mode="live",
            sanitized_base_url=self.sanitized_base_url,
            requested_model=self.model,
            request_sequence_number=record_base["sequence"],
            agent_role=context.get("agent_role"),
            workflow_stage=context.get("workflow_stage"),
            schema_name=output_schema.__name__,
            temperature=self.temperature,
            maximum_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            retry_count=record_base["retry_count"],
            provider_request_id=record_base.get("provider_request_id"),
            latency_ms=record_base["latency_ms"],
            structured_output_status=structured_status,  # type: ignore[arg-type]
            repair_attempt_count=repair_attempt,
            timestamp=datetime.now(UTC),
            success=False,
            error_type=structured_status,
            error_message=_safe_error(message, [self.api_key]),
            authentication_configured=bool(self.api_key),
        ))

    def _failure_record_base(self, sequence: int, retry: int, started: float) -> dict[str, Any]:
        return {"sequence": sequence, "retry_count": retry, "latency_ms": round((time.monotonic() - started) * 1000, 3)}

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_name:
            headers["X-Title"] = self.app_name
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient()
        return self.client

    def _validate_configuration(self) -> None:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for live OpenAI-compatible provider mode.")
        if not self.model:
            raise ProviderError("OPENAI_MODEL is required for live OpenAI-compatible provider mode.")
        if not self.base_url:
            raise ProviderError("OPENAI_BASE_URL is required for live OpenAI-compatible provider mode.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderError("OPENAI_BASE_URL must be an absolute http(s) URL.")


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    return stripped[start : end + 1]


def _usage(raw: Any) -> ModelUsage:
    if not isinstance(raw, dict):
        return ModelUsage()
    return ModelUsage(
        input_tokens=raw.get("prompt_tokens"),
        output_tokens=raw.get("completion_tokens"),
        total_tokens=raw.get("total_tokens"),
    )


def _backoff(attempt: int) -> float:
    return min(4.0, 0.5 * (2 ** attempt))


def _safe_error(message: str, secrets: list[str | None] | None = None) -> str:
    redacted = message
    for secret in [os.getenv("OPENAI_API_KEY"), *(secrets or [])]:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _content_excerpt(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"].get("content") or ""
    except Exception:
        content = ""
    return str(content)[:500]


def _failure_message(output_schema: type[BaseModel], detail: str, status: str) -> str:
    return f"Provider returned invalid {output_schema.__name__} ({status}): {detail}"
