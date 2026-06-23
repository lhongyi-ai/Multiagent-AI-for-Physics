from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ModelMode = Literal["mock", "live"]
StructuredOutputStatus = Literal[
    "success",
    "provider_error",
    "invalid_json",
    "schema_validation_error",
    "truncated",
    "refusal",
    "empty_response",
    "timeout",
]


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    provider_type: str = Field(min_length=1)
    model_mode: ModelMode
    sanitized_base_url: str | None = None
    requested_model: str | None = None
    returned_model: str | None = None
    request_sequence_number: int = Field(ge=1)
    agent_role: str | None = None
    workflow_stage: str | None = None
    prompt_template_version: str = "v1"
    schema_name: str = Field(min_length=1)
    temperature: float | None = None
    maximum_output_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    retry_count: int = Field(default=0, ge=0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    provider_request_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    structured_output_status: StructuredOutputStatus
    repair_attempt_count: int = Field(default=0, ge=0)
    timestamp: datetime
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    cache_behavior: str = "none"
    authentication_configured: bool = False
    response_debug: dict[str, Any] = Field(default_factory=dict)


class ModelProviderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_mode: ModelMode
    provider: str = Field(min_length=1)
    configured: bool
    live_model_enabled: bool
    authentication_configured: bool
    sanitized_base_url: str | None = None
    requested_model: str | None = None
    status: Literal["complete", "complete_with_warnings", "failed", "skipped", "dry_run"]
    message: str | None = None


class ModelUsageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model_mode: ModelMode
    provider: str = Field(min_length=1)
    call_count: int = Field(default=0, ge=0)
    successful_call_count: int = Field(default=0, ge=0)
    failed_call_count: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    structured_output_failures: int = Field(default=0, ge=0)
