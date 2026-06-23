from __future__ import annotations

from coscientist.schemas.model_provider import ModelCallRecord, ModelProviderStatus, ModelUsageSummary


def summarize_model_usage(provider: str, model_mode: str, records: list[ModelCallRecord]) -> ModelUsageSummary:
    input_tokens = _sum_optional([record.usage.input_tokens for record in records])
    output_tokens = _sum_optional([record.usage.output_tokens for record in records])
    total_tokens = _sum_optional([record.usage.total_tokens for record in records])
    return ModelUsageSummary(
        model_mode=model_mode,  # type: ignore[arg-type]
        provider=provider,
        call_count=len(records),
        successful_call_count=sum(1 for record in records if record.success),
        failed_call_count=sum(1 for record in records if not record.success),
        repair_attempts=sum(record.repair_attempt_count for record in records),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        structured_output_failures=sum(1 for record in records if record.structured_output_status != "success"),
    )


def provider_status(
    *,
    provider: str,
    model_mode: str,
    live_model_enabled: bool,
    authentication_configured: bool,
    sanitized_base_url: str | None,
    requested_model: str | None,
    records: list[ModelCallRecord],
    dry_run: bool = False,
) -> ModelProviderStatus:
    failed = any(not record.success for record in records)
    if dry_run:
        status = "dry_run"
    elif not live_model_enabled:
        status = "skipped"
    elif failed:
        status = "failed"
    else:
        status = "complete"
    return ModelProviderStatus(
        model_mode=model_mode,  # type: ignore[arg-type]
        provider=provider,
        configured=True,
        live_model_enabled=live_model_enabled,
        authentication_configured=authentication_configured,
        sanitized_base_url=sanitized_base_url,
        requested_model=requested_model,
        status=status,  # type: ignore[arg-type]
        message=None if records or dry_run else "No model calls were made.",
    )


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)
