from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from coscientist.providers.base import ProviderError, StructuredLLMProvider

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleProvider(StructuredLLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.max_retries = max_retries
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for --provider openai.")

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        schema = output_schema.model_json_schema()
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            repair = f"\nPrevious validation errors: {errors[-1]}" if errors else ""
            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You return only JSON matching the requested schema. "
                            "No Markdown, no prose outside JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}{repair}\n\nJSON schema:\n{json.dumps(schema)}"
                        ),
                    },
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
            text = await asyncio.to_thread(self._post_chat_completions, body)
            try:
                return output_schema.model_validate_json(text)
            except ValidationError as exc:
                errors.append(str(exc))
        raise ProviderError(
            f"Provider returned malformed {output_schema.__name__} after {self.max_retries + 1} attempts: {errors[-1]}"
        )

    def _post_chat_completions(self, body: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenAI-compatible provider HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"OpenAI-compatible provider request failed: {exc.reason}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible provider response did not contain choices[0].message.content.") from exc
