from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when a provider cannot return schema-valid output."""


class StructuredLLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        """Return one schema-validated structured result."""
