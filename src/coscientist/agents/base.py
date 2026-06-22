from __future__ import annotations

from dataclasses import dataclass

from coscientist.providers.base import StructuredLLMProvider


@dataclass(frozen=True)
class Agent:
    provider: StructuredLLMProvider
