from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    round_number: int = Field(ge=0)
    llm_call_count: int = Field(default=0, ge=0)
    maximum_llm_calls: int = Field(gt=0)
    active_hypothesis_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timestamps: dict[str, str] = Field(default_factory=dict)
