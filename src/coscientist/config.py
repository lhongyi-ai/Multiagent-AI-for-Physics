from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from coscientist.schemas.research_goal import ResearchGoal


DEFAULT_STRATEGIES = ["mechanistic", "analogy", "contrarian", "minimal-explanation"]


class LiteratureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = False
    search_providers: list[str] = Field(default_factory=lambda: ["mock"])
    metadata_resolvers: list[str] = Field(default_factory=lambda: ["mock"])
    full_text_locators: list[str] = Field(default_factory=lambda: ["mock"])
    max_results_per_provider: int = Field(default=10, ge=1, le=100)
    max_total_results: int = Field(default=20, ge=1, le=500)
    request_timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cache_enabled: bool = True
    cache_ttl_hours: int = Field(default=168, ge=0)
    allow_live_network: bool = False
    force_refresh: bool = False
    cache_dir: str = ".coscientist_cache/provider_responses"
    user_agent: str = "coscientist-mvp/0.1"


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    generators: int = Field(default=4, ge=1)
    hypotheses_per_generator: int = Field(default=3, ge=1)
    top_k_after_review: int = Field(default=6, ge=1)
    evolution_rounds: int = Field(default=2, ge=0)
    children_per_selected_hypothesis: int = Field(default=2, ge=1)
    final_top_k: int = Field(default=3, ge=1)
    max_llm_calls: int = Field(default=80, ge=1)
    max_parallel_agents: int = Field(default=4, ge=1)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    ranking_weights: dict[str, float] = Field(default_factory=lambda: {
        "correctness": 1.2,
        "novelty": 1.0,
        "testability": 1.2,
        "explanatory_power": 1.0,
        "feasibility": 1.0,
        "discriminative_power": 1.1,
        "evidence_quality": 1.0,
        "impact": 1.0,
        "parsimony": 0.8,
    })


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path = "config/default.yaml") -> WorkflowConfig:
    return WorkflowConfig.model_validate(_read_yaml(Path(path)))


def load_research_goal(path: str | Path) -> ResearchGoal:
    return ResearchGoal.model_validate(_read_yaml(Path(path)))
