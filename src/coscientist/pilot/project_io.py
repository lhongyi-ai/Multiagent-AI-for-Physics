from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from coscientist.schemas.literature import Paper
from coscientist.schemas.project import ResearchProjectSpec


def load_project_spec(path: str | Path) -> ResearchProjectSpec:
    project_path = Path(path)
    with project_path.open("r", encoding="utf-8") as handle:
        if project_path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{project_path} must contain a mapping")
    if isinstance(data.get("created_at"), str):
        data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
    return ResearchProjectSpec.model_validate(data)


def load_fixture_corpus(path: str | Path) -> list[Paper]:
    papers: list[Paper] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                papers.append(Paper.model_validate(json.loads(line)))
    return papers


def dump_jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [dump_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: dump_jsonable(value) for key, value in payload.items()}
    return payload
