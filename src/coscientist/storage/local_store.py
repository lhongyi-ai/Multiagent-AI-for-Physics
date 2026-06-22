from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class LocalStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def ensure_run(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: str, name: str, payload: Any) -> Path:
        path = self.ensure_run(run_id) / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def append_log(self, run_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
        path = self.ensure_run(run_id) / "run_log.jsonl"
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload or {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_text(self, run_id: str, name: str, text: str) -> Path:
        path = self.ensure_run(run_id) / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_jsonl(self, run_id: str, name: str, payloads: list[Any]) -> Path:
        path = self.ensure_run(run_id) / name
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(self._jsonable(payload), sort_keys=True) + "\n")
        return path

    @staticmethod
    def _jsonable(payload: Any) -> Any:
        if isinstance(payload, BaseModel):
            return payload.model_dump(mode="json")
        if isinstance(payload, list):
            return [LocalStore._jsonable(item) for item in payload]
        if isinstance(payload, dict):
            return {key: LocalStore._jsonable(value) for key, value in payload.items()}
        return payload
