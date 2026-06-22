from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class ProviderResponseCache:
    def __init__(self, root: str | Path, ttl_hours: int = 168, enabled: bool = True) -> None:
        self.root = Path(root)
        self.ttl = timedelta(hours=ttl_hours)
        self.enabled = enabled

    def key(self, provider: str, operation: str, request: dict[str, Any], api_version: str = "v1") -> str:
        safe_request = self._scrub(request)
        payload = {
            "provider": provider,
            "operation": operation,
            "request": safe_request,
            "api_version": api_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(record["cached_at"])
            if self.ttl.total_seconds() and datetime.now(UTC) - cached_at > self.ttl:
                return None
            return record["payload"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def set(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"cached_at": datetime.now(UTC).isoformat(), "payload": payload}
        fd, tmp_name = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, self._path(key))
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    @staticmethod
    def _scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("<redacted>" if key.lower() in {"api_key", "authorization", "email", "mailto"} else ProviderResponseCache._scrub(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ProviderResponseCache._scrub(item) for item in value]
        return value
