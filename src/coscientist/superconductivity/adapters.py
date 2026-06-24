from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from coscientist.pilot.artifacts import read_jsonl


class LiveAccessDisabledError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SuperConAdapter:
    provider_name = "nims_supercon_fixture"

    def load_snapshot(self, path: str | Path, *, live_network: bool = False) -> list[dict[str, Any]]:
        if str(path).startswith("http") and not live_network:
            raise LiveAccessDisabledError("SuperCon live access requires explicit live_network permission")
        rows = []
        snapshot = Path(path)
        with snapshot.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "schema_version": "v21",
                    "provider": self.provider_name,
                    "material_id": row["material_id"],
                    "formula": row["formula"],
                    "family": row["family"],
                    "tc_k": float(row["tc_k"]) if row.get("tc_k") else None,
                    "doping_label": row.get("doping_label") or None,
                    "source_id": row.get("source_id", "supercon-fixture"),
                    "original_row": row,
                    "snapshot_hash": sha256_file(snapshot),
                    "ambiguous_doping_preserved": bool(row.get("doping_label")),
                    "mechanism_inferred": False,
                })
        return rows


class MaterialsProjectAdapter:
    provider_name = "materials_project_fixture"

    def load_snapshot(self, path: str | Path, *, live_network: bool = False, api_key: str | None = None) -> list[dict[str, Any]]:
        if str(path).startswith("mp-live:"):
            if not live_network or not api_key:
                raise LiveAccessDisabledError("Materials Project live access requires explicit live_network permission and API key")
            return []
        return [{**item, "provider": self.provider_name, "computed_or_experimental": item.get("computed_or_experimental", "computed"), "snapshot_hash": sha256_file(path)} for item in read_jsonl(Path(path))]


class NomadAdapter:
    provider_name = "nomad_fixture"

    def load_snapshot(self, path: str | Path, *, live_network: bool = False) -> list[dict[str, Any]]:
        if str(path).startswith("nomad-live:") and not live_network:
            raise LiveAccessDisabledError("NOMAD live access requires explicit live_network permission")
        return [{**item, "provider": self.provider_name, "snapshot_hash": sha256_file(path)} for item in read_jsonl(Path(path))]


class OptimadeAdapter:
    provider_name = "optimade_fixture"

    def load_snapshot(self, path: str | Path, *, live_network: bool = False) -> list[dict[str, Any]]:
        if str(path).startswith("http") and not live_network:
            raise LiveAccessDisabledError("OPTIMADE live access requires explicit live_network permission")
        records = []
        for item in read_jsonl(Path(path)):
            records.append({
                **item,
                "provider": self.provider_name,
                "formula_only_merge_permitted": False,
                "provider_specific_fields_preserved": True,
                "snapshot_hash": sha256_file(path),
            })
        return records
