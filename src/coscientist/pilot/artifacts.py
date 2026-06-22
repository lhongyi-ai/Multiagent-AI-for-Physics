from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coscientist.pilot.project_io import dump_jsonable


REQUIRED_V1_ARTIFACTS = [
    "run_manifest.json",
    "project_snapshot.json",
    "corpus.jsonl",
    "normalized_papers.jsonl",
    "hypotheses_initial.jsonl",
    "reviews.jsonl",
    "evidence_verification.jsonl",
    "rankings.jsonl",
    "evolution_round_1.jsonl",
    "evolution_round_2.jsonl",
    "hypotheses_final.json",
    "evaluation_by_round.json",
    "round_comparison.json",
    "lineage.json",
    "report.md",
    "human_review.md",
]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(dump_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, payloads: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(dump_jsonable(payload), sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_v1_artifacts(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors = []
    for name in REQUIRED_V1_ARTIFACTS:
        if not (path / name).exists():
            errors.append(f"missing artifact: {name}")
    manifest_path = path / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid manifest JSON: {exc}")
        else:
            if manifest.get("artifact_schema_version") != "v1":
                errors.append("incompatible artifact_schema_version")
            for artifact in manifest.get("artifacts", []):
                if not (path / artifact).exists():
                    errors.append(f"manifest references missing artifact: {artifact}")
    for name in REQUIRED_V1_ARTIFACTS:
        artifact = path / name
        if not artifact.exists() or artifact.suffix not in {".json", ".jsonl"}:
            continue
        try:
            if artifact.suffix == ".jsonl":
                read_jsonl(artifact)
            else:
                read_json(artifact)
        except json.JSONDecodeError as exc:
            errors.append(f"corrupt JSON artifact {name}: {exc}")
    return errors
