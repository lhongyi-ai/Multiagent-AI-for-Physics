from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json


def compare_model_runs(mock_run_dir: str | Path, candidate_run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    left = Path(mock_run_dir)
    right = Path(candidate_run_dir)
    output = Path(output_dir) if output_dir else right
    output.mkdir(parents=True, exist_ok=True)
    comparison = build_model_run_comparison(left, right)
    write_json(output / "model_run_comparison.json", comparison)
    (output / "model_run_comparison.md").write_text(render_model_run_comparison(comparison), encoding="utf-8")
    return output / "model_run_comparison.json"


def build_model_run_comparison(mock_run_dir: Path, candidate_run_dir: Path) -> dict[str, Any]:
    left_manifest = read_json(mock_run_dir / "run_manifest.json")
    right_manifest = read_json(candidate_run_dir / "run_manifest.json")
    left_hypotheses = read_json(mock_run_dir / "hypotheses_final.json")
    right_hypotheses = read_json(candidate_run_dir / "hypotheses_final.json")
    left_verifications = read_jsonl(mock_run_dir / "evidence_verification.jsonl")
    right_verifications = read_jsonl(candidate_run_dir / "evidence_verification.jsonl")
    left_comparison = read_json(mock_run_dir / "round_comparison.json")
    right_comparison = read_json(candidate_run_dir / "round_comparison.json")
    left_usage = _read_optional_json(mock_run_dir / "model_usage.json")
    right_usage = _read_optional_json(candidate_run_dir / "model_usage.json")
    right_calls = _read_optional_jsonl(candidate_run_dir / "model_calls.jsonl")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mock_run": str(mock_run_dir),
        "candidate_run": str(candidate_run_dir),
        "disclaimer": (
            "The mock provider is a systems-test baseline, not a scientific-quality baseline. "
            "Evaluator scores are not objective truth, and one live run is insufficient for scientific conclusions."
        ),
        "runs": {
            "mock": _metrics(left_manifest, left_hypotheses, left_verifications, left_comparison, left_usage, []),
            "candidate": _metrics(right_manifest, right_hypotheses, right_verifications, right_comparison, right_usage, right_calls),
        },
    }


def render_model_run_comparison(comparison: dict[str, Any]) -> str:
    mock = comparison["runs"]["mock"]
    candidate = comparison["runs"]["candidate"]
    lines = [
        "# Model Run Comparison",
        "",
        comparison["disclaimer"],
        "",
        "| Metric | Mock | Candidate |",
        "| --- | ---: | ---: |",
    ]
    for key in [
        "hypothesis_count",
        "evidence_link_coverage",
        "unsupported_claim_count",
        "invalid_citation_count",
        "diversity",
        "prediction_specificity",
        "falsification_plan_quality",
        "token_total",
        "model_call_count",
        "structured_output_failures",
        "repair_attempts",
    ]:
        lines.append(f"| {key} | {mock.get(key)} | {candidate.get(key)} |")
    lines.append("")
    return "\n".join(lines)


def _metrics(
    manifest: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    comparison: dict[str, Any],
    usage: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    unsupported = sum(1 for item in verifications if item.get("status") in {"unsupported", "invalid_reference"})
    invalid = sum(1 for item in verifications if item.get("status") == "invalid_reference")
    linked = sum(1 for hypothesis in hypotheses if hypothesis.get("evidence_links"))
    final_diversity = (comparison.get("hypothesis_diversity") or {}).get("final")
    return {
        "run_id": manifest.get("run_id"),
        "model_mode": manifest.get("model_mode", "mock"),
        "model_provider": manifest.get("model_provider", "mock"),
        "hypothesis_count": len(hypotheses),
        "schema_valid_response_rate": _schema_valid_rate(calls),
        "evidence_link_coverage": round(linked / len(hypotheses), 3) if hypotheses else 0.0,
        "unsupported_claim_count": unsupported,
        "invalid_citation_count": invalid,
        "diversity": final_diversity,
        "prediction_specificity": (comparison.get("prediction_specificity") or {}).get("final"),
        "falsification_plan_quality": (comparison.get("falsification_plan_quality") or {}).get("final"),
        "token_total": usage.get("total_tokens"),
        "model_call_count": usage.get("call_count", len(calls)),
        "structured_output_failures": usage.get("structured_output_failures", 0),
        "repair_attempts": usage.get("repair_attempts", 0),
        "run_status": manifest.get("run_status", "complete"),
    }


def _schema_valid_rate(calls: list[dict[str, Any]]) -> float | None:
    if not calls:
        return None
    successes = sum(1 for call in calls if call.get("structured_output_status") == "success")
    return round(successes / len(calls), 3)


def _read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []
