from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from coscientist.pilot.artifacts import write_json, write_jsonl


REQUIRED_PHASE2_OBSERVABLES = [
    "tc_k",
    "gap_ev",
    "penetration_depth_nm",
    "isotope_alpha",
    "optical_spectral_weight_proxy",
]

TIER_A_REQUIRED_OBSERVABLE = "tc_k"
TIER_A_MECHANISM_SENSITIVE_OBSERVABLES = ["isotope_alpha", "optical_spectral_weight_proxy"]
TIER_B_OPTIONAL_OBSERVABLES = [
    "gap_ev",
    "penetration_depth_nm",
    "isotope_alpha",
    "optical_spectral_weight_proxy",
    "condensation_energy_ev_per_formula_unit",
]
TIER_C_RECOMMENDED_OBSERVABLES = ["condensation_energy_ev_per_formula_unit"]


def run_phase2_data_coverage_tool(run_dir: str | Path, *, source_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    observations, import_status = load_phase2_observations(source_path)
    coverage = phase2_coverage(observations)
    missing = phase2_missing_observables(observations)
    candidate_sources = phase2_candidate_sources(coverage)
    overall_status = coverage["overall_readiness_status"]
    evaluation = {
        "schema_version": "v24-phase2-data-tool",
        "tool_name": "phase2_data_coverage_tool",
        "status": overall_status,
        "import_status": import_status,
        "coverage": coverage,
        "required_observables": REQUIRED_PHASE2_OBSERVABLES,
        "tiered_requirements": {
            "tier_a_minimal_material_trend": {
                "requires": [TIER_A_REQUIRED_OBSERVABLE, f"one_of:{TIER_A_MECHANISM_SENSITIVE_OBSERVABLES}"],
                "claim_scope": "qualitative mechanism-sensitive trend only",
            },
            "tier_b_local_multi_observable": {
                "requires": ["tc_k", "at least two mechanism/consistency observables at representative dopings"],
                "claim_scope": "local multi-observable consistency; not full energy decomposition",
            },
            "tier_c_full_quantitative_separation": {
                "requires": REQUIRED_PHASE2_OBSERVABLES,
                "recommended": TIER_C_RECOMMENDED_OBSERVABLES,
                "claim_scope": "full material-level quantitative separation only after complete coverage and validation",
            },
        },
        "can_run_material_level_comparison": coverage["status"] == "sufficient",
        "can_run_partial_material_trend": coverage["tier_a"]["status"] in {"pass", "partial"} or coverage["tier_b"]["status"] in {"pass", "partial"},
        "can_run_local_multi_observable_check": coverage["tier_b"]["status"] in {"pass", "partial"},
        "can_claim_full_quantitative_separation": coverage["tier_c"]["status"] == "pass",
        "conclusion": (
            "Complete Tier C data are available for full held-out material-level comparison."
            if coverage["tier_c"]["status"] == "pass"
            else "Partial LSCO evidence can support scoped Tier A/B checks, but full quantitative material-level separation remains blocked."
            if coverage["tier_a"]["status"] in {"pass", "partial"} or coverage["tier_b"]["status"] in {"pass", "partial"}
            else "Material-level comparison remains blocked until mechanism-sensitive observables are imported."
        ),
    }
    write_jsonl(root / "phase2_imported_observations.jsonl", observations)
    write_json(root / "phase2_data_coverage.json", coverage)
    write_jsonl(root / "phase2_missing_observables.jsonl", missing)
    write_jsonl(root / "phase2_candidate_data_sources.jsonl", candidate_sources)
    write_json(root / "phase2_data_tool_evaluation.json", evaluation)
    (root / "phase2_data_template.csv").write_text(phase2_template_csv(), encoding="utf-8")
    return root


def validate_phase2_data_coverage_tool(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    required = [
        "phase2_imported_observations.jsonl",
        "phase2_data_coverage.json",
        "phase2_missing_observables.jsonl",
        "phase2_candidate_data_sources.jsonl",
        "phase2_data_tool_evaluation.json",
        "phase2_data_template.csv",
    ]
    errors = [f"missing phase2 data artifact: {name}" for name in required if not (root / name).exists()]
    if errors:
        return errors
    evaluation = json.loads((root / "phase2_data_tool_evaluation.json").read_text(encoding="utf-8"))
    if evaluation.get("status") not in {
        "sufficient_for_held_out_comparison",
        "partial_tier_a_minimal_material_trend",
        "partial_tier_b_local_multi_observable",
        "blocked_insufficient_existing_data",
    }:
        errors.append("invalid phase2 data tool status")
    return errors


def load_phase2_observations(source_path: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_path is None or not str(source_path).strip():
        return [], {"source_path": "", "status": "no_source_supplied", "message": "No local Phase 2 data file was supplied."}
    path = Path(source_path).expanduser()
    if not path.exists():
        return [], {"source_path": str(path), "status": "missing_source", "message": "The supplied Phase 2 data file does not exist."}
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        return [], {"source_path": str(path), "status": "unsupported_format", "message": "Use CSV or JSONL for Phase 2 observations."}
    normalized = [_normalize_observation(row, index) for index, row in enumerate(rows, start=1)]
    usable = [row for row in normalized if row["usable_for_fit"]]
    return normalized, {"source_path": str(path), "status": "imported", "row_count": len(normalized), "usable_row_count": len(usable)}


def phase2_coverage(observations: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in observations if row.get("usable_for_fit", True)]
    by_family: dict[str, dict[str, set[str]]] = {}
    by_family_material: dict[str, dict[tuple[str, str], set[str]]] = {}
    splits = {"train": 0, "validation": 0, "test": 0}
    for row in usable:
        family = str(row.get("material_family") or "unknown")
        material_id = str(row.get("material_id") or "unknown")
        doping = str(row.get("doping") or "unknown")
        observable = str(row.get("observable") or "")
        by_family.setdefault(family, {}).setdefault(doping, set()).add(observable)
        by_family_material.setdefault(family, {}).setdefault((material_id, doping), set()).add(observable)
        split = str(row.get("split") or "train")
        if split in splits:
            splits[split] += 1
    complete_dopings_by_family = {
        family: sorted(doping for doping, observed in doping_map.items() if set(REQUIRED_PHASE2_OBSERVABLES).issubset(observed))
        for family, doping_map in by_family.items()
    }
    complete_count = sum(len(dopings) for dopings in complete_dopings_by_family.values())
    has_train = splits["train"] >= 2
    has_held_out = splits["validation"] + splits["test"] >= 1
    has_series = any(len(dopings) >= 3 for dopings in complete_dopings_by_family.values())
    status = "sufficient" if has_series and has_train and has_held_out else "insufficient_existing_data"
    tier_a = _tier_a_status(by_family)
    tier_b = _tier_b_status(by_family, by_family_material)
    tier_c = {
        "schema_version": "v24-phase2-data-tool",
        "tier": "tier_c_full_quantitative_separation",
        "status": "pass" if status == "sufficient" else "blocked",
        "complete_doping_point_count": complete_count,
        "complete_dopings_by_family": complete_dopings_by_family,
        "required_observables": REQUIRED_PHASE2_OBSERVABLES,
        "recommended_observables": TIER_C_RECOMMENDED_OBSERVABLES,
        "claim_scope": "full quantitative separation across doping remains unavailable unless this tier passes",
    }
    overall_status = _overall_tier_status(tier_a, tier_b, tier_c)
    return {
        "schema_version": "v24-phase2-data-tool",
        "status": status,
        "overall_readiness_status": overall_status,
        "usable_observation_count": len(usable),
        "complete_doping_point_count": complete_count,
        "complete_dopings_by_family": complete_dopings_by_family,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "tier_c": tier_c,
        "split_counts": splits,
        "requires_at_least_three_complete_doping_points": not has_series,
        "requires_train_rows": not has_train,
        "requires_held_out_rows": not has_held_out,
    }


def _tier_a_status(by_family: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    qualifying: dict[str, list[str]] = {}
    partial: dict[str, list[str]] = {}
    for family, doping_map in by_family.items():
        for doping, observed in doping_map.items():
            has_tc = TIER_A_REQUIRED_OBSERVABLE in observed
            has_mechanism = bool(set(TIER_A_MECHANISM_SENSITIVE_OBSERVABLES).intersection(observed))
            if has_tc and has_mechanism:
                qualifying.setdefault(family, []).append(doping)
            elif has_tc or has_mechanism:
                partial.setdefault(family, []).append(doping)
    qualifying = {family: sorted(dopings) for family, dopings in qualifying.items()}
    partial = {family: sorted(dopings) for family, dopings in partial.items()}
    max_count = max((len(dopings) for dopings in qualifying.values()), default=0)
    status = "pass" if max_count >= 3 else "partial" if max_count >= 1 else "blocked"
    return {
        "schema_version": "v24-phase2-data-tool",
        "tier": "tier_a_minimal_material_trend",
        "status": status,
        "qualifying_dopings_by_family": qualifying,
        "partial_dopings_by_family": partial,
        "minimum_required_doping_points": 3,
        "required_observables": [TIER_A_REQUIRED_OBSERVABLE],
        "one_of_observables": TIER_A_MECHANISM_SENSITIVE_OBSERVABLES,
        "claim_scope": "qualitative trend comparison only; cannot decompose energy contributions",
    }


def _tier_b_status(
    by_family: dict[str, dict[str, set[str]]],
    by_family_material: dict[str, dict[tuple[str, str], set[str]]],
) -> dict[str, Any]:
    qualifying_family_dopings: dict[str, list[str]] = {}
    same_material_qualifying: dict[str, list[dict[str, str]]] = {}
    for family, doping_map in by_family.items():
        for doping, observed in doping_map.items():
            optional_count = len(set(TIER_B_OPTIONAL_OBSERVABLES).intersection(observed))
            if TIER_A_REQUIRED_OBSERVABLE in observed and optional_count >= 2:
                qualifying_family_dopings.setdefault(family, []).append(doping)
    for family, material_map in by_family_material.items():
        for (material_id, doping), observed in material_map.items():
            optional_count = len(set(TIER_B_OPTIONAL_OBSERVABLES).intersection(observed))
            if TIER_A_REQUIRED_OBSERVABLE in observed and optional_count >= 2:
                same_material_qualifying.setdefault(family, []).append({"material_id": material_id, "doping": doping})
    qualifying_family_dopings = {family: sorted(dopings) for family, dopings in qualifying_family_dopings.items()}
    max_family_count = max((len(dopings) for dopings in qualifying_family_dopings.values()), default=0)
    max_same_material_count = max((len(rows) for rows in same_material_qualifying.values()), default=0)
    status = "pass" if max_family_count >= 3 else "partial" if max_family_count >= 1 else "blocked"
    return {
        "schema_version": "v24-phase2-data-tool",
        "tier": "tier_b_local_multi_observable",
        "status": status,
        "qualifying_dopings_by_family": qualifying_family_dopings,
        "same_material_qualifying_points": same_material_qualifying,
        "same_material_complete_count": max_same_material_count,
        "minimum_required_doping_points": 3,
        "required_observables": ["tc_k", "at_least_two_of_optional_observables"],
        "optional_observables": TIER_B_OPTIONAL_OBSERVABLES,
        "claim_scope": "local multi-observable consistency; cross-source family-level merging must remain provenance-aware",
    }


def _overall_tier_status(tier_a: dict[str, Any], tier_b: dict[str, Any], tier_c: dict[str, Any]) -> str:
    if tier_c["status"] == "pass":
        return "sufficient_for_held_out_comparison"
    if tier_b["status"] in {"pass", "partial"}:
        return "partial_tier_b_local_multi_observable"
    if tier_a["status"] in {"pass", "partial"}:
        return "partial_tier_a_minimal_material_trend"
    return "blocked_insufficient_existing_data"


def phase2_missing_observables(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [row for row in observations if row.get("usable_for_fit", True)]
    if not usable:
        return [
            {
                "schema_version": "v24-phase2-data-tool",
                "material_family": "unknown",
                "material_id": "unknown",
                "doping": "unknown",
                "missing_observables": REQUIRED_PHASE2_OBSERVABLES,
                "reason": "no usable Phase 2 observations imported",
            }
        ]
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in usable:
        key = (str(row.get("material_family") or "unknown"), str(row.get("material_id") or "unknown"), str(row.get("doping") or "unknown"))
        grouped.setdefault(key, set()).add(str(row.get("observable") or ""))
    rows = []
    required = set(REQUIRED_PHASE2_OBSERVABLES)
    for (family, material_id, doping), observed in sorted(grouped.items()):
        missing = sorted(required - observed)
        if missing:
            rows.append(
                {
                    "schema_version": "v24-phase2-data-tool",
                    "material_family": family,
                    "material_id": material_id,
                    "doping": doping,
                    "missing_observables": missing,
                    "observed_observables": sorted(observed),
                    "reason": "incomplete doping point",
                }
            )
    return rows


def phase2_candidate_sources(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    status = "not_needed" if coverage["status"] == "sufficient" else "needed"
    return [
        {"schema_version": "v24-phase2-data-tool", "source_kind": "SuperCon", "status": status, "can_help_with": ["tc_k"], "cannot_supply_alone": REQUIRED_PHASE2_OBSERVABLES[1:]},
        {"schema_version": "v24-phase2-data-tool", "source_kind": "curated_spectroscopy_table", "status": status, "can_help_with": ["gap_ev"], "cannot_supply_alone": ["tc_k", "penetration_depth_nm", "isotope_alpha", "optical_spectral_weight_proxy"]},
        {"schema_version": "v24-phase2-data-tool", "source_kind": "curated_muSR_or_penetration_depth_table", "status": status, "can_help_with": ["penetration_depth_nm"], "cannot_supply_alone": ["gap_ev", "isotope_alpha", "optical_spectral_weight_proxy"]},
        {"schema_version": "v24-phase2-data-tool", "source_kind": "curated_isotope_effect_table", "status": status, "can_help_with": ["isotope_alpha"], "cannot_supply_alone": ["gap_ev", "penetration_depth_nm", "optical_spectral_weight_proxy"]},
        {"schema_version": "v24-phase2-data-tool", "source_kind": "curated_optical_conductivity_table", "status": status, "can_help_with": ["optical_spectral_weight_proxy"], "cannot_supply_alone": ["gap_ev", "penetration_depth_nm", "isotope_alpha"]},
    ]


def phase2_template_csv() -> str:
    return "\n".join(
        [
            "observation_id,material_family,material_id,doping,observable,value,unit,uncertainty,split,source_id,provenance,usable_for_fit",
            "obs-1,cuprate,LSCO,x=0.15,tc_k,38,K,,train,local-curated,local table,true",
            "obs-2,cuprate,LSCO,x=0.15,gap_ev,0.012,eV,,train,local-curated,local table,true",
            "obs-3,cuprate,LSCO,x=0.15,penetration_depth_nm,250,nm,,train,local-curated,local table,true",
            "obs-4,cuprate,LSCO,x=0.15,isotope_alpha,0.08,dimensionless,,train,local-curated,local table,true",
            "obs-5,cuprate,LSCO,x=0.15,optical_spectral_weight_proxy,0.02,relative,,test,local-curated,local table,true",
        ]
    ) + "\n"


def _normalize_observation(row: dict[str, Any], index: int) -> dict[str, Any]:
    value = row.get("value", "")
    try:
        parsed_value: float | str = float(value)
    except (TypeError, ValueError):
        parsed_value = str(value)
    usable = row.get("usable_for_fit", True)
    if isinstance(usable, str):
        usable = usable.strip().lower() not in {"false", "0", "no", "n"}
    return {
        "schema_version": "v24-phase2-data-tool",
        "observation_id": str(row.get("observation_id") or f"phase2-obs-{index}"),
        "material_family": str(row.get("material_family") or "unknown"),
        "material_id": str(row.get("material_id") or "unknown"),
        "doping": str(row.get("doping") or "unknown"),
        "observable": str(row.get("observable") or "").strip(),
        "value": parsed_value,
        "unit": str(row.get("unit") or ""),
        "uncertainty": row.get("uncertainty") or None,
        "split": str(row.get("split") or "train").strip().lower(),
        "source_id": str(row.get("source_id") or "local"),
        "provenance": str(row.get("provenance") or "user/local supplied"),
        "usable_for_fit": bool(usable),
    }
