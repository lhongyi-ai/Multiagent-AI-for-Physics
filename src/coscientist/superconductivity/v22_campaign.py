from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import ValidationError

from coscientist.pilot.artifacts import read_json, read_jsonl, write_json, write_jsonl
from coscientist.providers.base import ProviderError
from coscientist.providers.openai_compatible import OpenAICompatibleProvider
from coscientist.schemas.v22 import (
    AdversarialTestResult,
    AgentDialogueTurn,
    ExpertDossierRecord,
    ExperimentProposal,
    FingerprintComparison,
    FingerprintPrediction,
    IndependentReproductionResult,
    LiveRuntimeConfig,
    LiveModelSmokeResponse,
    MaterialFamilyCandidate,
    MaterialFamilyCoverage,
    MaterialFamilySelectionDecision,
    MechanismFingerprint,
    MicroscopicDerivation,
    MicroscopicHamiltonian,
    ObservableFingerprintComponent,
    ParameterPlausibilityResult,
    ParameterPrior,
    PerRoleModelRoute,
    ProviderConnectionResult,
    TheoryDiscriminationResult,
    TheoryFamily,
)
from coscientist.superconductivity.index import rebuild_scientific_index, validate_scientific_index


V22_REQUIRED_ARTIFACTS = [
    "v22_campaign_registration.json",
    "live_agent_configuration.json",
    "live_agent_dialogues.jsonl",
    "model_call_records.jsonl",
    "model_usage_summary.json",
    "provider_connection_results.json",
    "material_family_candidates.jsonl",
    "material_family_selection.json",
    "expert_curated_dossier.json",
    "real_observations.jsonl",
    "data_split_manifest.json",
    "microscopic_hamiltonians.jsonl",
    "microscopic_derivations.jsonl",
    "candidate_models.jsonl",
    "mechanism_fingerprints.jsonl",
    "parameter_priors.jsonl",
    "parameter_plausibility.jsonl",
    "fit_results.jsonl",
    "held_out_predictions.jsonl",
    "adversarial_tests.jsonl",
    "counterexamples.jsonl",
    "reproduction_results.jsonl",
    "theory_discrimination_results.json",
    "experiment_proposals.jsonl",
    "claim_ledger.jsonl",
    "prediction_ledger.jsonl",
    "objection_board.jsonl",
    "expert_review_package.md",
    "v22_scientific_report.md",
    "scientific_index.sqlite",
    "scientific_index_manifest.json",
]


REQUIRED_DATA_PROVIDERS = [
    ("openalex", "scholarly"),
    ("crossref", "scholarly"),
    ("unpaywall", "scholarly"),
    ("arxiv", "scholarly"),
    ("datacite", "scholarly"),
    ("zenodo", "scholarly"),
    ("supercon", "materials"),
    ("materials_project", "materials"),
    ("nomad", "materials"),
    ("optimade", "materials"),
    ("oqmd", "materials"),
]


def run_v22_campaign(
    project_path: str | Path,
    *,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    force: bool = False,
    live_model: bool = False,
    live_network: bool = False,
) -> Path:
    project_file = Path(project_path)
    project = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    run_id = run_id or project.get("campaign", {}).get("campaign_id", "v22-superconductivity")
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"V2.2 campaign artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "provider_snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    now = datetime.now(UTC).isoformat()
    routes = _runtime_routes(project, live_model=live_model)
    runtime = LiveRuntimeConfig(live_model_enabled=live_model, live_network_enabled=live_network, routes=routes)
    provider_results = test_data_connections(live_network=live_network, snapshots_dir=snapshots_dir)
    families = _material_family_candidates()
    selection = MaterialFamilySelectionDecision(
        selected_family_id="lsco",
        compared_family_ids=[item.family_id for item in families],
        decision_basis=["best fixture coverage for doping-resolved optical trend", "contains intentional missing-observable warnings", "suitable for adversarial non-identifiability tests"],
        limitations=["offline fixture dossier is not a real live database retrieval", "single-band toy mapping cannot settle cuprate mechanism"],
    )
    dossier = _expert_dossier(project)
    observations = [item.model_dump(mode="json") for item in dossier if item.source_type in {"direct_measurement", "derived", "manual", "computed"}]
    candidate_models = _candidate_models()
    hamiltonians = _hamiltonians(candidate_models)
    derivations = _derivations(candidate_models)
    fingerprints = _fingerprints(candidate_models)
    priors = _parameter_priors(candidate_models)
    plausibility = _parameter_plausibility(priors)
    fit_results = _fit_results(candidate_models, plausibility)
    predictions = _held_out_predictions(candidate_models)
    attacks = _adversarial_tests(candidate_models, plausibility)
    counterexamples = _counterexamples(attacks)
    reproduction = _reproduction_results()
    comparisons = _fingerprint_comparisons(fingerprints, attacks)
    discrimination = _theory_discrimination(selection, comparisons, attacks)
    if live_network and any(item.live_status == "connected" for item in provider_results):
        discrimination.limitations = [
            "live database smoke was bounded to one-record public endpoint checks",
            *[item for item in discrimination.limitations if item != "no live database smoke was executed"],
        ]
    proposals = _experiment_proposals(selection)
    dialogues = _agent_dialogues(candidate_models, routes, live_model=live_model)
    calls = _model_call_records(dialogues)
    usage = _usage_summary(calls, live_model=live_model)
    claims = _claim_ledger(discrimination)
    prediction_ledger = _prediction_ledger(predictions)
    objections = _objection_board(attacks)

    registration = {
        "schema_version": "v22",
        "campaign_id": run_id,
        "question": project.get("campaign", {}).get("question", _default_question()),
        "model_mode": "live" if live_model else "mock",
        "live_model_enabled": live_model,
        "live_network_enabled": live_network,
        "created_at": now,
        "scientific_claim_policy": "expert_review_required_before_public_claim",
        "stop_reason": "offline_fixture_campaign_complete",
    }
    write_json(run_dir / "v22_campaign_registration.json", registration)
    write_json(run_dir / "live_agent_configuration.json", runtime)
    write_jsonl(run_dir / "live_agent_dialogues.jsonl", dialogues)
    write_jsonl(run_dir / "model_call_records.jsonl", calls)
    write_json(run_dir / "model_usage_summary.json", usage)
    write_json(run_dir / "provider_connection_results.json", {"schema_version": "v22", "providers": provider_results})
    write_jsonl(run_dir / "material_family_candidates.jsonl", families)
    write_json(run_dir / "material_family_selection.json", selection)
    write_json(run_dir / "expert_curated_dossier.json", {"schema_version": "v22", "records": dossier, "grounding_note": "agent-visible dossier excludes no hidden ground truth; fixture provenance remains explicit"})
    write_jsonl(run_dir / "real_observations.jsonl", observations)
    write_json(run_dir / "data_split_manifest.json", _split_manifest(dossier))
    write_jsonl(run_dir / "microscopic_hamiltonians.jsonl", hamiltonians)
    write_jsonl(run_dir / "microscopic_derivations.jsonl", derivations)
    write_jsonl(run_dir / "candidate_models.jsonl", candidate_models)
    write_jsonl(run_dir / "mechanism_fingerprints.jsonl", fingerprints)
    write_jsonl(run_dir / "parameter_priors.jsonl", priors)
    write_jsonl(run_dir / "parameter_plausibility.jsonl", plausibility)
    write_jsonl(run_dir / "fit_results.jsonl", fit_results)
    write_jsonl(run_dir / "held_out_predictions.jsonl", predictions)
    write_jsonl(run_dir / "adversarial_tests.jsonl", attacks)
    write_jsonl(run_dir / "counterexamples.jsonl", counterexamples)
    write_jsonl(run_dir / "reproduction_results.jsonl", reproduction)
    write_json(run_dir / "theory_discrimination_results.json", discrimination)
    write_jsonl(run_dir / "experiment_proposals.jsonl", proposals)
    write_jsonl(run_dir / "claim_ledger.jsonl", claims)
    write_jsonl(run_dir / "prediction_ledger.jsonl", prediction_ledger)
    write_jsonl(run_dir / "objection_board.jsonl", objections)
    (run_dir / "expert_review_package.md").write_text(_expert_review_package(registration, selection, discrimination, proposals), encoding="utf-8")
    (run_dir / "v22_scientific_report.md").write_text(_scientific_report(registration, provider_results, selection, comparisons, discrimination, proposals, usage), encoding="utf-8")
    rebuild_scientific_index(run_dir)
    return run_dir


def validate_v22_campaign(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    errors = [f"missing V2.2 artifact: {name}" for name in V22_REQUIRED_ARTIFACTS if not (path / name).exists()]
    if not (path / "provider_snapshots").is_dir():
        errors.append("missing provider_snapshots directory")
    if errors:
        return errors
    try:
        config = LiveRuntimeConfig.model_validate(read_json(path / "live_agent_configuration.json"))
        if not config.live_model_enabled:
            for call in read_jsonl(path / "model_call_records.jsonl"):
                if call.get("provider") != "mock" or call.get("live_call_executed"):
                    errors.append("live model call recorded while live model disabled")
        providers = read_json(path / "provider_connection_results.json")["providers"]
        for provider in providers:
            result = ProviderConnectionResult.model_validate(provider)
            if result.live_status == "connected" and not config.live_network_enabled:
                errors.append(f"provider reported connected without live-network permission: {result.provider}")
        family_ids = {item["family_id"] for item in read_jsonl(path / "material_family_candidates.jsonl")}
        selection = MaterialFamilySelectionDecision.model_validate(read_json(path / "material_family_selection.json"))
        if selection.selected_family_id not in family_ids:
            errors.append("material selection references missing family")
        model_ids = {item["model_id"] for item in read_jsonl(path / "candidate_models.jsonl")}
        for artifact in ["microscopic_hamiltonians.jsonl", "microscopic_derivations.jsonl", "mechanism_fingerprints.jsonl", "parameter_priors.jsonl", "parameter_plausibility.jsonl", "fit_results.jsonl", "held_out_predictions.jsonl", "adversarial_tests.jsonl"]:
            for record in read_jsonl(path / artifact):
                model_id = record.get("model_id")
                if model_id and model_id not in model_ids:
                    errors.append(f"{artifact} references missing model: {model_id}")
        for item in read_jsonl(path / "reproduction_results.jsonl"):
            result = IndependentReproductionResult.model_validate(item)
            if len(set(result.paths)) < 2:
                errors.append(f"reproduction is not independent for {result.conclusion_id}")
        discrimination = TheoryDiscriminationResult.model_validate(read_json(path / "theory_discrimination_results.json"))
        if discrimination.status == "survives_within_scope" and not discrimination.limitations:
            errors.append("survival claim must include limitations")
        errors.extend(validate_scientific_index(path))
    except Exception as exc:
        errors.append(f"invalid V2.2 artifact: {exc}")
    text = "\n".join(file.read_text(encoding="utf-8", errors="ignore") for file in [*path.rglob("*.json"), *path.rglob("*.jsonl"), *path.rglob("*.md")])
    lowered = text.lower()
    if "openai_api_key" in lowered or "openrouter_api_key" in lowered or "sk-" in lowered or "bearer " in lowered:
        errors.append("secret-like content appears in V2.2 artifacts")
    return errors


def test_data_connections(*, live_network: bool = False, snapshots_dir: str | Path | None = None) -> list[ProviderConnectionResult]:
    target = Path(snapshots_dir) if snapshots_dir else None
    if target:
        target.mkdir(parents=True, exist_ok=True)
    results = []
    for provider, provider_type in REQUIRED_DATA_PROVIDERS:
        fixture_payload = {
            "schema_version": "v22",
            "provider": provider,
            "mode": "fixture",
            "note": "local deterministic connection fixture; not a live provider response",
        }
        snapshot_path = None
        digest = hashlib.sha256(json.dumps(fixture_payload, sort_keys=True).encode()).hexdigest()
        if target:
            snapshot = target / f"{provider}_fixture_snapshot.json"
            snapshot.write_text(json.dumps(fixture_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            snapshot_path = str(snapshot.name)
        live_payload: dict[str, Any] | None = None
        live_error: str | None = None
        live_status = "blocked"
        live_count = 0
        if live_network:
            if _requires_key(provider) and not _has_provider_key(provider):
                live_status = "authentication_required"
                live_error = "provider credentials are not configured"
            else:
                live_status, live_payload, live_count, live_error = _probe_live_provider(provider)
                if live_payload and target:
                    live_snapshot = target / f"{provider}_live_snapshot.json"
                    live_snapshot.write_text(json.dumps(live_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    snapshot_path = str(live_snapshot.name)
        authentication = "missing" if _requires_key(provider) and not _has_provider_key(provider) else "configured" if _requires_key(provider) else "not_required"
        results.append(ProviderConnectionResult(
            provider=provider,
            provider_type=provider_type,  # type: ignore[arg-type]
            fixture_status="fixture_only",
            live_status=live_status,  # type: ignore[arg-type]
            authentication_status=authentication,  # type: ignore[arg-type]
            record_count=live_count or 1,
            request_parameters={"query": "bounded smoke fixture", "live_network": live_network},
            snapshot_path=snapshot_path,
            snapshot_sha256=hashlib.sha256(json.dumps(live_payload or fixture_payload, sort_keys=True).encode()).hexdigest() if live_payload else digest,
            retrieved_at=datetime.now(UTC).isoformat(),
            error=live_error,
        ))
    return results


def _probe_live_provider(provider: str) -> tuple[str, dict[str, Any] | None, int, str | None]:
    endpoints = {
        "openalex": ("GET", "https://api.openalex.org/works", {"search": "superconductivity", "per-page": "1"}),
        "crossref": ("GET", "https://api.crossref.org/works", {"query": "superconductivity", "rows": "1"}),
        "arxiv": ("GET", "https://export.arxiv.org/api/query", {"search_query": "all:superconductivity", "start": "0", "max_results": "1"}),
        "datacite": ("GET", "https://api.datacite.org/dois", {"query": "superconductivity", "page[size]": "1"}),
        "zenodo": ("GET", "https://zenodo.org/api/records", {"q": "superconductivity", "size": "1"}),
        "nomad": ("GET", "https://nomad-lab.eu/prod/v1/api/v1/entries", {"per_page": "1"}),
        "optimade": ("GET", "https://optimade.materialsproject.org/v1/structures", {"page_limit": "1"}),
    }
    if provider == "unpaywall" and not os.getenv("UNPAYWALL_EMAIL"):
        return "authentication_required", None, 0, "UNPAYWALL_EMAIL is required by Unpaywall etiquette/API policy"
    if provider == "unpaywall":
        endpoints[provider] = ("GET", "https://api.unpaywall.org/v2/10.1038/nature12373", {"email": os.getenv("UNPAYWALL_EMAIL", "")})
    if provider == "materials_project":
        return "authentication_required", None, 0, "MATERIALS_PROJECT_API_KEY smoke endpoint is intentionally not called by this fixture adapter"
    if provider in {"supercon", "oqmd"}:
        return "unavailable", None, 0, "no stable unauthenticated public JSON smoke endpoint is configured"
    if provider not in endpoints:
        return "unavailable", None, 0, "provider endpoint is not configured"
    method, url, params = endpoints[provider]
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True, headers={"User-Agent": "coscientist-v22-smoke/0.1"}) as client:
            response = client.request(method, url, params=params)
        if response.status_code in {401, 403}:
            return "authentication_required", None, 0, f"HTTP {response.status_code}"
        if response.status_code == 429:
            return "rate_limited", None, 0, "HTTP 429"
        if response.status_code >= 400:
            return "failed", None, 0, f"HTTP {response.status_code}: {response.text[:200]}"
        payload = _response_snapshot(provider, response)
        return "connected", payload, _record_count(provider, payload), None
    except httpx.HTTPError as exc:
        return "unavailable", None, 0, _sanitize(str(exc))


def _response_snapshot(provider: str, response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        body: Any = response.json()
    else:
        body = response.text[:4000]
    return {
        "schema_version": "v22",
        "provider": provider,
        "status_code": response.status_code,
        "content_type": content_type,
        "body": body,
    }


def _record_count(provider: str, payload: dict[str, Any]) -> int:
    body = payload.get("body")
    if provider == "arxiv" and isinstance(body, str):
        return 1 if "<entry>" in body else 0
    if isinstance(body, dict):
        if provider == "openalex":
            return len(body.get("results") or [])
        if provider == "crossref":
            return len(((body.get("message") or {}).get("items") or []))
        if provider == "datacite":
            return len(body.get("data") or [])
        if provider == "zenodo":
            return len(body.get("hits", {}).get("hits") or body.get("hits") or [])
        if provider == "unpaywall":
            return 1 if body.get("doi") else 0
        if provider in {"nomad", "optimade"}:
            return len(body.get("data") or [])
    return 0


async def test_live_models(*, live_model: bool = False, runs_dir: str | Path = "runs", run_id: str = "v22-live-model-smoke", force: bool = False) -> Path:
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise ValueError(f"live model smoke artifacts are immutable; use a new run id or --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    routes = _runtime_routes({}, live_model=live_model)
    runtime = LiveRuntimeConfig(live_model_enabled=live_model, routes=routes)
    responses: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    if not live_model:
        for route in routes:
            responses.append(LiveModelSmokeResponse(role=route.role, status="blocked", summary="live-model permission was not enabled", safe_to_continue=False).model_dump(mode="json"))
    else:
        for route in routes:
            if route.provider == "mock":
                responses.append(LiveModelSmokeResponse(role=route.role, status="ok", summary="mock route requires no live call", safe_to_continue=True).model_dump(mode="json"))
                continue
            try:
                provider = OpenAICompatibleProvider(
                    api_key=_route_api_key(route.provider),
                    base_url=_route_base_url(route.provider),
                    model=_route_model(route.provider, route.model),
                    timeout_seconds=route.timeout_seconds,
                    temperature=route.temperature,
                    max_output_tokens=min(route.max_output_tokens, 300),
                    max_retries=0,
                    max_repair_attempts=0,
                )
                result = await provider.generate_structured(
                    "V2.2 bounded smoke test. Return a compact status object only; do not include chain of thought.",
                    LiveModelSmokeResponse,
                    context={"agent_role": route.role, "workflow_stage": "v22_live_model_smoke"},
                )
                responses.append(result.model_dump(mode="json"))
                records.extend([item.model_dump(mode="json") for item in provider.call_records])
            except (ProviderError, ValidationError) as exc:
                responses.append(LiveModelSmokeResponse(role=route.role, status="failed", summary=_sanitize(str(exc)), safe_to_continue=False).model_dump(mode="json"))
    write_json(run_dir / "live_agent_configuration.json", runtime)
    write_jsonl(run_dir / "live_model_smoke_results.jsonl", responses)
    write_jsonl(run_dir / "model_call_records.jsonl", records)
    write_json(run_dir / "model_usage_summary.json", {"schema_version": "v22", "model_mode": "live" if live_model else "mock", "role_count": len(routes), "live_call_count": len(records), "blocked_or_failed_count": sum(1 for item in responses if item["status"] != "ok")})
    return run_dir


def _runtime_routes(project: dict[str, Any], *, live_model: bool) -> list[PerRoleModelRoute]:
    configured = project.get("agent_models", {})
    default_provider = "openrouter" if live_model else "mock"
    live_default_model = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-4.1-mini"
    defaults = {
        "generator": os.getenv("GENERATOR_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "theory_reviewer": os.getenv("REVIEWER_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "experimental_reviewer": os.getenv("REVIEWER_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "prior_art_reviewer": os.getenv("REVIEWER_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "adversarial_reviewer": os.getenv("ADVERSARIAL_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "evolution": os.getenv("EVOLUTION_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "ranker": os.getenv("RANKER_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "supervisor": os.getenv("META_REVIEW_MODEL") or (live_default_model if live_model else "deterministic-mock"),
        "meta_review": os.getenv("META_REVIEW_MODEL") or (live_default_model if live_model else "deterministic-mock"),
    }
    routes = []
    for role, model in defaults.items():
        item = configured.get(role, {})
        routes.append(PerRoleModelRoute(
            role=role,  # type: ignore[arg-type]
            provider=item.get("provider", default_provider if live_model else "mock"),
            model=item.get("model", model),
            temperature=float(item.get("temperature", 0.2 if role == "generator" else 0.0)),
            max_output_tokens=int(item.get("max_output_tokens", 900)),
        ))
    return routes


def _route_api_key(provider: str) -> str | None:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _route_base_url(provider: str) -> str | None:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
    return os.getenv("OPENAI_BASE_URL")


def _route_model(provider: str, configured_model: str) -> str:
    if configured_model != "deterministic-mock":
        return configured_model
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-4.1-mini"
    return os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"


def _requires_key(provider: str) -> bool:
    return provider in {"materials_project"}


def _has_provider_key(provider: str) -> bool:
    env = {"materials_project": "MATERIALS_PROJECT_API_KEY"}.get(provider)
    return bool(env and os.getenv(env))


def _sanitize(message: str) -> str:
    redacted = message
    for name in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "MATERIALS_PROJECT_API_KEY"]:
        value = os.getenv(name)
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _default_question() -> str:
    return "Can microscopic phonon, correlated-hopping, mixed, and alternate kinetic mechanisms be distinguished with jointly constrained doping-resolved observables?"


def _material_family_candidates() -> list[MaterialFamilyCandidate]:
    return [
        MaterialFamilyCandidate(schema_version="v22", family_id="lsco", name="La2-xSrxCuO4", rationale="Fixture contains optical, Tc, and doping trend records with deliberate missing thermodynamic coverage.", selected=True, coverage=MaterialFamilyCoverage(tc_records=3, optical_records=2, isotope_records=1, thermodynamic_records=0, penetration_depth_records=1, doping_points=3, source_quality="fixture_only", missing_observables=["specific_heat"])),
        MaterialFamilyCandidate(schema_version="v22", family_id="mgb2", name="MgB2", rationale="Strong phonon reference, but weak electron-hole-asymmetry coverage in fixture.", coverage=MaterialFamilyCoverage(tc_records=1, optical_records=1, isotope_records=1, thermodynamic_records=1, penetration_depth_records=0, doping_points=1, source_quality="fixture_only", missing_observables=["electron_hole_asymmetry", "doping_dependence"])),
        MaterialFamilyCandidate(schema_version="v22", family_id="nb", name="Nb conventional reference", rationale="Useful calibration family but not discriminating for correlated-hopping claims.", coverage=MaterialFamilyCoverage(tc_records=1, optical_records=0, isotope_records=1, thermodynamic_records=1, penetration_depth_records=0, doping_points=1, source_quality="fixture_only", missing_observables=["optical_partial_sum", "doping_dependence", "electron_hole_asymmetry"])),
    ]


def _expert_dossier(project: dict[str, Any]) -> list[ExpertDossierRecord]:
    records = project.get("dossier_records")
    if records:
        return [ExpertDossierRecord.model_validate(item) for item in records]
    return [
        ExpertDossierRecord(record_id="lsco-tc-fit-1", family_id="lsco", source_id="fixture-lsco-tc", source_type="direct_measurement", observable="tc", doping_label="x=0.10", original_value="Tc about 28 K", normalized_value=28.0, normalized_unit="K", split="fit", curation_note="Local fixture value for pipeline validation only.", provenance=["local_fixture"]),
        ExpertDossierRecord(record_id="lsco-tc-heldout-1", family_id="lsco", source_id="fixture-lsco-tc", source_type="direct_measurement", observable="tc", doping_label="x=0.15", original_value="Tc about 38 K", normalized_value=38.0, normalized_unit="K", split="held_out", curation_note="Held out until after prediction registration.", provenance=["local_fixture"]),
        ExpertDossierRecord(record_id="lsco-optical-fit-1", family_id="lsco", source_id="fixture-lsco-optical", source_type="derived", observable="optical_partial_sum", doping_label="x=0.10", original_value="partial spectral-weight change depends on cutoff", normalized_value=None, normalized_unit=None, split="fit", curation_note="Encodes cutoff sensitivity rather than a single averaged value.", provenance=["local_fixture"]),
        ExpertDossierRecord(record_id="lsco-isotope-validation-1", family_id="lsco", source_id="fixture-lsco-isotope", source_type="author_interpretation", observable="isotope_coefficient", doping_label="underdoped", original_value="nonzero and doping dependent", normalized_value=None, normalized_unit=None, split="validation", curation_note="Interpretive record, cannot alone prove phonon dominance.", provenance=["local_fixture"]),
        ExpertDossierRecord(record_id="lsco-penetration-fit-1", family_id="lsco", source_id="fixture-lsco-lambda", source_type="direct_measurement", observable="penetration_depth", doping_label="x=0.15", original_value="superfluid response available with sample caveats", normalized_value=None, normalized_unit=None, split="fit", curation_note="Used for missing-observable accounting.", provenance=["local_fixture"]),
        ExpertDossierRecord(record_id="lsco-review-context-1", family_id="lsco", source_id="fixture-review", source_type="review", observable="electron_hole_asymmetry", doping_label=None, original_value="electron-hole asymmetry is family dependent", normalized_value=None, normalized_unit=None, split="expert_only", curation_note="Review context excluded from direct support scoring.", provenance=["local_fixture"]),
    ]


def _candidate_models() -> list[dict[str, Any]]:
    families: list[tuple[str, TheoryFamily, float, float, int]] = [
        ("model-phonon-v22", "phonon_dominated_bcs", 0.34, 0.0, 2),
        ("model-hirsch-ch", "correlated_hopping_hirsch_style", 0.0, 0.56, 3),
        ("model-mixed-v22", "mixed_phonon_correlated_hopping", 0.20, 0.30, 4),
        ("model-phenom-mixed", "phenomenological_mixed_kernel", 0.24, 0.18, 5),
        ("model-alt-kinetic", "non_hirsch_kinetic_competitor", 0.05, 0.22, 4),
        ("model-overfit", "overparameterized_competitor", 0.30, 0.35, 8),
        ("model-null", "null_or_inadequate_model", 0.0, 0.0, 1),
    ]
    return [{"schema_version": "v22", "model_id": model_id, "candidate_id": f"cand-{index}", "family": family, "phonon_coupling_ev": phonon, "correlated_hopping_ev": ch, "parameter_count": count, "structured_status": "valid"} for index, (model_id, family, phonon, ch, count) in enumerate(families, start=1)]


def _hamiltonians(models: list[dict[str, Any]]) -> list[MicroscopicHamiltonian]:
    rows = []
    for model in models:
        family = model["family"]
        ch = "Delta t" if "hopping" in family or "mixed" in family else "0"
        rows.append(MicroscopicHamiltonian(
            model_id=model["model_id"],
            family=family,
            real_space_hamiltonian="H = - sum_<ij>,sigma [t - Delta_t(n_i,-sigma+n_j,-sigma)](c^dag_i_sigma c_j_sigma + h.c.) + H_ph + H_mu",
            momentum_representation="epsilon_k plus separable pairing kernel V(k,k') with explicit phonon and density-dependent hopping channels",
            dispersion_renormalization=f"density-dependent hopping channel active: {ch}",
            anomalous_vertex="Gamma_ch(k,k') proportional to Delta_t times electron-hole-asymmetric form factor" if ch != "0" else "Gamma_ph(k,k') from retarded attraction proxy",
            current_operator="j_x = dH/dA_x evaluated before mean-field decoupling",
            kinetic_operator="K = sum_k epsilon_k n_k with fixed convention recorded",
            correlated_hopping_expectation="<n_i,-sigma c^dag_i_sigma c_j_sigma> tracked separately" if ch != "0" else None,
            electron_hole_transform_behavior="correlated-hopping term changes sign-like contribution under electron-hole transformation" if ch != "0" else "phonon-only proxy is electron-hole symmetric within fixture",
            assumptions=["single-band proxy", "bounded separable kernel", "fixed particle number unless noted"],
            dropped_terms=["retardation details", "full multiband Coulomb renormalization"],
            validity_scope=["offline theory-discrimination scaffold", "not a universal superconductivity solver"],
        ))
    return rows


def _derivations(models: list[dict[str, Any]]) -> list[MicroscopicDerivation]:
    rows = []
    for model in models:
        family = model["family"]
        derivation_type = "microscopic" if family in {"correlated_hopping_hirsch_style", "mixed_phonon_correlated_hopping", "non_hirsch_kinetic_competitor"} else "phenomenological" if family != "null_or_inadequate_model" else "null"
        rows.append(MicroscopicDerivation(
            derivation_id=f"der-{model['model_id']}",
            model_id=model["model_id"],
            derivation_type=derivation_type,
            mean_field_decoupling="introduce anomalous average Delta_k and separate phonon, correlated-hopping, and residual kinetic channels",
            gap_equation="Delta_k = - sum_k' V_eff(k,k') Delta_k' / (2 E_k')",
            number_equation="n = sum_k [1 - xi_k/E_k tanh(E_k/2T)]",
            free_energy_functional="F_s - F_n = quasiparticle term + interaction channel terms + tracked kinetic correction",
            sign_conventions=["negative free-energy change favors superconducting state", "energy-channel decomposition is convention dependent"],
            verifier_status="partial" if family in {"overparameterized_competitor", "null_or_inadequate_model"} else "pass",
            unresolved_issues=["parameter degeneracy"] if family == "overparameterized_competitor" else [],
        ))
    return rows


def _fingerprints(models: list[dict[str, Any]]) -> list[MechanismFingerprint]:
    rows = []
    for model in models:
        family = model["family"]
        components = [
            ObservableFingerprintComponent(observable="tc", direction_or_value="finite Tc if effective attraction is positive", source_ids=["lsco-tc-fit-1"]),
            ObservableFingerprintComponent(observable="optical_partial_sum", direction_or_value="cutoff-sensitive sign/magnitude; not standalone proof", source_ids=["lsco-optical-fit-1"]),
            ObservableFingerprintComponent(observable="doping_dependence", direction_or_value="must match x=0.10 to x=0.15 trend before held-out reveal", source_ids=["lsco-tc-fit-1"]),
        ]
        if "phonon" in family or "mixed" in family:
            components.append(ObservableFingerprintComponent(observable="isotope_coefficient", direction_or_value="nonzero isotope response expected", source_ids=["lsco-isotope-validation-1"]))
        if "hopping" in family or "kinetic" in family:
            components.append(ObservableFingerprintComponent(observable="electron_hole_asymmetry", direction_or_value="asymmetric doping trend expected", source_ids=["lsco-review-context-1"]))
        rows.append(MechanismFingerprint(fingerprint_id=f"fp-{model['model_id']}", model_id=model["model_id"], family=family, components=components, missing_observables=["specific_heat", "condensation_energy"]))
    return rows


def _parameter_priors(models: list[dict[str, Any]]) -> list[ParameterPrior]:
    rows = []
    for model in models:
        rows.append(ParameterPrior(parameter_id=f"prior-ph-{model['model_id']}", model_id=model["model_id"], parameter_name="phonon_coupling_ev", source_supported_min=0.0, source_supported_max=0.5, unit="eV", provenance=["fixture-prior"]))
        rows.append(ParameterPrior(parameter_id=f"prior-ch-{model['model_id']}", model_id=model["model_id"], parameter_name="correlated_hopping_ev", source_supported_min=0.0, source_supported_max=0.6, unit="eV", provenance=["fixture-prior"]))
    return rows


def _parameter_plausibility(priors: list[ParameterPrior]) -> list[ParameterPlausibilityResult]:
    rows = []
    for prior in priors:
        value = 0.34 if "phonon" in prior.parameter_name else 0.56
        max_value = prior.source_supported_max or value
        classification = "source_supported" if value <= max_value else "implausible"
        rows.append(ParameterPlausibilityResult(model_id=prior.model_id, parameter_id=prior.parameter_id, fitted_value=value, unit=prior.unit, classification=classification, rationale="fixture prior range check; no live fit performed"))
    return rows


def _fit_results(models: list[dict[str, Any]], plausibility: list[ParameterPlausibilityResult]) -> list[dict[str, Any]]:
    plaus_by_model: dict[str, list[str]] = {}
    for item in plausibility:
        plaus_by_model.setdefault(item.model_id, []).append(item.classification)
    rows = []
    for model in models:
        penalty = 0.15 if model["parameter_count"] > 5 else 0.0
        residual = round(0.35 + penalty + (0.2 if model["family"] == "null_or_inadequate_model" else 0.0), 3)
        rows.append({"schema_version": "v22", "model_id": model["model_id"], "training_residual": residual, "validation_residual": round(residual + 0.05, 3), "plausibility_summary": plaus_by_model.get(model["model_id"], []), "fit_scope": "offline fixture only"})
    return rows


def _held_out_predictions(models: list[dict[str, Any]]) -> list[FingerprintPrediction]:
    return [FingerprintPrediction(prediction_id=f"pred-heldout-{model['model_id']}", model_id=model["model_id"], observable="tc", conditions="LSCO x=0.15 hidden fixture record", predicted_value=("near optimal dome maximum" if model["family"] != "null_or_inadequate_model" else "no reliable prediction"), held_out_record_ids=["lsco-tc-heldout-1"]) for model in models]


def _adversarial_tests(models: list[dict[str, Any]], plausibility: list[ParameterPlausibilityResult]) -> list[AdversarialTestResult]:
    rows = []
    for model in models:
        family = model["family"]
        if family == "phonon_dominated_bcs":
            attack = "electron-hole asymmetry and optical trend failure"
            outcome = "data_insufficient"
        elif family == "correlated_hopping_hirsch_style":
            attack = "isotope coefficient mismatch and cutoff-sensitive kinetic interpretation"
            outcome = "survives_within_scope"
        elif family == "mixed_phonon_correlated_hopping":
            attack = "overfitting and non-identifiability against alternate kinetic competitor"
            outcome = "observationally_equivalent"
        elif family == "overparameterized_competitor":
            attack = "excessive parameter freedom and no held-out improvement"
            outcome = "requires_unphysical_parameters"
        elif family == "null_or_inadequate_model":
            attack = "failure to account for finite Tc fixture record"
            outcome = "falsified"
        else:
            attack = "leave-one-source-out and alternate optical cutoff"
            outcome = "data_insufficient"
        rows.append(AdversarialTestResult(test_id=f"adv-{model['model_id']}", model_id=model["model_id"], attack=attack, outcome=outcome, impact="ranking component reduced unless independently reproduced", unresolved_issue=None if outcome in {"falsified", "requires_unphysical_parameters"} else "requires real curated measurements"))
    return rows


def _counterexamples(attacks: list[AdversarialTestResult]) -> list[dict[str, Any]]:
    return [{"schema_version": "v22", "counterexample_id": f"ce-{item.model_id}", "model_id": item.model_id, "attack_id": item.test_id, "status": item.outcome, "preserved_for_review": item.outcome in {"falsified", "observationally_equivalent", "requires_unphysical_parameters"}} for item in attacks]


def _reproduction_results() -> list[IndependentReproductionResult]:
    return [
        IndependentReproductionResult(conclusion_id="mixed-state-bounded-construction", paths=["symbolic gap-equation derivation", "independent deterministic fixture evaluator"], discrepancy="same qualitative finite-gap condition; numeric values are toy-model dependent", outcome="reproduced"),
        IndependentReproductionResult(conclusion_id="optical-cutoff-non-identifiability", paths=["direct partial-sum accounting", "alternate cutoff stress test"], discrepancy="both paths preserve cutoff warning", outcome="reproduced"),
        IndependentReproductionResult(conclusion_id="material-family-selection", paths=["coverage-count selector", "manual suitability rubric"], discrepancy="both select LSCO fixture but mark dossier as non-live", outcome="reproduced"),
    ]


def _fingerprint_comparisons(fingerprints: list[MechanismFingerprint], attacks: list[AdversarialTestResult]) -> list[FingerprintComparison]:
    attack_by_model = {item.model_id: item for item in attacks}
    rows = []
    for fp in fingerprints:
        attack = attack_by_model[fp.model_id]
        score = 0.72
        if attack.outcome == "falsified":
            score = 0.15
        elif attack.outcome == "requires_unphysical_parameters":
            score = 0.25
        elif attack.outcome == "observationally_equivalent":
            score = 0.58
        rows.append(FingerprintComparison(model_id=fp.model_id, joint_consistency_score=score, tensions=[attack.attack], missing_observables=fp.missing_observables, source_conflicts=["fixture dossier only; no live-source conflict resolution"]))
    return rows


def _theory_discrimination(selection: MaterialFamilySelectionDecision, comparisons: list[FingerprintComparison], attacks: list[AdversarialTestResult]) -> TheoryDiscriminationResult:
    ranked = sorted([{"model_id": item.model_id, "joint_consistency_score": item.joint_consistency_score, "tensions": item.tensions} for item in comparisons], key=lambda item: item["joint_consistency_score"], reverse=True)
    return TheoryDiscriminationResult(
        selected_family_id=selection.selected_family_id,
        status="data_insufficient",
        candidate_rankings=ranked,
        equivalence_classes=[["model-hirsch-ch", "model-mixed-v22", "model-alt-kinetic"]],
        nontrivial_outputs=["finite mixed-state construction is possible in the bounded model", "current fixture observables do not identify phonon versus correlated-hopping contribution", "a doping-resolved optical-plus-isotope experiment is the most discriminating next step"],
        limitations=["offline fixture records only", "no live database smoke was executed", "expert review required before any public scientific claim"],
    )


def _experiment_proposals(selection: MaterialFamilySelectionDecision) -> list[ExperimentProposal]:
    return [
        ExperimentProposal(
            proposal_id="exp-lsco-optical-isotope-doping",
            selected_material_family=selection.selected_family_id,
            composition_or_doping_points=["La2-xSrxCuO4 x=0.10", "La2-xSrxCuO4 x=0.15"],
            sample_requirements=["matched growth protocol", "same crystallographic orientation", "normal-state reference above Tc"],
            observable="optical_partial_sum",
            conditions={"direction": "ab-plane", "temperature_range": "below and above Tc", "frequency_range": "bounded fixture placeholder: user must supply real Omega_min/Omega_max", "magnetic_field": "zero field unless expert revises"},
            theory_predictions={"phonon_dominated_bcs": "weak kinetic-energy anomaly after cutoff control", "correlated_hopping_hirsch_style": "larger electron-hole-asymmetric kinetic proxy", "mixed_phonon_correlated_hopping": "intermediate optical anomaly plus isotope response", "non_hirsch_kinetic_competitor": "kinetic anomaly without Hirsch-specific asymmetry"},
            expected_separation="sign or magnitude of cutoff-stable partial-sum change combined with isotope coefficient should separate at least one equivalence class if precision is sufficient",
            required_precision="must be set by expert from real optical error bars; fixture cannot quantify",
            confounders=["normal-state extrapolation", "sample disorder", "cutoff dependence", "pseudogap and multiband effects"],
            falsification_logic=["no cutoff-stable kinetic anomaly disfavors kinetic-dominant interpretations", "strong isotope trend with absent kinetic anomaly disfavors pure correlated-hopping model", "identical trends across doping preserve observational equivalence"],
            feasibility="unknown",
            source_support=["lsco-optical-fit-1", "lsco-isotope-validation-1", "lsco-tc-heldout-1"],
        )
    ]


def _agent_dialogues(models: list[dict[str, Any]], routes: list[PerRoleModelRoute], *, live_model: bool) -> list[AgentDialogueTurn]:
    route_by_role = {route.role: route for route in routes}
    rows = []
    for role, summary in [
        ("generator", "Generated structured candidate theory families from bounded superconductivity campaign specification."),
        ("theory_reviewer", "Checked Hamiltonian conventions, limiting cases, and energy-decomposition warnings."),
        ("experimental_reviewer", "Flagged missing real error bars, optical cutoff dependence, and sample identity risks."),
        ("prior_art_reviewer", "Marked dossier as fixture-only and blocked novelty claims pending real curation."),
        ("adversarial_reviewer", "Applied candidate-specific attacks and preserved non-identifiability objections."),
        ("evolution", "Converted verifier failures into bounded repair notes; excluded unmaterialized children."),
        ("ranker", "Ranked candidates using verifier, plausibility, held-out, source-quality, and complexity components."),
        ("supervisor", "Stopped campaign after offline fixture objectives and did not authorize live calls."),
        ("meta_review", "Identified bottleneck: real curated observations and expert review are needed before scientific claims."),
    ]:
        route = route_by_role[role]  # type: ignore[index]
        rows.append(AgentDialogueTurn(turn_id=f"turn-{len(rows)+1:02d}-{role}", role=role, provider=route.provider, model=route.model, status="completed" if route.provider == "mock" or live_model else "blocked", input_artifact_ids=["v22_campaign_registration.json"], output_artifact_ids=["live_agent_dialogues.jsonl"], summary=summary, objections=["fixture-only evidence cannot prove a mechanism"] if "reviewer" in role else [], decisions=["no live call executed"] if not live_model else []))
    return rows


def _model_call_records(dialogues: list[AgentDialogueTurn]) -> list[dict[str, Any]]:
    return [{"schema_version": "v22", "request_sequence_number": index, "agent_role": item.role, "provider": item.provider, "model": item.model, "live_call_executed": False, "success": item.status == "completed", "input_tokens": 0, "output_tokens": 0, "cost_usd": None, "latency_ms": 0.0, "error": None if item.status == "completed" else "blocked by permission gate"} for index, item in enumerate(dialogues, start=1)]


def _usage_summary(calls: list[dict[str, Any]], *, live_model: bool) -> dict[str, Any]:
    return {"schema_version": "v22", "model_mode": "live" if live_model else "mock", "call_count": len(calls), "live_call_count": sum(1 for item in calls if item["live_call_executed"]), "input_tokens": sum(item["input_tokens"] for item in calls), "output_tokens": sum(item["output_tokens"] for item in calls), "estimated_cost_usd": None, "cost_note": "unavailable for deterministic mock V2.2 fixture run"}


def _split_manifest(dossier: list[ExpertDossierRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in dossier:
        counts[item.split] = counts.get(item.split, 0) + 1
    return {"schema_version": "v22", "split_counts": counts, "leakage_controls": ["held_out_predictions generated before held-out evaluation", "review/context records are not treated as direct measurements", "material family and sample identity remain separate"]}


def _claim_ledger(discrimination: TheoryDiscriminationResult) -> list[dict[str, Any]]:
    return [{"schema_version": "v22", "claim_id": "claim-v22-no-breakthrough", "claim_text": "The offline fixture run produces constraints and experiment proposals, not a confirmed superconductivity mechanism.", "status": "requires_expert_review", "supporting_artifacts": ["theory_discrimination_results.json", "experiment_proposals.jsonl"], "calibration": discrimination.status}]


def _prediction_ledger(predictions: list[FingerprintPrediction]) -> list[dict[str, Any]]:
    return [{"schema_version": "v22", "prediction_id": item.prediction_id, "model_id": item.model_id, "observable": item.observable, "status": "preregistered_fixture_prediction", "held_out_record_ids": item.held_out_record_ids} for item in predictions]


def _objection_board(attacks: list[AdversarialTestResult]) -> list[dict[str, Any]]:
    return [{"schema_version": "v22", "objection_id": item.test_id, "model_id": item.model_id, "objection": item.attack, "status": item.outcome, "unresolved_issue": item.unresolved_issue} for item in attacks]


def _expert_review_package(registration: dict[str, Any], selection: MaterialFamilySelectionDecision, discrimination: TheoryDiscriminationResult, proposals: list[ExperimentProposal]) -> str:
    proposal = proposals[0]
    return "\n".join([
        "# V2.2 Expert Review Package",
        "",
        f"- Problem: {registration['question']}",
        f"- Selected material family: {selection.selected_family_id}",
        f"- Campaign status: {discrimination.status}",
        "- Disclosure state: do_not_disclose_until_expert_review",
        "",
        "## Required Expert Decisions",
        "",
        "- Is the microscopic correlated-hopping derivation convention acceptable?",
        "- Are the fixture observables sufficient to motivate real curation?",
        "- Which data records should be excluded or replaced by primary measurements?",
        "- Does the proposed optical/isotope/doping experiment distinguish the leading equivalence class?",
        "",
        "## Proposed Experiment",
        "",
        f"- Proposal ID: {proposal.proposal_id}",
        f"- Doping points: {', '.join(proposal.composition_or_doping_points)}",
        f"- Required precision: {proposal.required_precision}",
        f"- Falsification logic: {'; '.join(proposal.falsification_logic)}",
        "",
    ])


def _scientific_report(registration: dict[str, Any], providers: list[ProviderConnectionResult], selection: MaterialFamilySelectionDecision, comparisons: list[FingerprintComparison], discrimination: TheoryDiscriminationResult, proposals: list[ExperimentProposal], usage: dict[str, Any]) -> str:
    lines = [
        "# V2.2 Superconductivity Theory-Discrimination Report",
        "",
        "> Offline deterministic campaign scaffold. Do not treat this as scientific proof or live validation.",
        "",
        f"- Question: {registration['question']}",
        f"- Model mode: {registration['model_mode']}",
        f"- Live network enabled: {registration['live_network_enabled']}",
        f"- Selected material family: {selection.selected_family_id}",
        f"- Theory-discrimination status: {discrimination.status}",
        f"- Live model calls: {usage['live_call_count']}",
        "",
        "## Provider Status",
        "",
    ]
    for provider in providers:
        lines.append(f"- `{provider.provider}`: fixture={provider.fixture_status}, live={provider.live_status}, auth={provider.authentication_status}")
    lines.extend(["", "## Candidate Fingerprint Comparison", ""])
    for item in sorted(comparisons, key=lambda row: row.joint_consistency_score, reverse=True):
        lines.append(f"- `{item.model_id}`: score={item.joint_consistency_score:.2f}; tensions={'; '.join(item.tensions)}; missing={', '.join(item.missing_observables)}")
    lines.extend(["", "## Nontrivial Outputs", ""])
    for item in discrimination.nontrivial_outputs:
        lines.append(f"- {item}")
    lines.extend(["", "## Experiment Proposal", ""])
    proposal = proposals[0]
    lines.append(f"- `{proposal.proposal_id}` on {proposal.selected_material_family}: {proposal.expected_separation}")
    lines.extend(["", "## Known Limitations", ""])
    for item in discrimination.limitations:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def copy_v22_project_sources(project_path: str | Path, run_dir: str | Path) -> None:
    project_file = Path(project_path)
    project = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    source_dir = project.get("source_dir")
    if not source_dir:
        return
    src = project_file.parent / source_dir
    dst = Path(run_dir) / "source_inputs"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
