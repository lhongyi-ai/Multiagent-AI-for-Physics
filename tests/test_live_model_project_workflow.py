from __future__ import annotations

import json
from pathlib import Path

import httpx

from coscientist.cli import main
from coscientist.pilot.artifacts import validate_v1_artifacts
import coscientist.pilot.runner as runner
from coscientist.providers.openai_compatible import OpenAICompatibleProvider


PROJECT = "research-projects/code_assistant_fixture/project.yaml"


def test_run_project_requires_explicit_live_model_for_openai(tmp_path: Path) -> None:
    assert main([
        "run-project",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--provider",
        "openai",
    ]) == 2


def test_env_file_is_gitignored() -> None:
    assert ".env" in Path(".gitignore").read_text().splitlines()


def test_live_model_smoke_workflow_uses_mocked_http_and_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    fake_key = "test-secret-key"
    responses = iter([
        _chat(_hypothesis_batch()),
        _chat(_review_batch()),
        _chat(_ranking_batch()),
        _chat({"comparisons": []}),
    ])
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        assert request.headers["authorization"] == f"Bearer {fake_key}"
        return next(responses)

    def provider_factory() -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            api_key=fake_key,
            model="openrouter/test-model",
            base_url="https://openrouter.ai/api/v1",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
            max_repair_attempts=0,
        )

    monkeypatch.setattr(runner, "OpenAICompatibleProvider", provider_factory)
    assert main([
        "run-project",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--run-id",
        "live-smoke",
        "--provider",
        "openai",
        "--live-model",
        "--literature-mode",
        "fixture",
        "--smoke",
    ]) == 0
    run_dir = tmp_path / "live-smoke"
    assert validate_v1_artifacts(run_dir) == []
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["live_model_enabled"] is True
    assert manifest["model_mode"] == "live"
    assert manifest["model_provider"] == "openai"
    calls = (run_dir / "model_calls.jsonl").read_text()
    assert calls.count("\n") == 3
    assert fake_key not in calls
    usage = json.loads((run_dir / "model_usage.json").read_text())
    assert usage["call_count"] == 3
    assert usage["total_tokens"] == 30
    assert captured_urls == ["https://openrouter.ai/api/v1/chat/completions"] * 3

    assert main([
        "run-project",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--run-id",
        "mock-smoke",
        "--smoke",
    ]) == 0
    assert main([
        "compare-model-runs",
        str(tmp_path / "mock-smoke"),
        str(run_dir),
    ]) == 0
    assert (run_dir / "model_run_comparison.json").exists()
    assert "systems-test baseline" in (run_dir / "model_run_comparison.md").read_text()


def test_live_model_dry_run_writes_no_model_calls(tmp_path: Path) -> None:
    assert main([
        "run-project",
        PROJECT,
        "--runs-dir",
        str(tmp_path),
        "--run-id",
        "dry",
        "--provider",
        "openai",
        "--live-model",
        "--literature-mode",
        "fixture",
        "--dry-run",
    ]) == 0
    assert (tmp_path / "dry" / "model_calls.jsonl").read_text() == ""


def _chat(content: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "openrouter/test-model",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
    })


def _hypothesis_batch() -> dict:
    return {
        "hypotheses": [{
            "id": "hyp-live-smoke",
            "title": "Hybrid retrieval smoke hypothesis",
            "core_claim": "Hybrid retrieval may improve code assistant context relevance under a fixed budget.",
            "mechanism": "Lexical retrieval preserves exact symbols while semantic retrieval adds related examples.",
            "assumptions": ["The fixture corpus is incomplete."],
            "supporting_evidence": ["paper-code-lexical-hybrid"],
            "contradicting_evidence": ["paper-code-reranking"],
            "novelty_statement": "This is a smoke-test hypothesis, not a novelty claim.",
            "testable_predictions": ["Hybrid retrieval should improve relevance in a held-out fixture comparison."],
            "falsification_criteria": ["Hybrid retrieval performs no better than lexical-only retrieval."],
            "proposed_experiments": ["Run a fixed-budget retrieval ablation."],
            "uncertainty": 0.42,
            "generation_strategy": "mechanistic",
            "parent_ids": [],
            "version": 1,
            "status": "active",
            "change_summary": None,
            "evidence_links": [],
        }],
    }


def _review_batch() -> dict:
    return {
        "reviews": [{
            "hypothesis_id": "hyp-live-smoke",
            "fatal_flaws": [],
            "nonfatal_weaknesses": ["Needs a precise relevance metric."],
            "unsupported_claims": ["Production improvement is not established by the fixture."],
            "conflicting_evidence": ["Reranking can suppress rare useful examples."],
            "novelty_concerns": ["Novelty is unverified."],
            "suggested_repairs": ["Add a lexical-only baseline."],
            "recommendation": "keep",
            "confidence": 0.62,
        }],
    }


def _ranking_batch() -> dict:
    return {
        "rankings": [{
            "hypothesis_id": "hyp-live-smoke",
            "correctness": 6.0,
            "novelty": 5.0,
            "testability": 8.0,
            "explanatory_power": 6.5,
            "feasibility": 8.0,
            "discriminative_power": 7.0,
            "evidence_quality": 5.5,
            "impact": 6.0,
            "parsimony": 7.0,
            "weighted_total": 6.6,
            "pairwise_wins": 0,
            "pairwise_losses": 0,
            "judge_notes": ["Smoke ranking."],
        }],
    }
